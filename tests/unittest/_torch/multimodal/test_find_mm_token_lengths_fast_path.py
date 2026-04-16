# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for ``find_mm_token_lengths`` video routing + thread-safety."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm.inputs.multimodal import find_mm_token_lengths


def _make_mock_processor(image_count: int = 100, slow_path_video_count: int = 999) -> Mock:
    """Mock processor; video counter returns ``t*h*w`` when ``video_grid_thw``
    is passed, else ``slow_path_video_count``."""
    processor = Mock()
    processor.get_num_tokens_per_image = Mock(return_value=image_count)

    def _count_video(*, video, video_grid_thw=None, **kwargs):
        if video_grid_thw is not None:
            t, h, w = (int(x) for x in video_grid_thw)
            return t * h * w
        return slow_path_video_count

    processor.get_num_tokens_per_video = Mock(side_effect=_count_video)
    return processor


def _fake_pil_image():
    # Any non-Tensor placeholder skips the torch.Tensor → PIL branch.
    return object()


def _fake_video(num_frames: int = 4):
    return [_fake_pil_image() for _ in range(num_frames)]


def test_fast_path_routes_through_method_with_video_grid_thw():
    processor = _make_mock_processor()
    mm_data = {"video": [_fake_video(), _fake_video(), _fake_video()]}
    vgt = torch.tensor([[2, 14, 14], [1, 7, 7], [3, 28, 28]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    assert processor.get_num_tokens_per_video.call_count == 3
    for i, call in enumerate(processor.get_num_tokens_per_video.call_args_list):
        assert torch.equal(call.kwargs["video_grid_thw"], vgt[i])
    assert result == {"video": [2 * 14 * 14, 1 * 7 * 7, 3 * 28 * 28]}


def test_fallback_when_multimodal_data_is_none():
    processor = _make_mock_processor(slow_path_video_count=42)
    mm_data = {"video": [_fake_video(), _fake_video()]}

    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"video": [42, 42]}
    assert processor.get_num_tokens_per_video.call_count == 2
    for call in processor.get_num_tokens_per_video.call_args_list:
        assert call.kwargs.get("video_grid_thw") is None


def test_fallback_when_video_grid_thw_missing():
    processor = _make_mock_processor(slow_path_video_count=77)
    mm_data = {"video": [_fake_video()]}
    multimodal_data = {"image": {"something": "else"}}

    result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    assert result == {"video": [77]}
    processor.get_num_tokens_per_video.assert_called_once()
    assert processor.get_num_tokens_per_video.call_args.kwargs.get("video_grid_thw") is None


def test_fallback_when_video_grid_thw_shape_mismatch():
    processor = _make_mock_processor(slow_path_video_count=99)
    mm_data = {"video": [_fake_video(), _fake_video()]}
    # 3 rows, 2 videos — mismatch falls back and warns.
    vgt = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    with patch("tensorrt_llm.inputs.multimodal.logger.warning") as warn_mock:
        result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    assert result == {"video": [99, 99]}
    assert processor.get_num_tokens_per_video.call_count == 2
    for call in processor.get_num_tokens_per_video.call_args_list:
        assert call.kwargs.get("video_grid_thw") is None
    warn_mock.assert_called_once()
    assert "video_grid_thw" in warn_mock.call_args.args[0]


def test_image_only_request_unaffected():
    processor = _make_mock_processor(image_count=128)
    mm_data = {"image": [_fake_pil_image(), _fake_pil_image()]}

    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"image": [128, 128]}
    processor.get_num_tokens_per_video.assert_not_called()


@pytest.mark.parametrize("num_workers", [8, 16])
def test_fast_path_is_thread_safe_under_concurrent_calls(num_workers):
    """Regression for the cache-pop race: each worker's video_grid_thw
    fingerprint must come back unmodified under concurrent dispatch."""
    processor = _make_mock_processor()

    def call(worker_id: int) -> list:
        vgt = torch.tensor([[worker_id + 1, 2, 3], [worker_id + 1, 5, 7]])
        mm_data = {"video": [_fake_video(), _fake_video()]}
        multimodal_data = {"video": {"video_grid_thw": vgt}}
        result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)
        return result["video"]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(call, range(num_workers)))

    expected = [[(wid + 1) * 2 * 3, (wid + 1) * 5 * 7] for wid in range(num_workers)]
    assert results == expected
