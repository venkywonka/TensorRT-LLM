# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for `find_mm_token_lengths` video routing."""

from unittest.mock import Mock, patch

import torch

from tensorrt_llm.inputs.multimodal import find_mm_token_lengths


def _make_mock_processor(image_count: int = 100, slow_path_video_count: int = 999) -> Mock:
    """Mock processor; video counter returns `t*h*w` when `video_grid_thw`
    is passed, else `slow_path_video_count`."""
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
    # 3 videos, 3 grid rows — one call per video.
    mm_data = {"video": [_fake_video(), _fake_video(), _fake_video()]}
    vgt = torch.tensor([[2, 14, 14], [1, 7, 7], [3, 28, 28]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    assert processor.get_num_tokens_per_video.call_count == 3
    for i, call in enumerate(processor.get_num_tokens_per_video.call_args_list):
        assert torch.equal(call.kwargs["video_grid_thw"], vgt[i])
    assert result == {"video": torch.prod(vgt, dim=1).tolist()}


def test_fallback_when_multimodal_data_is_none():
    processor = _make_mock_processor(slow_path_video_count=42)
    # 2 videos, no hint — slow-path count returned twice.
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


def test_mixed_image_and_video_isolates_vgt_to_video_branch():
    # Regression guard: `video_grid_thw` must not leak into the image branch.
    # 2 images + 2 videos — one call per item on each branch.
    processor = _make_mock_processor(image_count=128)
    mm_data = {
        "image": [_fake_pil_image(), _fake_pil_image()],
        "video": [_fake_video(), _fake_video()],
    }
    vgt = torch.tensor([[2, 14, 14], [1, 7, 7]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    result = find_mm_token_lengths(mm_data,
                                   processor,
                                   multimodal_data=multimodal_data)

    assert result == {
        "image": [128, 128],
        "video": torch.prod(vgt, dim=1).tolist(),
    }
    assert processor.get_num_tokens_per_image.call_count == 2
    for call in processor.get_num_tokens_per_image.call_args_list:
        assert "video_grid_thw" not in call.kwargs
    assert processor.get_num_tokens_per_video.call_count == 2
    for i, call in enumerate(processor.get_num_tokens_per_video.call_args_list):
        assert torch.equal(call.kwargs["video_grid_thw"], vgt[i])
