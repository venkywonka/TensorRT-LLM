# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for ``find_mm_token_lengths`` routing through
``get_num_tokens_per_video(video_grid_thw=...)``.

Design under test (Option Y.a in PR #12944 review resolution):

- ``__call__`` places only ``video_grid_thw`` (already produced by the HF
  processor) into ``multimodal_data["video"]``.
- ``find_mm_token_lengths`` iterates ``video_grid_thw`` row-by-row and calls
  ``input_processor.get_num_tokens_per_video(video=item, video_grid_thw=row)``
  for each video — the input processor's method is the canonical compute
  path, not a bypass-able fallback.
- When ``video_grid_thw`` is not present (or shape-mismatched), the fast path
  falls back to calling the method without ``video_grid_thw`` — the method's
  slow path handles it (e.g. by running the HF processor directly).

This routing keeps all state thread-safe: per-request ``video_grid_thw``
lives in ``multimodal_data`` and is passed explicitly, never shared via
instance mutation on ``input_processor``.
"""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm.inputs.multimodal import find_mm_token_lengths


def _make_mock_processor(image_count: int = 100, slow_path_video_count: int = 999) -> Mock:
    """Build a mock input processor with video/image counters.

    The video counter returns ``t*h*w`` when ``video_grid_thw`` is passed
    (stand-in for the real ``t * (h//merge) * (w//merge)`` formula — we don't
    need to exercise the real formula here, just verify the routing).
    When ``video_grid_thw`` is not passed, returns the fixed
    ``slow_path_video_count`` so the slow-path branch is visibly distinct.
    """
    processor = Mock()
    processor.get_num_tokens_per_image = Mock(return_value=image_count)

    def _count_video(*, video, video_grid_thw=None, **kwargs):
        if video_grid_thw is not None:
            t = int(video_grid_thw[0])
            h = int(video_grid_thw[1])
            w = int(video_grid_thw[2])
            return t * h * w
        return slow_path_video_count

    processor.get_num_tokens_per_video = Mock(side_effect=_count_video)
    return processor


def _fake_pil_image():
    # find_mm_token_lengths only converts torch.Tensor → PIL; any non-Tensor
    # placeholder skips that branch cleanly.
    return object()


def _fake_video(num_frames: int = 4):
    return [_fake_pil_image() for _ in range(num_frames)]


def test_fast_path_routes_through_method_with_video_grid_thw():
    """When ``multimodal_data["video"]["video_grid_thw"]`` is present and
    shape-matched, ``find_mm_token_lengths`` calls
    ``get_num_tokens_per_video`` once per video with the corresponding
    ``video_grid_thw`` row.
    """
    processor = _make_mock_processor()
    mm_data = {"video": [_fake_video(), _fake_video(), _fake_video()]}
    vgt = torch.tensor([[2, 14, 14], [1, 7, 7], [3, 28, 28]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    # One call per video, each with its own video_grid_thw row.
    assert processor.get_num_tokens_per_video.call_count == 3
    for i, call in enumerate(processor.get_num_tokens_per_video.call_args_list):
        assert torch.equal(call.kwargs["video_grid_thw"], vgt[i]), (
            f"call {i} did not receive vgt[{i}]"
        )
    # Counts are derived per-video from the provided rows.
    assert result == {"video": [2 * 14 * 14, 1 * 7 * 7, 3 * 28 * 28]}


def test_fallback_when_multimodal_data_is_none():
    """Back-compat: callers that don't pass ``multimodal_data`` hit the slow
    path — the method is called without ``video_grid_thw``.
    """
    processor = _make_mock_processor(slow_path_video_count=42)
    mm_data = {"video": [_fake_video(), _fake_video()]}

    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"video": [42, 42]}
    assert processor.get_num_tokens_per_video.call_count == 2
    for call in processor.get_num_tokens_per_video.call_args_list:
        assert call.kwargs.get("video_grid_thw") is None


def test_fallback_when_video_grid_thw_missing():
    """``multimodal_data`` without a ``video_grid_thw`` key falls through to
    the slow path.
    """
    processor = _make_mock_processor(slow_path_video_count=77)
    mm_data = {"video": [_fake_video()]}
    multimodal_data = {"image": {"something": "else"}}

    result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    assert result == {"video": [77]}
    processor.get_num_tokens_per_video.assert_called_once()
    assert processor.get_num_tokens_per_video.call_args.kwargs.get("video_grid_thw") is None


def test_fallback_when_video_grid_thw_shape_mismatch():
    """Shape mismatch logs a warning and falls back to the slow path."""
    processor = _make_mock_processor(slow_path_video_count=99)
    mm_data = {"video": [_fake_video(), _fake_video()]}
    # 3 rows for 2 videos — mismatch.
    vgt = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    multimodal_data = {"video": {"video_grid_thw": vgt}}

    with patch("tensorrt_llm.inputs.multimodal.logger.warning") as warn_mock:
        result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)

    # Fell back to per-item call without video_grid_thw.
    assert result == {"video": [99, 99]}
    assert processor.get_num_tokens_per_video.call_count == 2
    for call in processor.get_num_tokens_per_video.call_args_list:
        assert call.kwargs.get("video_grid_thw") is None
    warn_mock.assert_called_once()
    assert "video_grid_thw" in warn_mock.call_args.args[0]


def test_image_only_request_unaffected():
    """Image-only requests don't touch the video fast-path code at all."""
    processor = _make_mock_processor(image_count=128)
    mm_data = {"image": [_fake_pil_image(), _fake_pil_image()]}

    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"image": [128, 128]}
    processor.get_num_tokens_per_video.assert_not_called()


@pytest.mark.parametrize("num_workers", [8, 16])
def test_fast_path_is_thread_safe_under_concurrent_calls(num_workers):
    """Regression for the cache-pop race that the refactor eliminates.

    Each worker submits a uniquely-fingerprinted ``video_grid_thw``; if any
    shared mutable state existed in the fast path, concurrent workers would
    see each other's values. This test asserts each worker reads back exactly
    its own fingerprint's per-video count.
    """
    processor = _make_mock_processor()

    def call(worker_id: int) -> list:
        vgt = torch.tensor(
            [
                [worker_id + 1, 2, 3],
                [worker_id + 1, 5, 7],
            ]
        )
        mm_data = {"video": [_fake_video(), _fake_video()]}
        multimodal_data = {"video": {"video_grid_thw": vgt}}
        result = find_mm_token_lengths(mm_data, processor, multimodal_data=multimodal_data)
        return result["video"]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(pool.map(call, range(num_workers)))

    expected = [[(wid + 1) * 2 * 3, (wid + 1) * 5 * 7] for wid in range(num_workers)]
    assert results == expected
