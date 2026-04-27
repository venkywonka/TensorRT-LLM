# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalRuntimeData cumsum math and the flat-mask producer."""

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData, find_mm_token_lengths
from tensorrt_llm.inputs.registry import maybe_compute_mm_embed_cumsum
from tensorrt_llm.inputs.utils import VideoData


def test_maybe_compute_mm_embed_cumsum_populates_py_multimodal_data():
    """Producer writes a flat int64 cumsum tensor at py_multimodal_data[multimodal_embed_mask_cumsum]."""

    class FakeProcessor:
        def get_vocab_size(self):
            return 1000

        def get_mm_token_ids(self):
            return None

        def get_mm_special_token_ids(self):
            return torch.tensor([2000])

    # [text, img, img, special, img, img, img, text]
    prompt_token_ids = [10, 1001, 1002, 2000, 1003, 1004, 1005, 20]
    extra = {"multimodal_data": {}}
    maybe_compute_mm_embed_cumsum(prompt_token_ids, extra, FakeProcessor())

    cumsum = extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
    assert torch.equal(
        cumsum,
        torch.tensor([0, 1, 2, 2, 3, 4, 5, 5], dtype=torch.int64),
    )


def test_runtime_data_cumsum_math_simplest():
    """All-True mask, full request, no cache."""
    is_embed = torch.ones(5, dtype=torch.bool)
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=5,
        embed_mask_cumsum=is_embed.to(torch.int64).cumsum(0),
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 5
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_chunk():
    """Chunk ends before end of mask."""
    is_embed = torch.tensor([True, True, False, True, True, False, True])
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=4,
        embed_mask_cumsum=is_embed.to(torch.int64).cumsum(0),
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_cache():
    """past_seen_token_num > 0: cached counts embeds before watermark."""
    is_embed = torch.tensor([True, True, False, True, True, False, True])
    rt = MultimodalRuntimeData(
        past_seen_token_num=3,
        chunk_end_pos=7,
        embed_mask_cumsum=is_embed.to(torch.int64).cumsum(0),
    )
    assert rt.num_cached_mm_tokens == 2
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_with_specials_mistral_shape():
    """Chunk boundary inside a unit with inline special (Mistral-shape)."""
    # [text, img, img, special, img, img, img, text]
    is_embed = torch.tensor([False, True, True, False, True, True, True, False])
    cumsum = is_embed.to(torch.int64).cumsum(0)

    rt0 = MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5, embed_mask_cumsum=cumsum)
    assert rt0.num_cached_mm_tokens == 0
    assert rt0.num_mm_tokens_in_chunk == 3
    assert rt0.total_embeds_in_request == 5

    rt1 = MultimodalRuntimeData(past_seen_token_num=5, chunk_end_pos=8, embed_mask_cumsum=cumsum)
    assert rt1.num_cached_mm_tokens == 3
    assert rt1.num_mm_tokens_in_chunk == 2
    assert rt1.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_negative_past_seen_rejected():
    """past_seen_token_num must be non-negative."""
    cumsum = torch.arange(1, 6, dtype=torch.int64)
    with pytest.raises(ValueError, match="past_seen_token_num must be non-negative"):
        MultimodalRuntimeData(past_seen_token_num=-1, chunk_end_pos=5, embed_mask_cumsum=cumsum)


def test_runtime_data_requires_cumsum():
    """embed_mask_cumsum is required."""
    with pytest.raises(TypeError):
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)


def _fake_video(num_frames: int = 4):
    """Video must be a list of frames per find_mm_token_lengths contract."""
    return [object() for _ in range(num_frames)]


def test_find_mm_token_lengths_video_data_preserves_frame_list_identity():
    """VideoData frames are passed through unchanged to the model counter.

    No `processed_multimodal_data` is provided, so `processed_item_metadata`
    must NOT appear in the call kwargs — receivers that don't accept it
    (no `**kwargs`, no default) should still work.
    """
    processor = Mock()
    processor.get_num_tokens_per_image = Mock(return_value=100)
    frames = _fake_video()

    def _count_video(*, video):
        assert video is frames
        return 98

    processor.get_num_tokens_per_video = Mock(side_effect=_count_video)

    mm_data = {"video": [VideoData(frames=frames, metadata={})]}
    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"video": [98]}
    # Lock in the contract: kwarg absent when nothing to slice.
    assert "processed_item_metadata" not in processor.get_num_tokens_per_video.call_args.kwargs


def test_find_mm_token_lengths_video_tensor_frames_passthrough():
    """Tensor video frames are forwarded unchanged; per-frame conversion is the model's job."""
    processor = Mock()
    frames = [torch.zeros((3, 4, 6), dtype=torch.float32)]

    def _count_video(*, video):
        assert video is frames
        assert isinstance(video[0], torch.Tensor)
        return 42

    processor.get_num_tokens_per_video = Mock(side_effect=_count_video)

    mm_data = {"video": [VideoData(frames=frames, metadata={})]}
    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"video": [42]}
    assert "processed_item_metadata" not in processor.get_num_tokens_per_video.call_args.kwargs


def test_find_mm_token_lengths_passes_processed_item_metadata():
    """Batched processor metadata is sliced into per-item processed_item_metadata payloads."""
    processor = Mock()
    frames = [_fake_video(), _fake_video()]
    processed_multimodal_data = {
        "video": {
            "video_grid_thw": torch.tensor([[1, 4, 6], [2, 8, 10]]),
            "frame_counts": [8, 16],
            "not_per_item": torch.tensor([[1, 2, 3]]),
            "pixel_values_videos": torch.zeros((2, 3, 4, 4)),
        }
    }
    seen_processed_item_metadata = []

    def _count_video(*, video, processed_item_metadata):
        assert video in frames
        assert "pixel_values_videos" not in processed_item_metadata
        assert "not_per_item" not in processed_item_metadata
        seen_processed_item_metadata.append(processed_item_metadata)
        return (
            int(processed_item_metadata["video_grid_thw"][0])
            + processed_item_metadata["frame_counts"]
        )

    processor.get_num_tokens_per_video = Mock(side_effect=_count_video)

    mm_data = {
        "video": [
            VideoData(frames=frames[0], metadata={}),
            VideoData(frames=frames[1], metadata={}),
        ]
    }
    result = find_mm_token_lengths(
        mm_data, processor, processed_multimodal_data=processed_multimodal_data
    )

    assert result == {"video": [9, 18]}
    assert torch.equal(seen_processed_item_metadata[0]["video_grid_thw"], torch.tensor([1, 4, 6]))
    assert torch.equal(seen_processed_item_metadata[1]["video_grid_thw"], torch.tensor([2, 8, 10]))


def test_find_mm_token_lengths_image_only_request_unaffected():
    """Image-only requests never invoke the video counter."""
    processor = Mock()
    processor.get_num_tokens_per_image = Mock(return_value=128)
    processor.get_num_tokens_per_video = Mock()

    mm_data = {"image": [object(), object()]}
    result = find_mm_token_lengths(mm_data, processor)

    assert result == {"image": [128, 128]}
    processor.get_num_tokens_per_video.assert_not_called()
