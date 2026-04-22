# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalRuntimeData cumsum math and the flat-mask producer."""

import pytest
import torch

from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData
from tensorrt_llm.inputs.registry import compute_mm_embed_mask_if_absent


def test_compute_mm_embed_mask_if_absent_populates_py_multimodal_data():
    """Producer writes a flat bool tensor at py_multimodal_data[multimodal_embed_mask]."""

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
    compute_mm_embed_mask_if_absent(prompt_token_ids, extra, FakeProcessor())

    mask = extra["multimodal_data"]["multimodal_embed_mask"]
    assert torch.equal(
        mask,
        torch.tensor(
            [False, True, True, False, True, True, True, False]),
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
    is_embed = torch.tensor(
        [False, True, True, False, True, True, True, False])
    cumsum = is_embed.to(torch.int64).cumsum(0)

    rt0 = MultimodalRuntimeData(past_seen_token_num=0,
                                chunk_end_pos=5,
                                embed_mask_cumsum=cumsum)
    assert rt0.num_cached_mm_tokens == 0
    assert rt0.num_mm_tokens_in_chunk == 3
    assert rt0.total_embeds_in_request == 5

    rt1 = MultimodalRuntimeData(past_seen_token_num=5,
                                chunk_end_pos=8,
                                embed_mask_cumsum=cumsum)
    assert rt1.num_cached_mm_tokens == 3
    assert rt1.num_mm_tokens_in_chunk == 2
    assert rt1.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_negative_past_seen_rejected():
    """past_seen_token_num must be non-negative."""
    cumsum = torch.arange(1, 6, dtype=torch.int64)
    with pytest.raises(ValueError,
                       match="past_seen_token_num must be non-negative"):
        MultimodalRuntimeData(past_seen_token_num=-1,
                              chunk_end_pos=5,
                              embed_mask_cumsum=cumsum)


def test_runtime_data_requires_cumsum():
    """embed_mask_cumsum is required."""
    with pytest.raises(TypeError):
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)
