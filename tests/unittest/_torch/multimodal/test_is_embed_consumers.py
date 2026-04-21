# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Consumer-side unit tests for the is_embed mask path.

See slop/mm_is_embed_migration/goals.md §7.2 and plan.md Commits 3-4.
"""

import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import find_input_mm_embeds
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams, MultimodalRuntimeData


def _mk_params(
    *,
    multimodal_is_embeds,
    multimodal_positions,
    multimodal_lengths,
    prompt_seq_len,
    past,
    chunk_end,
):
    """Construct MultimodalParams with a materialized is_embed_cumsum."""
    mi = MultimodalInput(
        multimodal_hashes=[[0] * 8] * len(multimodal_lengths),
        multimodal_positions=multimodal_positions,
        multimodal_lengths=multimodal_lengths,
        multimodal_is_embeds=multimodal_is_embeds,
    )
    prompt = torch.zeros(prompt_seq_len, dtype=torch.int64)
    mi.materialize_is_embed(prompt, vocab_size=999999)
    rt = MultimodalRuntimeData(
        past_seen_token_num=past,
        chunk_end_pos=chunk_end,
        is_embed_cumsum=mi.is_embed_cumsum,
    )
    return MultimodalParams(multimodal_input=mi, multimodal_runtime=rt)


def test_find_input_mm_embeds_chunked_partial_item():
    """Chunk ends mid-unit with specials — B returns only in-chunk embed rows.

    This is the code path that today's per-special-token subtraction got
    ambiguously wrong. The mask path derives the slice via cumsum diffs.
    """
    # Unit at positions 2..7 (length 6) with a special at relative pos 2.
    # Per-unit mask [T, T, F, T, T, T] -> 5 embeds total; embed positions
    # (absolute) = {2, 3, 5, 6, 7}.
    # Encoder output row[i] corresponds to the i-th True in is_embed_flat.
    encoder_out = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    params = _mk_params(
        multimodal_is_embeds=[torch.tensor([True, True, False, True, True, True])],
        multimodal_positions=[2],
        multimodal_lengths=[6],
        prompt_seq_len=10,
        past=0,
        chunk_end=5,  # covers mask[0..4] = [F,F,T,T,F] -> 2 embeds (rows 0,1)
    )
    out = find_input_mm_embeds([encoder_out], [params])
    assert len(out) == 1
    assert out[0].shape == (2, 4)
    assert torch.equal(out[0], encoder_out[0:2])


def test_find_input_mm_embeds_partial_cache_reuse():
    """past_seen > 0 — B returns only uncached encoder rows."""
    encoder_out = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    params = _mk_params(
        multimodal_is_embeds=[torch.tensor([True, True, False, True, True, True])],
        multimodal_positions=[2],
        multimodal_lengths=[6],
        prompt_seq_len=10,
        past=4,  # cumsum[3] = 2 -> 2 cached rows
        chunk_end=10,  # cumsum[9] = 5 -> in_chunk 3 rows (2,3,4)
    )
    out = find_input_mm_embeds([encoder_out], [params])
    assert len(out) == 1
    assert torch.equal(out[0], encoder_out[2:5])


def test_find_input_mm_embeds_all_cached_returns_empty():
    """All embeds cached and beyond current chunk — B returns empty list."""
    encoder_out = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    params = _mk_params(
        multimodal_is_embeds=[torch.tensor([True, True, True])],
        multimodal_positions=[1],
        multimodal_lengths=[3],
        prompt_seq_len=5,
        past=4,  # watermark past the unit
        chunk_end=5,
    )
    out = find_input_mm_embeds([encoder_out], [params])
    assert out == []


def test_find_input_mm_embeds_no_runtime_data_passthrough():
    """Legacy: params with multimodal_runtime=None -> return full mm_embeds."""
    encoder_out = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    mi = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[0],
        multimodal_lengths=[3],
    )
    params = MultimodalParams(multimodal_input=mi, multimodal_runtime=None)
    out = find_input_mm_embeds([encoder_out], [params])
    assert len(out) == 1
    assert torch.equal(out[0], encoder_out)
