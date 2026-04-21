# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalInput.materialize_embed_mask and derived views.

Each test gates a specific production branch of the embed-mask path.
"""

import pytest
import torch

from tensorrt_llm.inputs.multimodal import MultimodalInput


def test_materialize_embed_mask_vocab_predicate():
    """vocab_size branch: embed_mask_flat == (prompt_token_ids >= vocab_size).

    Gates the zero-regression predicate — byte-identical to legacy
    filter_mm_token_from_input_ids behavior when only vocab_size is known.
    """
    vocab_size = 1000
    # Prompt: [text, text, mm, mm, mm, text, mm, mm] with mm tokens >= vocab.
    prompt = torch.tensor([10, 20, 1001, 1002, 1003, 30, 1004, 1005])
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[2],
        multimodal_lengths=[3],
    )
    mm_input.materialize_embed_mask(prompt, vocab_size=vocab_size)
    expected = prompt >= vocab_size
    assert torch.equal(mm_input.embed_mask_flat, expected)


def test_materialize_embed_mask_mm_token_ids():
    """Isin branch: mask == isin(prompt, mm_token_ids)."""
    vocab_size = 1000
    mm_token_ids = torch.tensor([500, 501])
    prompt = torch.tensor([10, 500, 500, 501, 20, 500])
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[1],
        multimodal_lengths=[3],
    )
    mm_input.materialize_embed_mask(prompt, vocab_size=vocab_size, mm_token_ids=mm_token_ids)
    expected = torch.tensor([False, True, True, True, False, True])
    assert torch.equal(mm_input.embed_mask_flat, expected)


def test_materialize_embed_mask_specials_excluded():
    """Specials branch: is_embed is False at special-token positions.

    Even though specials are OOV tokens, the mask must exclude them so the
    stitched flat view lines up with the encoder's embed-slot count.
    """
    vocab_size = 1000
    prompt = torch.tensor([10, 1001, 1002, 2000, 1003, 1004, 1005, 20])
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[1],
        multimodal_lengths=[6],
    )
    mm_special_token_ids = torch.tensor([2000])
    mm_input.materialize_embed_mask(
        prompt, vocab_size=vocab_size, mm_special_token_ids=mm_special_token_ids
    )
    expected = torch.tensor([False, True, True, False, True, True, True, False])
    assert torch.equal(mm_input.embed_mask_flat, expected)


def test_materialize_embed_mask_idempotent():
    """Second call returns the same cached tensor object (not re-computed)."""
    vocab_size = 1000
    prompt = torch.tensor([10, 1001, 1002, 20])
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[1],
        multimodal_lengths=[2],
    )
    first = mm_input.materialize_embed_mask(prompt, vocab_size=vocab_size)
    second = mm_input.materialize_embed_mask(prompt, vocab_size=vocab_size)
    assert first is second


def test_materialize_embed_mask_per_unit_stitching():
    """Per-unit masks stitch into the correct flat mask.

    Text positions are False; inline specials inside a unit are False;
    declared embed positions are True.
    """
    # Unit 0 at positions 2..5 (length 4), is_embed=[T, F, T, T]
    # Unit 1 at positions 7..9 (length 3), is_embed=[T, T, F]
    # Text positions 0,1,6 all False.
    prompt_seq_len = 10
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8, [0] * 8],
        multimodal_positions=[2, 7],
        multimodal_lengths=[4, 3],
        multimodal_embed_mask=[
            torch.tensor([True, False, True, True]),
            torch.tensor([True, True, False]),
        ],
    )
    prompt = torch.zeros(prompt_seq_len, dtype=torch.int64)
    mm_input.materialize_embed_mask(prompt, vocab_size=999999)
    expected = torch.tensor(
        [
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
        ]
    )
    assert torch.equal(mm_input.embed_mask_flat, expected)


def test_compute_per_unit_embed_masks():
    """Companion helper builds per-unit bool masks; specials excluded.

    Specials are INCLUDED in span lengths (outer bounding box) but EXCLUDED
    from the per-unit mask.
    """
    from tensorrt_llm.inputs.multimodal import (
        compute_per_unit_embed_masks,
        find_contiguous_mm_spans,
    )

    vocab_size = 1000
    # Mistral-shape: [text, img, img, special, img, img, img, text]
    prompt = torch.tensor([10, 1001, 1002, 2000, 1003, 1004, 1005, 20])
    mm_special_token_ids = torch.tensor([2000])
    spans, offsets = find_contiguous_mm_spans(
        input_ids=prompt,
        vocab_size=vocab_size,
        mm_special_token_ids=mm_special_token_ids,
    )
    assert spans == [(1, 6)]
    assert offsets == [2]
    per_unit = compute_per_unit_embed_masks(
        input_ids=prompt,
        contiguous_spans=spans,
        vocab_size=vocab_size,
        mm_special_token_ids=mm_special_token_ids,
    )
    assert len(per_unit) == 1
    assert torch.equal(
        per_unit[0],
        torch.tensor([True, True, False, True, True, True]),
    )


def test_compute_mm_embed_mask_if_absent_populates_py_multimodal_data():
    """Producer writes per-unit masks into py_multimodal_data.

    Key used: 'multimodal_embed_mask'. Covers Task 1.8 plumbing.
    """
    from tensorrt_llm.inputs.registry import compute_mm_embed_mask_if_absent

    class FakeProcessor:
        def get_vocab_size(self):
            return 1000

        def get_mm_token_ids(self):
            return None

        def get_mm_special_token_ids(self):
            return torch.tensor([2000])

    prompt_token_ids = [10, 1001, 1002, 2000, 1003, 1004, 1005, 20]
    extra = {"multimodal_data": {}}
    compute_mm_embed_mask_if_absent(prompt_token_ids, extra, FakeProcessor())

    mm_data = extra["multimodal_data"]
    assert "multimodal_embed_mask" in mm_data
    assert len(mm_data["multimodal_embed_mask"]) == 1
    assert torch.equal(
        mm_data["multimodal_embed_mask"][0],
        torch.tensor([True, True, False, True, True, True]),
    )


def test_runtime_data_cumsum_math_simplest():
    """Simplest Consumer A path: all-True mask, full request, no cache.

    Gates the cumsum kwarg + minimal __post_init__ math.
    """
    from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData

    # All-True mask of length 5. embed_mask_cumsum = [1, 2, 3, 4, 5].
    is_embed = torch.ones(5, dtype=torch.bool)
    cumsum = is_embed.to(torch.int64).cumsum(0)
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=5,
        embed_mask_cumsum=cumsum,
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 5
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_chunk():
    """Chunk ends before end of mask — cached=0, in_chunk counts embeds in [0,chunk_end)."""
    from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData

    # Mask: [T, T, F, T, T, F, T] -> 5 embeds total. Chunk ends at pos 4 -> cumsum[3]=3.
    is_embed = torch.tensor([True, True, False, True, True, False, True])
    cumsum = is_embed.to(torch.int64).cumsum(0)
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=4,
        embed_mask_cumsum=cumsum,
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_cache():
    """past_seen_token_num > 0 — cached counts embeds before the watermark."""
    from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData

    is_embed = torch.tensor([True, True, False, True, True, False, True])
    cumsum = is_embed.to(torch.int64).cumsum(0)
    rt = MultimodalRuntimeData(
        past_seen_token_num=3,  # cumsum[2] = 2 -> 2 cached
        chunk_end_pos=7,  # cumsum[6] = 5 -> in_chunk = 5-2 = 3
        embed_mask_cumsum=cumsum,
    )
    assert rt.num_cached_mm_tokens == 2
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_with_specials_mistral_shape():
    """Mistral-shape reproducer — chunk boundary inside a unit with inline special.

    Today's interval path got ambiguous counts here; cumsum path is correct
    by construction. (Paired with the Mistral e2e reproducer in Commit 4.)
    """
    from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData

    # Prompt layout: [text, img, img, special, img, img, img, text] len=8
    # Unit spans positions 1..7 (length 6) with special at position 3.
    # is_embed over the whole prompt:
    # pos:       0      1     2     3      4     5     6     7
    # mask:      F      T     T     F      T     T     T     F
    is_embed = torch.tensor([False, True, True, False, True, True, True, False])
    cumsum = is_embed.to(torch.int64).cumsum(0)

    # Chunk 0: pos 0..5 -> covers mask[0..4]: [F,T,T,F,T] -> 3 embeds.
    rt0 = MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5, embed_mask_cumsum=cumsum)
    assert rt0.num_cached_mm_tokens == 0
    assert rt0.num_mm_tokens_in_chunk == 3
    assert rt0.total_embeds_in_request == 5

    # Chunk 1: pos 5..8 (rest of the request after chunk 0 was processed).
    rt1 = MultimodalRuntimeData(past_seen_token_num=5, chunk_end_pos=8, embed_mask_cumsum=cumsum)
    assert rt1.num_cached_mm_tokens == 3
    assert rt1.num_mm_tokens_in_chunk == 2
    assert rt1.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_negative_past_seen_rejected():
    """Guard: past_seen_token_num must be non-negative."""
    from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData

    cumsum = torch.arange(1, 6, dtype=torch.int64)
    with pytest.raises(ValueError, match="past_seen_token_num must be non-negative"):
        MultimodalRuntimeData(past_seen_token_num=-1, chunk_end_pos=5, embed_mask_cumsum=cumsum)


def test_runtime_data_requires_cumsum():
    """Guard: embed_mask_cumsum is required."""
    from tensorrt_llm.inputs.multimodal import MultimodalRuntimeData

    with pytest.raises(TypeError):
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)


@pytest.mark.parametrize(
    "chunk_start,chunk_end,expected_lo_hi",
    [
        (0, 2, (0, 0)),  # all text
        (2, 6, (0, 3)),  # fully in unit 0 (3 embeds: specials subtracted)
        (0, 10, (0, 5)),  # whole request (5 embeds total)
        (2, 4, (0, 1)),  # ends mid-unit
        (4, 6, (1, 3)),  # starts mid-unit
    ],
)
def test_get_chunk_embed_indices(chunk_start, chunk_end, expected_lo_hi):
    """Range query correctness + cumsum shape/dtype contract."""
    prompt_seq_len = 10
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8, [0] * 8],
        multimodal_positions=[2, 7],
        multimodal_lengths=[4, 3],
        multimodal_embed_mask=[
            torch.tensor([True, False, True, True]),
            torch.tensor([True, True, False]),
        ],
    )
    prompt = torch.zeros(prompt_seq_len, dtype=torch.int64)
    mm_input.materialize_embed_mask(prompt, vocab_size=999999)
    assert mm_input.embed_mask_cumsum.dtype == torch.int64
    assert mm_input.embed_mask_cumsum.shape == (prompt_seq_len,)
    assert mm_input.get_chunk_embed_indices(chunk_start, chunk_end) == expected_lo_hi
