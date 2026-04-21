# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalInput.materialize_is_embed and derived views.

Each test gates a specific production branch — see
slop/mm_is_embed_migration/goals.md §7.1 and plan.md Commit 1.
"""

import pytest
import torch

from tensorrt_llm.inputs.multimodal import MultimodalInput


def test_materialize_is_embed_vocab_predicate():
    """vocab_size branch: is_embed_flat == (prompt_token_ids >= vocab_size).

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
    mm_input.materialize_is_embed(prompt, vocab_size=vocab_size)
    expected = prompt >= vocab_size
    assert torch.equal(mm_input.is_embed_flat, expected)


def test_materialize_is_embed_mm_token_ids():
    """Isin branch: mask == isin(prompt, mm_token_ids)."""
    vocab_size = 1000
    mm_token_ids = torch.tensor([500, 501])
    prompt = torch.tensor([10, 500, 500, 501, 20, 500])
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[1],
        multimodal_lengths=[3],
    )
    mm_input.materialize_is_embed(prompt, vocab_size=vocab_size, mm_token_ids=mm_token_ids)
    expected = torch.tensor([False, True, True, True, False, True])
    assert torch.equal(mm_input.is_embed_flat, expected)


def test_materialize_is_embed_specials_excluded():
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
    mm_input.materialize_is_embed(
        prompt, vocab_size=vocab_size, mm_special_token_ids=mm_special_token_ids
    )
    expected = torch.tensor([False, True, True, False, True, True, True, False])
    assert torch.equal(mm_input.is_embed_flat, expected)


def test_materialize_is_embed_idempotent():
    """Second call returns the same cached tensor object (not re-computed)."""
    vocab_size = 1000
    prompt = torch.tensor([10, 1001, 1002, 20])
    mm_input = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[1],
        multimodal_lengths=[2],
    )
    first = mm_input.materialize_is_embed(prompt, vocab_size=vocab_size)
    second = mm_input.materialize_is_embed(prompt, vocab_size=vocab_size)
    assert first is second


def test_materialize_is_embed_per_unit_stitching():
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
        multimodal_is_embeds=[
            torch.tensor([True, False, True, True]),
            torch.tensor([True, True, False]),
        ],
    )
    prompt = torch.zeros(prompt_seq_len, dtype=torch.int64)
    mm_input.materialize_is_embed(prompt, vocab_size=999999)
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
    assert torch.equal(mm_input.is_embed_flat, expected)


def test_compute_per_unit_is_embeds():
    """Companion helper builds per-unit bool masks; specials excluded.

    Specials are INCLUDED in span lengths (outer bounding box) but EXCLUDED
    from the per-unit mask.
    """
    from tensorrt_llm.inputs.multimodal import compute_per_unit_is_embeds, find_contiguous_mm_spans

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
    per_unit = compute_per_unit_is_embeds(
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


def test_compute_mm_is_embed_if_absent_populates_py_multimodal_data():
    """Producer writes per-unit masks into py_multimodal_data.

    Key used: 'multimodal_is_embeds'. Covers Task 1.8 plumbing.
    """
    from tensorrt_llm.inputs.registry import compute_mm_contiguous_spans_if_absent

    class FakeProcessor:
        def get_vocab_size(self):
            return 1000

        def get_mm_token_ids(self):
            return None

        def get_mm_special_token_ids(self):
            return torch.tensor([2000])

    prompt_token_ids = [10, 1001, 1002, 2000, 1003, 1004, 1005, 20]
    extra = {"multimodal_data": {}}
    compute_mm_contiguous_spans_if_absent(prompt_token_ids, extra, FakeProcessor())

    mm_data = extra["multimodal_data"]
    assert "multimodal_is_embeds" in mm_data
    assert len(mm_data["multimodal_is_embeds"]) == 1
    assert torch.equal(
        mm_data["multimodal_is_embeds"][0],
        torch.tensor([True, True, False, True, True, True]),
    )


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
        multimodal_is_embeds=[
            torch.tensor([True, False, True, True]),
            torch.tensor([True, True, False]),
        ],
    )
    prompt = torch.zeros(prompt_seq_len, dtype=torch.int64)
    mm_input.materialize_is_embed(prompt, vocab_size=999999)
    assert mm_input.is_embed_cumsum.dtype == torch.int64
    assert mm_input.is_embed_cumsum.shape == (prompt_seq_len,)
    assert mm_input.get_chunk_embed_indices(chunk_start, chunk_end) == expected_lo_hi
