# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Consumer-side unit tests for the is_embed mask path.

See slop/mm_is_embed_migration/goals.md §7.2 and plan.md Commits 3-4.
"""

import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    fuse_input_embeds,
)
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


def test_fuse_input_embeds_mistral_specials_no_count_mismatch():
    """Mistral-shape chunk carries a partial unit with an inline special.

    Legacy filter_mm_token_from_input_ids raises count-mismatch because it
    counts EVERY mm id as an embedding consumer, including specials like
    [IMG_BREAK] which the encoder did not produce. The mask path derives the
    slice via per-unit is_embed mask, which excludes specials by construction.
    """
    from torch.nn import Embedding

    vocab_size = 1000
    hidden_dim = 4
    # Embedding table covers base vocab + mm placeholders + inline specials
    # (like Mistral's [IMG_BREAK]). num_embeddings=1100 > all ids used here.
    embedding_layer = Embedding(vocab_size + 100, hidden_dim)
    # input_ids layout [text, img, img, special, img]. Legacy path treats
    # every id >= vocab_size as an embed consumer -> 4 mm tokens counted,
    # encoder output 3 -> mismatch. Mask path excludes position 3 (special)
    # by construction.
    input_ids = torch.tensor([10, 1001, 1002, 1050, 1003])
    mm_embed = torch.arange(3 * hidden_dim, dtype=torch.float32).reshape(3, hidden_dim)
    # Unit bounding box = positions [1..4], length 4. is_embed mask:
    # [T, T, F, T] — 3 embed rows (non-special positions).
    mm_params = _mk_params(
        multimodal_is_embeds=[torch.tensor([True, True, False, True])],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        prompt_seq_len=5,
        past=0,
        chunk_end=5,
    )
    out = fuse_input_embeds(
        embedding_layer=embedding_layer,
        input_ids=input_ids,
        mm_embeds=[mm_embed],
        multimodal_params=[mm_params],
    )
    _, input_embeds = out[0], out[1]
    assert input_embeds is not None
    assert input_embeds.shape == (5, hidden_dim)
    # MM rows placed at positions 1, 2, 4 (specials skipped).
    assert torch.equal(input_embeds[1], mm_embed[0])
    assert torch.equal(input_embeds[2], mm_embed[1])
    assert torch.equal(input_embeds[4], mm_embed[2])


def test_fuse_input_embeds_backstop_materializes_when_intake_skipped():
    """Backstop fires when MultimodalInput reaches fuse_input_embeds without
    a materialized mask (out-of-band construction path). Uses the Qwen2-VL
    style OOV-sentinel pattern (mm ids >= num_embeddings).
    """
    from torch.nn import Embedding

    # Embedding table exactly vocab-size: mm placeholder ids 1001, 1002 are
    # OOV sentinels. Backstop must flag them via the vocab predicate without
    # being told mm_token_ids explicitly.
    num_embeddings = 1000
    hidden_dim = 4
    embedding_layer = Embedding(num_embeddings, hidden_dim)
    input_ids = torch.tensor([10, 1001, 1002, 20])
    mm_embed = torch.arange(2 * hidden_dim, dtype=torch.float32).reshape(2, hidden_dim)

    mi = MultimodalInput(
        multimodal_hashes=[[0] * 8],
        multimodal_positions=[1],
        multimodal_lengths=[2],
        # Intentionally NOT populating multimodal_is_embeds -> mask is None.
    )
    assert mi.is_embed_flat is None, "precondition: mask must not be materialized yet"
    params = MultimodalParams(multimodal_input=mi, multimodal_runtime=None)
    out = fuse_input_embeds(
        embedding_layer=embedding_layer,
        input_ids=input_ids,
        mm_embeds=[mm_embed],
        multimodal_params=[params],
    )
    _, input_embeds = out[0], out[1]
    assert mi.is_embed_flat is not None, "backstop must have populated is_embed_flat"
    assert torch.equal(mi.is_embed_flat, torch.tensor([False, True, True, False]))
    assert input_embeds.shape == (4, hidden_dim)
    assert torch.equal(input_embeds[1], mm_embed[0])
    assert torch.equal(input_embeds[2], mm_embed[1])
