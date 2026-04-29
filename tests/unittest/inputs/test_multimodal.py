# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalRuntimeData cumsum math and the flat-mask producer."""

import pytest
import torch

from tensorrt_llm.inputs.multimodal import (
    MultimodalInput,
    MultimodalRuntimeData,
    apply_mm_hashes,
    hexdigest_to_int32,
)
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    BaseMultimodalInputProcessor,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    create_input_processor_with_hash,
    maybe_compute_mm_embed_cumsum,
)


def _hash(seed):
    return [seed + i for i in range(8)]


def test_multimodal_input_defaults_encoder_lengths_for_legacy_callers():
    mm_input = MultimodalInput(
        multimodal_hashes=[_hash(0), _hash(10)],
        multimodal_positions=[2, 8],
        multimodal_lengths=[4, 6],
    )

    assert mm_input.multimodal_encoder_output_lengths == [4, 6]
    assert mm_input.multimodal_modalities is None

    hashes, positions, lengths = mm_input.to_tensor()
    torch.testing.assert_close(
        hashes,
        torch.tensor([_hash(0), _hash(10)], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(positions, torch.tensor([2, 8], dtype=torch.int32))
    torch.testing.assert_close(lengths, torch.tensor([4, 6], dtype=torch.int32))


def test_multimodal_input_preserves_mixed_item_order():
    mm_input = MultimodalInput.from_components(
        [_hash(0), _hash(10), _hash(20)],
        [3, 12, 30],
        [5, 9, 4],
        mm_uuids=["image-a", "video-b", "image-c"],
        mm_modalities=["image", "video", "image"],
        mm_encoder_output_lengths=[4, 12, 4],
    )

    assert mm_input.multimodal_modalities == ["image", "video", "image"]
    assert mm_input.multimodal_encoder_output_lengths == [4, 12, 4]
    assert mm_input.multimodal_uuids == ["image-a", "video-b", "image-c"]
    assert mm_input.multimodal_positions == [3, 12, 30]
    assert mm_input.multimodal_lengths == [5, 9, 4]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "multimodal_positions": [1],
                "multimodal_lengths": [2, 3],
            },
            "multimodal_positions length",
        ),
        (
            {
                "multimodal_positions": [1, 4],
                "multimodal_lengths": [2],
            },
            "multimodal_lengths length",
        ),
        (
            {
                "multimodal_positions": [1, 4],
                "multimodal_lengths": [2, 3],
                "multimodal_encoder_output_lengths": [2],
            },
            "multimodal_encoder_output_lengths length",
        ),
        (
            {
                "multimodal_positions": [1, 4],
                "multimodal_lengths": [2, 3],
                "multimodal_modalities": ["image"],
            },
            "multimodal_modalities length",
        ),
        (
            {
                "multimodal_positions": [1, 4],
                "multimodal_lengths": [2, 3],
                "multimodal_uuids": ["only-one"],
            },
            "multimodal_uuids length",
        ),
    ],
)
def test_multimodal_input_validates_aligned_item_fields(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MultimodalInput(multimodal_hashes=[_hash(0), _hash(10)], **kwargs)


def test_multimodal_input_validates_item_field_types():
    with pytest.raises(TypeError, match="multimodal_modalities"):
        MultimodalInput(
            multimodal_hashes=[_hash(0)],
            multimodal_positions=[1],
            multimodal_lengths=[2],
            multimodal_modalities=[123],
        )

    with pytest.raises(TypeError, match="multimodal_encoder_output_lengths"):
        MultimodalInput(
            multimodal_hashes=[_hash(0)],
            multimodal_positions=[1],
            multimodal_lengths=[2],
            multimodal_encoder_output_lengths=["2"],
        )


class _HashingFakeProcessor(BaseMultimodalInputProcessor):
    """Small fake that exercises create_input_processor_with_hash."""

    _registered_model_type = "test_mixed_hashing"

    def __init__(self, prompt_token_ids=None, extra_processed_inputs=None):
        self._multimodal_hashing_supported = None
        self._prompt_token_ids = prompt_token_ids or [1, 100, 100, 100, 2]
        self._extra_processed_inputs = extra_processed_inputs or {"multimodal_data": {"image": {}}}
        self.last_num_mm_tokens = None

    @property
    def processor(self):
        return None

    @property
    def tokenizer(self):
        return None

    @property
    def config(self):
        return None

    @property
    def dtype(self):
        return torch.float32

    def __call__(self, inputs, sampling_params):
        return self._prompt_token_ids, self._extra_processed_inputs

    def get_vocab_size(self):
        return 100

    def get_mm_token_ids(self):
        return None

    def get_mm_special_token_ids(self):
        return None

    def get_num_tokens_per_image(self, *, image):
        if isinstance(image, torch.Tensor) and image.numel() == 1:
            return int(image.item())
        return 3

    def get_num_tokens_per_video(self, *, video):
        return len(video) + 2

    def get_text_with_mm_placeholders(self, mm_counts):
        return "<image><video><image>"

    def expand_prompt_token_ids_for_mm(
        self, prompt_token_ids, num_mm_tokens, hf_processor_mm_kwargs=None
    ):
        self.last_num_mm_tokens = list(num_mm_tokens)
        return [1] + [100] * sum(num_mm_tokens)


def _register_mixed_hashing_placeholders():
    MULTIMODAL_PLACEHOLDER_REGISTRY.set_placeholder_metadata(
        "test_mixed_hashing",
        MultimodalPlaceholderMetadata(
            placeholder_map={
                "image": "<image>",
                "video": "<video>",
            },
            placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        ),
    )


def _mixed_mm_data():
    return {
        "image": [torch.tensor([2]), torch.tensor([3])],
        "video": [[torch.tensor([4]), torch.tensor([5])]],
    }


def _mixed_mm_uuids():
    return {
        "image": ["image-a", "image-c"],
        "video": ["video-b"],
    }


def test_hash_wrapper_populates_python_item_metadata():
    processor = _HashingFakeProcessor()
    wrapped = create_input_processor_with_hash(processor)

    _, extra = wrapped(
        {
            "prompt": "describe",
            "multi_modal_data": {"image": [torch.tensor([1, 2, 3])]},
        },
        sampling_params=None,
    )

    mm_input = extra["multimodal_input"]
    assert mm_input.multimodal_modalities == ["image"]
    assert mm_input.multimodal_lengths == [3]
    assert mm_input.multimodal_encoder_output_lengths == [3]
    assert extra["multimodal_data"]["layout_metadata"] == {
        "encoder_output_lengths": [3],
        "modalities": ["image"],
    }


def test_hash_wrapper_preserves_mixed_prompt_item_order():
    _register_mixed_hashing_placeholders()
    try:
        mm_data = _mixed_mm_data()
        mm_uuids = _mixed_mm_uuids()
        processor = _HashingFakeProcessor(
            prompt_token_ids=[1, 100, 100, 2, 100, 100, 100, 100, 3, 100, 100, 100, 4],
            extra_processed_inputs={
                "multimodal_data": {
                    "image": {},
                    "video": {},
                    "layout_metadata": {
                        "encoder_output_lengths": [20, 40, 30],
                    },
                }
            },
        )
        wrapped = create_input_processor_with_hash(processor)

        _, extra = wrapped(
            {
                "prompt": "<image> between <video> after <image>",
                "multi_modal_data": mm_data,
                "multi_modal_uuids": mm_uuids,
            },
            sampling_params=None,
        )

        hashes_by_modality, _ = apply_mm_hashes(mm_data, mm_uuids)
        expected_hashes = [
            hexdigest_to_int32(hashes_by_modality["image"][0]),
            hexdigest_to_int32(hashes_by_modality["video"][0]),
            hexdigest_to_int32(hashes_by_modality["image"][1]),
        ]

        mm_input = extra["multimodal_input"]
        assert mm_input.multimodal_hashes == expected_hashes
        assert mm_input.multimodal_positions == [1, 4, 9]
        assert mm_input.multimodal_lengths == [2, 4, 3]
        assert mm_input.multimodal_encoder_output_lengths == [20, 40, 30]
        assert mm_input.multimodal_modalities == ["image", "video", "image"]
        assert mm_input.multimodal_uuids == ["image-a", "video-b", "image-c"]
        assert extra["multimodal_data"]["layout_metadata"] == {
            "encoder_output_lengths": [20, 40, 30],
            "modalities": ["image", "video", "image"],
        }
    finally:
        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata("test_mixed_hashing")


def test_tokenized_hash_wrapper_uses_dummy_placeholders_for_mixed_order():
    mm_data = _mixed_mm_data()
    _register_mixed_hashing_placeholders()
    try:
        processor = _HashingFakeProcessor(
            extra_processed_inputs={
                "multimodal_data": {
                    "image": {},
                    "video": {},
                    "layout_metadata": {
                        "encoder_output_lengths": [20, 40, 30],
                    },
                }
            },
        )
        wrapped = create_input_processor_with_hash(processor)

        _, extra = wrapped(
            {
                "prompt_token_ids": [10, 11, 12],
                "multi_modal_data": mm_data,
                "multi_modal_uuids": _mixed_mm_uuids(),
            },
            sampling_params=None,
        )

        mm_input = extra["multimodal_input"]
        assert processor.last_num_mm_tokens == [2, 4, 3]
        assert mm_input.multimodal_positions == [1, 3, 7]
        assert mm_input.multimodal_lengths == [2, 4, 3]
        assert mm_input.multimodal_encoder_output_lengths == [20, 40, 30]
        assert mm_input.multimodal_modalities == ["image", "video", "image"]
        assert mm_input.multimodal_uuids == ["image-a", "video-b", "image-c"]
    finally:
        MULTIMODAL_PLACEHOLDER_REGISTRY.remove_placeholder_metadata("test_mixed_hashing")


def test_single_modality_hash_cache_does_not_force_mixed_hashing():
    class _NoPlaceholderFakeProcessor(_HashingFakeProcessor):
        _registered_model_type = "test_mixed_hashing_no_placeholders"

    processor = _NoPlaceholderFakeProcessor()
    wrapped = create_input_processor_with_hash(processor)

    _, single_extra = wrapped(
        {
            "prompt": "describe",
            "multi_modal_data": {"image": [torch.tensor([1, 2, 3])]},
        },
        sampling_params=None,
    )

    assert processor.multimodal_hashing_supported is True
    assert single_extra["multimodal_input"].multimodal_modalities == ["image"]

    processor._prompt_token_ids = [7, 8]
    processor._extra_processed_inputs = {
        "multimodal_data": {
            "image": {},
            "video": {},
        }
    }

    prompt_token_ids, mixed_extra = wrapped(
        {
            "prompt": "mixed prompt without registered placeholders",
            "multi_modal_data": _mixed_mm_data(),
        },
        sampling_params=None,
    )

    assert prompt_token_ids == [7, 8]
    assert "multimodal_input" not in mixed_extra
    assert "modalities" not in mixed_extra["multimodal_data"].get("layout_metadata", {})


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
    torch.testing.assert_close(
        cumsum,
        torch.tensor([0, 1, 2, 2, 3, 4, 5, 5], dtype=torch.int64),
        rtol=0,
        atol=0,
    )


def test_runtime_data_cumsum_math_simplest():
    """All-True mask, full request, no cache."""
    is_embed = torch.ones(5, dtype=torch.bool)
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=5,
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
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
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
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
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
    )
    assert rt.num_cached_mm_tokens == 2
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_with_specials_mistral_shape():
    """Chunk boundary inside a unit with inline special (Mistral-shape)."""
    # [text, img, img, special, img, img, img, text]
    is_embed = torch.tensor([False, True, True, False, True, True, True, False])
    cumsum = is_embed.cumsum(0, dtype=torch.int64)

    rt0 = MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5, embed_mask_cumsum=cumsum)
    assert rt0.num_cached_mm_tokens == 0
    assert rt0.num_mm_tokens_in_chunk == 3
    assert rt0.total_embeds_in_request == 5

    rt1 = MultimodalRuntimeData(past_seen_token_num=5, chunk_end_pos=8, embed_mask_cumsum=cumsum)
    assert rt1.num_cached_mm_tokens == 3
    assert rt1.num_mm_tokens_in_chunk == 2
    assert rt1.total_embeds_in_request == 5


def test_runtime_data_requires_cumsum():
    """embed_mask_cumsum is required."""
    with pytest.raises(TypeError):
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)
