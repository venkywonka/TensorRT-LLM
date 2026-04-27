# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase
from tensorrt_llm.inputs.multimodal import find_mm_token_lengths
from tensorrt_llm.inputs.utils import VideoData


def _make_qwen3vl_config():
    return SimpleNamespace(
        text_config=SimpleNamespace(
            dtype=torch.float32,
            vocab_size=151936,
            hidden_size=4096,
        ),
        vision_config=SimpleNamespace(
            deepstack_visual_indexes=[8, 16, 24],
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )


def _make_qwen3vl_processor(monkeypatch):
    mock_processor = Mock()
    mock_processor._get_num_multimodal_tokens.side_effect = AssertionError(
        "HF multimodal token plumbing should not be used for Qwen3-VL video tests"
    )

    monkeypatch.setattr(
        "tensorrt_llm._torch.models.modeling_qwen3vl.AutoProcessor.from_pretrained",
        lambda *args, **kwargs: mock_processor,
    )

    processor = Qwen3VLInputProcessorBase(
        model_path="dummy-model",
        config=_make_qwen3vl_config(),
        tokenizer=Mock(),
        trust_remote_code=True,
    )
    return processor, mock_processor


def test_get_num_tokens_per_video_uses_processed_video_grid_thw(monkeypatch):
    processor, mock_processor = _make_qwen3vl_processor(monkeypatch)
    frames = [Image.new("RGB", (6, 4))]

    assert (
        processor.get_num_tokens_per_video(
            video=frames,
            processed_item_metadata={"video_grid_thw": torch.tensor([1, 4, 6])},
        )
        == 6
    )
    assert (
        processor.get_num_tokens_per_video(
            video=frames,
            processed_item_metadata={"video_grid_thw": torch.tensor([1, 8, 10])},
        )
        == 20
    )
    mock_processor._get_num_multimodal_tokens.assert_not_called()


def test_find_mm_token_lengths_uses_qwen3vl_processed_video_items(monkeypatch):
    processor, mock_processor = _make_qwen3vl_processor(monkeypatch)
    first_frames = [torch.zeros((3, 4, 6), dtype=torch.float32)]
    second_frames = [torch.zeros((3, 8, 10), dtype=torch.float32)]
    mm_data = {
        "video": [
            VideoData(frames=first_frames, metadata={}),
            VideoData(frames=second_frames, metadata={}),
        ],
    }

    processed_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "pixel_values_videos": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        "video_grid_thw": torch.tensor([[1, 4, 6], [2, 8, 10]], dtype=torch.long),
    }
    monkeypatch.setattr(processor, "_preprocess", lambda *args, **kwargs: processed_inputs)
    monkeypatch.setattr(
        processor,
        "get_mrope_config",
        lambda *args, **kwargs: {
            "mrope_position_ids": torch.zeros((3, 1, 3), dtype=torch.long),
            "mrope_position_deltas": torch.zeros((1, 1), dtype=torch.int32),
        },
    )

    _, extra_processed_inputs = processor(
        {
            "prompt": "describe the videos",
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {},
        },
        Mock(),
    )

    assert extra_processed_inputs["multimodal_data"]["video"]["video_grid_thw"].tolist() == [
        [1, 4, 6],
        [2, 8, 10],
    ]
    assert find_mm_token_lengths(
        mm_data,
        processor,
        processed_multimodal_data=extra_processed_inputs["multimodal_data"],
    ) == {"video": [6, 40]}
    mock_processor._get_num_multimodal_tokens.assert_not_called()


def test_qwen3vl_processed_video_items_do_not_depend_on_hidden_state(monkeypatch):
    processor, mock_processor = _make_qwen3vl_processor(monkeypatch)
    frames = [Image.new("RGB", (6, 4))]

    assert (
        processor.get_num_tokens_per_video(
            video=frames,
            processed_item_metadata={"video_grid_thw": torch.tensor([2, 8, 10])},
        )
        == 40
    )
    assert (
        processor.get_num_tokens_per_video(
            video=frames,
            processed_item_metadata={"video_grid_thw": torch.tensor([1, 4, 6])},
        )
        == 6
    )
    mock_processor._get_num_multimodal_tokens.assert_not_called()


def test_get_num_tokens_per_video_raises_without_processed_video_grid_thw(monkeypatch):
    processor, mock_processor = _make_qwen3vl_processor(monkeypatch)

    with pytest.raises(RuntimeError, match="processed_item_metadata.*video_grid_thw"):
        processor.get_num_tokens_per_video(video=[Image.new("RGB", (6, 4))])

    mock_processor._get_num_multimodal_tokens.assert_not_called()
