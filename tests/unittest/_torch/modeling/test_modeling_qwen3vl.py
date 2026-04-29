# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalLM
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models import modeling_qwen3vl as qwen3vl_module
from tensorrt_llm._torch.models.checkpoints.hf.qwen3vl_weight_mapper import Qwen3VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VisionModelBase,
    Qwen3VLInputProcessorBase,
    Qwen3VLModel,
)
from tensorrt_llm._torch.shared_tensor import (
    SharedTensorContainer,
    _SharedTensorRebuildMethodRegistry,
)
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams

QWEN3_VL_8B_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 4,
        # NOTE: Only 4 layers for testing, 36 layers for full model.
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "_attn_implementation": "flash_attention_2",
    "_name_or_path": str(os.path.join(llm_models_root(), "Qwen3", "Qwen3-VL-8B-Instruct")),
}


class _MockQwen3Visual:
    def __init__(self, outputs_by_marker):
        self.outputs_by_marker = outputs_by_marker
        self.calls = []

    def __call__(self, pixel_values, grid_thw):
        marker = int(pixel_values[0, 0].item())
        self.calls.append((marker, grid_thw.clone()))
        return self.outputs_by_marker[marker], []


def _make_mock_qwen3vl_encoder(outputs_by_marker):
    encoder = Qwen3VisionModelBase.__new__(Qwen3VisionModelBase)
    encoder.config = SimpleNamespace(spatial_merge_size=1)
    encoder.model_dtype = torch.float32
    encoder.visual = _MockQwen3Visual(outputs_by_marker)
    return encoder


def _make_mixed_multimodal_param(modalities, *, metadata_source="layout"):
    encoder_lengths = [2 if modality == "image" else 3 for modality in modalities]
    multimodal_data = {
        "image": {
            "pixel_values": torch.ones((2, 1), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[2, 1, 1]], dtype=torch.int64),
        },
        "video": {
            "pixel_values_videos": torch.full((3, 1), 2.0, dtype=torch.float32),
            "video_grid_thw": torch.tensor([[3, 1, 1]], dtype=torch.int64),
        },
    }
    multimodal_input = None
    if metadata_source == "layout":
        multimodal_data["layout_metadata"] = {
            "modalities": modalities,
            "encoder_output_lengths": encoder_lengths,
        }
    else:
        multimodal_input = MultimodalInput.from_components(
            [[item] * 8 for item in range(len(modalities))],
            list(range(len(modalities))),
            encoder_lengths,
            mm_modalities=modalities,
            mm_encoder_output_lengths=encoder_lengths,
        )
    return MultimodalParams(
        multimodal_data=multimodal_data,
        multimodal_input=multimodal_input,
    )


def test_qwen3vl_encoder_preserves_image_then_video_layout_order():
    image_embeds = torch.tensor([[10.0], [11.0]], dtype=torch.float32)
    video_embeds = torch.tensor([[20.0], [21.0], [22.0]], dtype=torch.float32)
    encoder = _make_mock_qwen3vl_encoder(
        {
            1: image_embeds,
            2: video_embeds,
        }
    )
    multimodal_param = _make_mixed_multimodal_param(["image", "video"])

    result = Qwen3VisionModelBase.forward(encoder, [multimodal_param])

    assert len(result) == 1
    torch.testing.assert_close(result[0], torch.cat([image_embeds, video_embeds], dim=0))
    assert [marker for marker, _ in encoder.visual.calls] == [1, 2]


def test_qwen3vl_encoder_preserves_video_then_image_multimodal_input_order():
    image_embeds = torch.tensor([[10.0], [11.0]], dtype=torch.float32)
    video_embeds = torch.tensor([[20.0], [21.0], [22.0]], dtype=torch.float32)
    encoder = _make_mock_qwen3vl_encoder(
        {
            1: image_embeds,
            2: video_embeds,
        }
    )
    multimodal_param = _make_mixed_multimodal_param(
        ["video", "image"],
        metadata_source="multimodal_input",
    )

    result = Qwen3VisionModelBase.forward(encoder, [multimodal_param])

    assert len(result) == 1
    torch.testing.assert_close(result[0], torch.cat([video_embeds, image_embeds], dim=0))
    assert [marker for marker, _ in encoder.visual.calls] == [1, 2]


def test_qwen3vl_encoder_preserves_single_modality_without_order_metadata():
    image_embeds = torch.tensor([[10.0], [11.0]], dtype=torch.float32)
    encoder = _make_mock_qwen3vl_encoder({1: image_embeds})
    multimodal_param = MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": torch.ones((2, 1), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[2, 1, 1]], dtype=torch.int64),
            },
        },
    )

    result = Qwen3VisionModelBase.forward(encoder, [multimodal_param])

    assert len(result) == 1
    torch.testing.assert_close(result[0], image_embeds)
    assert [marker for marker, _ in encoder.visual.calls] == [1]


class _MockQwen3VLTokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, prompt, return_tensors):
        assert prompt == "prompt"
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([self.token_ids], dtype=torch.int64))


def _make_mock_qwen3vl_input_processor(token_ids):
    processor = Qwen3VLInputProcessorBase.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=151655,
        video_token_id=151656,
        text_config=SimpleNamespace(hidden_size=4, vocab_size=151936),
        vision_config=SimpleNamespace(deepstack_visual_indexes=[8, 16, 24]),
    )
    processor._tokenizer = _MockQwen3VLTokenizer(token_ids)
    processor.tllm_multimodal_token_id = processor.config.text_config.vocab_size + 1
    return processor


def _install_rejecting_shared_tensor_container(monkeypatch):
    def _unexpected_from_dict(cls, handle):
        raise AssertionError("get_prompt_token_ids must use tensor_size metadata")

    monkeypatch.setattr(SharedTensorContainer, "from_dict", classmethod(_unexpected_from_dict))


def _make_mm_handle(name, rows):
    return {
        "name": name,
        "method_key": _SharedTensorRebuildMethodRegistry.REBUILD_CPU,
        "tensor_size": [rows, 16],
        "tensor_stride": [16, 1],
        "tensor_storage_offset": 0,
        "storage_size": rows * 16,
        "storage_handle": f"unused-{name}",
        "storage_dtype": "torch.float32",
        "manager_handle": f"unused-{name}-manager",
    }


@pytest.mark.parametrize(
    "token_ids,handles,expected_ids,expected_lengths,expected_offsets",
    [
        pytest.param(
            [10, 151655, 11, 151656, 12],
            [
                _make_mm_handle("image", 2),
                _make_mm_handle("video", 3),
            ],
            [10, 151655, 151655, 11, 151656, 151656, 151656, 12],
            [2, 3],
            [1, 4],
            id="image-then-video",
        ),
        pytest.param(
            [10, 151656, 11, 151655, 12],
            [
                _make_mm_handle("video", 3),
                _make_mm_handle("image", 2),
            ],
            [10, 151656, 151656, 151656, 11, 151655, 151655, 12],
            [3, 2],
            [1, 5],
            id="video-then-image",
        ),
    ],
)
def test_qwen3vl_get_prompt_token_ids_expands_mixed_handles_in_prompt_order(
    monkeypatch,
    token_ids,
    handles,
    expected_ids,
    expected_lengths,
    expected_offsets,
):
    _install_rejecting_shared_tensor_container(monkeypatch)
    processor = _make_mock_qwen3vl_input_processor(token_ids)

    expanded_ids, lengths, offsets = processor.get_prompt_token_ids({"prompt": "prompt"}, handles)

    assert expanded_ids == expected_ids
    assert lengths == expected_lengths
    assert offsets == expected_offsets


def test_qwen3vl_get_prompt_token_ids_rejects_handle_count_mismatch(monkeypatch):
    _install_rejecting_shared_tensor_container(monkeypatch)
    processor = _make_mock_qwen3vl_input_processor([10, 151655, 11])

    with pytest.raises(ValueError, match="Expected 2 image/video placeholders"):
        processor.get_prompt_token_ids(
            {"prompt": "prompt"},
            [
                _make_mm_handle("image", 2),
                _make_mm_handle("video", 3),
            ],
        )


@pytest.mark.parametrize(
    "handle",
    [
        pytest.param({"name": "image"}, id="missing"),
        pytest.param(
            {
                "name": "image",
                "tensor_size": [2, 16, 1],
            },
            id="rank3",
        ),
    ],
)
def test_qwen3vl_get_prompt_token_ids_rejects_malformed_tensor_size(
    monkeypatch,
    handle,
):
    _install_rejecting_shared_tensor_container(monkeypatch)
    processor = _make_mock_qwen3vl_input_processor([10, 151655, 11])

    with pytest.raises(ValueError, match="rank-2 tensor_size"):
        processor.get_prompt_token_ids({"prompt": "prompt"}, [handle])


def test_qwen3vl_forward_passes_mm_token_ids_for_in_vocab_sentinels(monkeypatch):
    model = Qwen3VLModel.__new__(Qwen3VLModel)
    model.model_config = SimpleNamespace(pretrained_config=SimpleNamespace(disable_fuse_rope=True))
    model.use_deepstack = True
    model.deepstack_num_level = 3
    model.support_mm_disagg = True
    model.mm_token_ids = torch.tensor([151655, 151656, 151937], dtype=torch.int32)

    class _FakeLlm:
        def __init__(self):
            self.model = SimpleNamespace(embed_tokens=SimpleNamespace(num_embeddings=151936))
            self.forward_kwargs = None

        def forward(self, **kwargs):
            self.forward_kwargs = kwargs
            return torch.ones((kwargs["inputs_embeds"].shape[0], 1), dtype=torch.float32)

    fake_llm = _FakeLlm()
    model.llm = fake_llm
    captured = {}

    def _fake_find_input_mm_embeds(mm_embeds, multimodal_params):
        assert mm_embeds == []
        assert len(multimodal_params) == 1
        return [torch.ones((2, 16), dtype=torch.float32)]

    def _fake_fuse_input_embeds(
        embedding_layer, input_ids, mm_embeds, mm_token_ids=None, extra_embeds=None, **kwargs
    ):
        captured["mm_token_ids"] = mm_token_ids
        assert embedding_layer.num_embeddings == 151936
        assert len(mm_embeds) == 1
        return input_ids, torch.ones((input_ids.numel(), 4)), extra_embeds

    monkeypatch.setattr(qwen3vl_module, "_is_disagg", lambda: True)
    monkeypatch.setattr(qwen3vl_module, "find_input_mm_embeds", _fake_find_input_mm_embeds)
    monkeypatch.setattr(qwen3vl_module, "fuse_input_embeds", _fake_fuse_input_embeds)

    attn_metadata = SimpleNamespace(num_contexts=1, num_generations=0)
    multimodal_params = [
        MultimodalParams(
            multimodal_data={
                "multimodal_embedding": torch.ones((2, 16), dtype=torch.float32),
            }
        )
    ]

    result = qwen3vl_module.Qwen3VLModelBase.forward(
        model,
        attn_metadata=attn_metadata,
        input_ids=torch.tensor([1, 151655, 151655, 2], dtype=torch.int32),
        multimodal_params=multimodal_params,
    )

    torch.testing.assert_close(captured["mm_token_ids"], model.mm_token_ids)
    torch.testing.assert_close(
        fake_llm.forward_kwargs["inputs_embeds"], torch.ones((4, 4), dtype=torch.float32)
    )
    torch.testing.assert_close(result, torch.ones((4, 1), dtype=torch.float32))


@dataclass(repr=False)
class TestQwen3VLScenario(MultimodalScenario):
    disable_fuse_rope: bool = False

    def __repr__(self) -> str:
        """Generate a human-readable string representation of the scenario."""
        features = []
        features.append(f"modality:{self.modality.lower()}")
        if self.use_cuda_graph:
            features.append("cuda_graph")
        if self.disable_fuse_rope:
            features.append("no_fuse_rope")
        if self.chunked_prefill:
            features.append("chunked_prefill")
        if self.kv_cache_reuse:
            features.append("kv_cache_reuse")
        return "-".join(features)


class TestQwen3VL(TestModelingMultimodal):
    def get_model_config(self):
        """Return the model configuration dictionary."""
        return QWEN3_VL_8B_CONFIG

    def get_trtllm_model_class(self):
        return Qwen3VLModel

    def get_hf_model_class(self):
        return HFQwen3VLForConditionalLM

    def get_weight_mapper_class(self):
        return Qwen3VLHfWeightMapper

    def get_model_type(self):
        return "qwen3_vl"

    def get_model_config_class(self):
        return Qwen3VLConfig

    def get_trtllm_inputs(
        self,
        input_ids,
        multimodal_params_list,
        is_gen: bool = False,
        num_cached_tokens_per_seq: List[int] = None,
        total_prompt_len: Optional[int] = None,
    ):
        trtllm_inputs = super().get_trtllm_inputs(
            input_ids,
            multimodal_params_list,
            is_gen,
            num_cached_tokens_per_seq,
            total_prompt_len=total_prompt_len,
        )

        if is_gen:
            mrope_gen_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_gen_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]["mrope_position_deltas"]
                )
            mrope_gen_position_ids = torch.cat(mrope_gen_position_ids, dim=-1).to(self.device)
            trtllm_inputs["position_ids"] = (
                (trtllm_inputs["position_ids"] + mrope_gen_position_ids).expand(3, -1, 1).cuda()
            )
            gen_multimodal_params_list = []
            for multimodal_param in multimodal_params_list:
                multimodal_param.strip_for_generation()
                multimodal_param.to_device(
                    "multimodal_data",
                    self.device,
                    pin_memory=True,
                    target_keywords=["mrope_config.mrope_position_deltas"],
                )
                gen_multimodal_params_list.append(multimodal_param)
            trtllm_inputs["multimodal_params"] = gen_multimodal_params_list
        else:
            # Mrope position ids
            mrope_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]["mrope_position_ids"]
                )
            position_ids = torch.cat(mrope_position_ids, dim=-1)
            position_ids = position_ids.cuda()
            trtllm_inputs["position_ids"] = position_ids

        return trtllm_inputs

    def init_kv_cache_manager(self, scenario: TestQwen3VLScenario):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen3-VL model."""
        cache_config = self.get_kv_cache_config(scenario)
        tokens_per_block = cache_config["tokens_per_block"]
        max_seq_len = cache_config["max_seq_len"]
        batch_size = cache_config["batch_size"]

        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block

        self.kv_cache_manager = self.get_kv_cache_manager(
            dtype=self.model_config.pretrained_config.torch_dtype,
            config=self.model_config.pretrained_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        self.kv_cache_manager.add_dummy_requests(
            request_ids=[1],
            token_nums=[max_seq_len],
            # NOTE: Qwen3-VL model uses mrope
            use_mrope=True,
        )

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen3-VL model."""
        if not use_cuda_graph:
            trtllm_inputs["attn_metadata"].prepare()
            return self.trtllm_model.forward(**trtllm_inputs)
        else:
            # NOTE: Qwen3-VL model uses mrope
            graph_runner = create_mock_cuda_graph_runner(1, True)
            trtllm_inputs["attn_metadata"] = trtllm_inputs[
                "attn_metadata"
            ].create_cuda_graph_metadata(1)

            # Prepare metadata before capture (like in working Qwen2.5-VL test)
            trtllm_inputs["attn_metadata"].prepare()

            key = (1, 0, False)
            graph_runner.capture(
                key=key,
                forward_fn=lambda inputs: self.trtllm_model.forward(**inputs),
                initial_inputs=trtllm_inputs,
            )
            for _ in range(2):
                # Run it twice. This helps us catch problems if buffers are accidentally reallocated in prepare().
                trtllm_inputs["attn_metadata"].prepare()
                logits = graph_runner.replay(key=key, current_inputs=trtllm_inputs)
            return logits.clone()

    def get_scenarios(self) -> List[TestQwen3VLScenario]:
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestQwen3VLScenario(
                modality="video",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestQwen3VLScenario(
                modality="multiple_image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            # ==== CUDA Graph Scenarios ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=True,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            # ==== Chunked Prefill Scenarios ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=True,
                kv_cache_reuse=False,
            ),
            # ==== KV Cache Reuse Scenarios ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=True,
            ),
            # ==== Disable fuse rope scenarios ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=True,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
        ]
        return scenarios

    def setup_scenario(self, scenario: TestQwen3VLScenario):
        super().setup_scenario(scenario)
        if scenario.disable_fuse_rope:
            self.trtllm_model, self.model_config = self.create_trtllm_model(
                load_weights=True,
                hf_model_state_dict=self.hf_model.state_dict(),
                disable_fuse_rope=True,
            )
