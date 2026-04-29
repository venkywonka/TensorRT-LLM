# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import llm_request as llm_request_module
from tensorrt_llm._torch.pyexecutor.llm_request import (
    PyResult,
    executor_request_to_llm_request,
    get_mm_encoder_output_lengths,
    split_mm_embeddings_by_request,
)
from tensorrt_llm._torch.pyexecutor.sampler import (
    EarlyStopWithMMResult,
    MultimodalResult,
    SampleStateWithMMResult,
)

_MISSING = object()


def _request(prompt_lengths, encoder_lengths=None, direct_encoder_lengths=_MISSING):
    layout_metadata = {}
    if encoder_lengths is not None:
        layout_metadata["encoder_output_lengths"] = encoder_lengths
        layout_metadata["modalities"] = ["image"] * len(encoder_lengths)
    attrs = {
        "multimodal_lengths": prompt_lengths,
        "py_multimodal_data": {"layout_metadata": layout_metadata} if layout_metadata else {},
    }
    if direct_encoder_lengths is not _MISSING:
        attrs["py_mm_encoder_output_lengths"] = direct_encoder_lengths
    return SimpleNamespace(**attrs)


def test_request_encoder_output_lengths_prefer_direct_request_field():
    request = _request([5, 7], [2, 3], direct_encoder_lengths=[4, 6])

    assert get_mm_encoder_output_lengths(request) == [4, 6]


def test_request_encoder_output_lengths_falls_back_to_layout_metadata():
    request = _request([5, 7], [2, 3])

    assert get_mm_encoder_output_lengths(request) == [2, 3]


def test_executor_request_to_llm_request_attaches_direct_encoder_lengths(monkeypatch):
    class _FakeSamplingConfig:
        def __init__(self, executor_sampling_config):
            self.executor_sampling_config = executor_sampling_config

    class _FakeLlmRequest:
        def __init__(self, **kwargs):
            self.child_requests = []
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.py_end_id = self.end_id
            self.py_mm_encoder_output_lengths = kwargs.get("py_mm_encoder_output_lengths")

    monkeypatch.setattr(llm_request_module, "SamplingConfig", _FakeSamplingConfig)
    monkeypatch.setattr(llm_request_module, "LlmRequest", _FakeLlmRequest)

    request_type = next(iter(llm_request_module.REQUEST_TYPE_MAPPING))
    output_config = SimpleNamespace(
        return_log_probs=False,
        return_context_logits=False,
        return_perf_metrics=False,
        return_generation_logits=False,
        additional_model_outputs=None,
        exclude_input_from_output=False,
    )
    multimodal_input = SimpleNamespace(
        multimodal_hashes=[[1] * 8, [2] * 8],
        multimodal_positions=[0, 8],
        multimodal_lengths=[5, 7],
        multimodal_uuids=None,
        multimodal_encoder_output_lengths=[2, 3],
    )
    executor_request = SimpleNamespace(
        sampling_config=object(),
        input_token_ids=[10, 11],
        request_type=request_type,
        stop_words=None,
        bad_words=None,
        multimodal_input=multimodal_input,
        mrope_config=None,
        max_tokens=1,
        streaming=False,
        end_id=2,
        pad_id=0,
        embedding_bias=None,
        prompt_tuning_config=None,
        multimodal_embedding=None,
        lora_config=None,
        output_config=output_config,
        guided_decoding_params=None,
        client_id=None,
        priority=0,
        context_phase_params=None,
        cache_salt_id=None,
        kv_cache_retention_config=None,
    )

    request = executor_request_to_llm_request(
        req_id=42,
        executor_request=executor_request,
        child_req_ids=[],
        exclude_last_generation_logits=False,
    )

    assert request.py_mm_encoder_output_lengths == [2, 3]
    assert get_mm_encoder_output_lengths(request) == [2, 3]


def test_split_mm_embeddings_by_request_uses_encoder_output_lengths():
    requests = [_request([5, 7], [2, 3]), _request([9], [4])]
    mm_embeddings = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)

    split_embeddings = split_mm_embeddings_by_request(
        [mm_embeddings],
        requests,
        num_context_requests=2,
    )

    assert [embedding.shape[0] for embedding in split_embeddings] == [5, 4]
    torch.testing.assert_close(split_embeddings[0], mm_embeddings[:5])
    torch.testing.assert_close(split_embeddings[1], mm_embeddings[5:])


def test_split_mm_embeddings_by_request_falls_back_to_legacy_lengths():
    requests = [_request([5, 7]), _request([4])]
    mm_embeddings = torch.arange(16 * 2, dtype=torch.float32).reshape(16, 2)

    split_embeddings = split_mm_embeddings_by_request(
        [mm_embeddings],
        requests,
        num_context_requests=2,
    )

    assert [embedding.shape[0] for embedding in split_embeddings] == [12, 4]


def test_split_mm_embeddings_by_request_rejects_partial_lengths():
    requests = [
        _request([5, 7], [2, 3]),
        SimpleNamespace(multimodal_lengths=None, py_multimodal_data={}),
    ]
    mm_embeddings = torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2)

    with pytest.raises(ValueError, match="must be present for all requests"):
        split_mm_embeddings_by_request(
            [mm_embeddings],
            requests,
            num_context_requests=2,
        )


def test_split_mm_embeddings_by_request_rejects_modality_ordered_outputs():
    requests = [
        SimpleNamespace(
            multimodal_lengths=[2],
            py_multimodal_data={
                "layout_metadata": {
                    "encoder_output_lengths": [2],
                    "modalities": ["video"],
                }
            },
        ),
        SimpleNamespace(
            multimodal_lengths=[2],
            py_multimodal_data={
                "layout_metadata": {
                    "encoder_output_lengths": [2],
                    "modalities": ["image"],
                }
            },
        ),
    ]
    image_tensor = torch.full((2, 2), 1.0)
    video_tensor = torch.full((2, 2), 2.0)

    with pytest.raises(ValueError, match="single concatenated"):
        split_mm_embeddings_by_request(
            [image_tensor, video_tensor],
            requests,
            num_context_requests=2,
        )


def test_py_result_append_mm_embeddings_splits_item_handles_by_encoder_lengths(monkeypatch):
    class _FakeSharedTensorContainer:
        def __init__(self, tensor):
            self.tensor = tensor

        @classmethod
        def from_tensor(cls, tensor):
            return cls(tensor.clone())

        def dump_to_dict(self):
            return {
                "shape": list(self.tensor.shape),
                "first_column": self.tensor[:, 0].tolist(),
            }

    monkeypatch.setattr(llm_request_module, "SharedTensorContainer", _FakeSharedTensorContainer)

    request = _request([5, 7], [2, 3])
    mm_embeddings = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    result = PyResult(prompt_len=0, max_new_tokens=1)

    result.append_mm_embeddings(mm_embeddings, get_mm_encoder_output_lengths(request))

    assert result.mm_embedding_handles == [
        {
            "shape": [2, 2],
            "first_column": [0.0, 2.0],
        },
        {
            "shape": [3, 2],
            "first_column": [4.0, 6.0, 8.0],
        },
    ]


def test_early_stop_with_mm_result_uses_encoder_output_lengths():
    class _RecordingPyResult:
        def __init__(self):
            self.calls = []

        def append_mm_embeddings(self, mm_embeddings, multimodal_lengths):
            self.calls.append((mm_embeddings.clone(), list(multimodal_lengths)))

    class _FakeRequest:
        multimodal_lengths = [5, 7]
        py_multimodal_data = {
            "layout_metadata": {
                "encoder_output_lengths": [2, 3],
                "modalities": ["image", "video"],
            }
        }

        def __init__(self):
            self.py_result = _RecordingPyResult()
            self.finished_reason = None
            self.state = None

        def set_finished_reason(self, finish_reason, beam_idx):
            self.finished_reason = (finish_reason, beam_idx)

    request = _FakeRequest()
    mm_embeddings = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    state = SampleStateWithMMResult(
        requests=[request],
        data=MultimodalResult(mm_embeddings=[mm_embeddings]),
    )

    EarlyStopWithMMResult().update_requests(state)

    assert request.py_result.calls[0][1] == [2, 3]
    torch.testing.assert_close(request.py_result.calls[0][0], mm_embeddings)
