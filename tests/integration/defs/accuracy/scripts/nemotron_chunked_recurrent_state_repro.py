# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Nemotron V3 Omni forced-chunked recurrent-state repro.

This script is intentionally test-adjacent rather than production code.  It
uses the TRT-LLM LLM API and a local monkeypatch trace to run two deterministic
forced-chunked invocations, then compares the prompt hash, generated token,
first-generation top-k/logits, schedule signatures, and Mamba recurrent-state
content signatures.

Lyris/container invocation used for the original issue:

    cd /code/tensorrt_llm
    python3 -m pip install -r /code/tensorrt_llm/requirements.txt --quiet
    export NEMOTRON_REPRO_WHEEL=/path/to/tensorrt_llm-1.3.0rc13-*.whl
    export NEMOTRON_V3_OMNI_FP8=/path/to/nemotron-nano-v3-omni_vea-fp8
    python3 -m pip install "$NEMOTRON_REPRO_WHEEL" --quiet
    env MPI_ENV_CLEANUP=1 CONTAINER_REMAP_ROOT=0 SRUN_EXPORT_MODE=allowlist \
      TLLM_WORKER_USE_SINGLE_PROCESS=1 TLLM_LOG_LEVEL=error \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
      python3 tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py \
        --modality text \
        --model-path "$NEMOTRON_V3_OMNI_FP8" \
        --expect reproduced \
        --output-dir /tmp/nemotron_chunked_recurrent_state_repro

Expected issue reproduction prints a line beginning with "REPRODUCED:" and
exits zero because the default expectation is reproduced.  Control commands use
``--expect stable`` and fail nonzero if they reproduce.  On Lyris, launch this
script under ``mpirun -np 1`` and set ``TLLM_WORKER_USE_SINGLE_PROCESS=1`` so
the repeated runs stay in the traced Python process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_IMAGE_PATH = "tests/integration/test_input_files/merlion.png"
DEFAULT_MODEL_TYPE = "NemotronH_Nano_VL_V2"
DEFAULT_TEXT_PADDING_REPEATS = 240
PADDING_SENTENCE = (
    "This deterministic padding sentence records a stable observation about "
    "mountains, rivers, clouds, minerals, forest trails, and weather patterns."
)
REPO_ROOT = Path(__file__).resolve().parents[5]
if (REPO_ROOT / "tensorrt_llm").is_dir() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr).encode("utf-8")
    )


def _sha256_ints(values: list[int]) -> str:
    return _sha256_json(values)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _read_jsonl(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    events = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def _first_mismatch(left: list[Any], right: list[Any]) -> dict[str, Any] | None:
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return {"index": index, "left": left_item, "right": right_item}
    if len(left) == len(right):
        return None
    index = min(len(left), len(right))
    return {
        "index": index,
        "left": None if index >= len(left) else left[index],
        "right": None if index >= len(right) else right[index],
    }


def _tensor_signature(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    return {
        "shape": summary.get("shape"),
        "dtype": summary.get("dtype"),
        "numel": summary.get("numel"),
        "sha256": summary.get("sha256"),
        "norm": summary.get("norm"),
        "mean": summary.get("mean"),
    }


def _state_index_signature(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature = []
    for event in events:
        if event.get("event_type") != "state_index":
            continue
        signature.append(
            {
                "event_index": event.get("event_index"),
                "host_state_indices": event.get("host_state_indices"),
                "decisions": [
                    {
                        "request_id": item.get("request_id"),
                        "reason": item.get("reason"),
                        "context_current_position": item.get("context_current_position"),
                        "context_chunk_size": item.get("context_chunk_size"),
                        "context_end_next_step": item.get("context_end_next_step"),
                        "prompt_len_minus_1": item.get("prompt_len_minus_1"),
                        "selected_next_step": item.get("selected_next_step"),
                        "selected_block_index": item.get("selected_block_index"),
                    }
                    for item in event.get("decisions", [])
                ],
            }
        )
    return signature


def _mamba_content_signature(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature = []
    for event in events:
        if event.get("event_type") != "mamba_forward":
            continue
        signature.append(
            {
                "event_index": event.get("event_index"),
                "layer_idx": event.get("layer_idx"),
                "phase": event.get("phase"),
                "num_prefills": event.get("num_prefills"),
                "num_decodes": event.get("num_decodes"),
                "state_indices": event.get("state_indices"),
                "before_state_slots": [
                    {
                        "slot": item.get("slot"),
                        "conv": _tensor_signature(item.get("conv")),
                        "ssm": _tensor_signature(item.get("ssm")),
                    }
                    for item in event.get("before_state_slots", [])
                ],
                "after_state_slots": [
                    {
                        "slot": item.get("slot"),
                        "conv": _tensor_signature(item.get("conv")),
                        "ssm": _tensor_signature(item.get("ssm")),
                    }
                    for item in event.get("after_state_slots", [])
                ],
            }
        )
    return signature


def _prepare_signature(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signature = []
    for event in events:
        if event.get("event_type") != "prepare_tp_inputs":
            continue
        signature.append(
            {
                "event_index": event.get("event_index"),
                "num_context_requests": event.get("num_context_requests"),
                "num_generation_requests": event.get("num_generation_requests"),
                "context_requests": [
                    {
                        "request_id": item.get("request_id"),
                        "prompt_len": item.get("prompt_len"),
                        "prompt_sha256": item.get("prompt_sha256"),
                        "context_current_position": item.get("context_current_position"),
                        "context_chunk_size": item.get("context_chunk_size"),
                        "context_end_position": item.get("context_end_position"),
                        "context_slice_sha256": item.get("context_slice_sha256"),
                        "mm_data_keys": item.get("mm_data_keys"),
                    }
                    for item in event.get("context_requests", [])
                ],
                "generation_request_count": len(event.get("generation_requests", [])),
                "input_ids": _tensor_signature(event.get("input_ids")),
                "position_ids": _tensor_signature(event.get("position_ids")),
                "request_ids": event.get("request_ids"),
                "seq_lens": event.get("seq_lens"),
            }
        )
    return signature


def _topk_signature(record: dict[str, Any]) -> list[dict[str, Any]]:
    topk = record.get("generation_topk") or {}
    return [
        {
            "step": item.get("step"),
            "top1_token_id": item.get("top1_token_id"),
            "top1_logit": item.get("top1_logit"),
            "margin_1_2": item.get("margin_1_2"),
            "top": item.get("top"),
        }
        for item in topk.get("steps", [])
    ]


def compare_outputs(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _read_json(left_path)
    right = _read_json(right_path)
    left_events = _read_jsonl(left.get("trace_path"))
    right_events = _read_jsonl(right.get("trace_path"))

    left_prepare = _prepare_signature(left_events)
    right_prepare = _prepare_signature(right_events)
    left_state_index = _state_index_signature(left_events)
    right_state_index = _state_index_signature(right_events)
    left_content = _mamba_content_signature(left_events)
    right_content = _mamba_content_signature(right_events)
    left_topk = _topk_signature(left)
    right_topk = _topk_signature(right)

    same_prompt = left.get("prompt_len") == right.get("prompt_len") and left.get(
        "prompt_token_sha256"
    ) == right.get("prompt_token_sha256")
    prepare_match = left_prepare == right_prepare
    state_index_match = left_state_index == right_state_index
    content_match = left_content == right_content
    topk_match = left_topk == right_topk
    tokens_match = left.get("gen_token_ids") == right.get("gen_token_ids")
    schedule_match = same_prompt and prepare_match and state_index_match
    reproduced = schedule_match and (not content_match or not topk_match)

    if not same_prompt:
        hypothesis_call = "prompt_diverged"
    elif not prepare_match:
        hypothesis_call = "prepare_tp_inputs_schedule_diverged"
    elif not state_index_match:
        hypothesis_call = "mamba_state_index_schedule_diverged"
    elif not content_match:
        hypothesis_call = "recurrent_state_content_diverged"
    elif not topk_match:
        hypothesis_call = "first_generation_logits_diverged"
    elif not tokens_match:
        hypothesis_call = "generated_token_diverged"
    else:
        hypothesis_call = "no_divergence"

    return {
        "left_path": str(left_path),
        "right_path": str(right_path),
        "same_prompt": same_prompt,
        "prepare_signatures_match": prepare_match,
        "mamba_state_index_signatures_match": state_index_match,
        "mamba_content_signatures_match": content_match,
        "first_generation_topk_match": topk_match,
        "generated_tokens_match": tokens_match,
        "schedule_match_before_decode": schedule_match,
        "reproduced": reproduced,
        "hypothesis_call": hypothesis_call,
        "left_prompt_len": left.get("prompt_len"),
        "right_prompt_len": right.get("prompt_len"),
        "left_prompt_token_sha256": left.get("prompt_token_sha256"),
        "right_prompt_token_sha256": right.get("prompt_token_sha256"),
        "left_gen_token_ids": left.get("gen_token_ids"),
        "right_gen_token_ids": right.get("gen_token_ids"),
        "left_generation_topk": left_topk,
        "right_generation_topk": right_topk,
        "left_trace_event_count": len(left_events),
        "right_trace_event_count": len(right_events),
        "first_prepare_diff": _first_mismatch(left_prepare, right_prepare),
        "first_state_index_diff": _first_mismatch(left_state_index, right_state_index),
        "first_mamba_content_diff": _first_mismatch(left_content, right_content),
        "first_generation_topk_diff": _first_mismatch(left_topk, right_topk),
    }


def _jsonable(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _build_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        base = args.prompt
    elif args.modality == "image":
        base = "Describe the image in one concise sentence."
    elif args.modality == "video":
        base = "Describe the natural environment in the video."
    else:
        base = (
            "Summarize the diagnostic passage below in one sentence, preserving "
            "the ordered facts and avoiding speculation."
        )
    if args.text_padding_repeats <= 0:
        return base
    padding = "\n".join(
        f"{idx:04d}. {PADDING_SENTENCE}" for idx in range(args.text_padding_repeats)
    )
    return f"{base}\n\n{padding}"


def _ensure_bindings_attached() -> None:
    """Make editable precompiled installs expose bindings on the package object."""
    import importlib

    import tensorrt_llm as tllm

    bindings = importlib.import_module("tensorrt_llm.bindings")
    setattr(tllm, "bindings", bindings)
    for module_name in (
        "tensorrt_llm.bindings.internal",
        "tensorrt_llm.bindings.internal.batch_manager",
        "tensorrt_llm.bindings.executor",
    ):
        importlib.import_module(module_name)


def _reset_trace_event_counters() -> None:
    try:
        from tensorrt_llm._torch.pyexecutor import model_engine

        setattr(model_engine.PyTorchModelEngine, "_nemotron_repro_prepare_event_index", 0)
    except Exception:
        pass
    try:
        from tensorrt_llm._torch.modules.mamba import mamba2_mixer

        setattr(mamba2_mixer.Mamba2Mixer, "_nemotron_repro_forward_event_index", 0)
    except Exception:
        pass


def _append_jsonl(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=repr) + "\n")


def _active_trace_path(fallback: Path) -> Path:
    return Path(os.environ.get("NEMOTRON_CHUNKED_REPRO_TRACE_PATH", str(fallback)))


def _tensor_sha256(tensor: Any, max_hash_numel: int) -> str | None:
    try:
        if tensor is None or tensor.numel() > max_hash_numel:
            return None
        cpu = tensor.detach().contiguous().cpu()
        raw = cpu.view(__import__("torch").uint8).numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}:{exc}"


def _tensor_summary(tensor: Any, max_hash_numel: int) -> dict[str, Any] | None:
    if tensor is None:
        return None
    summary: dict[str, Any] = {
        "shape": _jsonable(getattr(tensor, "shape", None)),
        "dtype": str(getattr(tensor, "dtype", None)),
        "device": str(getattr(tensor, "device", None)),
        "numel": int(tensor.numel()) if hasattr(tensor, "numel") else None,
        "sha256": _tensor_sha256(tensor, max_hash_numel),
    }
    try:
        numeric = tensor.detach().float()
        summary.update(
            {
                "norm": float(numeric.norm().detach().cpu().item()),
                "mean": float(numeric.mean().detach().cpu().item()),
                "abs_max": float(numeric.abs().max().detach().cpu().item()),
            }
        )
    except Exception as exc:
        summary["stats_error"] = f"{type(exc).__name__}:{exc}"
    return summary


def _sha256_ints(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()


def _request_tokens(request: Any) -> list[int]:
    try:
        return list(request.get_tokens(0))
    except Exception:
        return []


def _request_summary(request: Any) -> dict[str, Any]:
    tokens = _request_tokens(request)
    begin_value = _safe_attr(request, "context_current_position")
    chunk_value = _safe_attr(request, "context_chunk_size")
    begin = begin_value if isinstance(begin_value, int) else None
    chunk = chunk_value if isinstance(chunk_value, int) else None
    end = begin + chunk if isinstance(begin, int) and isinstance(chunk, int) else None
    token_slice = tokens[begin:end] if end is not None else []
    mm_data = _safe_attr(request, "py_multimodal_data") or {}
    return {
        "request_id": _safe_attr(request, "py_request_id"),
        "prompt_len": len(tokens),
        "prompt_sha256": _sha256_ints(tokens) if tokens else None,
        "context_current_position": begin_value,
        "context_chunk_size": chunk_value,
        "context_end_position": end,
        "context_slice_sha256": _sha256_ints(token_slice) if token_slice else None,
        "mm_data_keys": sorted(mm_data.keys()) if isinstance(mm_data, dict) else [],
    }


def _state_slot_summaries(
    conv_states: Any, ssm_states: Any, slots: list[int], max_hash_numel: int
) -> list[dict[str, Any]]:
    summaries = []
    for slot in slots:
        entry: dict[str, Any] = {"slot": slot}
        try:
            entry["conv"] = _tensor_summary(conv_states[slot], max_hash_numel)
        except Exception as exc:
            entry["conv_error"] = f"{type(exc).__name__}:{exc}"
        try:
            entry["ssm"] = _tensor_summary(ssm_states[slot], max_hash_numel)
        except Exception as exc:
            entry["ssm_error"] = f"{type(exc).__name__}:{exc}"
        summaries.append(entry)
    return summaries


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        value = value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, int)]
    return []


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}:{exc}"}


def _safe_int_attr(obj: Any, name: str) -> int | None:
    value = _safe_attr(obj, name)
    return value if isinstance(value, int) else None


def _install_prepare_trace(trace_path: Path, max_hash_numel: int) -> None:
    from tensorrt_llm._torch.pyexecutor import model_engine

    engine_cls = model_engine.PyTorchModelEngine
    if getattr(engine_cls, "_nemotron_repro_prepare_sitepatched", False):
        return
    original_prepare = engine_cls._prepare_tp_inputs

    def traced_prepare(self, scheduled_requests, kv_cache_manager, attn_metadata, *args, **kwargs):
        event = {
            "event_type": "prepare_tp_inputs",
            "event_index": getattr(engine_cls, "_nemotron_repro_prepare_event_index", 0),
            "num_context_requests": getattr(scheduled_requests, "num_context_requests", None),
            "num_generation_requests": len(
                getattr(scheduled_requests, "generation_requests", []) or []
            ),
            "context_requests": [
                _request_summary(req) for req in getattr(scheduled_requests, "context_requests", [])
            ],
            "generation_requests": [
                _request_summary(req)
                for req in getattr(scheduled_requests, "generation_requests", [])
            ],
        }
        setattr(engine_cls, "_nemotron_repro_prepare_event_index", event["event_index"] + 1)
        result = original_prepare(
            self, scheduled_requests, kv_cache_manager, attn_metadata, *args, **kwargs
        )
        inputs = result[0] if isinstance(result, tuple) else result
        event["input_ids"] = (
            _tensor_summary(inputs.get("input_ids"), max_hash_numel)
            if isinstance(inputs, dict)
            else None
        )
        event["position_ids"] = (
            _tensor_summary(inputs.get("position_ids"), max_hash_numel)
            if isinstance(inputs, dict)
            else None
        )
        event["request_ids"] = _jsonable(getattr(attn_metadata, "request_ids", None))
        event["seq_lens"] = _jsonable(getattr(attn_metadata, "seq_lens", None))
        _append_jsonl(_active_trace_path(trace_path), event)
        return result

    engine_cls._prepare_tp_inputs = traced_prepare
    engine_cls._nemotron_repro_prepare_sitepatched = True


def _install_state_index_trace(trace_path: Path) -> None:
    from tensorrt_llm._torch.pyexecutor import mamba_cache_manager

    manager_cls = mamba_cache_manager.CppMambaHybridCacheManager
    if getattr(manager_cls, "_nemotron_repro_state_sitepatched", False):
        return
    original_setup = manager_cls._setup_state_indices

    def traced_setup(self) -> None:
        decisions = []
        for idx, req in enumerate(self.requests):
            begin = _safe_int_attr(req, "context_current_position")
            chunk = _safe_int_attr(req, "context_chunk_size")
            prompt_len = _safe_int_attr(req, "prompt_len")
            context_end_next_step = None
            if isinstance(begin, int) and isinstance(chunk, int):
                context_end_next_step = begin - 1 + chunk
            if _safe_attr(req, "is_context_finished") is True:
                next_step = self.get_num_tokens(req) - 1
                reason = "context_finished"
            elif self.kv_cache_config.enable_block_reuse and context_end_next_step is not None:
                next_step = context_end_next_step
                reason = "block_reuse_context_end"
            elif isinstance(prompt_len, int):
                next_step = prompt_len - 1
                reason = "nonreuse_prompt_len_minus_1"
            else:
                next_step = None
                reason = "unknown"
            block_index = next_step // self.tokens_per_block if isinstance(next_step, int) else None
            decisions.append(
                {
                    "ordinal": idx,
                    "request_id": _safe_attr(req, "py_request_id"),
                    "reason": reason,
                    "context_current_position": begin,
                    "context_chunk_size": chunk,
                    "context_end_next_step": context_end_next_step,
                    "prompt_len": prompt_len,
                    "prompt_len_minus_1": prompt_len - 1 if isinstance(prompt_len, int) else None,
                    "selected_next_step": next_step,
                    "tokens_per_block": self.tokens_per_block,
                    "selected_block_index": block_index,
                }
            )
        result = original_setup(self)
        event = {
            "event_type": "state_index",
            "event_index": getattr(self, "_nemotron_repro_state_event_index", 0),
            "manager": type(self).__name__,
            "request_count": len(self.requests),
            "decisions": decisions,
            "host_state_indices": _jsonable(getattr(self, "_host_state_indices", None)),
        }
        setattr(self, "_nemotron_repro_state_event_index", event["event_index"] + 1)
        _append_jsonl(_active_trace_path(trace_path), event)
        return result

    manager_cls._setup_state_indices = traced_setup
    manager_cls._nemotron_repro_state_sitepatched = True


def _install_mamba_forward_trace(trace_path: Path, max_hash_numel: int) -> None:
    from tensorrt_llm._torch.modules.mamba import mamba2_mixer

    mixer_cls = mamba2_mixer.Mamba2Mixer
    if getattr(mixer_cls, "_nemotron_repro_forward_sitepatched", False):
        return
    original_forward = mixer_cls.forward

    def traced_forward(self, *args, **kwargs):
        attn_metadata = kwargs.get("attn_metadata")
        if attn_metadata is None and len(args) > 1:
            attn_metadata = args[1]
        mamba_metadata = kwargs.get("mamba_metadata")
        if mamba_metadata is None and len(args) > 2:
            mamba_metadata = args[2]
        event = {
            "event_type": "mamba_forward",
            "event_index": getattr(mixer_cls, "_nemotron_repro_forward_event_index", 0),
            "layer_idx": getattr(self, "layer_idx", None),
            "module": type(self).__name__,
            "num_prefills": getattr(attn_metadata, "num_contexts", None),
            "seq_lens": _jsonable(getattr(attn_metadata, "seq_lens", None)),
            "state_indices": _as_int_list(getattr(mamba_metadata, "state_indices", None)),
        }
        setattr(mixer_cls, "_nemotron_repro_forward_event_index", event["event_index"] + 1)
        num_prefills = event["num_prefills"]
        seq_count = len(event["seq_lens"]) if isinstance(event["seq_lens"], list) else None
        num_decodes = (
            seq_count - num_prefills
            if isinstance(seq_count, int) and isinstance(num_prefills, int)
            else None
        )
        event["num_decodes"] = num_decodes
        if isinstance(num_prefills, int) and isinstance(num_decodes, int):
            if num_prefills > 0 and num_decodes == 0:
                event["phase"] = "context"
            elif num_prefills == 0 and num_decodes > 0:
                event["phase"] = "decode"
            else:
                event["phase"] = "mixed"
        slots = sorted({slot for slot in event["state_indices"] if isinstance(slot, int)})
        try:
            kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
            layer_idx = getattr(self, "layer_idx", None)
            event["before_state_slots"] = _state_slot_summaries(
                kv_cache_manager.get_conv_states(layer_idx),
                kv_cache_manager.get_ssm_states(layer_idx),
                slots,
                max_hash_numel,
            )
        except Exception as exc:
            event["before_state_error"] = f"{type(exc).__name__}:{exc}"
        result = original_forward(self, *args, **kwargs)
        try:
            kv_cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
            layer_idx = getattr(self, "layer_idx", None)
            event["after_state_slots"] = _state_slot_summaries(
                kv_cache_manager.get_conv_states(layer_idx),
                kv_cache_manager.get_ssm_states(layer_idx),
                slots,
                max_hash_numel,
            )
        except Exception as exc:
            event["after_state_error"] = f"{type(exc).__name__}:{exc}"
        _append_jsonl(_active_trace_path(trace_path), event)
        return result

    mixer_cls.forward = traced_forward
    mixer_cls._nemotron_repro_forward_sitepatched = True


def _install() -> None:
    trace_value = os.environ.get("NEMOTRON_CHUNKED_REPRO_TRACE_PATH")
    if not trace_value:
        return
    trace_path = Path(trace_value)
    max_hash_numel = int(os.environ.get("NEMOTRON_CHUNKED_REPRO_MAX_HASH_NUMEL", "50000000"))
    try:
        _install_prepare_trace(trace_path, max_hash_numel)
        _install_state_index_trace(trace_path)
        _install_mamba_forward_trace(trace_path, max_hash_numel)
    except Exception as exc:
        _append_jsonl(
            trace_path,
            {
                "event_type": "hook_install_failed",
                "error": f"{type(exc).__name__}:{exc}",
            },
        )


def install_trace_hooks_from_env() -> None:
    _install()


def _sitecustomize_source() -> str:
    return """
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

script_value = os.environ.get("NEMOTRON_CHUNKED_REPRO_SCRIPT_PATH")
if script_value:
    script_path = Path(script_value)
    spec = importlib.util.spec_from_file_location(
        "_nemotron_chunked_recurrent_state_repro_hooks", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load repro hook module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_trace_hooks_from_env()
"""


def _write_sitecustomize(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sitecustomize.py"
    path.write_text(_sitecustomize_source().lstrip(), encoding="utf-8")
    return path


def _ensure_sitecustomize_imported() -> None:
    install_trace_hooks_from_env()


def _install_current_process_hooks() -> None:
    _ensure_sitecustomize_imported()


def _torch_determinism(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_inputs(args: argparse.Namespace, llm: Any, prompt: str) -> list[Any]:
    if args.modality == "text":
        return [prompt]

    from tensorrt_llm.inputs import default_multimodal_input_loader, prompt_inputs

    if args.modality == "image":
        loaded = default_multimodal_input_loader(
            tokenizer=llm.tokenizer,
            model_dir=args.model_path,
            model_type=args.model_type,
            modality="image",
            prompts=[prompt],
            media=[args.image_path],
            image_data_format="pt",
            device="cpu",
        )
        return [prompt_inputs(item) for item in loaded]

    loaded = default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=args.model_path,
        model_type=args.model_type,
        modality="video",
        prompts=[prompt],
        media=[args.video_path],
        image_data_format="pt",
        num_frames=args.num_frames,
        device="cpu",
    )
    return [prompt_inputs(item) for item in loaded]


def _flatten_generation_logits(logits: Any) -> Any:
    import torch

    if logits is None:
        return None
    if isinstance(logits, torch.Tensor):
        tensor = logits.detach()
    elif isinstance(logits, (list, tuple)):
        tensors = [
            item
            for item in (_flatten_generation_logits(value) for value in logits)
            if item is not None
        ]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)
    else:
        return None
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    while tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor.cpu().float()


def _capture_generation_topk(logits: Any, top_k: int) -> dict[str, Any] | None:
    import torch

    flat = _flatten_generation_logits(logits)
    if flat is None:
        return None
    k = min(top_k, flat.shape[-1])
    values, indices = torch.topk(flat, k=k, dim=-1)
    steps = []
    for step, (step_indices, step_values) in enumerate(zip(indices, values)):
        top = [
            {"token_id": int(token_id), "logit": float(logit)}
            for token_id, logit in zip(step_indices.tolist(), step_values.tolist())
        ]
        steps.append(
            {
                "step": step,
                "top": top,
                "top1_token_id": top[0]["token_id"] if top else None,
                "top1_logit": top[0]["logit"] if top else None,
                "margin_1_2": top[0]["logit"] - top[1]["logit"] if len(top) >= 2 else None,
            }
        )
    return {"shape": list(flat.shape), "dtype": str(flat.dtype), "top_k": k, "steps": steps}


def run_single(args: argparse.Namespace) -> int:
    os.environ["NEMOTRON_CHUNKED_REPRO_TRACE_PATH"] = str(args.trace_path)
    os.environ["NEMOTRON_CHUNKED_REPRO_MAX_HASH_NUMEL"] = str(args.trace_max_hash_numel)
    Path(args.trace_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace_path).write_text("", encoding="utf-8")
    _torch_determinism(args.seed)

    import torch

    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams

    _ensure_bindings_attached()
    _install_current_process_hooks()
    _reset_trace_event_counters()

    prompt = _build_prompt(args)
    effective_max_num_tokens = (
        args.chunked_max_num_tokens
        if args.enable_chunked_prefill
        else args.nonchunked_max_num_tokens
    )
    mode = "chunked" if args.enable_chunked_prefill else "nonchunked"
    llm_kwargs: dict[str, Any] = {
        "max_batch_size": 1,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_tokens": effective_max_num_tokens,
        "kv_cache_config": KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=args.kv_cache_fraction,
            mamba_ssm_cache_dtype="float32",
        ),
        "gather_generation_logits": True,
    }
    llm_kwargs["cuda_graph_config"] = None if args.disable_cuda_graph else CudaGraphConfig()

    started = time.time()
    with LLM(args.model_path, trust_remote_code=True, **llm_kwargs) as llm:
        inputs = _build_inputs(args, llm, prompt)
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
            seed=args.seed,
            add_special_tokens=False,
            ignore_eos=True,
            logprobs=args.logprobs_top_k,
            return_generation_logits=True,
        )
        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)

    output = outputs[0]
    generated = output.outputs[0]
    prompt_token_ids = list(output.prompt_token_ids)
    gen_token_ids = list(generated.token_ids)
    generation_topk = _capture_generation_topk(generated.generation_logits, args.capture_top_k)
    result = {
        "mode": mode,
        "modality": args.modality,
        "model_type": args.model_type,
        "model_path": args.model_path,
        "image_path": args.image_path if args.modality == "image" else None,
        "video_path": args.video_path if args.modality == "video" else None,
        "num_frames": args.num_frames if args.modality == "video" else None,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_tokens": effective_max_num_tokens,
        "chunked_max_num_tokens": args.chunked_max_num_tokens,
        "nonchunked_max_num_tokens": args.nonchunked_max_num_tokens,
        "max_new_tokens": args.max_new_tokens,
        "text_padding_repeats": args.text_padding_repeats,
        "prompt_len": len(prompt_token_ids),
        "prompt_token_sha256": _sha256_ints(prompt_token_ids),
        "generated_token_count": len(gen_token_ids),
        "gen_token_ids": gen_token_ids,
        "gen_text": generated.text,
        "generation_topk": generation_topk,
        "generation_logprobs": _jsonable(generated.logprobs),
        "generation_logits_present": generated.generation_logits is not None,
        "trace_path": str(args.trace_path),
        "elapsed_s": time.time() - started,
        "host": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tllm_worker_use_single_process": os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS"),
        "tllm_log_level": os.environ.get("TLLM_LOG_LEVEL"),
    }
    Path(args.output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    forced = len(prompt_token_ids) > effective_max_num_tokens
    print(
        f"single-run {args.run_label}: mode={mode} prompt_len={len(prompt_token_ids)} "
        f"max_num_tokens={effective_max_num_tokens} forced={forced} "
        f"tokens={gen_token_ids} output={args.output_path}",
        flush=True,
    )
    if args.enable_chunked_prefill and not forced:
        raise RuntimeError(
            f"prompt_len={len(prompt_token_ids)} did not force chunking with "
            f"max_num_tokens={effective_max_num_tokens}"
        )
    return 0


def _subprocess_env(output_dir: Path, sitecustomize_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{sitecustomize_dir}{os.pathsep}{prior_pythonpath}"
        if prior_pythonpath
        else str(sitecustomize_dir)
    )
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    env.setdefault("TLLM_LOG_LEVEL", "error")
    env["NEMOTRON_CHUNKED_REPRO_SITE_DIR"] = str(sitecustomize_dir)
    env["NEMOTRON_CHUNKED_REPRO_SCRIPT_PATH"] = str(Path(__file__).resolve())
    env["NEMOTRON_CHUNKED_REPRO_OUTPUT_DIR"] = str(output_dir)
    return env


def _cleanup_after_single_process_run() -> None:
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def run_driver(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_in_current_process = _env_flag("TLLM_WORKER_USE_SINGLE_PROCESS")

    run_paths = []
    for index in range(args.runs):
        label = f"run{index + 1}"
        output_path = output_dir / f"{label}.json"
        trace_path = output_dir / f"{label}.trace.jsonl"
        if run_in_current_process:
            single_args = argparse.Namespace(**vars(args))
            single_args.single_run = True
            single_args.run_label = label
            single_args.output_path = str(output_path)
            single_args.trace_path = str(trace_path)
            run_single(single_args)
            _cleanup_after_single_process_run()
            run_paths.append(output_path)
            continue

        sitecustomize_dir = output_dir / "sitecustomize"
        _write_sitecustomize(sitecustomize_dir)
        env = _subprocess_env(output_dir, sitecustomize_dir)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single-run",
            "--run-label",
            label,
            "--output-path",
            str(output_path),
            "--trace-path",
            str(trace_path),
            "--model-path",
            args.model_path,
            "--modality",
            args.modality,
            "--model-type",
            args.model_type,
            "--chunked-max-num-tokens",
            str(args.chunked_max_num_tokens),
            "--nonchunked-max-num-tokens",
            str(args.nonchunked_max_num_tokens),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--text-padding-repeats",
            str(args.text_padding_repeats),
            "--seed",
            str(args.seed),
            "--kv-cache-fraction",
            str(args.kv_cache_fraction),
            "--capture-top-k",
            str(args.capture_top_k),
            "--logprobs-top-k",
            str(args.logprobs_top_k),
            "--trace-max-hash-numel",
            str(args.trace_max_hash_numel),
        ]
        if not args.enable_chunked_prefill:
            command.append("--disable-chunked-prefill")
        if args.prompt:
            command.extend(["--prompt", args.prompt])
        if args.disable_cuda_graph:
            command.append("--disable-cuda-graph")
        if args.modality == "image":
            command.extend(["--image-path", args.image_path])
        if args.modality == "video":
            command.extend(["--video-path", args.video_path, "--num-frames", str(args.num_frames)])
        print(f"launching {label}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, env=env, text=True)
        if completed.returncode != 0:
            return completed.returncode
        run_paths.append(output_path)

    if len(run_paths) < 2:
        raise RuntimeError("--runs must be at least 2")
    summary = compare_outputs(run_paths[0], run_paths[1])
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["reproduced"]:
        print(
            "REPRODUCED: prompt/state schedule matched while "
            f"{summary['hypothesis_call']} ({summary_path})",
            flush=True,
        )
    else:
        print(f"NOT_REPRODUCED: {summary['hypothesis_call']} ({summary_path})", flush=True)
    if args.expect == "reproduced" and not summary["reproduced"]:
        print(
            f"EXPECTATION_FAILED: expected reproduction but got {summary['hypothesis_call']}",
            flush=True,
        )
        return 2
    if args.expect == "stable" and summary["reproduced"]:
        print(
            f"EXPECTATION_FAILED: expected stability but got {summary['hypothesis_call']}",
            flush=True,
        )
        return 2
    if args.expect != "any":
        print(f"EXPECTATION_PASSED: {args.expect}", flush=True)
    return 0


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="nemotron_repro_selftest_") as tmp:
        root = Path(tmp)
        sitecustomize = _write_sitecustomize(root / "sitecustomize")
        compile(sitecustomize.read_text(encoding="utf-8"), str(sitecustomize), "exec")
        trace1 = root / "run1.trace.jsonl"
        trace2 = root / "run2.trace.jsonl"
        common_prepare = {
            "event_type": "prepare_tp_inputs",
            "event_index": 0,
            "num_context_requests": 1,
            "num_generation_requests": 0,
            "context_requests": [
                {
                    "request_id": 1,
                    "prompt_len": 300,
                    "prompt_sha256": "abc",
                    "context_current_position": 0,
                    "context_chunk_size": 128,
                    "context_end_position": 128,
                    "context_slice_sha256": "slice",
                    "mm_data_keys": [],
                }
            ],
            "generation_requests": [],
            "input_ids": {"shape": [128], "dtype": "torch.int64", "sha256": "ids"},
            "position_ids": {"shape": [128], "dtype": "torch.int64", "sha256": "pos"},
            "request_ids": [1],
            "seq_lens": [128],
        }
        common_state = {
            "event_type": "state_index",
            "event_index": 0,
            "host_state_indices": [0],
            "decisions": [
                {
                    "request_id": 1,
                    "reason": "nonreuse_prompt_len_minus_1",
                    "context_current_position": 0,
                    "context_chunk_size": 128,
                    "context_end_next_step": 127,
                    "prompt_len_minus_1": 299,
                    "selected_next_step": 299,
                    "selected_block_index": 2,
                }
            ],
        }
        left_forward = {
            "event_type": "mamba_forward",
            "event_index": 0,
            "layer_idx": 0,
            "phase": "context",
            "num_prefills": 1,
            "num_decodes": 0,
            "state_indices": [0],
            "before_state_slots": [{"slot": 0, "ssm": {"sha256": "zero"}}],
            "after_state_slots": [{"slot": 0, "ssm": {"sha256": "left"}}],
        }
        right_forward = {
            **left_forward,
            "after_state_slots": [{"slot": 0, "ssm": {"sha256": "right"}}],
        }
        trace1.write_text(
            "\n".join(json.dumps(item) for item in [common_prepare, common_state, left_forward])
            + "\n",
            encoding="utf-8",
        )
        trace2.write_text(
            "\n".join(json.dumps(item) for item in [common_prepare, common_state, right_forward])
            + "\n",
            encoding="utf-8",
        )
        for index, trace in enumerate([trace1, trace2], start=1):
            (root / f"run{index}.json").write_text(
                json.dumps(
                    {
                        "prompt_len": 300,
                        "prompt_token_sha256": "abc",
                        "gen_token_ids": [11],
                        "generation_topk": None,
                        "trace_path": str(trace),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        summary = compare_outputs(root / "run1.json", root / "run2.json")
        if not summary["reproduced"]:
            raise AssertionError(summary)
        if summary["hypothesis_call"] != "recurrent_state_content_diverged":
            raise AssertionError(summary)
    print("SELF_TEST_OK: argument-free compare and sitecustomize compile paths passed")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=os.environ.get("NEMOTRON_REPRO_MODEL") or os.environ.get("NEMOTRON_V3_OMNI_FP8"),
        help=(
            "Path to the model checkpoint. Defaults to NEMOTRON_REPRO_MODEL, "
            "then NEMOTRON_V3_OMNI_FP8."
        ),
    )
    parser.add_argument(
        "--model-type",
        default=os.environ.get("NEMOTRON_REPRO_MODEL_TYPE", DEFAULT_MODEL_TYPE),
        help=(
            "Model type passed to default_multimodal_input_loader for image/video "
            f"inputs. Defaults to NEMOTRON_REPRO_MODEL_TYPE or {DEFAULT_MODEL_TYPE}."
        ),
    )
    parser.add_argument("--modality", choices=("text", "image", "video"), default="text")
    parser.add_argument(
        "--image-path",
        default=os.environ.get("NEMOTRON_REPRO_IMAGE", DEFAULT_IMAGE_PATH),
        help=(
            "Image path used only with --modality image. Defaults to "
            "NEMOTRON_REPRO_IMAGE or a repo-local test image."
        ),
    )
    parser.add_argument(
        "--video-path",
        default=os.environ.get("NEMOTRON_REPRO_VIDEO"),
        help="Video path used only with --modality video. Defaults to NEMOTRON_REPRO_VIDEO.",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--prompt")
    parser.add_argument("--output-dir", default="/tmp/nemotron_chunked_recurrent_state_repro")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--disable-chunked-prefill",
        action="store_false",
        dest="enable_chunked_prefill",
        default=True,
        help="Control run: disable chunked prefill and expect stable output with --expect stable.",
    )
    parser.add_argument("--chunked-max-num-tokens", type=int, default=128)
    parser.add_argument("--nonchunked-max-num-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--text-padding-repeats", type=int, default=DEFAULT_TEXT_PADDING_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kv-cache-fraction", type=float, default=0.6)
    parser.add_argument("--capture-top-k", type=int, default=8)
    parser.add_argument("--logprobs-top-k", type=int, default=8)
    parser.add_argument("--trace-max-hash-numel", type=int, default=50_000_000)
    parser.add_argument("--disable-cuda-graph", action="store_true", default=True)
    parser.add_argument(
        "--expect",
        choices=("reproduced", "stable", "any"),
        default=None,
        help=(
            "Expected comparison outcome. Defaults to reproduced when chunked prefill "
            "is enabled and stable when --disable-chunked-prefill is set."
        ),
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-label", default="run", help=argparse.SUPPRESS)
    parser.add_argument("--output-path", help=argparse.SUPPRESS)
    parser.add_argument("--trace-path", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.runs < 2 and not args.single_run and not args.self_test:
        parser.error("--runs must be at least 2")
    if args.single_run and (not args.output_path or not args.trace_path):
        parser.error("--single-run requires --output-path and --trace-path")
    if not args.self_test and not args.model_path:
        parser.error("--model-path or NEMOTRON_V3_OMNI_FP8 is required")
    if args.expect is None:
        args.expect = "reproduced" if args.enable_chunked_prefill else "stable"
    if args.modality == "video" and not args.video_path:
        parser.error("--video-path or NEMOTRON_REPRO_VIDEO is required for --modality video")
    if args.enable_chunked_prefill and args.chunked_max_num_tokens <= 0:
        parser.error("--chunked-max-num-tokens must be positive")
    if not args.enable_chunked_prefill and args.nonchunked_max_num_tokens <= 0:
        parser.error("--nonchunked-max-num-tokens must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.single_run:
        return run_single(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
