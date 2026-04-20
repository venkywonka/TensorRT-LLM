# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for require_mm_spans_if_needed and _has_mm_payload_keys.

Covers the partial-iteration-aware gate introduced to replace
_check_mm_spans_present. See docs/superpowers/specs/
2026-04-20-mm-span-enforcement-redesign.md.
"""

import logging

import pytest
import torch

from tensorrt_llm.inputs.multimodal import _has_mm_payload_keys, require_mm_spans_if_needed
from tensorrt_llm.logger import logger as _trtllm_logger

# Key used by require_mm_spans_if_needed's one-shot warning. The TRT-LLM
# logger's `_appeared_keys` set is a process-wide singleton, so tests that
# assert on the warning must reset this key around each call to get
# deterministic, order-independent results.
_WARN_ONCE_KEY = "mm_spans_missing_non_partial"


@pytest.fixture(autouse=True)
def _reset_log_once_key():
    """Clear the warning-dedup key before and after each test."""
    _trtllm_logger._appeared_keys.discard(_WARN_ONCE_KEY)
    yield
    _trtllm_logger._appeared_keys.discard(_WARN_ONCE_KEY)


def _capture_trtllm_warnings(fn):
    """Run `fn` and return log records emitted on the TRT-LLM logger.

    The TRT-LLM logger uses the `TRT-LLM` name and sets `propagate=False`,
    so pytest's `caplog` (which hooks the root logger) doesn't capture its
    output. We attach a handler directly to the TRT-LLM logger instead.
    """
    trt_logger = logging.getLogger("TRT-LLM")
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _ListHandler(level=logging.WARNING)
    prev_level = trt_logger.level
    trt_logger.setLevel(logging.WARNING)
    trt_logger.addHandler(handler)
    try:
        fn()
    finally:
        trt_logger.removeHandler(handler)
        trt_logger.setLevel(prev_level)
    return records


# ---- _has_mm_payload_keys ------------------------------------------------


class TestHasMmPayloadKeys:
    def test_none(self):
        assert _has_mm_payload_keys(None) is False

    def test_empty_dict(self):
        assert _has_mm_payload_keys({}) is False

    def test_metadata_only_mrope(self):
        assert _has_mm_payload_keys({"mrope_config": {}}) is False

    def test_metadata_only_combined(self):
        data = {
            "mrope_config": {},
            "mm_contiguous_spans": [(0, 5)],
            "special_token_offsets": [1],
            "layout_metadata": {},
        }
        assert _has_mm_payload_keys(data) is False

    def test_image_payload(self):
        assert _has_mm_payload_keys({"image": {}}) is True

    def test_video_payload(self):
        assert _has_mm_payload_keys({"video": {}}) is True

    def test_payload_plus_metadata(self):
        data = {"image": {}, "mrope_config": {}}
        assert _has_mm_payload_keys(data) is True


# ---- require_mm_spans_if_needed -----------------------------------------


class TestRequireMmSpansIfNeeded:
    """Seven cases covering the partial-iteration-aware gate."""

    def test_no_mm_payload_never_raises(self):
        """Case 1: no MM payload → no-op under any partial-ness."""
        # Full prefill
        require_mm_spans_if_needed(None, begin_compute=0, end_compute=100, prompt_len=100)
        # Chunked
        require_mm_spans_if_needed(None, begin_compute=0, end_compute=50, prompt_len=100)
        # KV reuse
        require_mm_spans_if_needed(None, begin_compute=50, end_compute=100, prompt_len=100)

    def test_mm_payload_spans_present_partial_no_raise(self):
        """Case 2: MM payload + spans + partial → no-op."""
        data = {"image": {}, "mm_contiguous_spans": [(10, 20)]}
        require_mm_spans_if_needed(data, begin_compute=5, end_compute=50, prompt_len=100)

    def test_mm_payload_spans_present_full_no_raise(self):
        """Case 3: MM payload + spans + full prefill → no-op."""
        data = {"image": {}, "mm_contiguous_spans": [(10, 20)]}
        require_mm_spans_if_needed(data, begin_compute=0, end_compute=100, prompt_len=100)

    def test_mm_payload_spans_missing_full_warns(self):
        """Case 4: MM payload + no spans + non-partial → warn, no raise."""
        data = {"image": {"pixel_values": torch.zeros(1, 1)}}
        records = _capture_trtllm_warnings(
            lambda: require_mm_spans_if_needed(
                data, begin_compute=0, end_compute=100, prompt_len=100
            )
        )
        # The warning is emitted at least once. Assert on a stable substring.
        assert any("mm_contiguous_spans missing" in r.getMessage() for r in records)

    def test_mm_payload_spans_missing_kv_reuse_raises(self):
        """Case 5: MM payload + no spans + begin_compute > 0 → raise."""
        data = {"image": {"pixel_values": torch.zeros(1, 1)}}
        with pytest.raises(ValueError, match="partial iteration.*begin_compute=20"):
            require_mm_spans_if_needed(data, begin_compute=20, end_compute=100, prompt_len=100)

    def test_mm_payload_spans_missing_chunked_raises(self):
        """Case 6: MM payload + no spans + end_compute < prompt_len → raise."""
        data = {"image": {"pixel_values": torch.zeros(1, 1)}}
        with pytest.raises(ValueError, match="partial iteration.*end_compute=50"):
            require_mm_spans_if_needed(data, begin_compute=0, end_compute=50, prompt_len=100)

    def test_metadata_only_payload_never_raises(self):
        """Case 7: metadata-only payload (mrope_config etc.) → no-op."""
        data = {"mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 5)}}
        # All three scenarios: full, chunked, KV-reuse
        require_mm_spans_if_needed(data, begin_compute=0, end_compute=100, prompt_len=100)
        require_mm_spans_if_needed(data, begin_compute=0, end_compute=50, prompt_len=100)
        require_mm_spans_if_needed(data, begin_compute=20, end_compute=100, prompt_len=100)

    def test_warning_dedup(self):
        """Warning fires only once per process for the same key."""
        data = {"image": {"pixel_values": torch.zeros(1, 1)}}

        def _run_thrice():
            require_mm_spans_if_needed(data, begin_compute=0, end_compute=100, prompt_len=100)
            require_mm_spans_if_needed(data, begin_compute=0, end_compute=100, prompt_len=100)
            require_mm_spans_if_needed(data, begin_compute=0, end_compute=100, prompt_len=100)

        records = _capture_trtllm_warnings(_run_thrice)
        # logger.warning_once(key=...) should dedupe across three calls.
        # The autouse fixture resets the key before this test runs, so we
        # can assert on an exact count of 1 (first call emits, next two dedup).
        span_warnings = [r for r in records if "mm_contiguous_spans missing" in r.getMessage()]
        assert len(span_warnings) == 1, (
            f"Expected exactly 1 warning due to dedup; got "
            f"{len(span_warnings)}: "
            f"{[r.getMessage() for r in span_warnings]}"
        )
