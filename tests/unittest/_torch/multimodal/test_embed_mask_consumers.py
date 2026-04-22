# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Consumer-side tests for the multimodal embed-mask flow.

The embed mask is a flat bool tensor produced at intake
(``registry.compute_mm_embed_mask_if_absent`` stores it at
``py_multimodal_data["multimodal_embed_mask"]``). At the worker, it feeds
``MultimodalRuntimeData`` via a single ``cumsum``; ``find_input_mm_embeds``
slices encoder rows off that cumsum. Tests for slicing live in
``test_multimodal_runtime.py``; this file stays as a placeholder so removed
symbols can't silently reappear.
"""
