# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Multimodal utilities for handling images and other media types in TensorRT-LLM."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from blake3 import blake3
from torchvision.transforms import ToPILImage

import tensorrt_llm
from tensorrt_llm._utils import maybe_pin_memory
from tensorrt_llm.logger import logger

# Default hasher
default_hasher = blake3


@dataclass
class MultimodalInput:
    """Per-logical-unit multimodal metadata for KV-cache hashing (C++ layer).

    Indexed per logical unit (one image, one video), NOT per contiguous token
    run. Non-contiguous tokens (e.g. video frames with text separators) are
    tracked via ``mm_contiguous_spans`` in ``py_multimodal_data``.
    """

    multimodal_hashes: List[List[int]]
    """Hash digest per logical unit (list of 8 int32 each)."""

    multimodal_positions: List[int]
    """Starting token position of each logical unit. For non-contiguous units
    this is the position of the *first* token."""

    multimodal_lengths: List[int]
    """Total token count per logical unit, including special tokens.
    For non-contiguous units this is the sum across all contiguous runs."""

    multimodal_uuids: Optional[List[Optional[str]]] = None
    """Optional user-provided UUIDs for logical multimodal units.

    When provided, these UUIDs will be returned in KV cache events instead of the
    computed hash hex string. This enables deterministic cache identification across
    sessions using user-defined stable identifiers.

    Each element can be:
    - A string UUID: Used as the cache identifier (returned in events)
    - None: Falls back to content-based hashing for that unit

    If the UUID string is longer than 32 bytes, it will be hashed internally
    for cache key computation, but the original UUID string is preserved and
    returned in KV cache events.
    """

    multimodal_is_embeds: Optional[List[Optional[torch.Tensor]]] = None
    """Per-logical-unit bool mask over outer-box positions. None iff not yet
    materialized; each entry None iff unit has no inline specials. Python-only —
    NOT forwarded to the C++ tle::MultimodalInput.  See
    slop/mm_is_embed_migration/goals.md §3.1.
    """

    def __post_init__(self):
        """Validate input data structure and consistency."""
        # Validate multimodal_hashes
        if not isinstance(self.multimodal_hashes, list):
            raise TypeError("multimodal_hashes must be a list")

        # Check that hashes are lists of consistent length containing integers
        if not all(isinstance(h, list) for h in self.multimodal_hashes):
            raise TypeError("Each element in multimodal_hashes must be a list")

        # Check consistent length of hash arrays
        hash_lengths = [len(h) for h in self.multimodal_hashes]
        if min(hash_lengths) != max(hash_lengths):
            raise ValueError(
                f"All hash arrays must have the same length, got lengths: {hash_lengths}"
            )

        # Check that positions and lengths are valid
        if not all(isinstance(x, int) for x in self.multimodal_positions):
            raise TypeError("multimodal_positions must contain only integers")

        if not all(isinstance(x, int) for x in self.multimodal_lengths):
            raise TypeError("multimodal_lengths must contain only integers")

        # Check position and length arrays match in size
        if len(self.multimodal_positions) != len(self.multimodal_lengths):
            raise ValueError(
                f"Position and length arrays must match in size: "
                f"positions={len(self.multimodal_positions)}, lengths={len(self.multimodal_lengths)}"
            )

        # Validate multimodal_uuids if provided
        if self.multimodal_uuids is not None:
            if not isinstance(self.multimodal_uuids, list):
                raise TypeError("multimodal_uuids must be a list")
            if len(self.multimodal_uuids) != len(self.multimodal_hashes):
                raise ValueError(
                    f"multimodal_uuids length ({len(self.multimodal_uuids)}) must match "
                    f"multimodal_hashes length ({len(self.multimodal_hashes)})")
            for i, uuid in enumerate(self.multimodal_uuids):
                if uuid is not None and not isinstance(uuid, str):
                    raise TypeError(
                        f"multimodal_uuids[{i}] must be a string or None, got {type(uuid)}"
                    )

    @classmethod
    def from_components(
        cls,
        mm_hashes: List[List[int]],
        mm_positions: List[int],
        mm_lengths: List[int],
        mm_uuids: Optional[List[Optional[str]]] = None,
    ) -> 'MultimodalInput':
        return cls(multimodal_hashes=mm_hashes,
                   multimodal_positions=mm_positions,
                   multimodal_lengths=mm_lengths,
                   multimodal_uuids=mm_uuids)

    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert data to tensors"""
        return (
            # int32 to match the type in TRTLLM SizeType32
            torch.tensor(self.multimodal_hashes, dtype=torch.int32),
            torch.tensor(self.multimodal_positions, dtype=torch.int32),
            torch.tensor(self.multimodal_lengths, dtype=torch.int32))

    def materialize_is_embed(
        self,
        prompt_token_ids: torch.Tensor,
        vocab_size: int,
        mm_token_ids: Optional[torch.Tensor] = None,
        mm_special_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Idempotent materialization of is_embed_flat.

        Source order (see slop/mm_is_embed_migration/goals.md §4.2-§4.3):
          1. self.multimodal_is_embeds if populated -> stitch into flat.
          2. mm_token_ids if available -> isin(prompt, mm_token_ids), then
             subtract specials when declared.
          3. vocab_size fallback -> prompt >= vocab_size, then subtract specials.

        Paths 2 and 3 preserve the legacy `filter_mm_token_from_input_ids`
        predicate byte-for-byte — the zero-regression guarantee.
        """
        # Second call is a no-op (backstop may fire after intake already ran).
        if self.__dict__.get("_is_embed_flat") is not None:
            return self.__dict__["_is_embed_flat"]

        if not isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids = torch.as_tensor(prompt_token_ids)

        # Path 1: stitch from per-unit masks if the processor populated them.
        if self.multimodal_is_embeds is not None:
            mask = torch.zeros(prompt_token_ids.shape[0],
                               dtype=torch.bool,
                               device=prompt_token_ids.device)
            for i, per_unit in enumerate(self.multimodal_is_embeds):
                pos = self.multimodal_positions[i]
                length = self.multimodal_lengths[i]
                if per_unit is None:
                    mask[pos:pos + length] = True
                else:
                    per_unit = per_unit.to(device=prompt_token_ids.device,
                                           dtype=torch.bool)
                    mask[pos:pos + length] = per_unit
            return self._store_is_embed(mask)

        # Path 2/3: predicate over prompt_token_ids.
        if mm_token_ids is not None:
            mm_token_ids = mm_token_ids.to(device=prompt_token_ids.device,
                                           dtype=prompt_token_ids.dtype)
            mask = torch.isin(prompt_token_ids, mm_token_ids)
        else:
            mask = prompt_token_ids >= vocab_size

        if mm_special_token_ids is not None:
            mm_special_token_ids = mm_special_token_ids.to(
                device=prompt_token_ids.device, dtype=prompt_token_ids.dtype)
            mask = mask & ~torch.isin(prompt_token_ids, mm_special_token_ids)

        return self._store_is_embed(mask)

    def _store_is_embed(self, mask: torch.Tensor) -> torch.Tensor:
        # Write mask into the backing slot AND invalidate any stale
        # @cached_property values (which may have been populated with None by a
        # pre-materialization access). Without this, is_embed_flat and
        # is_embed_cumsum return stale None after materialize_is_embed runs.
        self.__dict__["_is_embed_flat"] = mask
        self.__dict__.pop("is_embed_flat", None)
        self.__dict__.pop("is_embed_cumsum", None)
        return mask

    @cached_property
    def is_embed_flat(self) -> Optional[torch.Tensor]:
        """Return the cached bool[request_seq_len] mask, or None if not materialized.

        Note: materialize_is_embed writes into self.__dict__["_is_embed_flat"]
        directly, so this property reads the cache when available; when it
        hasn't been called, returns None without attempting lazy stitching
        (caller must pass prompt_token_ids + vocab via materialize_is_embed).
        """
        return self.__dict__.get("_is_embed_flat")

    @cached_property
    def is_embed_cumsum(self) -> Optional[torch.Tensor]:
        """Return int64 prefix sum of is_embed_flat, or None when mask is absent.

        Enables O(1) range queries via get_chunk_embed_indices.
        """
        if self.is_embed_flat is None:
            return None
        return self.is_embed_flat.to(dtype=torch.int64).cumsum(0)

    def get_chunk_embed_indices(self, chunk_start: int,
                                chunk_end: int) -> Tuple[int, int]:
        """Translate [chunk_start, chunk_end) to [enc_lo, enc_hi) via cumsum.

        Input positions are in the full prompt; output positions are row
        indices into the encoder output tensor. See goals.md §5.2.
        """
        cs = self.is_embed_cumsum
        enc_lo = int(cs[chunk_start - 1]) if chunk_start > 0 else 0
        enc_hi = int(cs[chunk_end - 1]) if chunk_end > 0 else 0
        return enc_lo, enc_hi


@dataclass
class MultimodalRuntimeData:
    """Runtime data for tracking multimodal token caching and reuse per request sequence.

    Two construction modes:

    - LEGACY (sparse): pass ``mm_contiguous_spans`` + ``special_token_offsets``;
      counts derived via interval arithmetic. Path will be dropped in Commit 6.
    - CUMSUM (mask-based): pass ``is_embed_cumsum`` (from
      ``MultimodalInput.is_embed_cumsum``); counts derived via three O(1)
      cumsum lookups. Preferred for non-contiguous MM with specials.

    Attributes:
        past_seen_token_num: Total tokens already processed in previous iterations (cached)
        chunk_end_pos: End position of the current chunk for chunked prefill
        mm_contiguous_spans: LEGACY — list of (start_position, token_count) per run
        special_token_offsets: LEGACY — indices of specials in the flat MM token union
        is_embed_cumsum: CUMSUM — int64 prefix sum of is_embed_flat from MultimodalInput

        num_cached_mm_tokens: Number of MM tokens that are cached (computed)
        num_mm_tokens_in_chunk: Number of MM tokens in the current chunk (computed)
        total_mm_tokens_in_request: Total MM tokens (legacy name; kept for back-compat)
        total_embeds_in_request: Total embed slots (cumsum path; excludes specials)

        num_cached_special_tokens, num_special_tokens_in_chunk, total_special_tokens_in_request:
            LEGACY — dropped in Commit 6
    """
    past_seen_token_num: int
    chunk_end_pos: int

    mm_contiguous_spans: Optional[List[Tuple[int, int]]] = None
    special_token_offsets: Optional[List[int]] = None
    is_embed_cumsum: Optional[torch.Tensor] = None

    num_cached_mm_tokens: Optional[int] = None
    num_mm_tokens_in_chunk: Optional[int] = None
    total_mm_tokens_in_request: Optional[int] = None
    total_embeds_in_request: Optional[int] = None

    num_cached_special_tokens: Optional[int] = 0
    num_special_tokens_in_chunk: Optional[int] = 0
    total_special_tokens_in_request: Optional[int] = 0

    def __post_init__(self):
        if self.past_seen_token_num < 0:
            raise ValueError(
                f"past_seen_token_num must be non-negative, got {self.past_seen_token_num}"
            )

        # CUMSUM path: preferred when mask is available.
        if self.is_embed_cumsum is not None:
            cs = self.is_embed_cumsum
            self.num_cached_mm_tokens = (int(cs[self.past_seen_token_num - 1])
                                         if self.past_seen_token_num > 0 else 0)
            self.num_mm_tokens_in_chunk = (int(cs[self.chunk_end_pos - 1]) -
                                           self.num_cached_mm_tokens
                                           if self.chunk_end_pos > 0 else 0)
            self.total_embeds_in_request = int(cs[-1])
            # Mirror to legacy field name for consumers not yet migrated.
            self.total_mm_tokens_in_request = self.total_embeds_in_request
            return

        # LEGACY interval-arithmetic path. Requires mm_contiguous_spans.
        if self.mm_contiguous_spans is None:
            raise ValueError(
                "MultimodalRuntimeData requires either is_embed_cumsum or "
                "mm_contiguous_spans; both were None.")

        if self.total_mm_tokens_in_request is None:
            self.total_mm_tokens_in_request = sum(
                length for _, length in self.mm_contiguous_spans)

        if any(length <= 0 for _, length in self.mm_contiguous_spans):
            raise ValueError(
                f"All span lengths must be positive, got {self.mm_contiguous_spans}"
            )

        if any(pos < 0 for pos, _ in self.mm_contiguous_spans):
            raise ValueError(
                f"All span positions must be non-negative, got {self.mm_contiguous_spans}"
            )

        if self.special_token_offsets is None:
            self.special_token_offsets = []

        remainder = 0
        if self.num_cached_mm_tokens is None or self.num_mm_tokens_in_chunk is None:
            self.num_cached_mm_tokens = 0
            self.num_mm_tokens_in_chunk = 0
            for pos, length in self.mm_contiguous_spans:
                span_end = pos + length
                if span_end <= self.past_seen_token_num:
                    self.num_cached_mm_tokens += length
                elif pos < self.past_seen_token_num:
                    self.num_cached_mm_tokens += self.past_seen_token_num - pos
                    self.num_mm_tokens_in_chunk += min(
                        self.chunk_end_pos, span_end) - self.past_seen_token_num
                else:
                    if span_end > self.chunk_end_pos:
                        if pos < self.chunk_end_pos:
                            self.num_mm_tokens_in_chunk += self.chunk_end_pos - pos
                        else:
                            remainder += length
                    else:
                        self.num_mm_tokens_in_chunk += length

        if len(self.special_token_offsets) > 0:
            mm_tokens_end_pos = (self.num_cached_mm_tokens +
                                 self.num_mm_tokens_in_chunk)
            self.num_cached_special_tokens = sum(
                1 for offset in self.special_token_offsets
                if offset < self.num_cached_mm_tokens)
            self.num_special_tokens_in_chunk = sum(
                1 for offset in self.special_token_offsets
                if self.num_cached_mm_tokens <= offset < mm_tokens_end_pos)

            self.total_special_tokens_in_request = len(
                self.special_token_offsets)

        if (self.num_cached_mm_tokens + self.num_mm_tokens_in_chunk + remainder
                > self.total_mm_tokens_in_request):
            raise ValueError(
                f"num_cached_mm_tokens ({self.num_cached_mm_tokens}) + "
                f"num_mm_tokens_in_chunk ({self.num_mm_tokens_in_chunk}) + "
                f"remainder ({remainder}) must be <= total "
                f"({self.total_mm_tokens_in_request})")


@dataclass
class MultimodalParams:
    """Unified container for multimodal parameters.

    This class encapsulates all multimodal-related data that flows through the system,
    providing a clean interface for handling multimodal inputs across different models.

    Attributes:
        multimodal_input: Multimodal input data with hashing information.
        multimodal_data: Processed multimodal data containing embeddings, configurations,
                        and modality-specific data organized by type.
        multimodal_runtime: Runtime data for tracking multimodal token caching and reuse
                           during KV cache scenarios. Contains information about cached
                           tokens, multimodal token positions, and lengths for efficient
                           processing during inference.

    Structure of multimodal_data:
        {
            "mrope_config": {
                "mrope_rotary_cos_sin": torch.Tensor,    # Rotary embeddings (Qwen2/2.5-VL)
                "mrope_position_deltas": torch.Tensor,   # Position deltas (Qwen2/2.5-VL)
            },
            "multimodal_embedding": torch.Tensor,        # Pre-computed vision embeddings
            "image": {
                "pixel_values": torch.Tensor,
                "image_height": torch.Tensor | List[int],
                "image_width": torch.Tensor | List[int],
            },
            "video": {
                "pixel_values": torch.Tensor,
                "video_height": torch.Tensor | List[int],
                "video_width": torch.Tensor | List[int],
            },
            "special_token_offsets": List[int],          # List of starting positions of special tokens in the union of all multimodal token chunks, if available
            # ... other modalities
        }
    """

    multimodal_input: Optional[MultimodalInput] = None
    multimodal_data: Optional[Dict[str, Any]] = field(default_factory=dict)
    multimodal_runtime: Optional[MultimodalRuntimeData] = None

    def __post_init__(self):
        """Ensure default values are properly set."""
        if self.multimodal_data is None:
            self.multimodal_data = {}

    def _is_shared_tensor_dict(self, obj: Any) -> bool:
        """Check if an object is a shared tensor dictionary.

        Args:
            obj: Object to check

        Returns:
            True if the object is a shared tensor dictionary, False otherwise
        """
        if not isinstance(obj, dict):
            return False

        # Check for required keys that uniquely identify a shared tensor dict
        required_keys = {'method_key'}
        if not required_keys.issubset(obj.keys()):
            return False

        # Additional validation based on method_key
        method_key = obj.get('method_key')

        # Import here to avoid circular imports
        from tensorrt_llm._torch.shared_tensor import \
            _SharedTensorRebuildMethodRegistry

        if method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CUDA:
            cuda_keys = {'tensor_size', 'storage_handle', 'storage_device'}
            return cuda_keys.issubset(obj.keys())
        elif method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CPU:
            cpu_keys = {'tensor_size', 'storage_handle', 'manager_handle'}
            return cpu_keys.issubset(obj.keys())

        return False

    def _apply_tensor_operation(
            self, input_data: Union[torch.Tensor, List, dict, None],
            operation: str, **kwargs) -> Union[torch.Tensor, List, dict, None]:
        """Apply tensor operations recursively to nested data structures.

        This method handles three types of operations:
        - "to_handle": Convert tensors to shared tensor dictionaries
        - "to_tensor": Convert shared tensor dictionaries back to tensors
        - "to_device": Move tensors to specified device

        Args:
            input_data: Input data structure (tensor, list, dict, or None)
            operation: Operation to apply
            **kwargs: Additional arguments for the operation

        Returns:
            Transformed data structure
        """
        # Handle None case
        if input_data is None:
            return None

        # Handle list case - recursively process each element
        if isinstance(input_data, list):
            return [
                self._apply_tensor_operation(item, operation, **kwargs)
                for item in input_data
            ]

        # Handle dictionary case
        if isinstance(input_data, dict):
            if operation == "to_tensor" and self._is_shared_tensor_dict(
                    input_data):
                # Convert shared tensor dict back to tensor
                try:
                    # Import here to avoid circular imports
                    from tensorrt_llm._torch.shared_tensor import \
                        SharedTensorContainer

                    return SharedTensorContainer.from_dict(
                        input_data).get_local_view()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to restore tensor from shared tensor dict: {e}"
                    )
            else:
                # Regular dictionary - recursively process values
                return {
                    key: self._apply_tensor_operation(value, operation,
                                                      **kwargs)
                    for key, value in input_data.items()
                }

        # Handle tensor case
        if isinstance(input_data, torch.Tensor):
            if operation == "to_handle":
                try:
                    # Import here to avoid circular imports
                    from tensorrt_llm._torch.shared_tensor import \
                        SharedTensorContainer
                    return SharedTensorContainer.from_tensor(
                        input_data).dump_to_dict()
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to convert tensor to shared tensor: {e}")
            elif operation == "to_device":
                device = kwargs.get('device')
                if device is None:
                    raise ValueError(
                        "Device must be specified for 'to_device' operation")

                pin_memory = kwargs.get('pin_memory', False)
                try:
                    if pin_memory and input_data.device.type == 'cpu':
                        return maybe_pin_memory(input_data).to(
                            device, non_blocking=True)
                    else:
                        return input_data.to(device, non_blocking=True)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to move tensor to device {device}: {e}")

        # For any other type, return as-is
        return input_data

    def to_handle(self, element: str) -> None:
        """Move specified multimodal data element to shared tensor.

        Args:
            element: Element to move (only "multimodal_data" is supported)

        Raises:
            ValueError: If element is not "multimodal_data"
            RuntimeError: If tensor conversion fails
        """
        if element != "multimodal_data":
            raise ValueError(
                f"Unsupported element '{element}'. Only 'multimodal_data' is supported."
            )

        data = getattr(self, element)
        if data is None:
            return  # Nothing to convert

        transformed_data = self._apply_tensor_operation(data, "to_handle")
        setattr(self, element, transformed_data)

    def to_tensor(self, element: str) -> None:
        """Move specified multimodal data element from shared tensor.

        Args:
            element: Element to restore (only "multimodal_data" is supported)

        Raises:
            ValueError: If element is not "multimodal_data"
            RuntimeError: If tensor restoration fails
        """
        if element != "multimodal_data":
            raise ValueError(
                f"Unsupported element '{element}'. Only 'multimodal_data' is supported."
            )

        data = getattr(self, element)
        if data is None:
            return  # Nothing to restore

        restored_data = self._apply_tensor_operation(data, "to_tensor")
        setattr(self, element, restored_data)

    def to_device(self,
                  element: str,
                  device: str,
                  pin_memory: bool = False,
                  target_keywords: Optional[List[str]] = None) -> None:
        """Move specified multimodal data element to target device.

        Args:
            element: Element to move (only "multimodal_data" is supported)
            device: Target device (e.g., "cuda", "cpu")
            pin_memory: Whether to pin memory for asynchronous transfers
            target_keywords: Optional list of keyword paths to filter which data to move.
                    Each string can be a simple key or dot-separated path
                    (e.g., ["image.pixel_values", "mrope_config"])
                    If provided, only data matching these paths will be moved to device.

        Raises:
            ValueError: If element is not "multimodal_data" or device is invalid
            RuntimeError: If device transfer fails
        """
        if element != "multimodal_data":
            raise ValueError(
                f"Unsupported element '{element}'. Only 'multimodal_data' is supported."
            )

        data = getattr(self, element)
        if data is None:
            return  # Nothing to move

        # If keyword is specified, only move data for those keyword paths
        if target_keywords is not None:
            if not isinstance(data, dict):
                raise ValueError(
                    f"multimodal_data must be a dictionary when keyword is specified, "
                    f"got {type(data)}")

            # Process multiple keyword paths
            transformed_data = self._move_multiple_paths_to_device(
                data, target_keywords, device, pin_memory)
        else:
            # Move all data as before
            transformed_data = self._apply_tensor_operation(
                data, "to_device", device=device, pin_memory=pin_memory)

        setattr(self, element, transformed_data)

    def _move_multiple_paths_to_device(self, data: Dict[str, Any],
                                       target_keywords: List[str], device: str,
                                       pin_memory: bool) -> Dict[str, Any]:
        """Move multiple nested data paths to device.

        Args:
            data: The multimodal data dictionary
            target_keywords: List of keyword paths (can be dot-separated)
            device: Target device
            pin_memory: Whether to pin memory

        Returns:
            Updated data dictionary with specified paths moved to device
        """
        result = data
        for keyword_path in target_keywords:
            # Parse each keyword path
            if '.' in keyword_path:
                key_path = keyword_path.split('.')
            else:
                key_path = [keyword_path]

            # Navigate to the target location and move data
            current = result
            parent_path = key_path[:-1]
            target_key = key_path[-1]

            # Navigate to the parent dictionary
            for key in parent_path:
                if not isinstance(current, dict) or key not in current:
                    # Path doesn't exist, skip this keyword path
                    break
                current = current[key]
            else:
                # Check if the target key exists and move it to device
                if isinstance(current, dict) and target_key in current:
                    current[target_key] = self._apply_tensor_operation(
                        current[target_key],
                        "to_device",
                        device=device,
                        pin_memory=pin_memory)

        return result

    def strip_for_generation(self) -> None:
        """Strip multimodal data for generation processing.

        Keeps only mrope_position_deltas and removes all other multimodal data
        (embeddings, images, etc.) as they're not needed during generation.
        """
        if not self.multimodal_data:
            return

        # Extract mrope_position_deltas before clearing
        mrope_position_deltas = None
        if 'mrope_config' in self.multimodal_data:
            mrope_config = self.multimodal_data['mrope_config']
            if isinstance(mrope_config,
                          dict) and 'mrope_position_deltas' in mrope_config:
                mrope_position_deltas = mrope_config['mrope_position_deltas']

        # Clear all data and restore only position deltas if they exist
        self.multimodal_data = {}
        if mrope_position_deltas is not None:
            self.multimodal_data['mrope_config'] = {
                'mrope_position_deltas': mrope_position_deltas
            }

    def has_content(self) -> bool:
        """Check if this object contains any multimodal data."""
        return bool(self.multimodal_input or self.multimodal_data)


@dataclass
class MultimodalServerConfig():
    media_io_kwargs: Optional[dict] = None


# adopt from vllm : https://github.com/vllm-project/vllm/blob/main/vllm/vllm/multimodal/hash.py
def serialize_item(obj: object) -> bytes:
    # Simple cases
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, (int, float)):
        return np.array(obj).tobytes()

    if isinstance(obj, PIL.Image.Image):
        return np.array(obj.convert("RGBA")).tobytes()
    if isinstance(obj, torch.Tensor):
        return obj.numpy().tobytes()
    if isinstance(obj, np.ndarray):
        return obj.tobytes()
    if isinstance(obj, (tuple, list)):
        # Support compound types like audio (np.ndarray, sample_rate).
        # Use length-delimited framing so sequences with different element
        # boundaries (e.g. ["ab", "c"] vs ["a", "bc"]) cannot collide.
        container_tag = b"T" if isinstance(obj, tuple) else b"L"
        parts = [container_tag, len(obj).to_bytes(8, "big", signed=False)]
        for x in obj:
            payload = serialize_item(x)
            parts.append(len(payload).to_bytes(8, "big", signed=False))
            parts.append(payload)
        return b"".join(parts)

    raise ValueError(f"Unsupported object type: {type(obj)}")


def apply_mm_hashes(
    mm_data: Dict[str, Any],
    mm_uuids: Optional[Dict[str, List[Optional[str]]]] = None,
    hash_lib=default_hasher
) -> Tuple[Dict[str, List[str]], Optional[List[Optional[str]]]]:
    """Apply hashing to multimodal data, one hash per logical multimodal unit.

    When a UUID is provided for a unit, the hash is computed from both the UUID
    and the content together: BLAKE3(UUID || Content). This ensures:
    - Cache correctness: Different content always produces different hashes
    - User isolation: Same content with different UUIDs produces different hashes
    - The original UUID string is preserved and returned in KV cache events

    Args:
        mm_data: Dictionary of modality -> data items
        mm_uuids: Optional dictionary of modality -> list of UUID strings.
                  Use None for units that should use content-based hashing only.
        hash_lib: Hash function to use (default: blake3)

    Returns:
        Tuple of:
        - Dictionary of modality -> list of hash hex strings (64 chars each)
        - Flattened list of original UUID strings (or None for content-hashed units)
    """

    def _hash_content(hasher, item):
        """Hash the content of a multimodal item into the provided hasher."""
        if isinstance(item, torch.Tensor):
            # Ensure tensor is on CPU and contiguous for consistent hashing
            item = item.detach().cpu().contiguous()
            hasher.update(serialize_item(item))
        elif isinstance(item, list):
            # Hash each frame with a separator to avoid collisions between [A,B] and [AB]
            for frame in item:
                hasher.update(b"<frame>")
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().contiguous()
                hasher.update(serialize_item(frame))
        elif isinstance(item, tensorrt_llm.inputs.utils.VideoData):
            frames = item.frames
            for frame in frames:
                hasher.update(b"<frame>")
                if isinstance(frame, torch.Tensor):
                    frame = frame.detach().cpu().contiguous()
                hasher.update(serialize_item(frame))
        else:
            hasher.update(serialize_item(item))

    def _hash_item(item):
        """Hash only the content of a multimodal item (no UUID)."""
        # TODO: possible hash collision w/ this simplified version (vllm/PR/17378)
        hasher = hash_lib()
        _hash_content(hasher, item)
        return hasher.hexdigest()

    def _hash_item_with_uuid(item, uuid: str):
        """Hash UUID and content together: BLAKE3(UUID || Content).

        This creates a unique hash that incorporates both the user-provided
        identifier and the actual content, ensuring cache correctness while
        supporting user-defined cache isolation.
        """
        hasher = hash_lib()
        # Hash UUID first with delimiters to prevent length-extension ambiguity
        hasher.update(b"<uuid>")
        hasher.update(uuid.encode('utf-8'))
        hasher.update(b"</uuid>")
        # Then hash the content
        hasher.update(b"<content>")
        _hash_content(hasher, item)
        hasher.update(b"</content>")
        return hasher.hexdigest()

    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }

    # Collect UUIDs in the same order as items
    all_uuids: List[Optional[str]] = []
    mm_hashes: Dict[str, List[str]] = {}

    for modality, items in mm_items.items():
        modality_uuids = None
        if mm_uuids is not None and modality in mm_uuids:
            modality_uuids = mm_uuids[modality]
            if not isinstance(modality_uuids, list):
                modality_uuids = [modality_uuids]
            if len(modality_uuids) != len(items):
                raise ValueError(
                    f"UUID list length ({len(modality_uuids)}) doesn't match "
                    f"data items length ({len(items)}) for modality '{modality}'"
                )

        hashes = []
        for i, item in enumerate(items):
            uuid = modality_uuids[i] if modality_uuids else None
            if uuid is not None:
                # Hash UUID + content together for cache correctness
                hashes.append(_hash_item_with_uuid(item, uuid))
                all_uuids.append(uuid)  # Store original UUID
            else:
                # Fall back to content-only hashing
                hashes.append(_hash_item(item))
                all_uuids.append(None)

        mm_hashes[modality] = hashes

    # Return None for uuids if no UUIDs were provided at all
    return mm_hashes, all_uuids if mm_uuids is not None else None


def hexdigest_to_int32(hex_digest: str) -> List[int]:
    """Convert a 256-bit hexadecimal digest to 8 int32 values."""
    if len(hex_digest) != 64:
        raise ValueError(
            f"Expected 64 character hexadecimal string, got {len(hex_digest)}")

    result = []
    for i in range(0, 64, 8):
        hex_chunk = hex_digest[i:i + 8]
        value = int(hex_chunk, 16)
        if value > 0x7FFFFFFF:  # Check if the highest bit is set (value > 2^31-1)
            value = value - 0x100000000  # Convert to signed by subtracting 2^32
        result.append(value)
    return result


def int32_to_hexdigest(int32_values: List[int]) -> str:
    """Convert 8 int32 values back to a 64-character hexadecimal digest.

    This is the inverse of hexdigest_to_int32.

    Args:
        int32_values: List of 8 signed int32 values

    Returns:
        64-character hexadecimal string representing the 32-byte hash
    """
    if len(int32_values) != 8:
        raise ValueError(f"Expected 8 int32 values, got {len(int32_values)}")

    result = []
    for value in int32_values:
        # Convert signed int32 back to unsigned
        if value < 0:
            value = value + 0x100000000
        # Format as 8 hex characters (zero-padded)
        result.append(f'{value:08x}')
    return ''.join(result)


def find_mm_token_lengths(
    mm_data: Dict[str, Any],
    input_processor: Any,
    *,
    multimodal_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[int]]:
    """Get the token lengths of each multimodal item.

    Returns the total token count for each multimodal item, including any special tokens
    (e.g., image_begin, image_end, image_break) that may be mixed with the actual
    multimodal content tokens. This mm_token_lengths represents the full chunk from beginning
    to end, not just pure image/video/audio tokens.

    When `multimodal_data["video"]["video_grid_thw"]` is present and its row
    count matches the number of videos in `mm_data`, each row is forwarded
    to `input_processor.get_num_tokens_per_video` as a kwarg. Processors are
    free to use it for a faster token-count computation or to ignore it;
    falls back to calling the method without the kwarg on mismatch / absence.
    """

    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }
    num_mm_tokens = {}

    mm_video_dict = (multimodal_data or {}).get("video") or {}
    video_grid_thw = mm_video_dict.get("video_grid_thw")

    for modality, items in mm_items.items():
        if not hasattr(input_processor, f"get_num_tokens_per_{modality}"):
            raise AttributeError(
                f"Input processor {type(input_processor).__name__} does not have 'get_num_tokens_per_{modality}' method required for multimodal hashing."
            )

        fast_path_vgt = None
        if modality == "video" and video_grid_thw is not None:
            if len(video_grid_thw) == len(items):
                fast_path_vgt = video_grid_thw
            else:
                logger.warning(
                    "find_mm_token_lengths: video_grid_thw row count "
                    f"({len(video_grid_thw)}) does not match number of "
                    f"videos in mm_data ({len(items)}); falling back to "
                    "per-item recompute without video_grid_thw.")

        modality_token_lengths = []
        for idx, item in enumerate(items):
            if modality == "image":
                if isinstance(item, torch.Tensor):
                    item = ToPILImage()(item)
                num_tokens = input_processor.get_num_tokens_per_image(
                    image=item, )
                modality_token_lengths.append(num_tokens)
            elif modality == "video":
                if isinstance(item, tensorrt_llm.inputs.utils.VideoData):
                    item = item.frames
                assert isinstance(item, list), "Video must be a list of frames"
                if isinstance(item[0], torch.Tensor):
                    item = [ToPILImage()(frame) for frame in item]
                call_kwargs = {"video": item}
                if fast_path_vgt is not None:
                    # Note: forwarding video_grid_thw does not guarantee the
                    # processor uses it; get_num_tokens_per_video is
                    # processor-dependent. Qwen3-VL consumes it (fast path);
                    # other processors may ignore it via **kwargs.
                    call_kwargs["video_grid_thw"] = fast_path_vgt[idx]
                num_tokens = input_processor.get_num_tokens_per_video(
                    **call_kwargs)
                modality_token_lengths.append(num_tokens)
            elif modality == "audio":
                num_tokens = input_processor.get_num_tokens_per_audio(
                    audio=item)
                modality_token_lengths.append(num_tokens)
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        num_mm_tokens[modality] = modality_token_lengths

    return num_mm_tokens  # flatten all mm instances to a single list


# Keys in py_multimodal_data that carry metadata (not vision/audio content).
# If py_multimodal_data has ONLY these keys, the request has no real MM
# payload (e.g. mrope-only warmup on an mrope-enabled model) and the
# require_mm_spans_if_needed gate short-circuits.
_MM_METADATA_ONLY_KEYS = frozenset({
    "mrope_config",
    "mm_contiguous_spans",
    "special_token_offsets",
    "layout_metadata",
})


def _has_mm_payload_keys(py_multimodal_data: Optional[dict]) -> bool:
    """True iff py_multimodal_data contains vision/video/audio content keys.

    Metadata-only payloads (``mrope_config`` on mrope warmup,
    ``mm_contiguous_spans`` alone, ``special_token_offsets`` alone,
    ``layout_metadata``) return False — those don't carry real MM content
    that the model needs to fuse embeddings for.
    """
    if not py_multimodal_data:
        return False
    return bool(set(py_multimodal_data.keys()) - _MM_METADATA_ONLY_KEYS)


def require_mm_spans_if_needed(
    py_multimodal_data: Optional[dict],
    *,
    begin_compute: int,
    end_compute: int,
    prompt_len: int,
) -> None:
    """Raise iff this iteration is partial AND MM data is present without spans.

    A partial iteration is one where either:
      * ``begin_compute > 0`` — a prefix was reused from KV cache, OR
      * ``end_compute < prompt_len`` — the scheduler chose to chunk.

    Partial iterations require ``mm_contiguous_spans`` to compute
    ``num_cached_mm_tokens`` and ``num_mm_tokens_in_chunk`` correctly in
    ``MultimodalRuntimeData``. Full-prefill, no-reuse iterations do not:
    ``MultimodalRuntimeData`` stays ``None`` and ``find_input_mm_embeds``
    handles the full payload via mask-based position lookup.

    When spans are missing on a non-partial iteration, log a one-shot warning
    via ``logger.warning_once`` and proceed.
    """
    if not _has_mm_payload_keys(py_multimodal_data):
        return
    if py_multimodal_data.get("mm_contiguous_spans") is not None:
        return

    is_partial = (begin_compute > 0) or (end_compute < prompt_len)
    mm_keys = set(py_multimodal_data.keys()) - _MM_METADATA_ONLY_KEYS

    if is_partial:
        raise ValueError(
            f"Request requires mm_contiguous_spans for partial iteration "
            f"(begin_compute={begin_compute}, end_compute={end_compute}, "
            f"prompt_len={prompt_len}) but py_multimodal_data has keys "
            f"{mm_keys} with no spans. The input processor may be missing a "
            f"discriminator (override get_mm_token_ids or ensure get_vocab_size "
            f"resolves).")

    logger.warning_once(
        "mm_contiguous_spans missing on multimodal request (keys=%s); "
        "running without span-aware accounting. This is fine for full-prefill "
        "iterations but will fail if this request is later chunked or reuses "
        "KV cache.",
        mm_keys,
        key="mm_spans_missing_non_partial",
    )


def find_contiguous_mm_spans(
    input_ids: Union[torch.Tensor, List[int], np.ndarray],
    vocab_size: Optional[int] = None,
    mm_token_ids: Optional[torch.Tensor] = None,
    mm_special_token_ids: Optional[torch.Tensor] = None,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Scan input_ids for contiguous runs of multimodal tokens.

    Lightweight alternative to find_mm_token_positions that does not require
    num_mm_tokens. Suitable for any code path that has token IDs and needs
    to know where contiguous blocks of MM tokens sit in the sequence.

    At least one of vocab_size or mm_token_ids must be provided.
    If mm_token_ids is provided, vocab_size is ignored.

    Args:
        input_ids: Token sequence (tensor, list, or numpy array).
        vocab_size: Vocabulary size; tokens >= vocab_size are considered MM.
        mm_token_ids: Explicit token IDs that represent multimodal tokens.
        mm_special_token_ids: Token IDs for special MM tokens (e.g. image_break).

    Returns:
        A 2-tuple of:
        - contiguous_spans: List of (start_position, length) for each contiguous
            run of MM tokens in input_ids.
        - special_token_offsets: Indices into the flat list of all MM token
            positions where special tokens occur.
    """
    if mm_token_ids is None and vocab_size is None:
        raise ValueError(
            "Provide either mm_token_ids or vocab_size to find multimodal token positions"
        )
    if mm_token_ids is not None and vocab_size is not None:
        logger.debug(
            "Both mm_token_ids and vocab_size are provided, using mm_token_ids and ignoring vocab_size"
        )

    # Convert input_ids to tensor if needed
    if not isinstance(input_ids, torch.Tensor):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        elif isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)

    if input_ids.numel() == 0:
        return [], []

    # Create mask for multimodal tokens including special tokens if provided
    if mm_token_ids is None:
        mm_mask = input_ids >= vocab_size
        if mm_special_token_ids is not None:
            mm_special_token_ids = mm_special_token_ids.to(
                device=input_ids.device, dtype=input_ids.dtype)
            mm_mask = mm_mask | torch.isin(input_ids, mm_special_token_ids)
    else:
        mm_token_ids = mm_token_ids.to(device=input_ids.device,
                                       dtype=input_ids.dtype)
        if mm_token_ids.ndim != 1:
            raise ValueError("mm_token_ids must be a 1D tensor")
        if mm_special_token_ids is not None:
            mm_special_token_ids = mm_special_token_ids.to(
                device=input_ids.device, dtype=input_ids.dtype)
            mm_token_ids = torch.unique(
                torch.cat([mm_token_ids, mm_special_token_ids]))
        else:
            mm_token_ids = torch.unique(mm_token_ids)
        mm_mask = torch.isin(input_ids, mm_token_ids)

    # If no multimodal tokens found, return empty
    if not torch.any(mm_mask):
        return [], []

    mm_positions = torch.where(mm_mask)[0]

    # Identify special token offsets within the flat mm_positions list
    special_token_offsets: List[int] = []
    if mm_special_token_ids is not None:
        tokens_at_mm_positions = input_ids[mm_positions]
        special_mask = torch.isin(tokens_at_mm_positions, mm_special_token_ids)
        special_token_offsets = torch.where(special_mask)[0].tolist()

    # diffs[i] = mm_positions[i+1] - mm_positions[i]. Where diffs != 1 there is
    # a gap, so the next contiguous span starts at index (i + 1) in mm_positions
    # (i.e. gap_indices are indices *into mm_positions*, not 1-based positions).
    diffs = torch.diff(mm_positions)
    gap_indices = torch.where(diffs != 1)[0] + 1
    span_starts_idx = torch.cat([mm_positions.new_zeros(1), gap_indices])
    span_ends_idx = torch.cat(
        [gap_indices, mm_positions.new_tensor([len(mm_positions)])])
    span_starts = mm_positions[span_starts_idx]
    span_lengths = span_ends_idx - span_starts_idx
    contiguous_spans = list(
        zip(span_starts.tolist(), span_lengths.tolist(), strict=True))

    return contiguous_spans, special_token_offsets


def compute_per_unit_is_embeds(
    input_ids: Union[torch.Tensor, List[int], np.ndarray],
    contiguous_spans: List[Tuple[int, int]],
    vocab_size: Optional[int] = None,
    mm_token_ids: Optional[torch.Tensor] = None,
    mm_special_token_ids: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Return per-span bool[length] masks with True at embeds, False at specials.

    Companion to ``find_contiguous_mm_spans`` — keeps the producer API back-compat
    (still 2-tuple) while providing the per-unit mask needed for
    MultimodalInput.multimodal_is_embeds. See slop/mm_is_embed_migration/goals.md §3.1.
    """
    if not isinstance(input_ids, torch.Tensor):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        elif isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)

    # Build the embed-only mask over the whole prompt, then slice per span.
    if mm_token_ids is None:
        embed_mask = input_ids >= vocab_size
    else:
        mm_token_ids = mm_token_ids.to(device=input_ids.device,
                                       dtype=input_ids.dtype)
        embed_mask = torch.isin(input_ids, mm_token_ids)
    if mm_special_token_ids is not None:
        mm_special_token_ids = mm_special_token_ids.to(device=input_ids.device,
                                                       dtype=input_ids.dtype)
        embed_mask = embed_mask & ~torch.isin(input_ids, mm_special_token_ids)
    return [
        embed_mask[start:start + length].clone()
        for start, length in contiguous_spans
    ]


def find_mm_token_positions(
    input_ids: Union[torch.Tensor, List[int], np.ndarray],
    num_mm_tokens: List[int],
    vocab_size: Optional[int] = None,
    mm_token_ids: Optional[torch.Tensor] = None,
    mm_special_token_ids: Optional[torch.Tensor] = None
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Get positions of multimodal token chunks using known lengths.

    Finds multimodal tokens (with IDs > vocab_size or matching mm_token_ids)
    and uses the provided lengths in num_mm_tokens to identify where each chunk starts.
    Each logical unit in num_mm_tokens may span a non-contiguous range of token positions
    when text tokens (e.g., video frame separators) are interleaved with multimodal tokens.

    Note: at least one of vocab_size or mm_token_ids must be provided. If mm_token_ids
    is provided, vocab_size is ignored.

    Args:
        input_ids: Token sequence (tensor, list, or numpy array)
        num_mm_tokens: List of token counts for each logical multimodal unit
        vocab_size: Size of the model's vocabulary (used to identify tokens > vocab_size)
        mm_token_ids: Specific token IDs that represent multimodal tokens
        mm_special_token_ids: Specific token IDs that represent special multimodal tokens

    Returns:
        A 3-tuple of:
        - start_positions: List of starting positions for each logical multimodal unit
            (one entry per image/video).
        - start_special_token_positions: List of positions of special tokens
            in the union of all chunks (indices into the flat mm token list).
        - contiguous_spans: List of (start, length) tuples for each contiguous run
            of MM tokens in ``input_ids``. Used by MultimodalRuntimeData for exact
            counting during chunked prefill. A single logical unit may produce
            multiple contiguous spans when its tokens are non-contiguous.
    """
    # Delegate mask creation, position scanning, span compression, and
    # special-token detection to the lighter find_contiguous_mm_spans.
    contiguous_spans, start_special_token_positions = find_contiguous_mm_spans(
        input_ids=input_ids,
        vocab_size=vocab_size,
        mm_token_ids=mm_token_ids,
        mm_special_token_ids=mm_special_token_ids,
    )

    if not contiguous_spans:
        return [], [], []

    # Reconstruct flat mm_positions from contiguous_spans via vectorized arange+cat.
    spans_t = torch.tensor(contiguous_spans)  # (N, 2) — [start, length]
    mm_positions = torch.cat(
        [torch.arange(s, s + n) for s, n in spans_t.tolist()])

    # Validate total token count against num_mm_tokens.
    lengths_t = torch.tensor(num_mm_tokens)
    assert mm_positions.numel() == lengths_t.sum().item(), (
        f"Number of multimodal tokens ({mm_positions.numel()}) does not match "
        f"sum of all lengths ({lengths_t.sum().item()}): "
        f"num_mm_tokens={num_mm_tokens}, contiguous_spans={contiguous_spans}")

    # Gather start_positions at cumsum offsets (exclusive prefix sum).
    offsets = torch.zeros(len(num_mm_tokens), dtype=torch.long)
    if len(num_mm_tokens) > 1:
        torch.cumsum(lengths_t[:-1], dim=0, out=offsets[1:])
    start_positions = mm_positions[offsets].tolist()

    return start_positions, start_special_token_positions, contiguous_spans


def validate_mm_inputs(prompt_token_ids: Union[torch.Tensor, List[int],
                                               np.ndarray],
                       mm_hashes: List[List[int]], start_positions: List[int],
                       num_mm_tokens: List[int]) -> None:
    """Validates multimodal inputs for consistency and correctness."""
    # Validate number of hashes matches number of chunks
    if len(mm_hashes) != len(num_mm_tokens):
        raise AssertionError(
            f"Number of hashes ({len(mm_hashes)}) does not match "
            f"number of multimodal chunks ({len(num_mm_tokens)})")

    # Validate number of start positions matches number of chunks
    if len(start_positions) != len(num_mm_tokens):
        raise AssertionError(
            f"Number of start positions ({len(start_positions)}) does not match "
            f"number of multimodal chunks ({len(num_mm_tokens)})")
    # Validate each chunk's position and length
    prompt_len = len(prompt_token_ids)
    # Verify start_positions are sorted
    if not all(start_positions[i] < start_positions[i + 1]
               for i in range(len(start_positions) - 1)):
        raise AssertionError(
            "start_positions must be sorted in ascending order")
    for chunk_idx, (start_pos,
                    chunk_len) in enumerate(zip(start_positions,
                                                num_mm_tokens)):
        if start_pos < 0:
            raise AssertionError(
                f"Invalid negative start position {start_pos} for chunk {chunk_idx}"
            )

        if start_pos + chunk_len > prompt_len:
            raise AssertionError(
                f"Multimodal chunk {chunk_idx} at position {start_pos} with length {chunk_len} "
                f"exceeds input sequence length {prompt_len}")

        # Check for overlap with next chunk
        if chunk_idx < len(start_positions) - 1:
            next_start = start_positions[chunk_idx + 1]
            if start_pos + chunk_len > next_start:
                raise AssertionError(
                    f"Multimodal chunk {chunk_idx} at position {start_pos} with length {chunk_len} "
                    f"overlaps with chunk {chunk_idx + 1} at position {next_start}"
                )
