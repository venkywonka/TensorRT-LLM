"""Multimodal utilities for handling images and other media types in TensorRT-LLM."""

import bisect
from dataclasses import dataclass, field
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

# Shared key for mm_contiguous_spans in the untyped multimodal_data dict.
# Used by registry.py (producer) and model_engine.py / ad_executor.py (consumers).
MM_CONTIGUOUS_SPANS_KEY = "mm_contiguous_spans"


@dataclass
class MultimodalInput:
    """Logical multimodal metadata — one entry per source media (image/video).

    Used for KV-cache hashing (C++ layer).

    Fields here are indexed per *logical* multimodal unit (one entry per
    image or video), NOT per contiguous multimodal token run.  For models
    with non-contiguous tokens (e.g. video frames separated by text), a
    single logical unit may span multiple disjoint contiguous runs.  The
    physical contiguous-run layout is stored separately as
    ``mm_contiguous_spans`` in ``py_multimodal_data``
    (see :data:`MM_CONTIGUOUS_SPANS_KEY`) and consumed by
    :class:`MultimodalRuntimeData` for chunked-prefill accounting.
    """

    multimodal_hashes: List[List[int]]
    """Hash values for each logical multimodal unit (e.g., one image, one video).

    Each element is a list of 8 integers representing the hash digest of one logical unit.
    """

    multimodal_positions: List[int]
    """Starting token position of each *logical* multimodal unit in the token sequence.

    One entry per logical unit (image, video, …).  For units whose tokens
    are non-contiguous (e.g. video with interleaved text separators), this
    is the position of the *first* token of the unit — it does NOT imply
    that all ``multimodal_lengths[i]`` tokens starting here are contiguous.
    """

    multimodal_lengths: List[int]
    """Total token count of each *logical* multimodal unit, including any special tokens.

    One entry per logical unit.  May include special tokens mixed with
    actual multimodal tokens (e.g. image_end_token, image_break_token for
    mistral3).  For non-contiguous units this is the *sum* across all
    contiguous runs belonging to that unit, not the length of a single
    contiguous run.
    """

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


@dataclass
class MultimodalRuntimeData:
    """Runtime data for tracking multimodal token caching and reuse per request sequence.

    This class tracks which multimodal tokens are cached vs. need to be processed
    for each request sequence during both KV cache reuse and chunked prefill scenarios.

    Attributes:
        past_seen_token_num: Total number of tokens already processed in previous
            iterations (KV cache reuse or prior chunks).
        mm_contiguous_spans: List of (start_position, token_count) tuples for each
            contiguous run of MM tokens. A single logical unit (image/video) may
            produce one contiguous span or multiple (e.g., video frames separated
            by text tokens).
        chunk_end_pos: End position (exclusive) of the current chunk for chunked prefill.
        special_token_offsets: Sorted indices of special tokens (e.g., image_start,
            image_end) within the flat union of all MM token positions. These tokens
            occupy MM positions but get text embeddings, not encoder embeddings.

        num_unseen_mm_tokens: Number of MM tokens already processed in prior chunks,
            used as a skip offset when slicing encoder embeddings (computed).
        num_mm_tokens_in_chunk: Number of MM tokens in the current chunk (computed).
        total_mm_tokens_in_request: Total MM tokens across all chunks (computed).

        num_unseen_special_tokens: Special tokens in the already-processed region (computed).
        num_special_tokens_in_chunk: Special tokens in the current chunk (computed).
        total_special_tokens_in_request: Total special tokens in the request (computed).
    """
    past_seen_token_num: int
    mm_contiguous_spans: List[Tuple[int, int]]
    chunk_end_pos: int
    special_token_offsets: List[int]

    num_unseen_mm_tokens: Optional[int] = None
    num_mm_tokens_in_chunk: Optional[int] = None
    total_mm_tokens_in_request: Optional[int] = None

    num_unseen_special_tokens: Optional[int] = 0
    num_special_tokens_in_chunk: Optional[int] = 0
    total_special_tokens_in_request: Optional[int] = 0

    def __post_init__(self):
        if self.total_mm_tokens_in_request is None:
            self.total_mm_tokens_in_request = sum(
                length for _, length in self.mm_contiguous_spans)

        if self.past_seen_token_num < 0:
            raise ValueError(
                f"past_seen_token_num must be non-negative, got {self.past_seen_token_num}"
            )

        if any(length <= 0 for _, length in self.mm_contiguous_spans):
            raise ValueError(
                f"All span lengths must be positive, got {self.mm_contiguous_spans}"
            )

        if any(pos < 0 for pos, _ in self.mm_contiguous_spans):
            raise ValueError(
                f"All span positions must be non-negative, got {self.mm_contiguous_spans}"
            )

        remainder = 0
        if self.num_unseen_mm_tokens is None or self.num_mm_tokens_in_chunk is None:
            self.num_unseen_mm_tokens = 0
            self.num_mm_tokens_in_chunk = 0
            for pos, length in self.mm_contiguous_spans:
                span_end = pos + length
                if span_end <= self.past_seen_token_num:
                    self.num_unseen_mm_tokens += length
                elif pos < self.past_seen_token_num:
                    self.num_unseen_mm_tokens += self.past_seen_token_num - pos
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
            # special_token_offsets are sorted indices into the mm token union
            s = self.special_token_offsets
            self.num_unseen_special_tokens = bisect.bisect_left(
                s, self.num_unseen_mm_tokens)
            mm_tokens_end_pos = self.num_unseen_mm_tokens + self.num_mm_tokens_in_chunk
            self.num_special_tokens_in_chunk = (
                bisect.bisect_left(s, mm_tokens_end_pos) -
                self.num_unseen_special_tokens)

            self.total_special_tokens_in_request = len(
                self.special_token_offsets)

        total = sum(length for _, length in self.mm_contiguous_spans)
        if self.num_unseen_mm_tokens + self.num_mm_tokens_in_chunk + remainder > total:
            raise ValueError(
                f"num_unseen_mm_tokens ({self.num_unseen_mm_tokens}) + "
                f"num_mm_tokens_in_chunk ({self.num_mm_tokens_in_chunk}) + "
                f"remainder ({remainder}) must be <= total ({total})")


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
    """Apply hashing to multimodal data, one hash per logical unit (image/video).

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


def find_mm_token_lengths(mm_data: Dict[str, Any],
                          input_processor: Any) -> List[int]:
    """Get the token lengths of each logical multimodal unit.

    Returns the total token count for each logical unit (image or video), including any special tokens
    (e.g., image_begin, image_end, image_break) that may be mixed with the actual
    multimodal content tokens. This mm_token_lengths represents the full chunk from beginning
    to end, not just pure image/video/audio tokens.
    """

    mm_items = {
        modality: items if isinstance(items, list) else [items]
        for modality, items in mm_data.items()
    }
    num_mm_tokens = {}

    for modality, items in mm_items.items():
        if not hasattr(input_processor, f"get_num_tokens_per_{modality}"):
            raise AttributeError(
                f"Input processor {type(input_processor).__name__} does not have 'get_num_tokens_per_{modality}' method required for multimodal hashing."
            )

        modality_token_lengths = []
        for item in items:
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
                num_tokens = input_processor.get_num_tokens_per_video(
                    video=item, )
                modality_token_lengths.append(num_tokens)
            elif modality == "audio":
                num_tokens = input_processor.get_num_tokens_per_audio(
                    audio=item)
                modality_token_lengths.append(num_tokens)
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        num_mm_tokens[modality] = modality_token_lengths

    return num_mm_tokens  # flatten all mm instances to a single list


def find_contiguous_mm_spans(
    input_ids: Union[torch.Tensor, List[int], np.ndarray],
    vocab_size: Optional[int] = None,
    mm_token_ids: Optional[torch.Tensor] = None,
    mm_special_token_ids: Optional[torch.Tensor] = None,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Scan input_ids for contiguous runs of multimodal tokens.

    Lightweight alternative to find_mm_token_positions that does not require
    num_mm_tokens.  Suitable for any code path that has token IDs and needs
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

    # Handle empty input
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

    # Get positions of all multimodal tokens
    mm_positions = torch.where(mm_mask)[0].tolist()

    # Identify special token offsets within the flat mm_positions list
    special_token_offsets: List[int] = []
    if mm_special_token_ids is not None:
        mm_token_values = input_ids[mm_positions]
        special_mask = torch.isin(mm_token_values, mm_special_token_ids)
        special_token_offsets = torch.where(special_mask)[0].tolist()

    # Compress flat mm_positions into contiguous spans: (start, length)
    contiguous_spans: List[Tuple[int, int]] = []
    if mm_positions:
        span_start = mm_positions[0]
        span_len = 1
        for i in range(1, len(mm_positions)):
            if mm_positions[i] == mm_positions[i - 1] + 1:
                span_len += 1
            else:
                contiguous_spans.append((span_start, span_len))
                span_start = mm_positions[i]
                span_len = 1
        contiguous_spans.append((span_start, span_len))

    return contiguous_spans, special_token_offsets


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
    assert mm_positions.numel() == lengths_t.sum().item(), \
        "Number of multimodal tokens does not match sum of all lengths"

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
