from typing import List
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds, get_multimodal_embeddings)
from tensorrt_llm._torch.pyexecutor.model_engine import _check_mm_spans_present
from tensorrt_llm.inputs.multimodal import (MultimodalParams,
                                            MultimodalRuntimeData,
                                            find_contiguous_mm_spans,
                                            find_mm_token_positions)
from tensorrt_llm.inputs.registry import compute_mm_contiguous_spans_if_absent


class TestMultimodalRuntimeData:
    """Test cases for MultimodalRuntimeData computation logic, testing both KV cache reuse and chunked prefill."""

    def test_fully_cached_multimodal_tokens(self):
        """Test when all multimodal tokens are cached (KV cache reuse scenario)."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=20,
            mm_contiguous_spans=[(0, 5), (5, 8), (13, 7)],  # Total: 20 tokens
            chunk_end_pos=20,
            special_token_offsets=[])

        # All tokens should be cached since past_seen_token_num (20) >= all positions + lengths
        assert runtime.num_cached_mm_tokens == 20
        assert runtime.num_mm_tokens_in_chunk == 0

    def test_no_cached_multimodal_tokens(self):
        """Test when no multimodal tokens are cached (KV cache reuse scenario)."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=10,
            mm_contiguous_spans=[(10, 5), (18, 8), (30, 7)],  # Total: 20 tokens
            chunk_end_pos=40,
            special_token_offsets=[])

        # No multimodal tokens should be cached
        assert runtime.num_cached_mm_tokens == 0
        assert runtime.num_mm_tokens_in_chunk == 20

    def test_partial_caching_with_chunk_boundaries(self):
        """Test partial caching with chunk boundaries (chunked prefill scenario)."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=15,
            mm_contiguous_spans=[(10, 5), (18, 8), (25, 7)],  # Total: 20 tokens
            chunk_end_pos=30,
            special_token_offsets=[])

        # Span [10,15): 5 tokens fully cached
        # Span [18,26): 0 cached, 8 in chunk
        # Span [25,32): 0 cached, 5 in chunk [25,30), 2 beyond
        assert runtime.num_cached_mm_tokens == 5
        assert runtime.num_mm_tokens_in_chunk == 13

    def test_chunk_boundary_case1(self):
        """Test case chunk around chunk boundaries."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=12,
            mm_contiguous_spans=[(8, 6), (16, 4), (22, 8)],  # Total: 18 tokens
            chunk_end_pos=20,
            special_token_offsets=[])

        # Span [8,14): 4 cached (8-12), 2 in chunk (12-14)
        # Span [16,20): 0 cached, 4 in chunk (16-20)
        # Span [22,30): beyond chunk_end_pos
        assert runtime.num_cached_mm_tokens == 4
        assert runtime.num_mm_tokens_in_chunk == 6

    def test_chunk_boundary_case2(self):
        """Test chunk end is very large."""
        runtime = MultimodalRuntimeData(past_seen_token_num=30,
                                        mm_contiguous_spans=[(0, 3), (5, 4),
                                                             (10, 5), (15, 6),
                                                             (25, 7), (35, 8)],
                                        chunk_end_pos=100,
                                        special_token_offsets=[])

        expected_cached = 3 + 4 + 5 + 6 + 5  # 23 tokens
        expected_current_chunk = 2 + 8  # 10 tokens
        assert runtime.num_cached_mm_tokens == expected_cached
        assert runtime.num_mm_tokens_in_chunk == expected_current_chunk

    def test_validation_errors(self):
        """Test validation logic for invalid inputs."""
        # Test negative past_seen_token_num
        with pytest.raises(ValueError,
                           match="past_seen_token_num must be non-negative"):
            MultimodalRuntimeData(past_seen_token_num=-1,
                                  mm_contiguous_spans=[(0, 5)],
                                  chunk_end_pos=10,
                                  special_token_offsets=[])

        # Test non-positive span lengths
        with pytest.raises(ValueError,
                           match="All span lengths must be positive"):
            MultimodalRuntimeData(past_seen_token_num=10,
                                  mm_contiguous_spans=[(0, 5), (5, 0), (10, 7)],
                                  chunk_end_pos=20,
                                  special_token_offsets=[])

        # Test negative span positions
        with pytest.raises(ValueError,
                           match="All span positions must be non-negative"):
            MultimodalRuntimeData(past_seen_token_num=10,
                                  mm_contiguous_spans=[(0, 5), (-5, 8),
                                                       (10, 7)],
                                  chunk_end_pos=20,
                                  special_token_offsets=[])

    def test_single_item_multiple_spans(self):
        """One video item whose MM tokens are non-contiguous — THE bug scenario.
        Video with 3 temporal groups of 196 tokens, separated by 8-token text gaps.
        Layout: text[0,10) group1[10,206) text[206,214) group2[214,410) text[410,418) group3[418,614)
        Chunk [0, 256) should contain 196 + 42 = 238 MM tokens."""
        runtime = MultimodalRuntimeData(past_seen_token_num=0,
                                        mm_contiguous_spans=[(10, 196),
                                                             (214, 196),
                                                             (418, 196)],
                                        chunk_end_pos=256,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 0
        assert runtime.num_mm_tokens_in_chunk == 238
        assert runtime.total_mm_tokens_in_request == 588


class TestNonContiguousMultimodalRuntimeData:
    """Test cases for MultimodalRuntimeData with non-contiguous MM regions.

    These simulate Qwen3-VL-style layouts where images/video frames are
    separated by text tokens, creating gaps between MM regions.
    """

    def test_two_images_separated_by_text_chunk_spans_both(self):
        """Two images at positions [5,15) and [30,40) with text gap at [15,30).
        Chunk [0, 50) covers everything."""
        runtime = MultimodalRuntimeData(past_seen_token_num=0,
                                        mm_contiguous_spans=[(5, 10), (30, 10)],
                                        chunk_end_pos=50,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 0
        assert runtime.num_mm_tokens_in_chunk == 20  # both full regions

    def test_two_images_separated_chunk_in_text_gap(self):
        """Chunk [15, 30) lands entirely in the text gap between two images.
        Images at [5,15) and [30,40). No MM tokens in chunk."""
        runtime = MultimodalRuntimeData(past_seen_token_num=15,
                                        mm_contiguous_spans=[(5, 10), (30, 10)],
                                        chunk_end_pos=30,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 10  # first image fully cached
        assert runtime.num_mm_tokens_in_chunk == 0  # gap has no MM tokens

    def test_two_images_separated_chunk_hits_second_only(self):
        """Chunk [25, 45) starts in the gap and covers the second image.
        Images at [5,15) and [30,40)."""
        runtime = MultimodalRuntimeData(past_seen_token_num=25,
                                        mm_contiguous_spans=[(5, 10), (30, 10)],
                                        chunk_end_pos=45,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 10  # first image fully cached
        assert runtime.num_mm_tokens_in_chunk == 10  # second image fully in chunk

    def test_two_images_separated_chunk_straddles_second(self):
        """Chunk [25, 35) starts in gap, partially covers second image.
        Images at [5,15) and [30,40)."""
        runtime = MultimodalRuntimeData(past_seen_token_num=25,
                                        mm_contiguous_spans=[(5, 10), (30, 10)],
                                        chunk_end_pos=35,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 10  # first image fully cached
        assert runtime.num_mm_tokens_in_chunk == 5  # 5 tokens of second image [30,35)

    def test_three_images_chunk_hits_middle_only(self):
        """Three images at [5,15), [30,40), [60,70). Chunk [25, 45) hits middle only."""
        runtime = MultimodalRuntimeData(past_seen_token_num=25,
                                        mm_contiguous_spans=[(5, 10), (30, 10),
                                                             (60, 10)],
                                        chunk_end_pos=45,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 10  # first image cached
        assert runtime.num_mm_tokens_in_chunk == 10  # second image fully in chunk

    def test_three_images_chunk_partial_first_full_second_miss_third(self):
        """Three images at [5,15), [30,40), [60,70). Chunk [10, 45).
        First image partial (5 tokens cached, 5 in chunk), second full in chunk, third missed."""
        runtime = MultimodalRuntimeData(past_seen_token_num=10,
                                        mm_contiguous_spans=[(5, 10), (30, 10),
                                                             (60, 10)],
                                        chunk_end_pos=45,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 5  # [5,10) of first image cached
        assert runtime.num_mm_tokens_in_chunk == 15  # 5 from first + 10 from second

    def test_scattered_video_frames_qwen3vl_style(self):
        """Simulate Qwen3-VL timestamp-separated frames:
        text[0,10) frame1[10,30) text[30,35) frame2[35,55) text[55,60) frame3[60,80)
        Chunk [30, 60) should get: 0 from frame1 (cached), full frame2, 0 from frame3."""
        runtime = MultimodalRuntimeData(past_seen_token_num=30,
                                        mm_contiguous_spans=[(10, 20), (35, 20),
                                                             (60, 20)],
                                        chunk_end_pos=60,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 20  # frame1 fully cached
        assert runtime.num_mm_tokens_in_chunk == 20  # frame2 fully in chunk [35,55)

    def test_scattered_frames_chunk_straddles_gap_and_frame(self):
        """Frames at [10,30) and [50,70). Chunk [25, 55).
        Partial first frame (5 tokens [25,30)), then gap [30,50), then partial second (5 tokens [50,55))."""
        runtime = MultimodalRuntimeData(past_seen_token_num=25,
                                        mm_contiguous_spans=[(10, 20),
                                                             (50, 20)],
                                        chunk_end_pos=55,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 15  # [10,25) of first frame cached
        assert runtime.num_mm_tokens_in_chunk == 10  # 5 from first [25,30) + 5 from second [50,55)

    def test_large_gap_between_images(self):
        """Images at [5,10) and [500,505) with a huge text gap.
        Chunk [0, 100). First image fully in chunk, second not reached."""
        runtime = MultimodalRuntimeData(past_seen_token_num=0,
                                        mm_contiguous_spans=[(5, 5), (500, 5)],
                                        chunk_end_pos=100,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 0
        assert runtime.num_mm_tokens_in_chunk == 5  # only first image


class TestNonContiguousWithContiguousSpans:
    """Test cases for MultimodalRuntimeData with multi-group spans.

    These simulate the actual bug: a single video entry where MM tokens are
    scattered across a wider range than a single (pos, length) due to text gaps
    (e.g., <vision_end> + timestamp + <vision_start> between temporal groups).
    With mm_contiguous_spans, each group is its own span so counting is exact.
    """

    def test_video_three_groups_chunk_hits_first_two(self):
        """Video with 3 temporal groups of 196 MM tokens each:
        Group1: [10, 206), Gap: [206, 214), Group2: [214, 410), Gap: [410, 418), Group3: [418, 614)
        Total MM tokens: 588. Chunk [0, 256) should contain 196 + 42 = 238 MM tokens.
        """
        runtime = MultimodalRuntimeData(past_seen_token_num=0,
                                        mm_contiguous_spans=[(10, 196),
                                                             (214, 196),
                                                             (418, 196)],
                                        chunk_end_pos=256,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 0
        assert runtime.num_mm_tokens_in_chunk == 238

    def test_video_three_groups_chunk_in_middle(self):
        """Same video, chunk [256, 512). past_seen=256."""
        runtime = MultimodalRuntimeData(past_seen_token_num=256,
                                        mm_contiguous_spans=[(10, 196),
                                                             (214, 196),
                                                             (418, 196)],
                                        chunk_end_pos=512,
                                        special_token_offsets=[])

        # Cached: 196 (group1) + 42 (group2 partial [214,256)) = 238
        # In chunk: 154 (group2 remainder [256,410)) + 94 (group3 partial [418,512)) = 248
        assert runtime.num_cached_mm_tokens == 238
        assert runtime.num_mm_tokens_in_chunk == 248

    def test_video_three_groups_last_chunk(self):
        """Same video, chunk [512, 627). past_seen=512."""
        runtime = MultimodalRuntimeData(past_seen_token_num=512,
                                        mm_contiguous_spans=[(10, 196),
                                                             (214, 196),
                                                             (418, 196)],
                                        chunk_end_pos=627,
                                        special_token_offsets=[])

        # Cached: 196 + 196 + 94 = 486
        # In chunk: 102 (group3 remainder [512,614))
        assert runtime.num_cached_mm_tokens == 486
        assert runtime.num_mm_tokens_in_chunk == 102

    def test_single_span_matches_contiguous(self):
        """When MM tokens are contiguous, a single span behaves identically."""
        runtime = MultimodalRuntimeData(past_seen_token_num=50,
                                        mm_contiguous_spans=[(10, 100)],
                                        chunk_end_pos=80,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 40  # [10,50)
        assert runtime.num_mm_tokens_in_chunk == 30  # [50,80)

    def test_all_cached_with_spans(self):
        """All MM tokens cached."""
        runtime = MultimodalRuntimeData(past_seen_token_num=700,
                                        mm_contiguous_spans=[(10, 196),
                                                             (214, 196)],
                                        chunk_end_pos=800,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 392
        assert runtime.num_mm_tokens_in_chunk == 0

    def test_chunk_in_text_gap(self):
        """Chunk falls entirely in a text gap between groups."""
        runtime = MultimodalRuntimeData(past_seen_token_num=60,
                                        mm_contiguous_spans=[(10, 50),
                                                             (200, 50)],
                                        chunk_end_pos=200,
                                        special_token_offsets=[])

        assert runtime.num_cached_mm_tokens == 50  # group1 fully cached
        assert runtime.num_mm_tokens_in_chunk == 0  # gap has no MM tokens


class TestFindInputMmEmbed:
    """Focused test cases for find_input_mm_embeds function - testing both KV cache reuse and chunked prefill."""

    def create_mock_runtime(self,
                            num_cached_mm_tokens: int,
                            num_mm_tokens_in_chunk: int,
                            mm_token_lengths: List[int],
                            num_cached_special_tokens: int = 0,
                            num_special_tokens_in_chunk: int = 0,
                            total_special_tokens_in_request: int = 0):
        """Helper to create a mock MultimodalRuntimeData."""
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.num_cached_mm_tokens = num_cached_mm_tokens
        runtime.num_mm_tokens_in_chunk = num_mm_tokens_in_chunk
        runtime.total_mm_tokens_in_request = sum(mm_token_lengths)
        runtime.num_cached_special_tokens = num_cached_special_tokens
        runtime.num_special_tokens_in_chunk = num_special_tokens_in_chunk
        runtime.total_special_tokens_in_request = total_special_tokens_in_request

        return runtime

    def create_multimodal_params(self, num_cached_mm_tokens: int,
                                 num_mm_tokens_in_chunk: int,
                                 mm_token_lengths: List[int]):
        """Helper to create MultimodalParams with runtime data."""
        runtime = self.create_mock_runtime(num_cached_mm_tokens,
                                           num_mm_tokens_in_chunk,
                                           mm_token_lengths)
        return MultimodalParams(multimodal_runtime=runtime)

    def test_mm_embed_not_batched(self):
        """
        Test individual batching mode where each mm_embed corresponds to one param.
        This tests the case where len(mm_embeds) == len(multimodal_params) > 1.
        """
        mm_embeds = [
            torch.randn(10, 512),  # Batch 1: 10 tokens
            torch.randn(15, 512),  # Batch 2: 15 tokens
            torch.randn(8, 512)  # Batch 3: 8 tokens
        ]
        multimodal_params = [
            self.create_multimodal_params(
                3, 7, [5, 5]),  # 3 unseen, 7 in current chunk
            self.create_multimodal_params(8, 7,
                                          [15]),  # 8 unseen, 7 in current chunk
            self.create_multimodal_params(
                0, 8, [4, 4])  # 0 unseen, 8 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Should return individual slices for each batch
        assert len(result) == 3
        assert result[0].shape == (7, 512)  # 7 tokens in current chunk
        assert result[1].shape == (7, 512)  # 7 tokens in current chunk
        assert result[2].shape == (8, 512)  # 8 tokens in current chunk

        # Verify the slices are correct
        torch.testing.assert_close(result[0], mm_embeds[0][3:10])
        torch.testing.assert_close(result[1], mm_embeds[1][8:15])
        torch.testing.assert_close(result[2], mm_embeds[2][0:8])

    def test_mm_embed_batched(self):
        """
        Test batching (concatenated) mm_embeds with fused mm_embeds for each batch.
        This tests the case where len(mm_embeds) == 1
        """
        mm_embeds = [torch.randn(33,
                                 512)]  # Pre-concatenated: 10 + 13 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(4, 6,
                                          [10]),  # 4 cached, 6 in current chunk
            self.create_multimodal_params(
                7, 6, [6, 7]),  # 7 cached, 6 in current chunk
            self.create_multimodal_params(
                3, 7, [4, 6])  # 3 cached, 7 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices:
        # Batch 1: [4:10] = 6 tokens
        # Batch 2: [10+7:10+13] = [17:23] = 6 tokens
        # Batch 3: [23+3:23+10] = [26:33] = 7 tokens
        # Total: 6 + 6 + 7 = 19 tokens
        assert len(result) == 1
        assert result[0].shape == (19, 512)

        # Verify the slices are correct
        expected = torch.cat(
            [
                mm_embeds[0][4:10],  # Batch 1: 6 tokens
                mm_embeds[0][17:23],  # Batch 2: 6 tokens
                mm_embeds[0][26:33]  # Batch 3: 7 tokens
            ],
            dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_mixed_caching_with_fully_cached_batches(self):
        """
        Test mixed scenarios where some batches are fully cached (should be skipped).
        """
        mm_embeds = [torch.randn(25, 512)]  # Pre-concatenated: 8 + 9 + 8 tokens
        multimodal_params = [
            self.create_multimodal_params(
                8, 0, [8]),  # All unseen - should be skipped
            self.create_multimodal_params(
                3, 6, [6, 3]),  # 3 unseen, 6 in current chunk
            self.create_multimodal_params(8, 0,
                                          [8])  # All unseen - should be skipped
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Only batch 2 should contribute: [8+3:8+9] = [11:17] = 6 tokens
        assert len(result) == 1
        assert result[0].shape == (6, 512)

        # Verify the slice is correct
        torch.testing.assert_close(result[0], mm_embeds[0][11:17])

    def test_all_batches_fully_unseen(self):
        """
        Test edge case where all batches are fully unseen.
        """
        mm_embeds = [torch.randn(30,
                                 512)]  # Pre-concatenated: 10 + 10 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(10, 0, [10]),  # All unseen
            self.create_multimodal_params(10, 0, [10]),  # All unseen
            self.create_multimodal_params(10, 0, [10])  # All unseen
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Should return empty list
        assert result == []

    def test_no_batches_cached(self):
        """
        Test edge case where no batches have any cached tokens.
        """
        mm_embeds = [torch.randn(30,
                                 512)]  # Pre-concatenated: 10 + 10 + 10 tokens
        multimodal_params = [
            self.create_multimodal_params(
                0, 10, [10]),  # No unseen, 10 in current chunk
            self.create_multimodal_params(
                0, 10, [10]),  # No unseen, 10 in current chunk
            self.create_multimodal_params(
                0, 10, [10])  # No unseen, 10 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Should return the full embeddings
        assert len(result) == 1
        torch.testing.assert_close(result[0], mm_embeds[0])

    def test_chunked_prefill_scenario(self):
        """
        Test chunked prefill scenario where some tokens are cached and some are in current chunk.
        """
        mm_embeds = [torch.randn(25, 512)]  # Pre-concatenated: 8 + 9 + 8 tokens
        multimodal_params = [
            self.create_multimodal_params(5, 3,
                                          [8]),  # 5 unseen, 3 in current chunk
            self.create_multimodal_params(2, 7,
                                          [9]),  # 2 unseen, 7 in current chunk
            self.create_multimodal_params(6, 2,
                                          [8])  # 6 unseen, 2 in current chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices:
        # Batch 1: [5:8] = 3 tokens
        # Batch 2: [8+2:8+9] = [10:17] = 7 tokens
        # Batch 3: [17+6:17+8] = [23:25] = 2 tokens
        # Total: 3 + 7 + 2 = 12 tokens
        assert len(result) == 1
        assert result[0].shape == (12, 512)

        # Verify the slices are correct
        expected = torch.cat(
            [
                mm_embeds[0][5:8],  # Batch 1: 3 tokens
                mm_embeds[0][10:17],  # Batch 2: 7 tokens
                mm_embeds[0][23:25]  # Batch 3: 2 tokens
            ],
            dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_error_handling_mismatched_counts(self):
        """
        Test error handling when mm_embeds and multimodal_params counts don't match
        in individual batching mode.
        """
        mm_embeds = [torch.randn(10, 512), torch.randn(15, 512)]  # 2 embeddings
        multimodal_params = [self.create_multimodal_params(0, 10, [10])
                             ]  # Only 1 param

        with pytest.raises(
                ValueError,
                match=
                "Number of mm_embeds \\(2\\) does not match number of multimodal params \\(1\\)"
        ):
            find_input_mm_embeds(mm_embeds, multimodal_params)

    def test_single_batch_scenarios(self):
        """
        Test various single batch scenarios.
        """
        # Single batch, no caching
        mm_embeds = [torch.randn(20, 512)]
        multimodal_params = [self.create_multimodal_params(0, 20, [20])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert len(result) == 1
        torch.testing.assert_close(result[0], mm_embeds[0])

        # Single batch, partial caching
        multimodal_params = [self.create_multimodal_params(5, 15, [20])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert len(result) == 1
        assert result[0].shape == (15, 512)
        torch.testing.assert_close(result[0], mm_embeds[0][5:20])

        # Single batch, all cached
        multimodal_params = [self.create_multimodal_params(20, 0, [20])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert result == []

    def test_different_devices(self):
        """
        Test with tensors on different devices (if CUDA is available).
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Test CPU tensors
        mm_embeds = [torch.randn(10, 512, device='cpu')]
        multimodal_params = [self.create_multimodal_params(3, 7, [10])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert result[0].device == mm_embeds[0].device

        # Test CUDA tensors
        mm_embeds = [torch.randn(10, 512, device='cuda')]
        multimodal_params = [self.create_multimodal_params(3, 7, [10])]
        result = find_input_mm_embeds(mm_embeds, multimodal_params)
        assert result[0].device == mm_embeds[0].device

    def test_noncontiguous_two_requests_batched_chunk_in_gap(self):
        """
        Non-contiguous: two requests in a batch, where one request's chunk
        falls entirely in a text gap between MM regions.
        Request 1: 10 MM tokens, 5 cached, 5 in chunk.
        Request 2: 20 MM tokens (two regions of 10 each), but chunk is in the
                   text gap — 10 cached, 0 in chunk.
        Pre-concatenated: 10 + 20 = 30 tokens.
        """
        mm_embeds = [torch.randn(30, 512)]
        multimodal_params = [
            self.create_multimodal_params(5, 5, [10]),  # 5 cached, 5 in chunk
            self.create_multimodal_params(
                10, 0, [10, 10]),  # 10 cached, 0 in chunk (gap)
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Only request 1 contributes: [5:10] = 5 tokens
        # Request 2 contributes nothing (0 in chunk)
        assert len(result) == 1
        assert result[0].shape == (5, 512)
        torch.testing.assert_close(result[0], mm_embeds[0][5:10])

    def test_noncontiguous_individual_batching_mixed_gaps(self):
        """
        Non-contiguous: individual batching mode, three requests with
        different non-contiguous patterns.
        """
        mm_embeds = [
            torch.randn(20, 512),  # Request 1: 20 tokens (two regions of 10)
            torch.randn(15, 512),  # Request 2: 15 tokens (one region)
            torch.randn(20, 512),  # Request 3: 20 tokens (two regions of 10)
        ]
        multimodal_params = [
            self.create_multimodal_params(
                10, 10,
                [10, 10
                 ]),  # 10 cached (first region), 10 in chunk (second region)
            self.create_multimodal_params(0, 15,
                                          [15]),  # nothing cached, all in chunk
            self.create_multimodal_params(
                20, 0, [10, 10]),  # all cached, nothing in chunk
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        assert len(result) == 3
        assert result[0].shape == (10, 512)  # second region of request 1
        assert result[1].shape == (15, 512)  # all of request 2
        assert result[2].shape == (0, 512)  # nothing from request 3

        torch.testing.assert_close(result[0], mm_embeds[0][10:20])
        torch.testing.assert_close(result[1], mm_embeds[1][0:15])

    def test_special_tokens_in_batched_mode(self):
        """Test special token handling in batched mode."""
        mm_embeds = [torch.randn(12, 512)
                     ]  # Pre-concatenated: (8-2) + (10-4) = 6 + 6 = 12 tokens
        multimodal_params = [
            self.create_mock_runtime(num_cached_mm_tokens=2,
                                     num_mm_tokens_in_chunk=6,
                                     mm_token_lengths=[8],
                                     num_cached_special_tokens=1,
                                     num_special_tokens_in_chunk=1,
                                     total_special_tokens_in_request=2),
            self.create_mock_runtime(num_cached_mm_tokens=4,
                                     num_mm_tokens_in_chunk=6,
                                     mm_token_lengths=[10],
                                     num_cached_special_tokens=2,
                                     num_special_tokens_in_chunk=2,
                                     total_special_tokens_in_request=4)
        ]
        multimodal_params = [
            MultimodalParams(multimodal_runtime=runtime)
            for runtime in multimodal_params
        ]

        result = find_input_mm_embeds(mm_embeds, multimodal_params)

        # Expected slices accounting for special tokens:
        # Batch 1: local_start = 2-1=1, local_end = 1+(6-1)=6, slice [1:6] = 5 tokens
        # Batch 2: local_start = 4-2=2, local_end = 2+(6-2)=6, slice [6+2:6+6] = [8:12] = 4 tokens
        # Total: 5 + 4 = 9 tokens
        assert len(result) == 1
        assert result[0].shape == (9, 512)

        # Verify the slices are correct
        expected = torch.cat(
            [
                mm_embeds[0][1:6],  # Batch 1: 5 tokens
                mm_embeds[0][8:12]  # Batch 2: 4 tokens
            ],
            dim=0)
        torch.testing.assert_close(result[0], expected)


class TestGetMultimodalEmbeddings:
    """Test cases for get_multimodal_embeddings function - testing caching and encoder forward optimization."""

    def create_mock_runtime(self,
                            total_mm_tokens: int,
                            total_special_tokens: int = 0):
        """Helper to create a mock MultimodalRuntimeData with total_mm_tokens and special_tokens."""
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.total_mm_tokens_in_request = total_mm_tokens
        runtime.total_special_tokens_in_request = total_special_tokens
        return runtime

    def create_multimodal_params_with_data(self,
                                           has_cached_embedding: bool = False,
                                           total_mm_tokens: int = 10,
                                           total_special_tokens: int = 0,
                                           cached_embedding=None):
        """Helper to create MultimodalParams with optional cached embeddings."""
        runtime = self.create_mock_runtime(total_mm_tokens,
                                           total_special_tokens)

        multimodal_data = {
            # Add some dummy multimodal data to ensure has_content() returns True
            "image": {
                "pixel_values": torch.randn(3, 224, 224)
            }
        }
        if has_cached_embedding:
            if cached_embedding is None:
                cached_embedding = torch.randn(total_mm_tokens, 512)
            multimodal_data["multimodal_embedding"] = cached_embedding

        param = MultimodalParams(multimodal_data=multimodal_data,
                                 multimodal_runtime=runtime)
        return param

    def test_no_multimodal_params(self):
        """Test with empty multimodal_params list."""

        def mock_encoder(params):
            return [torch.randn(10, 512)]

        result = get_multimodal_embeddings(mock_encoder, [])
        assert result == []

    def test_all_params_need_processing(self):
        """Test when all params need encoder processing (no cached embeddings)."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            # Return concatenated embeddings for all params
            total_tokens = sum(
                param.multimodal_runtime.total_mm_tokens_in_request
                for param in params)
            return [torch.randn(total_tokens, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=5),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=8),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=7)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once
        assert encoder_call_count == 1

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (20, 512)  # 5 + 8 + 7 = 20 tokens

        # All params should now have cached embeddings
        for param in multimodal_params:
            assert "multimodal_embedding" in param.multimodal_data
            assert param.multimodal_data["multimodal_embedding"] is not None

    def test_all_params_already_cached(self):
        """Test when all params already have cached embeddings."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            return [torch.randn(10, 512)]

        # Create params with pre-cached embeddings
        cached_emb1 = torch.randn(5, 512)
        cached_emb2 = torch.randn(8, 512)
        cached_emb3 = torch.randn(7, 512)

        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=5,
                cached_embedding=cached_emb1),
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=8,
                cached_embedding=cached_emb2),
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=7,
                cached_embedding=cached_emb3)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should not be called
        assert encoder_call_count == 0

        # Should return concatenated cached embeddings
        assert len(result) == 1
        assert result[0].shape == (20, 512)  # 5 + 8 + 7 = 20 tokens

        # Verify the embeddings are correct
        expected = torch.cat([cached_emb1, cached_emb2, cached_emb3], dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_mixed_cached_and_uncached(self):
        """Test mix of cached and uncached params."""
        encoder_call_count = 0
        processed_params = []

        def mock_encoder(params):
            nonlocal encoder_call_count, processed_params
            encoder_call_count += 1
            processed_params = params
            # Return embeddings for uncached params only
            total_tokens = sum(
                param.multimodal_runtime.total_mm_tokens_in_request
                for param in params)
            return [torch.randn(total_tokens, 512)]

        # Mix: cached, uncached, cached
        cached_emb = torch.randn(5, 512)
        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=5,
                cached_embedding=cached_emb),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=8),
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=7,
                cached_embedding=torch.randn(7, 512))
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once, only for uncached param
        assert encoder_call_count == 1
        assert len(processed_params) == 1  # Only the middle param
        assert processed_params[0] == multimodal_params[1]

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (20, 512)  # 5 + 8 + 7 = 20 tokens

        # Uncached param should now have cached embedding
        assert "multimodal_embedding" in multimodal_params[1].multimodal_data
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"] is not None

    def test_missing_multimodal_runtime(self):
        """Test handling when multimodal_runtime is missing."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            return [torch.randn(10, 512)]

        # Create param without multimodal_runtime but with content
        param = MultimodalParams(multimodal_data={
            "image": {
                "pixel_values": torch.randn(3, 224, 224)
            }
        })

        result = get_multimodal_embeddings(mock_encoder, [param])

        # Should call encoder and return its output directly (no caching)
        assert encoder_call_count == 1
        assert len(result) == 1
        assert result[0].shape == (10, 512)

        # Should not have cached embedding due to missing runtime
        assert "multimodal_embedding" not in param.multimodal_data

    def test_missing_total_mm_tokens(self):
        """Test handling when total_mm_tokens is None."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            return [torch.randn(10, 512)]

        # Create runtime without total_mm_tokens
        runtime = Mock(spec=MultimodalRuntimeData)
        runtime.total_mm_tokens_in_request = None

        param = MultimodalParams(multimodal_data={
            "image": {
                "pixel_values": torch.randn(3, 224, 224)
            }
        },
                                 multimodal_runtime=runtime)

        result = get_multimodal_embeddings(mock_encoder, [param])

        # Should call encoder and return its output directly (no caching)
        assert encoder_call_count == 1
        assert len(result) == 1
        assert result[0].shape == (10, 512)

    def test_multiple_modalities_early_return(self):
        """Test early return when encoder outputs multiple modalities."""

        def mock_encoder(params):
            # Return multiple embeddings (multiple modalities)
            return [torch.randn(5, 512), torch.randn(8, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=5)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return encoder output directly without caching
        assert len(result) == 2
        assert result[0].shape == (5, 512)
        assert result[1].shape == (8, 512)

        # Should not have cached anything
        assert "multimodal_embedding" not in multimodal_params[
            0].multimodal_data

    def test_caching_with_torch_split(self):
        """Test that caching uses torch.split correctly for multiple params."""

        def mock_encoder(params):
            # Return single concatenated tensor for all params
            return [torch.randn(20, 512)]  # 5 + 8 + 7 = 20 tokens

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=5),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=8),
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=7)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Check that embeddings were split correctly
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (5, 512)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (8, 512)
        assert multimodal_params[2].multimodal_data[
            "multimodal_embedding"].shape == (7, 512)

        # Verify the result is correct concatenation
        assert result[0].shape == (20, 512)
        expected = torch.cat([
            multimodal_params[0].multimodal_data["multimodal_embedding"],
            multimodal_params[1].multimodal_data["multimodal_embedding"],
            multimodal_params[2].multimodal_data["multimodal_embedding"]
        ],
                             dim=0)
        torch.testing.assert_close(result[0], expected)

    def test_different_devices(self):
        """Test with tensors on different devices (if CUDA is available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        def mock_encoder(params):
            return [torch.randn(10, 512, device='cuda')]

        multimodal_params = [
            self.create_multimodal_params_with_data(has_cached_embedding=False,
                                                    total_mm_tokens=10)
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Result should be on CUDA
        assert result[0].device.type == 'cuda'
        # Cached embedding should also be on CUDA
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].device.type == 'cuda'

    def test_special_tokens_basic_caching(self):
        """Test caching behavior with special tokens present."""

        def mock_encoder(params):
            # Return embeddings for non-special tokens only
            # Total: (10-2) + (8-1) + (6-3) = 8 + 7 + 3 = 18 tokens
            return [torch.randn(18, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=10,
                total_special_tokens=2),  # 8 actual embedding tokens
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=8,
                total_special_tokens=1),  # 7 actual embedding tokens
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=6,
                total_special_tokens=3)  # 3 actual embedding tokens
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return concatenated embeddings
        assert len(result) == 1
        assert result[0].shape == (18, 512)  # 8 + 7 + 3 = 18 tokens

        # Check that embeddings were split correctly based on non-special token counts
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (8, 512)  # 10 - 2
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (7, 512)  # 8 - 1
        assert multimodal_params[2].multimodal_data[
            "multimodal_embedding"].shape == (3, 512)  # 6 - 3

    def test_special_tokens_all_special(self):
        """Test edge case where all tokens are special tokens."""

        def mock_encoder(params):
            # Should return empty tensor when no actual embedding tokens
            return [torch.randn(0, 512)]

        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=5,
                total_special_tokens=5),  # All tokens are special
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=3,
                total_special_tokens=3)  # All tokens are special
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Should return empty embeddings
        assert len(result) == 1
        assert result[0].shape == (0, 512)

        # Cached embeddings should also be empty
        assert multimodal_params[0].multimodal_data[
            "multimodal_embedding"].shape == (0, 512)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (0, 512)

    def test_special_tokens_mixed_with_cached(self):
        """Test special tokens with mixed cached and uncached params."""
        encoder_call_count = 0

        def mock_encoder(params):
            nonlocal encoder_call_count
            encoder_call_count += 1
            # Only process uncached param: 12 - 3 = 9 tokens
            return [torch.randn(9, 512)]

        # Mix: cached (with special tokens), uncached (with special tokens)
        cached_emb = torch.randn(4, 512)  # 6 - 2 = 4 actual tokens
        multimodal_params = [
            self.create_multimodal_params_with_data(
                has_cached_embedding=True,
                total_mm_tokens=6,
                total_special_tokens=2,
                cached_embedding=cached_emb),
            self.create_multimodal_params_with_data(
                has_cached_embedding=False,
                total_mm_tokens=12,
                total_special_tokens=3)  # 9 actual embedding tokens
        ]

        result = get_multimodal_embeddings(mock_encoder, multimodal_params)

        # Encoder should be called once for uncached param
        assert encoder_call_count == 1

        # Should return concatenated embeddings: 4 + 9 = 13 tokens
        assert len(result) == 1
        assert result[0].shape == (13, 512)

        # Verify cached embedding is preserved and uncached is now cached
        torch.testing.assert_close(
            multimodal_params[0].multimodal_data["multimodal_embedding"],
            cached_emb)
        assert multimodal_params[1].multimodal_data[
            "multimodal_embedding"].shape == (9, 512)


class TestFindMmTokenPositions:
    """Test cases for find_mm_token_positions — verifies 3-tuple return
    (start_positions, special_positions, contiguous_spans)."""

    def test_early_return_no_mm_tokens(self):
        """When input has no MM tokens, should return three empty lists."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        result = find_mm_token_positions(
            input_ids=input_ids,
            num_mm_tokens=[],
            vocab_size=100,
        )
        assert result == ([], [], [])

    def test_early_return_no_match(self):
        """When mm_token_ids don't match anything in input_ids."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        result = find_mm_token_positions(
            input_ids=input_ids,
            num_mm_tokens=[2],
            mm_token_ids=torch.tensor([99]),
        )
        assert result == ([], [], [])

    def test_basic_contiguous_tokens(self):
        """Basic case: contiguous MM tokens identified by out-of-vocab IDs."""
        # vocab_size=10, tokens >= 10 are MM tokens
        input_ids = torch.tensor([1, 2, 10, 11, 12, 3, 4, 10, 11, 5])
        start_pos, special_pos, spans = find_mm_token_positions(
            input_ids=input_ids,
            num_mm_tokens=[3, 2],
            vocab_size=10,
        )
        assert start_pos == [2, 7]
        assert special_pos == []
        assert spans == [(2, 3), (7, 2)]

    def test_with_mm_token_ids(self):
        """MM tokens identified by explicit token IDs."""
        input_ids = torch.tensor([1, 5, 5, 5, 2, 3, 5, 5, 4])
        start_pos, special_pos, spans = find_mm_token_positions(
            input_ids=input_ids,
            num_mm_tokens=[3, 2],
            mm_token_ids=torch.tensor([5]),
        )
        assert start_pos == [1, 6]
        assert spans == [(1, 3), (6, 2)]

    def test_with_special_tokens(self):
        """Special tokens (e.g., image_break, image_end) detected within MM region."""
        # Token 5 = MM placeholder, Token 6 = image_break (special), Token 7 = image_end (special)
        input_ids = torch.tensor([1, 5, 5, 6, 5, 7, 2])
        start_pos, special_pos, spans = find_mm_token_positions(
            input_ids=input_ids,
            num_mm_tokens=[5],
            mm_token_ids=torch.tensor([5]),
            mm_special_token_ids=torch.tensor([6, 7]),
        )
        assert start_pos == [1]
        # special_pos are indices into the flat mm token list where specials occur
        assert special_pos == [2, 4]
        # All 5 MM tokens are contiguous at positions 1-5
        assert spans == [(1, 5)]

    def test_non_contiguous_tokens(self):
        """MM tokens scattered with text gaps between them."""
        # Two groups of MM tokens separated by text
        input_ids = torch.tensor([1, 100, 100, 2, 3, 100, 100, 100, 4])
        start_pos, special_pos, spans = find_mm_token_positions(
            input_ids=input_ids,
            num_mm_tokens=[5],  # Single item spanning non-contiguous positions
            vocab_size=10,
        )
        assert start_pos == [1]
        # Two contiguous groups: [1,2] and [5,6,7]
        assert spans == [(1, 2), (5, 3)]

    def test_non_contiguous_multiple_items(self):
        """Multiple items, each with non-contiguous tokens."""
        # Item 1: 3 tokens at [1,2] and [5] (gap at 3,4)
        # Item 2: 2 tokens at [8,9]
        input_ids = torch.tensor([0, 100, 100, 0, 0, 100, 0, 0, 100, 100, 0])
        start_pos, special_pos, spans = find_mm_token_positions(
            input_ids=input_ids,
            num_mm_tokens=[3, 2],
            vocab_size=10,
        )
        assert start_pos == [1, 8]
        # Three contiguous groups across both items: [1,2], [5], [8,9]
        assert spans == [(1, 2), (5, 1), (8, 2)]

    def test_raises_without_vocab_size_or_mm_token_ids(self):
        """Should raise ValueError when neither vocab_size nor mm_token_ids provided."""
        with pytest.raises(ValueError,
                           match="Provide either mm_token_ids or vocab_size"):
            find_mm_token_positions(
                input_ids=torch.tensor([1, 2, 3]),
                num_mm_tokens=[1],
            )


class TestMultimodalRuntimeDataPreset:
    """Test MultimodalRuntimeData when num_cached_mm_tokens and
    num_mm_tokens_in_chunk are pre-set (skipping the computation block).
    This covers the remainder NameError path we fixed."""

    def test_preset_counts_skip_computation(self):
        """When both num_unseen and num_in_chunk are pre-set, __post_init__
        should skip the counting block and not raise NameError on remainder."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=10,
            mm_contiguous_spans=[(0, 5), (5, 5)],
            chunk_end_pos=20,
            special_token_offsets=[],
            num_cached_mm_tokens=5,
            num_mm_tokens_in_chunk=5,
        )
        # Pre-set values should be preserved
        assert runtime.num_cached_mm_tokens == 5
        assert runtime.num_mm_tokens_in_chunk == 5
        assert runtime.total_mm_tokens_in_request == 10

    def test_preset_counts_with_special_tokens(self):
        """Pre-set counts should still allow special token computation."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=10,
            mm_contiguous_spans=[(0, 8), (10, 8)],
            chunk_end_pos=20,
            special_token_offsets=[1, 5, 9, 13],
            num_cached_mm_tokens=8,
            num_mm_tokens_in_chunk=8,
        )
        assert runtime.num_cached_mm_tokens == 8
        assert runtime.num_mm_tokens_in_chunk == 8
        # Special tokens at offsets [1, 5] are < 8 (unseen), [9, 13] are in [8, 16)
        assert runtime.num_cached_special_tokens == 2
        assert runtime.num_special_tokens_in_chunk == 2

    def test_preset_only_unseen_still_computes_chunk(self):
        """When only num_unseen is pre-set but num_in_chunk is None,
        should still run computation (the 'or' condition)."""
        runtime = MultimodalRuntimeData(
            past_seen_token_num=5,
            mm_contiguous_spans=[(0, 10)],
            chunk_end_pos=8,
            special_token_offsets=[],
            num_cached_mm_tokens=5,
            # num_mm_tokens_in_chunk=None triggers computation
        )
        # Should compute num_mm_tokens_in_chunk from spans
        assert runtime.num_mm_tokens_in_chunk == 3  # positions 5..8


class TestCheckMmSpansPresent:
    """Test cases for _check_mm_spans_present — the fail-fast discriminator."""

    def test_none_mm_data_no_raise(self):
        """None py_multimodal_data is fine — no multimodal content."""
        _check_mm_spans_present(None)  # should not raise

    def test_mrope_only_no_raise(self):
        """mrope-only metadata dict should NOT trigger fail-fast."""
        mm_data = {"mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 5)}}
        _check_mm_spans_present(mm_data)  # should not raise

    def test_spans_present_no_raise(self):
        """When mm_contiguous_spans is present, no error regardless of other keys."""
        mm_data = {
            "multimodal_embedding": torch.zeros(5, 10),
            "mm_contiguous_spans": [(0, 5)],
        }
        _check_mm_spans_present(mm_data)  # should not raise

    def test_vision_data_without_spans_raises(self):
        """Vision data present but mm_contiguous_spans missing — must raise."""
        mm_data = {"multimodal_embedding": torch.zeros(5, 10)}
        with pytest.raises(ValueError, match="mm_contiguous_spans"):
            _check_mm_spans_present(mm_data)

    def test_pixel_values_without_spans_raises(self):
        """Pixel values present but mm_contiguous_spans missing — must raise."""
        mm_data = {"image": {"pixel_values": torch.zeros(1, 3, 224, 224)}}
        with pytest.raises(ValueError, match="mm_contiguous_spans"):
            _check_mm_spans_present(mm_data)

    def test_metadata_only_keys_no_raise(self):
        """All metadata-only keys, no vision content — should not raise."""
        mm_data = {
            "mrope_config": {},
            "special_token_offsets": [1, 2],
            "layout_metadata": {},
            "item_types": ["image"],
        }
        _check_mm_spans_present(mm_data)  # should not raise

    def test_empty_dict_no_raise(self):
        """Empty py_multimodal_data should not raise."""
        _check_mm_spans_present({})  # should not raise


class _MockProcessor:
    """Minimal mock of BaseMultimodalInputProcessor for compute_mm_contiguous_spans_if_absent tests."""

    def __init__(self,
                 vocab_size=100,
                 mm_token_ids=None,
                 mm_special_token_ids=None):
        self._vocab_size = vocab_size
        self._mm_token_ids = mm_token_ids
        self._mm_special_token_ids = mm_special_token_ids

    def get_vocab_size(self):
        return self._vocab_size

    def get_mm_token_ids(self):
        return self._mm_token_ids

    def get_mm_special_token_ids(self):
        return self._mm_special_token_ids


class TestEnsureMmContiguousSpans:
    """Test cases for compute_mm_contiguous_spans_if_absent — the idempotent post-processing helper."""

    def test_none_extra_is_noop(self):
        """No crash when extra_processed_inputs is None."""
        compute_mm_contiguous_spans_if_absent([1, 2, 3], None, _MockProcessor())

    def test_no_multimodal_data_key_is_noop(self):
        """No crash when multimodal_data key is absent."""
        extra = {"some_other_key": {}}
        compute_mm_contiguous_spans_if_absent([1, 2, 3], extra,
                                              _MockProcessor())
        assert "mm_contiguous_spans" not in extra

    def test_already_present_is_idempotent(self):
        """Existing spans are NOT overwritten."""
        original_spans = [(0, 5)]
        extra = {"multimodal_data": {"mm_contiguous_spans": original_spans}}
        compute_mm_contiguous_spans_if_absent([100, 101, 102, 103, 104], extra,
                                              _MockProcessor(vocab_size=100))
        assert extra["multimodal_data"]["mm_contiguous_spans"] is original_spans

    def test_computes_spans_when_absent(self):
        """Spans computed from token IDs when not already present."""
        extra = {"multimodal_data": {"multimodal_embedding": "placeholder"}}
        compute_mm_contiguous_spans_if_absent([1, 100, 101, 2, 102], extra,
                                              _MockProcessor(vocab_size=100))
        assert extra["multimodal_data"]["mm_contiguous_spans"] == [(1, 2),
                                                                   (4, 1)]

    def test_no_mm_tokens_stores_empty(self):
        """When no MM tokens found, stores empty list (not None)."""
        extra = {"multimodal_data": {"some_key": "value"}}
        compute_mm_contiguous_spans_if_absent([1, 2, 3], extra,
                                              _MockProcessor(vocab_size=100))
        assert extra["multimodal_data"]["mm_contiguous_spans"] == []

    def test_stores_special_token_offsets(self):
        """Special token offsets stored when mm_special_token_ids provided."""
        proc = _MockProcessor(vocab_size=None,
                              mm_token_ids=torch.tensor([50]),
                              mm_special_token_ids=torch.tensor([60]))
        extra = {"multimodal_data": {"embed": "x"}}
        # input: [1, 50, 60, 50, 2] → mm positions [1,2,3], special at index 1
        compute_mm_contiguous_spans_if_absent([1, 50, 60, 50, 2], extra, proc)
        assert extra["multimodal_data"]["special_token_offsets"] == [1]

    def test_no_vocab_and_no_mm_ids_is_noop(self):
        """When processor provides neither vocab_size nor mm_token_ids, no crash."""
        proc = _MockProcessor(vocab_size=None, mm_token_ids=None)
        extra = {"multimodal_data": {"embed": "x"}}
        compute_mm_contiguous_spans_if_absent([100, 101], extra, proc)
        # Should not have set spans since we can't identify MM tokens
        assert "mm_contiguous_spans" not in extra["multimodal_data"]


class TestFindContiguousMmSpans:
    """Test cases for find_contiguous_mm_spans — the lightweight span scanner
    that does NOT require num_mm_tokens."""

    def test_no_mm_tokens_returns_empty(self):
        """Input with no MM tokens returns empty spans and offsets."""
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[1, 2, 3, 4, 5],
            vocab_size=100,
        )
        assert spans == []
        assert offsets == []

    def test_single_contiguous_run_vocab_size(self):
        """Single contiguous run detected via vocab_size threshold."""
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[1, 100, 101, 2],
            vocab_size=100,
        )
        assert spans == [(1, 2)]
        assert offsets == []

    def test_two_runs_separated_by_text(self):
        """Two contiguous runs separated by text tokens."""
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[100, 101, 5, 102, 103],
            vocab_size=100,
        )
        assert spans == [(0, 2), (3, 2)]
        assert offsets == []

    def test_with_mm_token_ids(self):
        """MM tokens identified by explicit token IDs instead of vocab_size."""
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[1, 50, 50, 2, 50, 3],
            mm_token_ids=torch.tensor([50]),
        )
        assert spans == [(1, 2), (4, 1)]
        assert offsets == []

    def test_with_special_tokens(self):
        """Special tokens included in mask and reported as offsets."""
        # Token 5 = MM, Token 6 = special (image_break)
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[1, 5, 5, 6, 5, 2],
            mm_token_ids=torch.tensor([5]),
            mm_special_token_ids=torch.tensor([6]),
        )
        # All 4 tokens [1..4] are contiguous (5,5,6,5 — special 6 is also MM)
        assert spans == [(1, 4)]
        # In the flat mm_positions [1,2,3,4], index 2 is the special token (value 6)
        assert offsets == [2]

    def test_tensor_input(self):
        """Accepts torch.Tensor input_ids."""
        spans, offsets = find_contiguous_mm_spans(
            input_ids=torch.tensor([1, 100, 101, 2]),
            vocab_size=100,
        )
        assert spans == [(1, 2)]

    def test_numpy_input(self):
        """Accepts numpy array input_ids."""
        import numpy as np
        spans, offsets = find_contiguous_mm_spans(
            input_ids=np.array([1, 100, 101, 2]),
            vocab_size=100,
        )
        assert spans == [(1, 2)]

    def test_raises_without_vocab_size_or_mm_token_ids(self):
        """Should raise ValueError when neither vocab_size nor mm_token_ids provided."""
        with pytest.raises(ValueError,
                           match="Provide either mm_token_ids or vocab_size"):
            find_contiguous_mm_spans(input_ids=[1, 2, 3])

    def test_empty_input_ids(self):
        """Empty input returns empty spans."""
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[],
            vocab_size=100,
        )
        assert spans == []
        assert offsets == []

    def test_all_mm_tokens(self):
        """All tokens are MM — single span covering entire input."""
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[100, 101, 102],
            vocab_size=100,
        )
        assert spans == [(0, 3)]

    def test_multiple_special_tokens(self):
        """Multiple special tokens at different positions."""
        # 5=MM, 6=special, 7=special
        spans, offsets = find_contiguous_mm_spans(
            input_ids=[1, 5, 6, 5, 7, 2],
            mm_token_ids=torch.tensor([5]),
            mm_special_token_ids=torch.tensor([6, 7]),
        )
        assert spans == [(1, 4)]
        # flat mm_positions: [1,2,3,4] → index 1 is token 6, index 3 is token 7
        assert offsets == [1, 3]


if __name__ == "__main__":
    pytest.main([__file__])
