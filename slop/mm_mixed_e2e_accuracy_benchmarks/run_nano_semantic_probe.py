# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Controlled Nano mixed-modality semantic probe.

This is intentionally a semantic gate, not a transport smoke. It fails if the
model generates but refuses or misses the controlled image/video facts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

for key in list(os.environ):
    if key.startswith(("SLURM_", "PMI_", "PMIX_")):
        os.environ.pop(key, None)

from tensorrt_llm import LLM
from tensorrt_llm.inputs.utils import load_image, load_video
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams


def _write_probe_image(path: Path) -> None:
    image = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((92, 92, 292, 292), fill=(220, 20, 30), outline=(0, 0, 0), width=6)
    draw.text((92, 320), "IMAGE", fill=(0, 0, 0))
    image.save(path)


def _write_probe_video(path: Path) -> None:
    width = 384
    height = 384
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    for _ in range(8):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        points = np.array([[192, 70], [78, 300], [306, 300]], np.int32)
        cv2.fillPoly(frame, [points], color=(255, 80, 20))
        cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 0), thickness=6)
        cv2.putText(
            frame,
            "VIDEO",
            (104, 352),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _format_prompt(tokenizer: Any, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"User: {user_text}\nAssistant:"


def _contains_all(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return all(word in lower for word in words)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--asset-dir",
        default="slop/mm_mixed_e2e_accuracy_benchmarks/nano_semantic_assets",
    )
    parser.add_argument(
        "--output",
        default="slop/mm_mixed_e2e_accuracy_benchmarks/nano_semantic_result.json",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-num-tokens", type=int, default=32768)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--max-batch-size", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_path = asset_dir / "probe_image_red_square.png"
    video_path = asset_dir / "probe_video_blue_triangle.mp4"
    _write_probe_image(image_path)
    _write_probe_video(video_path)

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.55,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="float32",
    )
    with LLM(
        model=str(model_dir),
        backend="pytorch",
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_num_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        kv_cache_config=kv_cache_config,
    ) as llm:
        image = load_image(str(image_path), format="pt", device="cpu")
        video = load_video(str(video_path), num_frames=args.num_frames, format="pt", device="cpu")
        prompts = {
            "image_only": _format_prompt(
                llm.tokenizer,
                "Look at this image: <image>\n"
                "What color and shape is the main object? Answer in five words or fewer.",
            ),
            "video_only": _format_prompt(
                llm.tokenizer,
                "Look at this video: <video>\n"
                "What color and shape is the main object? Answer in five words or fewer.",
            ),
            "mixed_image_video": _format_prompt(
                llm.tokenizer,
                "Look at this image first: <image>\n"
                "Then look at this video: <video>\n"
                "Answer with the image object's color and shape, then the video object's color and shape.",
            ),
        }
        requests = [
            {"prompt": prompts["image_only"], "multi_modal_data": {"image": [image]}},
            {"prompt": prompts["video_only"], "multi_modal_data": {"video": [video]}},
            {
                "prompt": prompts["mixed_image_video"],
                "multi_modal_data": {"image": [image], "video": [video]},
            },
        ]
        outputs = llm.generate(
            requests,
            SamplingParams(max_tokens=args.max_tokens, min_tokens=1, temperature=0.0, top_k=1),
        )

    names = list(prompts)
    texts = {
        name: output.outputs[0].text
        for name, output in zip(names, outputs)
    }
    token_counts = {
        name: len(output.outputs[0].token_ids)
        for name, output in zip(names, outputs)
    }
    checks = {
        "image_generates_tokens": token_counts["image_only"] > 0,
        "video_generates_tokens": token_counts["video_only"] > 0,
        "mixed_generates_tokens": token_counts["mixed_image_video"] > 0,
        "image_mentions_red_square": _contains_all(texts["image_only"], ["red", "square"]),
        "video_mentions_blue_triangle": _contains_all(texts["video_only"], ["blue", "triangle"]),
        "mixed_mentions_both": _contains_all(
            texts["mixed_image_video"],
            ["red", "square", "blue", "triangle"],
        ),
    }
    result = {
        "model_dir": str(model_dir),
        "image_path": str(image_path),
        "video_path": str(video_path),
        "num_frames": args.num_frames,
        "tp_size": args.tp_size,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "token_counts": token_counts,
        "outputs": texts,
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
