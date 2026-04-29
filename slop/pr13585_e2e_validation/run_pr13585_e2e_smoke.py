# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import struct
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


QWEN_IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
QWEN_VIDEO = "<|vision_start|><|video_pad|><|vision_end|>"


def _write_probe_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((92, 92, 292, 292),
                   fill=(220, 20, 30),
                   outline=(0, 0, 0),
                   width=6)
    draw.text((92, 320), "IMAGE", fill=(0, 0, 0))
    image.save(path)


def _write_probe_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (384, 384),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")

    for _ in range(8):
        frame = np.full((384, 384, 3), 255, dtype=np.uint8)
        points = np.array([[192, 70], [78, 300], [306, 300]], np.int32)
        cv2.fillPoly(frame, [points], color=(255, 80, 20))
        cv2.polylines(frame, [points],
                      isClosed=True,
                      color=(0, 0, 0),
                      thickness=6)
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


def _read_pcm16_wav(path: Path) -> tuple[np.ndarray, int]:
    data = path.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")

    pos = 12
    sample_rate = None
    channels = None
    bits_per_sample = None
    payload = None
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        chunk_start = pos + 8
        chunk_end = min(chunk_start + size, len(data))
        if chunk_id == b"fmt ":
            audio_format, channels, sample_rate, _, _, bits_per_sample = (
                struct.unpack_from("<HHIIHH", data, chunk_start))
            if audio_format != 1 or bits_per_sample != 16:
                raise ValueError("Only PCM16 WAV is supported by this probe")
        elif chunk_id == b"data":
            payload = data[chunk_start:] if size == 0xFFFFFFFF else data[
                chunk_start:chunk_end]
            break
        pos = chunk_end + (chunk_end % 2)

    if sample_rate is None or channels is None or bits_per_sample is None or payload is None:
        raise ValueError(f"Could not parse WAV metadata from {path}")

    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, int(sample_rate)


def _video_frame(idx: int) -> np.ndarray:
    frame = np.full((384, 384, 3), 255, dtype=np.uint8)
    points = np.array([[192, 70], [78, 300], [306, 300]], np.int32)
    cv2.fillPoly(frame, [points], color=(255, 80, 20))
    cv2.polylines(frame, [points],
                  isClosed=True,
                  color=(0, 0, 0),
                  thickness=6)
    cv2.putText(
        frame,
        f"AUDIO {idx}",
        (74, 352),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _write_audio_video(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("mpeg4", rate=2)
        video_stream.width = 384
        video_stream.height = 384
        video_stream.pix_fmt = "yuv420p"

        audio_stream = container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = "mono"

        for idx in range(8):
            frame = av.VideoFrame.from_ndarray(_video_frame(idx), format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        frame_size = 1024
        for start in range(0, len(pcm), frame_size):
            chunk = pcm[start:start + frame_size]
            if chunk.size == 0:
                continue
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1),
                                               format="s16",
                                               layout="mono")
            frame.sample_rate = sample_rate
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)


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


def _generate(
    llm: LLM,
    requests: list[dict[str, Any]],
    names: list[str],
    max_tokens: int,
) -> dict[str, str]:
    outputs = llm.generate(
        requests,
        SamplingParams(max_tokens=max_tokens, temperature=0.0, top_k=1),
    )
    return {
        name: output.outputs[0].text
        for name, output in zip(names, outputs)
    }


def _run_nano(args: argparse.Namespace, image_path: Path,
              video_path: Path) -> dict[str, Any]:
    if args.nano_model_dir is None:
        return {"status": "SKIPPED", "reason": "--nano-model-dir not provided"}

    model_dir = Path(args.nano_model_dir)
    if not model_dir.exists():
        return {"status": "FAIL", "reason": f"{model_dir} does not exist"}

    names = ["image_only", "video_only", "mixed_image_video"]
    with LLM(
            str(model_dir),
            max_batch_size=5,
            max_num_tokens=args.max_num_tokens,
            kv_cache_config=KvCacheConfig(
                free_gpu_memory_fraction=args.free_gpu_memory_fraction,
                enable_block_reuse=False),
    ) as llm:
        image = load_image(str(image_path), format="pt", device="cpu")
        video = load_video(str(video_path),
                           num_frames=args.num_frames,
                           format="pt",
                           device="cpu")

        prompts = {
            "image_only":
            _format_prompt(
                llm.tokenizer,
                "Look at this image: <image>\n"
                "What color and shape is the main object? "
                "Answer in five words or fewer.",
            ),
            "video_only":
            _format_prompt(
                llm.tokenizer,
                "Look at this video: <video>\n"
                "What color and shape is the main object? "
                "Answer in five words or fewer.",
            ),
            "mixed_image_video":
            _format_prompt(
                llm.tokenizer,
                "Look at this image first: <image>\n"
                "Then look at this video: <video>\n"
                "Answer with the image object's color and shape, then "
                "the video object's color and shape.",
            ),
        }
        requests = [
            {
                "prompt": prompts["image_only"],
                "multi_modal_data": {
                    "image": [image]
                },
            },
            {
                "prompt": prompts["video_only"],
                "multi_modal_data": {
                    "video": [video]
                },
            },
            {
                "prompt": prompts["mixed_image_video"],
                "multi_modal_data": {
                    "image": [image],
                    "video": [video]
                },
            },
        ]

        audio_wav = Path(args.audio_wav) if args.audio_wav else None
        if audio_wav is not None and audio_wav.exists():
            audio_samples, audio_sample_rate = _read_pcm16_wav(audio_wav)
            video_audio = load_video(str(video_path),
                                     num_frames=args.num_frames,
                                     format="pt",
                                     device="cpu")
            video_audio.metadata["audio_samples"] = audio_samples
            video_audio.metadata["audio_sample_rate"] = audio_sample_rate
            prompts["video_metadata_audio"] = _format_prompt(
                llm.tokenizer,
                "Watch and listen to this video: <video>\n"
                "What question is spoken in the audio? Also name the "
                "main video shape.",
            )
            prompts["mixed_image_video_audio"] = _format_prompt(
                llm.tokenizer,
                "Look at this image: <image>\n"
                "Then watch and listen to this video: <video>\n"
                "Answer with the image object and the exact question "
                "spoken in the audio.",
            )
            names.extend(["video_metadata_audio", "mixed_image_video_audio"])
            requests.extend([
                {
                    "prompt": prompts["video_metadata_audio"],
                    "multi_modal_data": {
                        "video": [video_audio]
                    },
                },
                {
                    "prompt": prompts["mixed_image_video_audio"],
                    "multi_modal_data": {
                        "image": [image],
                        "video": [video_audio],
                    },
                },
            ])

        texts = _generate(llm, requests, names, args.max_tokens)

    checks = {
        "image_mentions_red_square":
        _contains_all(texts["image_only"], ["red", "square"]),
        "video_mentions_blue_triangle":
        _contains_all(texts["video_only"], ["blue", "triangle"]),
        "mixed_mentions_both":
        _contains_all(texts["mixed_image_video"],
                      ["red", "square", "blue", "triangle"]),
    }
    if "video_metadata_audio" in texts:
        checks.update({
            "video_audio_mentions_spoken_question":
            _contains_all(texts["video_metadata_audio"],
                          ["what", "shown", "image"]),
            "video_audio_mentions_triangle":
            "triangle" in texts["video_metadata_audio"].lower(),
            "mixed_audio_mentions_image_red_square":
            _contains_all(texts["mixed_image_video_audio"], ["red", "square"]),
            "mixed_audio_mentions_spoken_question":
            _contains_all(texts["mixed_image_video_audio"],
                          ["what", "shown", "image"]),
        })

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model_dir": str(model_dir),
        "outputs": texts,
        "checks": checks,
    }


def _run_qwen3vl(args: argparse.Namespace, image_path: Path,
                 video_path: Path) -> dict[str, Any]:
    if args.qwen3vl_model_dir is None:
        return {
            "status": "SKIPPED",
            "reason": "--qwen3vl-model-dir not provided",
        }

    model_dir = Path(args.qwen3vl_model_dir)
    if not model_dir.exists():
        return {"status": "FAIL", "reason": f"{model_dir} does not exist"}

    with LLM(str(model_dir), max_num_tokens=args.max_num_tokens) as llm:
        image = load_image(str(image_path), format="pt", device="cpu")
        video = load_video(str(video_path),
                           num_frames=args.num_frames,
                           format="pt",
                           device="cpu")
        prompt = _format_prompt(
            llm.tokenizer,
            f"{QWEN_IMAGE}{QWEN_VIDEO}\n"
            "Name the main object in the image and the main object in "
            "the video. Include color and shape for each.",
        )
        texts = _generate(
            llm,
            [{
                "prompt": prompt,
                "multi_modal_data": {
                    "image": [image],
                    "video": [video],
                },
            }],
            ["mixed_image_video"],
            args.max_tokens,
        )

    checks = {
        "mixed_mentions_both":
        _contains_all(texts["mixed_image_video"],
                      ["red", "square", "blue", "triangle"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model_dir": str(model_dir),
        "outputs": texts,
        "checks": checks,
    }


def _run_pyav_extract(args: argparse.Namespace, asset_dir: Path,
                      video_path: Path) -> dict[str, Any]:
    if not args.run_pyav_extract_smoke:
        return {
            "status": "SKIPPED",
            "reason": "--run-pyav-extract-smoke not provided",
        }
    if args.audio_wav is None:
        return {"status": "FAIL", "reason": "--audio-wav not provided"}

    audio_path = Path(args.audio_wav)
    if not audio_path.exists():
        return {"status": "FAIL", "reason": f"{audio_path} does not exist"}

    os.environ["TRTLLM_ENABLE_PYAV"] = "1"
    audio, sample_rate = _read_pcm16_wav(audio_path)
    output = asset_dir / "pyav_audio_video.mp4"
    _write_audio_video(output, audio, sample_rate)
    video = load_video(str(output),
                       num_frames=args.num_frames,
                       format="pt",
                       device="cpu",
                       extract_audio=True)
    metadata = video.metadata
    extracted = metadata.get("audio_samples")
    checks = {
        "has_audio_samples": extracted is not None,
        "has_audio_sample_rate": metadata.get("audio_sample_rate") == sample_rate,
        "has_frames": len(video.frames) > 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "input_video": str(video_path),
        "output_video": str(output),
        "metadata_keys": sorted(metadata.keys()),
        "audio_sample_rate": metadata.get("audio_sample_rate"),
        "audio_num_samples": None
        if extracted is None else int(extracted.shape[-1]),
        "frames": len(video.frames),
        "checks": checks,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano-model-dir")
    parser.add_argument("--qwen3vl-model-dir")
    parser.add_argument(
        "--audio-wav",
        default="slop/pr13585_e2e_validation/what_is_shown_in_this_image.wav",
    )
    parser.add_argument("--asset-dir",
                        default="slop/pr13585_e2e_validation/assets")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--max-num-tokens", type=int, default=32768)
    parser.add_argument("--free-gpu-memory-fraction", type=float, default=0.55)
    parser.add_argument("--run-pyav-extract-smoke", action="store_true")
    parser.add_argument("--no-strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asset_dir = Path(args.asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    image_path = asset_dir / "probe_image_red_square.png"
    video_path = asset_dir / "probe_video_blue_triangle.mp4"
    _write_probe_image(image_path)
    _write_probe_video(video_path)

    result = {
        "image_path": str(image_path),
        "video_path": str(video_path),
        "audio_wav": args.audio_wav,
        "nano": _run_nano(args, image_path, video_path),
        "pyav_extract": _run_pyav_extract(args, asset_dir, video_path),
        "qwen3vl": _run_qwen3vl(args, image_path, video_path),
    }
    print(json.dumps(result, indent=2))

    if args.no_strict:
        return
    failed = [
        name for name, section in result.items()
        if isinstance(section, dict) and section.get("status") == "FAIL"
    ]
    if failed:
        raise SystemExit(f"Validation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
