<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# PR 13585 E2E Validation Commands

Run from the Lyris/EOS worktree once it exists:

```bash
cd /home/gvenkatarama/scratch/TensorRT-LLM-Worktrees/mm-mixed-on-pr13585/TensorRT-LLM
git rev-parse --short HEAD
```

Expected commit:

```text
b24e5a4c5f
```

Inside the validation container, set up the editable runtime and PyAV layer:

```bash
cd /workspace/TensorRT-LLM

TRTLLM_PRECOMPILED_LOCATION=https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/36106/aarch64-linux-gnu/tensorrt_llm-1.3.0rc13-cp312-cp312-linux_aarch64.whl \
  python3 -m pip install -e .

python3 -m pip install --disable-pip-version-check --no-input \
  --cache-dir slop/pr13585_e2e_validation/pip_cache_pyav \
  --target slop/pr13585_e2e_validation/pyav_site \
  av

export PYTHONPATH="$PWD/slop/pr13585_e2e_validation/pyav_site:${PYTHONPATH:-}"
export TRTLLM_ENABLE_PYAV=1
python3 -c 'import av; print(av.__version__)'
```

Nano required smoke:

```bash
PYTHONUNBUFFERED=1 python3 slop/pr13585_e2e_validation/run_pr13585_e2e_smoke.py \
  --nano-model-dir /home/gvenkatarama/scratch/checkpoints/nemotron-nano-v3/nemotron-nano-v3-omni_vea-fp8 \
  --audio-wav slop/pr13585_e2e_validation/what_is_shown_in_this_image.wav \
  --run-pyav-extract-smoke \
  --max-tokens 96 \
  --max-num-tokens 32768
```

Optional Qwen3VL mixed image/video smoke, only if this model cache is visible
inside the Lyris container:

```bash
PYTHONUNBUFFERED=1 python3 slop/pr13585_e2e_validation/run_pr13585_e2e_smoke.py \
  --nano-model-dir /home/gvenkatarama/scratch/checkpoints/nemotron-nano-v3/nemotron-nano-v3-omni_vea-fp8 \
  --qwen3vl-model-dir /lustre/share/coreai_comparch_trtllm/llm-models/Qwen3/Qwen3-VL-8B-Instruct \
  --audio-wav slop/pr13585_e2e_validation/what_is_shown_in_this_image.wav \
  --run-pyav-extract-smoke \
  --max-tokens 96 \
  --max-num-tokens 32768
```

Use this `srun` shape after filling in an available `JOB_ID` and
`CONTAINER_IMAGE`:

```bash
srun --jobid="$JOB_ID" --overlap --ntasks=1 --nodes=1 \
  --container-name="trtllm-pr13585-e2e-${JOB_ID}" \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/home/gvenkatarama/scratch/TensorRT-LLM-Worktrees/mm-mixed-on-pr13585/TensorRT-LLM:/workspace/TensorRT-LLM,/home/gvenkatarama/scratch:/home/gvenkatarama/scratch,/lustre/share/coreai_comparch_trtllm:/lustre/share/coreai_comparch_trtllm \
  bash -lc 'cd /workspace/TensorRT-LLM && \
    TRTLLM_PRECOMPILED_LOCATION=https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/LLM/main/L0_MergeRequest_PR/36106/aarch64-linux-gnu/tensorrt_llm-1.3.0rc13-cp312-cp312-linux_aarch64.whl python3 -m pip install -e . && \
    python3 -m pip install --disable-pip-version-check --no-input --cache-dir slop/pr13585_e2e_validation/pip_cache_pyav --target slop/pr13585_e2e_validation/pyav_site av && \
    export PYTHONPATH="$PWD/slop/pr13585_e2e_validation/pyav_site:${PYTHONPATH:-}" && \
    export TRTLLM_ENABLE_PYAV=1 && \
    PYTHONUNBUFFERED=1 python3 slop/pr13585_e2e_validation/run_pr13585_e2e_smoke.py --nano-model-dir /home/gvenkatarama/scratch/checkpoints/nemotron-nano-v3/nemotron-nano-v3-omni_vea-fp8 --audio-wav slop/pr13585_e2e_validation/what_is_shown_in_this_image.wav --run-pyav-extract-smoke --max-tokens 96 --max-num-tokens 32768'
```
