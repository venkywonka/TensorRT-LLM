# Nemotron Chunked Recurrent-State Repro

Purpose: provide a small Jira/NVBug repro for the Nemotron V3 Omni FP8 forced
chunked-prefill recurrent-state divergence.  The cheapest repro is
`--modality text`; optional `image` and `video` modes are present only to help
confirm that the same symptom is not text-only.

What this repro isolates:

- The two runs use the same prompt and the same pre-decode schedule.
- The failure is called only after prompt hashes, `prepare_tp_inputs` schedule,
  and Mamba state-index schedule match.
- A `REPRODUCED` result means recurrent-state contents, first generated-token
  logits/top-k, or generated tokens diverged despite that matched schedule.
- The cheapest signal seen so far is `recurrent_state_content_diverged`.
- The discriminating condition is model-specific, not a common chunked-prefill
  or harness issue: Qwen3VL with chunked prefill was strictly stable in this
  harness, while Nemotron V3 Omni showed Mamba/logit nondeterminism. Nemotron
  with chunked prefill disabled was generated-token stable on the text repro,
  but not strict recurrent-state/logit stable.

## When This Shows

Observed campaign behavior at the time this repro was written:

| Case | Expected Result | Command expectation | Notes |
| --- | --- | --- | --- |
| Nemotron V3 Omni FP8, `--modality text`, forced chunked repeat | Shows | `--expect reproduced` | Cheapest repro. This means the issue does not require image/video token interleaving or multimodal embed slicing. |
| Nemotron V3 Omni FP8, `--modality image`, forced chunked repeat | Shows | `--expect reproduced` | Confirms the issue is not video-only or caused only by non-contiguous video frame layout. |
| Nemotron V3 Omni FP8, `--modality video`, forced chunked repeat | Shows | `--expect reproduced` | Original user-visible lane; Video-MME can surface this as divergent answers, empty outputs, or failed predictions depending on prompt budget and server settings. |
| Nemotron V3 Omni FP8, same prompt, chunked prefill disabled | Strict Mamba/logit divergence still shows; generated token stayed stable in the text run | Use `--disable-chunked-prefill --expect reproduced` for strict-state repro, or inspect `generated_tokens_match` for token-level stability | This falsifies the stronger claim that disabling chunked prefill makes Nemotron strictly deterministic. It does not reproduce the same user-visible token flip on the text prompt. |
| Qwen3VL, same text harness, chunked prefill enabled | Does not show | `--expect stable` | Control proving the harness and forced chunking are not sufficient by themselves; swapping the model away from Nemotron should pass. |
| Nemotron V3 Omni FP8, same Video-MME row, `max_batch_size=1`, chunked off, cache reuse on | Separate failure can show | Outside this minimal repro | This can trip an HTTP 400 token/embedding mismatch, for example 223 image tokens vs 254 image embeddings. That is a related cache/reuse symptom, but it is not the minimal recurrent-state repro in this script. |
| Nemotron V3 Omni FP8, same Video-MME row, `max_batch_size=1`, chunked off, cache reuse off | Did not show in the one-row sanity run | Outside this minimal repro | Row `2247` returned `B`, matched ground truth `B`, had `0/1` failed predictions, and scored `1.000`. This argues against "main video is generally broken." |

The key takeaway is that this repro is for a Nemotron hybrid/Mamba
recurrent-state issue. It is not primarily a batch-size issue, not primarily a
Video-MME scoring issue, and not currently explained by multimodal embed masks
alone because text-only reproduces.

Run from the Lyris TRT-LLM dev container:

```bash
cd /code/tensorrt_llm
python3 -m pip install -r /code/tensorrt_llm/requirements.txt --quiet

export NEMOTRON_REPRO_WHEEL=/path/to/tensorrt_llm-1.3.0rc13-*.whl
export NEMOTRON_V3_OMNI_FP8=/path/to/nemotron-nano-v3-omni_vea-fp8
export QWEN3VL_MODEL=/path/to/Qwen3-VL-8B-Instruct
python3 -m pip install "$NEMOTRON_REPRO_WHEEL" --quiet

env MPI_ENV_CLEANUP=1 CONTAINER_REMAP_ROOT=0 SRUN_EXPORT_MODE=allowlist \
  TLLM_WORKER_USE_SINGLE_PROCESS=1 TLLM_LOG_LEVEL=error \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  mpirun --allow-run-as-root --oversubscribe -np 1 \
  python3 tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py \
    --modality text \
    --model-path "$NEMOTRON_V3_OMNI_FP8" \
    --expect reproduced \
    --output-dir /tmp/nemotron_chunked_recurrent_state_repro
```

Control runs:

```bash
# Same Nemotron model, but chunked prefill disabled.
# Lyris verification showed generated-token stability, but strict Mamba/logit
# repeatability still diverged, so use --expect reproduced for strict tracing.
env TLLM_WORKER_USE_SINGLE_PROCESS=1 TLLM_LOG_LEVEL=error \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  mpirun --allow-run-as-root --oversubscribe -np 1 \
  python3 tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py \
    --modality text \
    --model-path "$NEMOTRON_V3_OMNI_FP8" \
    --disable-chunked-prefill \
    --nonchunked-max-num-tokens 8192 \
    --expect reproduced \
    --output-dir /tmp/nemotron_nonchunked_recurrent_state_control

# Same forced-chunked harness, but Qwen3VL instead of Nemotron: must be stable.
env TLLM_WORKER_USE_SINGLE_PROCESS=1 TLLM_LOG_LEVEL=error \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
  mpirun --allow-run-as-root --oversubscribe -np 1 \
  python3 tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py \
    --modality text \
    --model-path "$QWEN3VL_MODEL" \
    --expect stable \
    --output-dir /tmp/qwen3vl_chunked_recurrent_state_control
```

## Lyris Verification Snapshot

These are the validated text-mode runs from the same Lyris/container setup:

| Case | Output dir | Result | Key evidence |
| --- | --- | --- | --- |
| Nemotron V3 Omni FP8, chunked on | `slop/nemotron_chunked_repro_verify/nemotron_chunked_text_v5` | `REPRODUCED`, `recurrent_state_content_diverged` | Prompt hashes matched; `prepare_tp_inputs` and Mamba state-index schedules matched; traces had `1450` events each; first Mamba content diff appeared at layer `2`; generated token stayed `[11]`; first-token logits/top-k diverged. |
| Nemotron V3 Omni FP8, chunked off | `slop/nemotron_chunked_repro_verify/nemotron_nonchunked_text_v5` | `REPRODUCED`, `recurrent_state_content_diverged` under strict tracing | Prompt hashes and schedules matched; traces had `100` events each; generated token stayed `[11]`; first-token logits/top-k diverged. This is strict-state/logit nondeterminism, not token divergence on this text prompt. |
| Qwen3VL, chunked on | `slop/nemotron_chunked_repro_verify/qwen3vl_chunked_text_v6` | `NOT_REPRODUCED`, `no_divergence`, `EXPECTATION_PASSED: stable` | Prompt hashes matched; schedules matched; traces had `35` events each; generated token `[151645]`, first-token logits/top-k, and all captured signatures matched exactly. |

For image/video variants, pass paths explicitly instead of embedding
user-specific filesystem locations:

```bash
export NEMOTRON_REPRO_IMAGE=/path/to/image.png
export NEMOTRON_REPRO_VIDEO=/path/to/video.mp4

python3 tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py \
  --modality image \
  --model-path "$NEMOTRON_V3_OMNI_FP8" \
  --image-path "$NEMOTRON_REPRO_IMAGE" \
  --expect reproduced \
  --output-dir /tmp/nemotron_chunked_recurrent_state_repro_image

python3 tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py \
  --modality video \
  --model-path "$NEMOTRON_V3_OMNI_FP8" \
  --video-path "$NEMOTRON_REPRO_VIDEO" \
  --expect reproduced \
  --output-dir /tmp/nemotron_chunked_recurrent_state_repro_video
```

`TLLM_WORKER_USE_SINGLE_PROCESS=1` is the preferred local/Jira repro process
mode. It is a TRT-LLM LLM API knob for TP1 that keeps the worker in the launch
process, which makes the local trace hooks observe the same code path. This
script also uses that env var to keep the repeated runs in the current Python
process rather than launching child Python processes. It is not a chunking
toggle and should not change `enable_chunked_prefill` semantics. On Lyris, keep
the `mpirun --allow-run-as-root --oversubscribe -np 1` wrapper to avoid the
direct `srun` OpenMPI/PMI import failure.

When using an editable install backed by precompiled binaries, the command also
needs the extracted library paths in `LD_LIBRARY_PATH`, for example:

```bash
export LD_LIBRARY_PATH=/code/tensorrt_llm/tensorrt_llm/libs:/usr/local/tensorrt/lib:${LD_LIBRARY_PATH:-}
```

The script explicitly imports and attaches `tensorrt_llm.bindings` before model
construction because some editable precompiled installs make the extension
importable without attaching it to the parent package object early enough for
`tensorrt_llm.bindings.internal...` lookups.

The script also inserts its checkout root into `sys.path`. That matters because
executing a file under `tests/.../scripts` normally puts only the script
directory on `sys.path[0]`; without the checkout root, Python can silently import
an installed `tensorrt_llm` wheel instead of the repro checkout.

In single-process mode, the trace hooks are installed once but read
`NEMOTRON_CHUNKED_REPRO_TRACE_PATH` dynamically on every event. The driver resets
that env var and the event counters before each repeated run, so `run1.trace`
and `run2.trace` remain independently comparable. Hook installation happens
after importing `tensorrt_llm` and attaching bindings, but before constructing
the `LLM`, so the first run is traced as well as later repeats.
Request metadata reads are exception-safe because some warmup requests reject
properties such as `context_chunk_size` outside context/generation-init phases.

Expected issue output:

```text
REPRODUCED: prompt/state schedule matched while recurrent_state_content_diverged
```

If the script prints `NOT_REPRODUCED`, inspect `summary.json` before concluding
the bug is gone. The useful discriminator is `hypothesis_call`:

| `hypothesis_call` | Meaning |
| --- | --- |
| `recurrent_state_content_diverged` | Best repro signal: prompt/schedule/state-index metadata matched, but recurrent-state content changed across repeats. |
| `first_generation_logits_diverged` | Recurrent-state content signatures did not catch the first mismatch, but first generated-token top-k/logits diverged. |
| `generated_token_diverged` | Visible output token IDs diverged, but earlier captured summaries did not isolate the cause. |
| `prompt_diverged` | The prompt/input construction changed; this is not a valid recurrent-state repro. |
| `prepare_tp_inputs_schedule_diverged` | The pre-decode schedule changed; compare inputs and chunking before blaming recurrent state. |
| `mamba_state_index_schedule_diverged` | Mamba state-index metadata changed; this is a scheduler/cache-index issue rather than same-schedule state-content drift. |
| `no_divergence` | The chosen command did not reproduce in that run. Try the text mode first, then image/video with a fresh output directory. |

Known output fields:

- `run1.json`, `run2.json`: prompt length/hash, generated token IDs, decoded
  text, first-generation top-k/logit rows, and trace path.
- `run1.trace.jsonl`, `run2.trace.jsonl`: `prepare_tp_inputs`, Mamba
  state-index, and Mamba forward recurrent-state summaries.
- `summary.json`: `schedule_match_before_decode`,
  `mamba_content_signatures_match`, `first_generation_topk_match`,
  `hypothesis_call`, and first-diff payloads.

Local validation without GPUs:

```bash
env PYTHONPYCACHEPREFIX=/tmp/nemotron_repro_pycache \
  python3 -m py_compile tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py
python3 tests/integration/defs/accuracy/scripts/nemotron_chunked_recurrent_state_repro.py --self-test
```

Stash-style fork ref for citation:

```bash
git fetch origin refs/heads/stash/nemotron-mamba-chunked-repro-20260428-160823
git checkout FETCH_HEAD
```

This ref is for Jira/NVBug evidence only and is not intended as a PR.
