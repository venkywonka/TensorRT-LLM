# PR 13585 Mixed-Modality E2E Validation

## Goal

Mirror `mm-mixed-on-pr13585` to the EOS/Lyris filesystem and run end-to-end smoke validation against the adapted branch.

## Scope

- Use the adapted branch head `b24e5a4c5f` based on `upstream/pr/13585`.
- Mirror exact local HEAD `b24e5a4c5f74b32156f9a1d64ede9c0b1f32c402`.
- Lyris mirror target: `/home/gvenkatarama/eos-scratch/TensorRT-LLM-Worktrees/mm-mixed-on-pr13585/TensorRT-LLM` on the workstation, mapping to `~/scratch/TensorRT-LLM-Worktrees/mm-mixed-on-pr13585/TensorRT-LLM` on EOS/Lyris.
- Run end-to-end generation/semantic smoke tests, not narrow unit tests.
- Prioritize Nano image-only, video-only, mixed image+video, video metadata audio, and image+video+audio via video metadata.
- Include Qwen3VL mixed image/video if model cache and runtime infra are available.
- Keep reusable scripts/logs in this folder.

## Done

- Spawned infra readiness, test-plan, and execution/TDD subagents.
- Confirmed the local checkout is already `mm-mixed-on-pr13585` at `b24e5a4c5f`.
- Prepared a self-contained E2E smoke harness in this task folder:
  `run_pr13585_e2e_smoke.py`.
- Copied the known speech probe WAV into this task folder:
  `what_is_shown_in_this_image.wav`.
- Wrote runnable Lyris command templates in `commands.md`.

## Remaining

- Mirror the branch to EOS/Lyris.
- Confirm active Lyris allocation/container/build environment.
- Run smoke tests and collect logs.
- Locate or provide a Lyris-visible validation container image.
- Use Qwen3VL only if `/lustre` exposes the model cache.
