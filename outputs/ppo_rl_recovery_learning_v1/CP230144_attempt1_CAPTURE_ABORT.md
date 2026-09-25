# CP230144 deterministic attempt 1 — capture abort evidence

Status: sealed media artifact failure; **not** a physical task terminal and not a formal CP230144 evaluation result.

- Run: `runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1200016881929Z_g59e868f3e223_682c728eb6c54cdfa895ae0c3a080c98`.
- Runtime/checkpoint: HEAD `59e868f3e223c589e7645a0f5d63f91fa6119fb6`, CP230144 (`230144 / 1760 / 35200`).
- Sealed source manifest SHA-256: `fdc8eb9f0d05f22725b21e7956c5bc40c788b913fa2536c90e9b74a1851e6919`.
- Sealed run manifest SHA-256: `53c68d77612369ee5a08b7521d1860dc2de8e6428852ae28a94a9ee79387b001`.
- Viewport manifest SHA-256: `10147ef089dd9a89ff7625975b9ccf00f31fdbf235b72adeb9235672203ee91f`.
- Exact failure: `VideoArtifactError: VIDEO_OR_ARTIFACT_ERROR: encode failed: RuntimeError: viewport callback_count=0 after 3 waits, expected 1`.
- The writer sealed `116` real frames (`7.733334 s`) at physics tick `936`; those frames fully decode at 15 fps, have continuous monotonic PTS, 116 unique checksums, and zero black-like frames. They remain partial diagnostic evidence only.
- `termination_reason` and `termination_source` are null. `physical_task_success=false` at an interrupted recording is not a completed task-failure outcome.

The viewport identity and render product remained unchanged, completed frames have one callback per render, and there were no extra app updates or renders. The failing render exhausted the existing three callback waits; this is the known intermittent viewport callback path, not evidence of a control or task failure. The bounded recovery is one fresh-process rerun with the same checkpoint, runtime, seed, and evaluation settings. The formal three-video destination remains unused; no frame is synthesized, duplicated, or appended from this partial attempt.
