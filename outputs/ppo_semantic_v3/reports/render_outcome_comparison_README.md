# Neutral A/C outcome comparison — unexecuted artifact helper

Status: **draft only, not imported, tested or executed**. No Python, ffmpeg or Isaac was run while preparing this file. It lives under outputs/reports, not in the production runtime. It is not called by training and cannot be an optimizer prerequisite.

Scope: a new same-condition v3 A source may be incomplete, physically failed or successful; C must be an independently validated saved-checkpoint task success. The result is named `fsm_vs_ppo_outcomes.mp4`, with each actual outcome on screen. Neither a successful C video nor this comparison establishes improved stability. Failed C recordings remain available as raw diagnostics; this first bounded helper does not publish failed-C comparisons.

The existing v2 A video with64 extra physical warmup ticks is intentionally ineligible. Capture fresh v3 A/C sources with the video wrapper's `-SemanticVersion v3`, identical committed runtime, seed4001, camera and natural P01. A can finish with a diagnostic/nonzero wrapper status: its complete raw artifacts are still the input. Do not rerun A until successful merely to satisfy publication.

After all live simulation and training processes have exited, a future supervised invocation is:

```powershell
$env:PYTHONPATH = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\src'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -P `
  'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\render_outcome_comparison.py' `
  --a-source '<absolute new v3 A run>\source' `
  --c-source '<absolute new v3 C run>\source' `
  --output-dir 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\videos\outcome_comparison_NEW_UNIQUE_ID' `
  --acknowledge-untested-helper
```

Replace placeholders with actual completed sources. The acknowledgement is a reminder of the untested implementation, not permission to bypass any validation. The helper uses current pinned production validators and will refuse if HEAD/configuration no longer matches the sources; do not edit historical manifests to make them match.

Checks include managed source/run binding, unchanged frozen/runtime/configuration inventory, common camera and seed, natural P01, original180-settle sequence with zero added ticks, actuator audit continuity, no state writes, independent physical replay, real C checkpoint/hash/load provenance, and full H.264/yuv420p/15fps decode with native120Hz-to15Hz frame/PTS ledger. The existing bound video CLI uses deterministic actor means; the C success validator retains its learned-state unchanged assertion. A's diagnostic path allows only the existing physical-task-noncompletion error, not an interface, recorder or codec error masquerading as a task outcome.

Every recorded source frame is retained. No trimming, interpolation, speed filter or frame-rate conversion is used. The shorter side holds its final recorded frame only after its full source ends, visibly labelled `SOURCE ENDED - last recorded frame held`; this is not simulated post-roll or continued task credit. Final partial physics ticks keep their original ledger timing rather than being falsely assigned to a later image. The combined image is2560×720,15fps,H.264,yuv420p, at most200s, with full decode/PTS checks.

Output goes into a fresh directory only, with `comparison.started.json`, then either `comparison_manifest.json` or `comparison_failed.json`. Existing files are never overwritten. Failed partial outputs are retained and must not be presented as validated. No stability metric is synthesized or compared across unequal episode windows. Any later stability claim needs its separately declared matched physical window and evidence.

Known preparation limit: this code has received source-level review only. Its first actual execution and final visual review remain pending. Source validation can legitimately reject a malformed/incomplete recording, an unsupported post-success stability failure, or an old runtime. Such a rendering issue must not block or reset PPO training.
