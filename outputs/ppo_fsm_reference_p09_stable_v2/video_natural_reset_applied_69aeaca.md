# Natural-P01 video reset repair applied: 69aeaca

Recorded 2026-09-10 (UTC). This is an applied-change receipt, not a successful-video or full-task certificate. The original deferred draft and failed capture remain unchanged.

## Applied scope

Verified commit: `69aeaca777dcc8653e60f19da56ae1cd002e5271`, “Verify natural P01 video resets from backend provenance”, committed 2026-09-10 05:48:35 -0400. Git records three files, 266 insertions and six deletions:

- Production: `src/wlr50_clean/ppo/semantic_video.py`.
- Updated fixture: `tests/unit/test_semantic_video_fsm_reference_v2.py`.
- New regression tests: `tests/unit/test_semantic_video_natural_reset.py`.

The only changed runtime file has SHA-256 `c480d482054a1e1124c3b8b89fe137f4dd0f13db15d6b93790417ed7f6ad781b` before and `3d8b23596c88ac7bc2771ea3725a206c56a523e9659adcc6ace5a5fecc86cc85` after, as recorded in the exact migration. Backend reset, sensing, control, physics, nominal, reward, observation and six experiment configurations are unchanged.

For `fsm_reference_p09_stable_v2` only, capture and independent source replay now share a persisted natural-reset proof. The existing semantic `training_phase_snapshot="P01"` label is accepted only with actual semantic-natural restoration metadata, first reset, empty options, P01/tick-zero/zero-decision entry, clock/prime evidence and appropriate B/C runtime identity. Legacy A uses its actual natural-reset receipt. Real snapshots, explicit reset options, repeated resets, missing or inconsistent proof and historical restoration are rejected. Older experiments retain their previous routing; their existing artifacts are not retrospectively required to contain this new proof.

## Retained failure and tests

The original capture is preserved at:

`runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T0930063395586Z_g64c03243ac05_4bb994da17ed47bea5c9e0b04c3a2a41/source/semantic_video_source_manifest.json`

It failed the old first-reset guard before checkpoint loading or policy actions: reset count one, options `{}`, semantic marker `P01`, zero issued decisions, `success_candidate=false`. This was a video initialization/provenance rejection, not an observed PPO task failure. It has not been reclassified as successful.

Actual XML receipts under this output directory:

| Receipt | Tests | Passed | Failures | Errors / skipped | XML seconds |
| --- | ---: | ---: | ---: | --- | ---: |
| `video_natural_reset_regression.xml` | 173 | 172 | 1 | 0 / 0 | 8.613 |
| `video_natural_reset_regression_v2.xml` | 316 | 316 | 0 | 0 / 0 | 20.092 |

The first failure was the real-CPU checkpoint-loader test's launcher-environment assertion: `CUDA_VISIBLE_DEVICES` was absent instead of the explicitly required empty string. The corrected CPU invocation passed; no production workaround for that assertion was introduced. The parent console rounded the final run to 20.11 seconds; the table uses the XML's 20.092 seconds.

The final selection comprises the existing 231-test group, 66 new reset-proof cases and 19 backend tests. Coverage includes real-backend-style A/B/C receipts, snapshot/repeated-reset/missing-proof negatives, persisted-proof tampering and historical routing, as well as the prior CPU HISTORY372 saved/reloaded policy and nonempty-Adam checks. These are CPU tests, not a repaired live camera run or task-success evaluation.

## Exact continuation plan

Read and hashed plan:

`migrations/video_natural_reset_from_000142848_to_69aeaca.json`

- Plan file SHA-256: `42d4eea45cd9d5ee39246477fd09fa265065108e8565c2586f8073d5f01ce51e`.
- Source immutable checkpoint: `checkpoints/history/checkpoint_step_000142848.pt`, source commit `64c03243ac05e43e7a5843eaee252eef1136c925`.
- Source manifest counters: 142,848 policy decisions, 1,081 PPO updates, 21,620 optimizer steps; effective last-update learning rate `1e-5`.
- Target commit: `69aeaca777dcc8653e60f19da56ae1cd002e5271`.
- Actual allowed runtime delta: only `src/wlr50_clean/ppo/semantic_video.py`, through the separate strict video-instrumentation factor.
- Plan target contract digest: `52117e4f6df4d987941b36bd69389a93aa5052e24f13600cafbd1df72c75a9ed`.
- Parent-validated target runtime fingerprint: `21ad0d667c84e5c53e5e8aa37239421f8dcc4f717f5bcf6d1c817c2dd95293a4`; this is distinct from the plan's target contract digest above.

The plan preserves actor/critic weights, learned standard deviation, Adam state and effective learning rate, identity normalizers, RNG and lifetime counters, with the same 372-observation / 12-action HISTORY policy contract. Old rollout storage is discarded and fresh data must be collected. No new MDP or expanded compatibility factor is authorized by this change.

## Live status boundary

The parent reported that ordinary deterministic P01 evaluation of checkpoint 142848 completed at 41.075 seconds with P05 incomplete, then started a 4,096-decision natural-P01 training request under the new commit (session 54301). A started request is not 4,096 completed decisions or additional completed updates; this receipt does not audit the active process or claim its outcomes.

No repaired live video attempt had yet been performed when this receipt was requested. Therefore the new guard has CPU evidence only, and there is no current successful full-task video, live guard-pass claim or FSM-vs-PPO improvement claim here. The deferred source/test mirrors remain the original pre-application audit, and global progress/RECOVERY records are owned separately.
