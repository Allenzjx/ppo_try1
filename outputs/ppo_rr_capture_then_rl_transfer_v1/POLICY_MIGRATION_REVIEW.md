# RR capture policy and explicit checkpoint migration

Implementation ready for integrated review; no real migration publication, GPU, simulation or optimizer update was performed by this subtask.

## Observation and inference contract

- Namespace: `rr_capture_then_rl_transfer_v1`.
- Policy: `rr_capture_then_rl_transfer_history_v1`.
- Layout: `role389_rr_capture_transfer_v1`; dimension410 is calculated from the inherited389, the assist module's14 declared feature names, and seven task-context fields.
- Columns0:389 retain the complete existing encoding. Columns389:403 encode `frame.info.rr_capture_assist`; columns403:410 encode the exact seven boolean keys in `frame.info.rr_capture_transfer_context`. Missing fields, extra context keys, and non-boolean context values fail closed.
- The actor/critic input layers retain every existing column and append21 zero columns. All other learned parameters, including learned mean and sigma heads, are retained. HISTORY uses the unchanged389 prefix; the receiving-wheel sigma kernel uses the unchanged372 prefix.
- Original Gaussian samples and log probabilities remain the PPO data. FL/RR post-sample actuator assistance is separately observed and never substituted for a raw action.
- A frozen checkpoint-policy prefix requires the already saved/reloaded410 checkpoint with exactly matching runtime and policy contract. It cannot relabel an old389 file as the new actor.

## Migration API

Module `wlr50_clean.ppo.semantic_rr_capture_migration` exports:

- `SCHEMA = "wlr50_clean.rr_capture_transfer_append.v1"`
- `FACTOR_KEY = "rr_capture_transfer_factor"`
- `build_rr_capture_migration(checkpoint, current_contract, *, reason, reviewed_code_sha256, project_root=None)`
- `validate_rr_capture_migration(checkpoint, current_contract, plan_path, *, project_root=None)`
- `publish_rr_capture_checkpoint(checkpoint, current_contract, plan_path, output_checkpoint)`

The target must be a new, clean committed runtime. Runtime code changes have an explicit bounded inventory and exact reviewed hashes. The old six configuration files remain protected. New action/quality/reward files must be byte-identical to the source; execution adds only the declared hip-only RR mode, wheel mode `off`, and revision. The task spec adds only `rr_capture_continuation_semantics`. The observation schema retains all old389 fields and appends the explicit groups/version marker.

All Adam steps, groups, actual LR and existing moments are retained; only the two first-layer moment tensors gain zero columns. Identity and complete RNG are preserved. Five historical origins/migration records and complete four-event AUX ledger103/104 (front96/96, RR7/8), plus separate historical7/8, remain intact. A new `rr_capture_transfer_branch` begins at the actual source counters with zero new credit. Its migration and origin are carried by normal PPO saves. Old incomplete rollout/physical state is not inherited. The publisher uses the source device, official save and independent fresh reload; it does not promote a latest pointer.

The audited actual source is CP220544/1688/33760, SHA256 `56239937cd0aaccdc8b7ea36c6266a41b3bed9df67b00f84a06806d2a6fa09b7`. Its completed training/checkpoint evidence is valid; the original outer run remains FAILED due to the postflight pinned-HEAD mismatch. The new explicit migration binds the original6ac7 source contract to the new reviewed runtime; it neither edits that failure nor waives checkpoint contracts.

## Bounded verification

18 dedicated CPU tests pass, covering all13 phase distribution/critic continuity, original raw likelihood, exact parameter/Adam zero extension and source immutability, strict observation fields, frozen410 prefix identity, official synthetic checkpoint publication/fresh reload, subsequent normal-save lineage, and rejection of committed/rehashed unrelated configuration changes. A preceding combined regression with old P05 and checkpoint-prefix/entrypoint tests passed78 tests. `git diff --check` passed; only line-ending informational warnings appeared.

These are synthetic wiring/migration checks, not physical success or added PPO/AUX credit. All CPU helpers exited. Integrated physical diagnostics and real publication remain the root task's next actions after review and runtime freeze.
