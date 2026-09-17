# Explicit execution-composition repair boundary (draft, no production change)

The diagnosed defect is reuse of a previous combined final target as a nominal geometry reference, followed by adding the current PPO residual again. Correcting this changes the effective action-to-transition execution mapping; it is not a reward-only migration and cannot be described as unchanged physical MDP.

## Minimal route

- Add `execution_composition_review` to `build_migration_plan`, an independent exclusive factor named `execution_composition_factor`, schema `wlr50_clean.independent_post_mapper_residual_same372_fix.v1`.
- Require same `task_first_recovery_v1`, v3, N1, canonical tempered HISTORY372/12 contract; preserve all six selected configuration bytes/bindings. No new profile/config namespace is needed: existing `independent_post_mapper_residual.v1` already declares the intended composition.
- Allow only exact changed runtime files among `semantic_residual_adapter.py`, `semantic_nominal_geometry.py`, `semantic_migration.py`, `semantic_training.py`, and `semantic_cli.py`. Require at least one core composition implementation file changed. No backend, reward, task detector, physical asset, actor, caps, observation, or sampling-temperature change.
- Factor explicitly records `action_execution_changed=true` and `transition_execution_semantics_changed=true`; physical scene/actuator capability, nominal source schedule, task acceptance, reward epsilon0, ranges, and policy kernel stay fixed. No `physical_mdp_changed=false` claim.
- Use normal exact model loader: identity parameter mapping, verify actor/critic/complete Adam/Identity hashes, restore RNG, preserve source effective Adam LR and absolute counters. Fresh rollout required; no migration decisions/updates. No mean-head reset.
- Loader only needs the factor's observation receipt added to its existing reviewed same372 path. Existing CLI preflight already routes arbitrary reviewed plans through exact `validate_migration_plan` plus runner/layout checks; no new CLI switch or config change appears necessary.
- Extend plan revalidation to reconstruct this factor. Old factors and their historical receipts remain unchanged. New factor is mutually exclusive with reward-only, topology, timing, instrumentation and temperature factors.

## Source selection / timing

Do not hardcode CP176768. Select the actual latest checkpoint emitted at the parent-owned `stop_after_update` legal boundary and bind its immutable SHA, sidecar, counters and runtime. No production changes until parent confirms the Isaac process has finished. The draft builder is in `execution_composition_factor_draft.py` and is not imported by production.

## Focused tests after safe boundary

Positive: build/revalidate new factor; unchanged six bindings; official same372 identity load preserves weights/Adam/Identity/RNG/counters with empty128x1x372/12 storage. Negative: protected source/reward/caps/HISTORY/physics/config change, duplicate or missing delta/hash, receipt tamper, mixed factor, partial old rollout. Re-run existing reward-only and migration loader tests. These are wiring tests, not physical success evidence; root follows with scoped real execution and continued PPO.
