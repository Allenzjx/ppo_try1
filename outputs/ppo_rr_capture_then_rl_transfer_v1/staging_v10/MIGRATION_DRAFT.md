# v10 earned-reserve migration — staged, not applied or tested

Prepared only under this outputs directory. No source/config/test production file, real checkpoint, optimizer or simulator was changed or executed by this preparation.

The staged assist change removes exactly three uses of the +25 mm ceiling: earning reserve, continuing reserve, and validating the public mode6 snapshot. It preserves the -15 mm lower bound, measured physical/contact predicates, current support/tracking/progress checks, and the existing 53 degree /45 active-second total. The 25 mm TOP sensor rule is not changed. v9 wheel bytes, task/reward/schema/kernel/caps and all14+7 feature definitions remain unchanged. Existing public mode6 eligibility changes control behavior, so this is an explicit different control-MDP version, not same-MDP resume.

## Draft files and API

- `semantic_rr_capture_reserve_migration.py` is the proposed production module. Schema `wlr50_clean.rr_capture_reserve_same410.v10`, factor `rr_capture_reserve_v10_factor`, persistent receipt `rr_capture_reserve_v10_migration`. Exports `build_rr_capture_reserve_migration`, `validate_rr_capture_reserve_migration`, `record_loaded_rr_capture_reserve`, `validate_v10_branch_receipt`.
- `integration.patch` adds the strict generic-loader hooks, ordinary-save receipt carry, highest-priority v10 same-branch CLI/direct-training validation and the profile markers. It does not relax any historical validator.
- `test_semantic_rr_capture_reserve_migration.py` contains small source/scope/AUX/route negatives and a synthetic official save/fresh-reload identity fixture. Not executed and not real PPO evidence.
- `assist_production.patch` and its focused control tests are independently owned by the assist reviewer. Do not duplicate their runtime edit.

Target profile revision is proposed as `rr_capture_progress_reserve_v10`; target feedback is `progress_earned_capture_reserve_incremental_v10`. The exact search string is bound to the staged assist literal. Runtime delta is exactly six paths: assist, execution_profile, the new migration module, semantic_cli, semantic_training and semantic_migration. Five other selected config files must stay byte-identical. Whole-assist AST validation permits only the three comparisons, four named public/error literals, removal of the now-unused upper-bound import and docstring changes. It does not permit changed travel/exposure, feature scales, contact rules, hidden state or extra helpers.

## Source selection remains pending

The actual v9 block has sealed at CP222592 /1704 PPO /34080 Adam. Plain learned checkpoint SHA `554e0ea53763aefaa5452f2ddff87ecfd40ac452eb037ac44e4ad973df40c77c`, sidecar SHA `35c8f268109348182bb51bdb91fcd180092724d73dac9067f0f68fa433217ab4`. This is a fact, not yet a v10 source registration: root is reviewing a possible separate bounded front AUX update at these same counters. Do not register the plain checkpoint and silently substitute a later AUX file.

`SOURCE_REGISTRY` is deliberately empty and fails closed. After the actual source decision, register exact checkpoint SHA, sidecar SHA, all three counters, source role and the full `rr_capture_transfer_branch.front_retention_auxiliary` digest (or explicit null if absent). Source must be the existing v9 branch and retain at least one actual completed update beyond the v9 publication221952 /1699 /33980. No rollback to221952 or220544, and no borrowing main-branch640 credit. Source may be either the actual sealed PPO checkpoint or its explicitly approved official AUX result; all existing source metadata, including the new AUX ledger if present, is preserved wholesale.

Source runtime remains frozen `3edda51732f4ff85717fcb3491bf5c8c5766474d`. The future target HEAD is an explicit runtime parameter, never guessed. Same branch `ancestor220544_signed_wheel_v8`, unchanged historic origin220544 /1688 /33760 and unchanged old receipts/events. New migration counter origin is the selected actual source counters; migration itself adds0 PPO,0 Adam and0 AUX.

## At the later authorized boundary

1. Finalize source registration only from actual sealed source/sidecar and optional official AUX receipt. Review/apply the six-path change plus focused tests; do not alter historical source artifacts.
2. Run the new focused migration tests, existing v9 routing/migration tests and the staged assist tests under CPU-only/one-thread settings. These fixtures are not actual learning.
3. Commit the reviewed actual target runtime. Build/validate a plan with explicit source SHA, actual target HEAD and exact reviewed changed-path hashes.
4. Use the established official source-device load/save/fresh-reload protocol. Preserve exact actor/critic/all buffers, full Adam moments/steps/groups/effective LR, Identity and Python/NumPy/CPU/CUDA RNG. Clear only unfinished rollout. Save a unique same-branch history filename; do not promote either latest pointer during migration.
5. Natural-P01 and checkpoint-policy suffix training use the same explicit branch. Current v10 receipt takes priority; malformed v10 cannot fall back to inherited valid v9. Ordinary subsequent saves carry both receipts and complete branch/AUX metadata.

No publisher execution, real source registration, fitting budget or physical success is claimed by this draft.
