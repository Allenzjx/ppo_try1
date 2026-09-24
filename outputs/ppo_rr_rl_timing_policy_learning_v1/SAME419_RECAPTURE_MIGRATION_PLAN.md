# Minimal next same419 continuation — read-only proposal

No implementation, publication or optimizer action in this note. Finish the fixed fa4b98ed course block and its deterministic evaluation first. Bind the actual last sealed checkpoint/sidecar/counts afterward; planned totals are not a source registration. Keep the existing namespace and original branch origin 220544/1688/33760. Do not repeat the selected-ancestor migration or borrow unrelated main-branch updates.

## Existing path and exact limitations

- `semantic_cli.py` already accepts `--resume-migration`; `run_semantic_ppo.ps1` exposes `-ResumeMigration`. No new launcher framework is needed.
- `semantic_migration.validate_migration_plan` dispatches explicit schema validators. Add one same419 schema route rather than loosening old 410/389 factors.
- Existing `rear_policy_timing_factor` is specifically the pinned 410→419 ancestor migration. Its loader appends nine zero columns and creates the initial branch: **it must not be reused for 419→419**.
- `NewMdpWarmStart` creates fresh Adam in the existing implementation and is unsuitable for the requested full-state continuation.
- `semantic_training._verify_reviewed_same410_identity_state` hardcodes 410. Do not call it unchanged or weaken its legacy scope. The ordinary strict checkpoint loader already verifies actual actor/critic/Adam/normalizer hashes, embedded metadata versus sidecar, original seed and restored RNG. Reuse that loader as a source-state leaf.
- Two concrete routing guards need updating: CLI currently assumes every rear-timing resume migration comes from the old RR-capture namespace; `validate_rear_policy_namespace` currently requires the original append receipt's target hash to equal the current runtime. A new runtime must chain an additional receipt, not overwrite that ancestor receipt or skip this check.

## Proposed small module and hooks

New `src/wlr50_clean/ppo/semantic_rear_recapture_migration.py`:

- `SCHEMA = wlr50_clean.rear_recapture_same419.v1`
- `FACTOR_KEY = rear_recapture_same419_factor`
- `MIGRATION = rear_recapture_migration`
- `build_rear_recapture_migration`, `validate_rear_recapture_migration`, `load_rear_recapture_migration`, `publish_rear_recapture_checkpoint`.

The dedicated loader first verifies the explicit plan, then calls existing `load_semantic_checkpoint(runner, source, contract=source_metadata.runtime_contract, seed=original_seed)` with no migration argument. Same actor/layout/runner configuration makes this an identity load, with no tensor remapping, zero append, head initialization or optimizer recreation. After exact-state checks it returns the same metadata plus the new runtime and immutable revision receipt. Empty storage `(128,1,419)`, raw actions `(128,1,12)`, transition actions absent, complete Identity normalizers and actual effective LR must be checked explicitly. Restore original full RNG after any construction/comparison that consumes randomness.

Small production integration only:

1. `semantic_migration.py`: dedicated schema dispatch.
2. `semantic_training.py`: early dedicated loader dispatch, and add `rear_recapture_migration` to the ordinary-save metadata carry tuple. No PPO optimizer/objective changes.
3. `semantic_cli.py`: distinguish the original append factor (old source namespace) from the new same419 factor (current namespace). Keep seed/layout checks and strict source path resolution.
4. `semantic_rear_policy_timing_migration.py`: preserve the old append validator; extend only current-namespace lineage validation so initial receipt target == new receipt source contract, and new receipt target == current runtime. Reject malformed/missing chains, wrong origin or old-runtime fallback.

These hooks plus the new module are the migration-specific scope. Actual task-state/retention files and config marker paths will be bound after the controller delta is finalized. No whole-function AST equivalence matrix or arbitrary changed-path waiver is needed.

## Minimal binding fields

Plan/receipt:

- Actual source checkpoint absolute path + SHA256 + manifest SHA256; source contract/runtime hash and source HEAD; actual sealed global decisions/PPO updates/Adam steps.
- Explicit target committed HEAD and contract/runtime hash; exact changed-runtime paths with before/after hashes. Reject unrelated changed files. Target must be frozen/clean, not a guessed future identity.
- Source/target policy contract exactly unchanged: `rr_rl_timing_policy_learning_history_v1`, `role410_rear_policy_timing_v1`, 419 observations, 12 actions, N=1. Same numeric input has exactly the same Gaussian/kernel; physical-state encoding and therefore closed-loop Gaussian may differ.
- `parameter_mapping=identity_all_actor_critic_parameters_and_buffers`; exact Adam moments/steps/options, actual LR, Identity, RNG; preserved original branch and all inherited receipts/AUX digests. Preserve the entire original `rear_policy_timing_branch` object and counts.
- `same_mdp_claimed=false`, `physical_dynamics_changed=false`, `reward_potential_semantics_changed=true`, explicit changed public task-state meanings and retention rule. Existing task potential index17 changes; any of indices410–413 whose meaning changes must be named. No added observation dimensions or hidden timers.
- `revision_counter_origin` = actual source three counters, for revision-local reporting only, without resetting the existing branch or global counters.
- `discard_old_rollout_storage=true`; all added learning/AUX counters zero; reset physical simulation legally for the next run, no simulator-state equivalence claim.

Preserve action order/masks/caps/HISTORY/physics/return hyperparameters; rear task assists remain OFF and existing explicit FL assist unchanged. Any additional handoff measurement change must be finalized and recorded before the plan is built, not admitted as an unspecified wildcard. If the numerical sigma helper or network policy contract actually changes, stop and extend the declared contract explicitly rather than calling it same-kernel.

## Publication and later continuation

Use an outputs-only publisher following `publish_initial.py`, but take explicit `--source`, `--source-sha`, `--source-manifest-sha` and actual target HEAD. Construct on the source recorded device with unchanged CUDA visibility, official save and independent fresh load. Unique target filename, e.g. `checkpoint_rear_recapture_same419_step_<actual>_g<actual-head>.pt`; never overwrite the sealed ordinary checkpoint, old migration receipt or latest pointer during zero-update publication.

Publish/reload before any checkpoint-policy prefix. Existing `semantic_checkpoint_prefix_policy.py` already requires source/effective/runtime identities to match an actually saved checkpoint for this actor: keep that guard. Once published, natural P01 and checkpoint-policy suffix use the existing course API and namespace normally; subsequent actual PPO saves advance the usual pointer/counts and carry the new receipt. Original historical curriculum/terminal outcomes remain untouched.

## Bounded tests at the later legal boundary

Reuse fixtures in `test_semantic_rear_policy_timing_migration.py` and `test_semantic_rear_policy_training_audit.py`:

- Synthetic complete419 Adam/RNG fixture: source→migration→official save→fresh load exact tensors/full Adam/options/LR/Identity/RNG/counts/old receipts; new learning0 and empty rollout.
- Reject wrong source or manifest hash, changed runner/action/physics/reward coefficients, wrong target marker, unrelated runtime delta and malformed lineage; existing initial410→419 route remains strict.
- One ordinary128 synthetic PPO update checks receipt/whole-branch carry and fresh raw419/12 likelihood path (synthetic, not real training credit).
- CLI same-namespace migration and migrated checkpoint natural/suffix resolution; no prefix contract relaxation. The control agent separately tests current-bearing recapture versus valid ongoing RL AIR and physical acceptance.

No need to rerun historical rollout scans or establish a new success gate to authorize optimizer startup. The next actual source identity and effective LR must come from its sealed metadata, not this plan.
