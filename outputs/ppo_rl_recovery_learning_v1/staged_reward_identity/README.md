# Isolated same439 reward identity candidate — not applied

Base HEAD: `f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b`.
Do not apply, import model helpers, run CPU Torch tests, or publish while the current Isaac/video process is active. Wait for root's confirmation of natural termination and legal boundary. This directory changes no production bytes.

`migration_only.apply_patch` is an explicit apply_patch-tool patch for one new module, four production routing/save changes and one test file. `migration_only.patch` is the equivalent standard unified diff; `git apply --check` and AST parsing passed without applying it. `tree/` contains mechanically generated candidate bytes only. None of these checks is a model test or physics result.

The reward agent owns `reward/`; its `semantic_reward.py` must join these five runtime paths before publication. The builder intentionally requires **exactly all six runtime changes**, so the migration-only candidate cannot authorize a boundary by itself. No YAML, observation schema, distribution, mapper, owner, actuator, wrapper or physical asset change is allowed. The reward change is revision `rr_recapture_retention_reward_only_v1`: reward-local Phi only; shared task progress/439 observations are unchanged.

## API and invariants

New production module proposal: `wlr50_clean.ppo.semantic_rr_retention_migration`.

```python
record = build_rr_retention_migration(
    checkpoint, current_contract, reason=reason,
    expected_source_sha256=actual_checkpoint_sha,
    expected_manifest_sha256=actual_sidecar_sha,
)
receipt = publish_rr_retention_checkpoint(
    checkpoint, current_contract, plan_path, unique_output_checkpoint,
)
```

The current known source is the **learned** immutable CP226048, not a hardcoded module constant:

- `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000226048.pt`
- SHA256 `fbd27dea193153c7150881b1258dbf5ce249cfd5d1529e22b066be0abf0da973`
- Sidecar SHA256 `93f41386de6d46fc204f2a22cb2c712ea83eb9b1d6b29c32a7276e2d9da6df53`
- Actual counts: 226048 decisions / 1731 PPO / 34620 Adam steps; LR `1e-5`.

Re-resolve the actual latest complete source at application time. The builder accepts explicit hashes/immutable checkpoint path and reads its counters; a subsequent valid f6d439 source is not rejected merely for having newer counts. `last_update` must agree with source global decisions, PPO count, actor hash, LR and complete Adam-minibatch count. No initial zero-credit owner publication and no 422 ancestor qualifies.

Migration preserves all learned439 weights including nonzero new17 columns, critic, Adam moments/steps/options, effective LR, Identity normalizers, RNG, branch/counters and used stage quantities. It does not zero-append, create a new actor profile, or warm-start a new optimizer. The old owner receipt is unchanged. New outer lineage reconstructs f6d and validates that historical receipt; it also binds the actual selected source checkpoint and sidecar. Ordinary descendants may update weights/Adam/LR/RNG/counters; only immutable receipt/route/kernel ancestry remains frozen.

Publication is a new immutable same-branch checkpoint, zero PPO/AUX credit, independently reloaded, with no main-pointer promotion. It declares fresh legal P01 reset and empty128x1x439 rollout. It does not claim bitwise simulator continuation. Normal HISTORY semantics and in-episode continuity remain unchanged; fresh prefixes are recollected and uncredited.

## Mechanical assembly, still no production application

From the project root, using base Python only:

```powershell
& 'C:\Program Files\Python313\python.exe' outputs/ppo_rl_recovery_learning_v1/staged_reward_identity/assemble_patch.py
# When the reward agent's complete candidate file is available:
& 'C:\Program Files\Python313\python.exe' outputs/ppo_rl_recovery_learning_v1/staged_reward_identity/assemble_patch.py --reward-candidate outputs/ppo_rl_recovery_learning_v1/staged_reward_identity/reward/v2/semantic_reward.py
```

The second command creates `complete_reward_identity.apply_patch`, standard `.patch`, candidate tree and exact before/after hash manifest. It parses Python AST and checks standard patch applicability only; it never imports the production/model modules or applies the patch. Reward tests remain separately owned in `reward/` and should be applied alongside these migration tests.

## Tests after root confirms Isaac exit and applies the reviewed patch

From project root, use the existing environment, CPU only. Do not run this command before the natural video termination.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:WLR_SAME439_SOURCE_CHECKPOINT = (Join-Path (Get-Location) 'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000226048.pt')
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m pytest -q tests/unit/test_semantic_rr_retention_migration.py
```

Coverage candidates: exact dynamic-source binding and contracts; wrong source/sidecar/unreviewed files/physics/budgets rejected; original owner receipt unchanged; mutable descendants accepted and stale origin/route/counters rejected; populated learned439/Adam state survives publication/reload; identical numerical input mean/value/raw log likelihood; partial storage/pending transition/wrong seed/mixed factor rejected; one synthetic128-row PPO update carries both receipts and real raw439x12 storage. Synthetic tests carry no robot/PPO-training credit.

Then run the reward agent's targeted positives/negatives, existing relevant owner/edge/front interfaces, and the existing raw-likelihood numerical test (avoid interpreting old source-pinned migration-plan tests as authority for a new boundary):

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m pytest -q tests/unit/test_semantic_rear_owner_policy.py -k 'actual_raw_request_audit_and_frozen_prefix or one_synthetic439_update_raw_likelihood_and_five_exposures'
```

Unset/restore test-only `CUDA_VISIBLE_DEVICES` before the real CUDA publication/Isaac run; do not leak test environment into formal evaluation.

## Publication command template after tests, committed new HEAD, no Isaac

`publish_rr_retention.py` remains an outputs-only helper. All source/output/hash values are explicit and there is no model reset or permanent runner. First omit `--publish` to validate; add it once to write unique plan/checkpoint/receipt. Replace placeholders, and use newer complete source values if any exist:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs/ppo_rl_recovery_learning_v1/staged_reward_identity/publish_rr_retention.py --checkpoint outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000226048.pt --source-sha256 fbd27dea193153c7150881b1258dbf5ce249cfd5d1529e22b066be0abf0da973 --manifest-sha256 93f41386de6d46fc204f2a22cb2c712ea83eb9b1d6b29c32a7276e2d9da6df53 --expected-head <NEW_FULL_HEAD> --output-checkpoint outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_rr_retention_CP226048_g<NEW_HEAD12>.pt --plan outputs/ppo_rl_recovery_learning_v1/rr_retention_CP226048_g<NEW_HEAD12>_migration.json --receipt outputs/ppo_rl_recovery_learning_v1/rr_retention_CP226048_g<NEW_HEAD12>_publication.json --reason 'Reward-only RR retention correction; unchanged observed task potential and full learned439 training state'
```

Do not paste the angle-bracket placeholders as PowerShell syntax. After explicit publication use ordinary same-runtime resume from the new immutable checkpoint; or transport the verified plan through existing `-ResumeMigration` at the initial boundary. Default `NewMdpWarmStart` is not involved. New rollout data must be recollected under the fixed version before PPO updates. Save/reload/natural-P01 assessment and source-version video labels remain the root workflow.
