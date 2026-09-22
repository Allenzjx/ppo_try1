# P05 pre-edge recovery: unapplied same389 migration

Status: candidate only. No production patch, real checkpoint write, optimizer update, or simulation was performed by this preparation. The CPU helper completed with exit code 0; 25 tests passed.

The opt-in is exactly `nominal.p05_preedge_approach_recovery: p05_preedge_approach_recovery_v1`. It is a control-MDP change, not a reward-only change and not the same MDP. The actor input codec, 389 dimensions, policy kernel, caps, sigma, reward, physics and task acceptance rules remain unchanged. No new controller timer/latch is claimed. Identical numeric inputs preserve the policy mapping; identical physical trajectories/actions are not claimed.

## APIs

Proposed module: `wlr50_clean.ppo.semantic_p05_preedge_migration`.

- `SCHEMA = "wlr50_clean.p05_preedge_approach_recovery_same389.v1"`
- `build_p05_preedge_migration(checkpoint, current_contract, *, reason, reviewed_code_sha256, project_root=None)`
- `validate_p05_preedge_migration(checkpoint, current_contract, plan_path, *, project_root=None)`
- `publish_p05_preedge_checkpoint(checkpoint, current_contract, plan_path, output_checkpoint)`

The builder binds the actual source checkpoint/sidecar and new committed runtime HEAD. Its only permitted runtime differences are the supervisor, task spec, new dedicated migration module, semantic migration route and semantic training route. The task-spec object may only gain that exact nested key; the other five configuration files must retain their exact bytes.

## Identity transfer and lineage

The existing source-device same389 publisher performs official save and independent fresh load. Actor/critic parameters and buffers, full Adam state/groups/actual LR, Identity normalizers, full training RNG and runner configuration are preserved. Rollout storage is empty and physical state is not inherited. Migration adds zero policy decisions, PPO updates, Adam steps or AUX updates; no latest pointer is promoted.

All four existing branch origins, counts and migration receipts are preserved, together with the entire four-event-or-later AUX ledger (current front 96/96, RR 7/8, mixed 103/104) and the separate historical 7/8 ledger. Opaque future AUX entries are not truncated. Existing `resume_migration` evidence is retained inside the new factor before the generic loader records its new resume receipt.

The new `p05_preedge_approach_recovery_branch` has an actual source-counter origin; its `_migration` receipt and `_branch_counts` are carried/recomputed by normal PPO saves. The intended real source is CP216448 / 1656 PPO / 33120 Adam, SHA256 `8ec7784a9078ae7e9bb5f1d7298a4aa655c0d979ddb906e8a4d577bf8079a8f4`; the validator binds actual source metadata rather than trusting a filename or hardcoding counters/LR.

## Tests and artifacts

The in-memory patched-module runner exercised 25 CPU tests, including strict negative cases; a temporary synthetic Git source/target and checkpoint; official source-device publication and independent reload; preservation of nonempty Adam, RNG, normalizers and all lineage; and a subsequent normal save carrying the new origin/counts. The synthetic source LR was 2.25e-5 to verify preservation rather than accidental hardcoding to the current real 1e-5. No real checkpoint fitting/publication occurred.

- `migration.patch`: SHA256 `2c1976584b4fcf678f1f826eb38b4213bb957680fc595a15fedc853b49636c4a`
- `tests.patch`: SHA256 `815efcac591d1478c8a7776509adb676297b7dcda888f484329f4e80314971a5`
- Candidate new module: SHA256 `f3afa7117c008b337c6c9f8f52e9e44cb31d6db8a6caee7f97908ad54bba6c60`

Both patches are apply_patch-compatible and remain unapplied. The supervisor/task-spec patch is independently owned by the primary-methods review agent. Adoption/publication requires the root's legal execution boundary and new frozen runtime.
