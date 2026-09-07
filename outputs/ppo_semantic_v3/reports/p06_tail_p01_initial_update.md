# P06 finite-tail revision — P01 initial migration and first saved update

Status: RUNNING; this report is fixed to the completed, published boundary **68352 / 499 PPO updates / 9980 optimizer steps**. It covers only globals **68225–68352** (128 decisions, one new PPO update, 20 optimizer steps). The requested 4096-decision block is not claimed complete.

Run: `runs/ppo_semantic_v3/train/20260906T1650134938238Z_g73e937039708_7681a31f01714528b75509300dfac00f`.
Runtime HEAD: `73e937039708a7306b2b2c941e1f9108b8fa8b3d`. N1, seed 1001, `full_episode`, natural P01, teacher offset 0, explicit `NewMdpWarmStart` from checkpoint 68224. Checks below used PowerShell JSON/source reads only: no tensor load, Python, additional Isaac, production edits or repeated large-checkpoint hashes.

## 1. Source → independent initial checkpoint

Source: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000068224.pt`, with its immutable manifest.

New immutable initial: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000068224_s0bc86a834f7b_g73e937039708_de8dfb8e98bdd00d437abcb722ebe99d38079b4ed637040223affa8ab41871f2.pt`.
Its manifest records checkpoint SHA `7047bca41eabae1fe55c86cfdc1931f3b26bb4912abc39f6cb69e60f03d7feae`, `stage=initial_v3_warm_start`, and `save_load_round_trip=true`. This report compares saved metadata and the runtime's round-trip evidence; it does not claim a second independent tensor/hash verification.

| Verified source/initial field | Result |
| --- | --- |
| Full actor parameter hash, including learned distribution parameters | Equal: `d4b129487079242acdb524ae4b87e6485edc3e54243a04e47aa92a1b8e4c1c42` |
| Critic parameter hash | Equal: `b2e7cc3b0b54d43bab3ea8c1031219c53d90e13e02eb768c2ee3b186a3ac8f84` |
| Normalizer state hash | Equal: `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` |
| Complete saved training RNG JSON | Exactly equal |
| Complete policy contract and runner configuration JSON | Exactly equal |
| Global decisions / lifetime PPO / lifetime optimizer steps | Unchanged: 68224 / 498 / 9960 |
| Existing full / suffix / smoke spent ledger | Unchanged: 25088 / 33024 / 0 |
| Original v3 budget origin | Unchanged: 10112 |

The policy remains `heteroscedastic_log_v1`, `HeteroscedasticGaussianDistribution`, state-dependent log standard deviation, 324 observations and 12 unbounded Gaussian latent actions before the existing tanh projection. There is no scalar-Gaussian conversion or sigma reset in this boundary.

Adam is intentionally **not** preserved: the explicit new-MDP record says `reset_all_moments`, initial LR `3e-5`. Source optimizer hash `880a27017fe02cb36cff98c1b950c990e21ff7ff78bd9b3f9d3c784956dda378` becomes fresh-Adam hash `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`; source recorded LR was `1e-5`. This is not an exact-MDP optimizer resume.

`old_rollout_buffer_inherited=false`, `physical_state_inherited=false`, `physical_env_state_saved=false`, and `resume_physics=legal_reset_not_bitwise_continuation`. The initial checkpoint adds zero training credit. A fresh official 128 × 1 rollout supplies the first update; the first audit global is 68225 at the new episode's physics tick 8.

**Initial metadata caveat:** `implemented_reset_sampling` in this initial sidecar still contains the source P06-prefix string. Its `sampling` and `execution_topology.reset_sampling` both already say `P01_full_task_only_initial_version`; `phase_suffix_curriculum_implemented=false`. The launch arguments, physical first tick and first trained sidecar independently agree on natural P01. At checkpoint 68352, both sampling fields correctly say P01. The inherited initial description is not used here as evidence of a teacher prefix or of an execution reset.

## 2. Explicit runtime/configuration boundary

Source HEAD `68cd9f5fca6c9ba06c38666fb55024b9ba04f9b0` → target HEAD above. Recorded runtime-content SHA changes from `b7d80703ee87e23902156fa0b2a5a9016ba1b66d21fc9d04444508497079f494` to `de8dfb8e98bdd00d437abcb722ebe99d38079b4ed637040223affa8ab41871f2`.

The migration's exact changed runtime-file list is:

- `configs/ppo_semantic_v3/stage_task_spec.yaml` — explicit P06 measured-workspace finite-source-tail revision.
- `src/wlr50_clean/ppo/semantic_supervisor.py` — owner-local tail implementation.
- `src/wlr50_clean/ppo/semantic_prefix.py` — current nominal diagnostic getter.

The source/target `frozen_A_files` maps contain 29 entries and compare exactly equal. The commit diff contains those three runtime files plus their tests, with no frozen-file change. Source/target action schema, execution profile (including action ranges), observation schema, reward config and fixed quality-score hashes are equal; only stage-task-spec differs among the six selected configurations. Local-runtime version records are equal: Python 3.11.15, Torch 2.7.0+cu128, RSL-RL 5.0.1, TensorDict 0.12.2, Isaac Lab 0.54.3, Isaac Sim 5.1.0.0; timing remains 120 Hz physics / 15 Hz decisions / 200 s task horizon.

The initial same-state diagnostic is P01, tick 0, 324 observations. All six recorded 12-channel old/new bounded/scaled/masked/rate/safe/applied differences are exactly zero. `optimizer_updates=0` and `actual_environment_or_bridge_history_modified=false`. Its declared scope is logical projected targets, **not** native dispatch or subsequent trajectory; it cannot show that the later P06 tail is inactive or equivalent.

## 3. First completed optimization and published checkpoint

The first `optimizer_updates.jsonl` record is PPO update 499 at global 68352. Actor-before equals the source and initial actor hash above; actor-after is `e06ab411a15659bdbd5bd166179187bcf432bfdbacd256ccaf1f623ad07422c4`, matching the 68352 sidecar. `actor_parameters_changed=true`, `finite_nonzero_gradient_observed=true`, 20 optimizer steps.

Recorded gradient norm min/max: 1.0010694365 / 1.3618201221; mean KL 0.0205086392; clip fraction 0.2640625; entropy −5.3396528959; value loss 0.00099443193; surrogate loss −0.0409774460. The recorded **update-end** LR is `1e-5`; this does not assert that every minibatch used that LR or negate the fresh initial `3e-5`.

Published checkpoint: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000068352.pt`. Pointer/sidecar record checkpoint SHA `b2a9960b65d66f2b6899dd580705079bc29f994b51aac569e23d34bef9503c88` and manifest SHA `f44a8d27f7afd8a624c31db99f06495b62c54ff66654edd0fd8b011a533773f6`. Sidecar: `save_load_round_trip=true`, correct current run/source, full-episode stage, 68352 / 499 / 9980, full spent 25216, suffix 33024, smoke 0. Normalizer hash remains unchanged. These are recorded publication identities, not newly recomputed file hashes.

## 4. Fixed 128-decision physical and policy audit

The first 128 JSONL rows were read once, checked for exact contiguous globals 68225–68352, and not extended to later collected rows.

| Item | Actual fixed-window result |
| --- | --- |
| Credited phase samples | P01 = 1; P02 = 127; P03–P13 = 0 |
| Episode physics | 1024 ticks, first decision ends at tick 8, last at tick 1024 / 8.533333333 s |
| Completed episodes / terminal decisions | 0 / 0; final current phase P02, no terminal reason |
| Native audit verified ticks | 1024 / 1024; all per-decision all-ticks flags true |
| Actual native residual-effect ticks | 1024 |
| Own-phase-request effect ticks | 1023; P02 first decision has 7 of 8 while native effect still has 8 |
| Last-tick native reconstruction checks | All 128: verified, setters equal dispatch, mapping equals dispatch, same-tick counterfactual |
| In-episode root pose / velocity / force-or-impulse / gravity writes | All four counters zero in all 128 rows |
| No in-episode state-write verification | True in all 128 rows |

The one own-request count exclusion occurs at global 68226, immediately after the real P01→P02 transition at tick 8. It is not an unverified native tick. Native buffers and same-tick counterfactual are separate audit evidence; `actual_drive_target_full12` alone is a canonical double command, not a native float32 readback.

All 128 rows contain 12 raw actions, old means and old standard deviations. All saved standard deviations are positive and finite, with window-wide range 0.1188909784–0.2044543177. A PowerShell double-precision reconstruction of the diagonal **raw Gaussian** log probability differs from logged old log probability by at most `1.0409839e-6` (float32-origin evidence). It does not substitute projected actions into the PPO density or infer a causal benefit from the std values.

There is no prefix-evidence file; actual launch/topology is natural P01, episode tick accounting is exactly 128 × 8, and no pre-credit teacher span is present in this window. Reset/settle are excluded from PPO credit; neither a historical AIR snapshot nor teacher rear-leg progress is credited.

**Scope stop:** these samples reach only P02. They provide initial migration, real update and native-interface evidence, but no P06 source-endpoint/tail activation, retirement, rear-leg progress, P13 stop or whole-task success evidence. Later run progress is deliberately outside this report.
