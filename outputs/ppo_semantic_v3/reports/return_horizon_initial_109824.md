# Return-horizon new-MDP initial — checkpoint 109,824

Status: immutable initial publication is present and records a successful save/load round trip. This is a fixed initial-boundary audit, not a completed training-block audit or a task-success claim. No post-initial decision/update stream was read.

## Sources and scope

- Run: `runs/ppo_semantic_v3/train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8`.
- Source: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000109824.pt` and its `_manifest.json`.
- Initial: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000109824_s065bf5de9604_gf4bfe2560bfd_70cad3ecadd1db615bbad69728d32a15907c83f1c952f064f06da1a4c90622a4.pt` and its `_manifest.json`.
- Read the two sidecars, `run_manifest.started.json`, `new_mdp_warm_start.json`, and `new_mdp_initial_action_comparison.json` with PowerShell. No PT load, tensor read, Python, tests, Isaac, optimizer stream, or repeated checkpoint hashing. The small action-comparison JSON alone was hashed to verify its recorded binding.

The source revision is `f1a9bbf650b1b8f80d5169e09173fcfc68797d99`; the target is frozen `f4bfe2560bfd228541f7829d830fa441054eab1d`. The initial runtime contract equals the started-manifest contract, and its complete new-MDP record equals the run's migration JSON.

## Published preservation and reset evidence

| Field | Source → immutable initial |
| --- | --- |
| Global decisions / PPO updates / optimizer steps | `109824 / 823 / 16460` → exactly unchanged |
| Original new-MDP origin | `10112` → unchanged |
| Spent budgets: full / suffix / smoke | `49408 / 50304 / 0` → exactly unchanged |
| Actor parameter SHA, including learned state-dependent log-std parameters | `db277144127e0653082346a04ce45156586465db9b27187c5b7c3c8c30768b70` → identical |
| Critic parameter SHA | `db638d05f704451f49f510ca7e088a1cbfbff64efc3d2bde1cd4243cf94b5403` → identical |
| Identity-normalizer state SHA | `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` → identical |
| Optimizer state SHA | `8c54af06b3523b09cbdb41f81218cd3703d9fa59ee65246a954072e01321c427` → `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614` |
| Recorded actual optimizer learning rate | `1e-5` → `3e-5` |
| Save/load round trip | Both sidecars record `true` |

The complete serialized `training_rng_state` objects compare exactly equal, including Python, NumPy, Torch CPU and one recorded CUDA-device RNG state; seed remains `1001`. This audit compares the stored CUDA record without accessing CUDA. Policy contracts are exactly equal: 324 observations, 12 raw Gaussian actions, `[256,256]` ELU actor, `HeteroscedasticGaussianDistribution`, learned log std, and identity RSL normalization. There is no separate std fingerprint in these sidecars; its preservation is covered by the full actor fingerprint, not a new per-state sigma-output experiment.

The migration explicitly declares fresh Adam with `reset_all_moments`, initial LR `3e-5`, `old_rollout_buffer_inherited=false`, and `physical_state_inherited=false`; the return transition separately records `rollout_storage_inherited=false`. The initial records `physical_env_state_saved=false` and `resume_physics=legal_reset_not_bitwise_continuation`. These are the actual published save/verification receipts; this read-only audit did not independently open the optimizer or rollout tensors.

Recorded source checkpoint SHA is `065bf5de9604da5f6732d652dea1a3d9154750000d688e103c492b3b5b36067a`, equal in the source sidecar, migration and initial resume-source binding. The source sidecar binding is `133f2af375575e1363a1ebdebc555ee2d9acdec508a97346e053f87e0af4c64c`. The initial checkpoint's recorded SHA is `997a112c0aab038c3461f82b58674c005c460dece57d4cd19551eec8a748711b`; both named PT files exist. No independent rehash or reload of either PT was performed here.

## Versioned return change, not a physics/profile change

| Return field | Source | Initial target |
| --- | --- | --- |
| Profile | `legacy_gamma_0995_lambda_095_v1` (runner marker absent) | `v3_gamma_09985_lambda_099_v1` (explicit marker) |
| Gamma | `0.995` | `0.9985` |
| GAE lambda | `0.95` | `0.99` |
| Rollout length | `128` | `128` |

The migration flags both `reward_discount_changed=true` and `gae_estimator_changed=true`. Its reward binding changes from SHA `da574107eb0c0ba02ef02ff77cd00c7e9cfa4ef3208abae96435fafd1bc38bcb` to `7874da47f19f1a619bc8d9dbc8d2f01d78bf4a7628cd277881ab19d764db338e` at the same `configs/ppo_semantic_v3/reward_config.yaml` path. The full runner configurations compare equal after restoring only gamma/lambda and removing the new return marker; no other runner option changed in that comparison. This is deliberately `exact_mdp_resume=false`, not an ordinary resume with silently different discounting.

Source/target contract hashes are identical for stage task spec, execution profile, observation schema, action schema and quality score; the frozen-A inventories are identical. In particular, the execution-profile SHA remains `b22673e7ab57dbf719ae6fcebd2c380b98c391944bed2c32d423e1d1f6b416dd`. The six changed runtime paths are the reward YAML and the return-profile integration in `semantic_migration.py`, `semantic_policy_distribution.py`, `semantic_return_profile.py`, `semantic_reward.py`, and `semantic_training.py`.

## Same-state physical-profile action comparison

The initial sidecar binds the comparison by path and SHA, not by embedding the full comparison. Its SHA `fc967265ea6f547d626acae91584d381d9b1b2d4d2bb13278c49388865c8a18e` matches the small JSON's computed hash.

The comparison is at natural P01, physics tick `0`, simulation time `0`, 324 observations, one common deterministic actor mean, zero previous residual and `dt=1/120 s`. Source and target execution-profile hashes are identical. All 12 components in each of the six `new_minus_old` vectors are exactly zero: bounded, scaled, masked, rate-projected and safe-projected residual, and applied action. The record says `optimizer_updates=0` and `actual_environment_or_bridge_history_modified=false`.

Its explicit scope is logical output/projected target on the same state, **not native dispatch or the subsequent trajectory**. It therefore supports unchanged immediate physical-profile mapping in this comparison, not equal future training trajectories, full-task success, or improved stability.

## Initial accounting boundary

The started record says RUNNING, N1/seed1001, `full_episode`, natural P01, offset `0`, explicit new-MDP warm start and **4096 planned** decisions. The initial's current sampling and topology both identify natural P01; `curriculum_epoch.prefix_request=null`, and `phase_suffix_curriculum_implemented=false`. The generic CLI `prefix_source=frozen_fsm` selection is not an active teacher request here.

The initial stage is `initial_v3_warm_start`; its inherited `source_run` and `last_update.global_policy_decisions=109824` are source lineage, not a new update. This audit credits **zero** new decisions, PPO updates or optimizer steps to initialization, and makes no statement about later block completion. Master accounting and historical task failures were not modified.
