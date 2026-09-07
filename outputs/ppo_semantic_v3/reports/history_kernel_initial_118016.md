# #37 history-kernel initial checkpoint — fixed initial boundary

Scope: PowerShell/JSON and source-only review of the published initial checkpoint, not the later training stream. No PT/Torch/Python, GPU, simulation, optimizer, or file hashing was invoked by this review. A `save_load_round_trip: true` below is the publisher's actual load/save receipt, not an independent tensor load by this reviewer. The first attempted broad JSON serialization was truncated; conclusions use subsequent explicit scalar/subfield reads, not that truncated output.

## Published evidence

- Runtime: `e8462f06690873c8ac4c4fec453f451f5d10d0e8`.
- Run: `runs/ppo_semantic_v3/train/20260907T0736098630371Z_ge8462f066908_a07ce6b30c64439e8e8735a0926c4cc6`.
- Source: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000118016.pt` and its manifest.
- Published initial: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000118016_s57f8c5e70820_ge8462f066908_a12a177d1de13baf57915cb4c6cb48c78610716f4e8840339dbd8dc2a15302ab_p1b5981b6d057.pt` and corresponding `_manifest.json`.
- Initial stage is `initial_v3_warm_start`; `save_load_round_trip` is **true**. The initial filename binds source global/source checkpoint SHA, target Git/runtime inventory, and target policy-contract digest.
- Run JSONs `new_mdp_warm_start.json`, `new_mdp_initial_policy_kernel_comparison.json`, and `new_mdp_initial_action_comparison.json` are published; the initial manifest records their comparison references.

## Preserved learned state and accounting

| Item | Source → published initial |
|---|---|
| Global policy decisions / PPO updates / optimizer steps | 118016 / 887 / 17740 → identical |
| Original v3 origin | 10112 → 10112 |
| Spent full episode / phase suffix / smoke | 53504 / 54400 / 0 → identical |
| Actor parameter SHA, including learned log-std head | `49515718e9c0b6d9bccde781973e7395a5bcc8b7be55fed16f108f236945baad` → identical |
| Critic parameter SHA | `2ea462337c629a745f7a3dd21df12e0ef6c647a155deeb4dfa022e266dfc8423` → identical |
| Normalizer state SHA | `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` → identical |
| Complete saved training RNG JSON | Exactly equal in the two manifests |

The accounting identity remains `10112 + 53504 + 54400 = 118016`. This initial boundary adds **0 policy decisions, 0 PPO updates, 0 optimizer steps**. The requested 1024 decisions remain a plan; this review makes no first-update or completed-block count.

The publisher records fixed schema preprocessing with identity RSL normalizers. The actor hash covers all actor state, not just the mean head, so the receipt supports preservation of the learned state-dependent std parameters. It does not imply equal old/new conditional action means.

Adam intentionally changes: source optimizer SHA `20df54c71d1f1f4e7fa38601559da1970bb0ad37df7dd071c064134f52166af4`; initial SHA `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`. The actual initial `optimizer_learning_rate` is **0.00003**. The migration specifies reset-all-moments, `old_rollout_buffer_inherited=false`, and no inherited physical state. `semantic_training.py:575` checks fresh `(128,1,12)` storage, 324 observations, `storage.step == 0`, and no pending action before source loading; lines 594–612 verify source actor/critic/Adam/normalizer state, require identity normalizers, instantiate fresh Adam at 3e-5, and restore the saved RNG. Thus empty Adam/rollout are supported by the executed successful publication path plus its source checks, not by independently reading their tensors here. `physical_env_state_saved=false` in the initial manifest.

## What the policy change actually does

The record explicitly identifies `heteroscedastic_log_v1` → `history_conditioned_heteroscedastic_log_v1`, with `conditional_policy_changed=true`, `fixed_mean_control_changed=true`, `parameter_layout_changed=false`, `physical_mdp_changed=false`, and `reward_changed=false`. Although the CLI uses the existing explicit warm-start entry point, this record does not claim a physical/reward MDP change.

The 324/12 protocol is unchanged. The actor reads stored observation history `[195:207]`, already bounded by the existing observation clip ±20, and uses `mu_new = 0.1*mu_base + 0.9*previous_raw`. Its learned heteroscedastic log-std remains the innovation sigma, without stationary rescaling or a private actor history cache.

The actual same-observation comparison is at P06, physical tick **4864**, task time **40.53333333333333 s**; observation SHA is `91be17ca4acddba973bc3a4a0de2c379e0be3b574b5b6291000c86e47ec993e5`. Previous raw history is all zero, so the new conditional mean is one tenth of the old mean. Representative wheel raw means are:

| Canonical wheel channel | Old conditional mean | New conditional mean | Shared sigma |
|---|---:|---:|---:|
| 8 | −0.2568612993 | −0.0256861299 | 0.2803962231 |
| 9 | −0.1890034229 | −0.0189003423 | 0.1728143245 |
| 10 | −0.09473872185 | −0.009473872371 | 0.1363206357 |
| 11 | +0.07775072008 | +0.007775072008 | 0.1434370875 |

The shared critic value is **−7.23592472076416**; recorded old-to-new conditional KL is **4.623683929443359** at this one observation. The comparison records `learned_weights_changed=false` and `rng_or_sampling_cache_changed=false`. These are conditional distribution calculations, not a new physical rollout, native command measurement, stability result, or task success.

## Physical configuration and current curriculum

All six source/target configuration SHA pairs are equal: stage task spec, execution profile, reward config, observation schema, action schema, and quality score. Runtime-inventory differences are exactly the eight scoped policy/integration files: `semantic_history_actor.py`, `semantic_policy_distribution.py`, `semantic_training.py`, `semantic_migration.py`, `semantic_cli.py`, `semantic_checkpoint_prefix.py`, `semantic_checkpoint_prefix_policy.py`, and `scripts/run_semantic_ppo.ps1`. The recorded frozen-A inventory is equal; no other inventory path changes. This is a comparison of the bound inventory records, not a re-hash of live files.

The return profile is unchanged: gamma .9985, lambda .99, rollout 128; 120 Hz physics, 15 Hz decisions, 200 s task horizon, and no task-terminal bootstrap. The action-comparison JSON uses the **target actor on both physical profiles**, correctly labels that limitation, and reports exact zero differences in bounded/scaled/masked/rate/safe residuals and applied targets. It must not be mistaken for old-policy/new-policy equivalence.

The initial sampling and topology are coherently updated to `natural_P01_A_teacher_prefix_then_semantic_suffix_N1.v2:P06:offset_160`: N1, seed 1001, frozen FSM reset-only teacher, task horizon includes prefix, ordinary phase changes do not end the episode, 324/12, 128 decisions per rollout, no peer reset, and `phase_suffix_curriculum_implemented=true`. There is no checkpoint-policy-prefix provenance. The initial comparison records zero optimizer updates and no environment/bridge-history modification. No teacher-prefix work is added to the retained policy/PPO/optimizer ledger by this initial publication.

Conclusion: actual initial publication and its recorded preservation/reset semantics are present and internally consistent. This review stops at that initial boundary; it does not certify later training, the physical efficacy of the history kernel, or a full/suffix task success.
