# Tracking-reference first live update: fixed global 101377–101504

Scope: runtime `42b91e857a0a2da256dee85e8092642ca8e64aeb`, run `20260907T0134080107746Z_g42b91e857a0a_a862752b4ec84772b24f061befcd1676`, N1/seed1001/P01/full_episode/NewMdpWarmStart. This is the first completed 128-decision update only. The requested 4096 is a plan, not a completed count. No subsequent samples or checkpoint were included.

Method: read small JSON receipts and fixed first 128 audit records with PowerShell, plus relevant production source. No Python, Torch/tensor loading, GPU, Isaac, checkpoint rehash, production/test/config changes or optimizer execution by this audit. Saved round-trip and tensor-state assertions below are recorded runtime evidence, not an independent `torch.load`. One initially overlarge tool serialization was truncated; the parent authorized one compact reread of exactly the same window. This was a tool-output handling failure, not a training failure.

## Actual migration and immutable save

Source: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000101376.pt` and its sidecar.

New initial sidecar: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000101376_sb4f23b0ff604_g42b91e857a0a_37cad9a1279dacd8091fde0ed41adbae50c15f7613468d5f109be_manifest.json`.

First updated sidecar: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000101504_manifest.json`; its `source_run` equals the run named above and `save_load_round_trip=true`.

| Recorded state | Source | New initial | First updated |
|---|---:|---:|---:|
| Global policy decisions | 101376 | 101376 | 101504 |
| Lifetime PPO updates | 757 | 757 | 758 |
| Lifetime optimizer steps | 15140 | 15140 | 15160 |
| Full-episode budget spent | 43520 | 43520 | 43648 |
| Suffix budget spent | 47744 | 47744 | 47744 |
| Smoke budget spent | 0 | 0 | 0 |
| Original v3 global origin | 10112 | 10112 | 10112 |
| Effective Adam LR | 1e-5 | 3e-5 | 1e-5 |

The budget identity remains `10112 + 43648 + 47744 = 101504`; migration added no policy/PPO/optimizer credit and did not erase previous v3 expenditure.

Source and initial sidecars contain exactly equal actor, critic and normalizer fingerprints:

- Actor: `529984f97c26de203c796617545b36c18e690b67280a2079c8d1e2bf4e079dd6`.
- Critic: `ce98f9f3680533d092f8fb026c9edc80735fbd72f58d9a494ee8aee9a6dde3d1`.
- Identity normalizer: `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`.

Complete serialized source/initial `training_rng_state` and `policy_contract` compare equal in PowerShell. The retained contract is `heteroscedastic_log_v1`, official `HeteroscedasticGaussianDistribution`, log std, 324 observations/12 raw actions, ELU/256/256. The actor fingerprint covers all named parameters (`semantic_training.py:67`), including the learned state-dependent std head; this is not a claim that std output is constant across observations.

Optimizer fingerprint changes from source `9663d3efd0222ce014223516c3890b07d28a8b1490847183b4ceb8f4c2e05797` to initial `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`. The explicit migration receipt specifies fresh Adam moments and LR3e-5, preserved verified RNG, no inherited rollout or physical state. `_load_v3_warm_start` (`semantic_training.py:512–560`) checks fresh `(128,1,12)` storage, 324 observations, storage step0/no pending action, verifies loaded source actor/critic/optimizer/normalizer against metadata, then creates a new Adam and restores RNG. Thus old Adam was temporarily validated during loading but not used for any new optimizer step. Initial saved round-trip is true.

This is the explicit new-MDP tracking-reference revision, not a policy-distribution reset. `new_mdp_warm_start.json` lists changes only in the execution profile and tracking-reference/backend/adapter/native-audit/video implementation files. Its same-state comparison uses the actual reset P01 observation at episode tick0/time0: optimizer updates0, environment/bridge history unmodified, all logical bounded/scaled/projected/applied differences zero. The comparison explicitly excludes native dispatch and subsequent trajectories, so these zeros do not demonstrate that the new feedback route has no effect.

## First real PPO update and raw-storage evidence

The first `optimizer_updates.jsonl` row and checkpoint `last_update` agree on global101504/update758 and **20 actual optimizer steps**, with:

- actor changed to `4acdeb195d43bae2772578222f6eeaa1444a8a3ccd71e2a3fed1a3f44d07701d`;
- finite nonzero gradients observed; recorded norm min/max `1.0026161107393996 / 1.4142128300558532`;
- KL mean `0.01916691716760548`, clip fraction `0.2671875`;
- value loss `0.002020745718618855`, surrogate loss `-0.03331987801939249`, entropy `-4.903095126152039`;
- end-update adaptive LR `1e-5`, distinct from the initial fresh-Adam LR3e-5.

The first128 records are consecutive globals101377–101504. All have finite raw12, old mean12, strictly positive finite std12, old log probability, old value and reward. All raw requests match the corresponding environment audit raw request exactly. Observed std values span `0.11079240590333939–0.221418097615242` across this fixed sample/channel set.

Production `semantic_training.py:710–746` clones the actual sampled raw latent, pre-step mean/std/log probability/value, writes them to this audit, and after `process_env_step` checks exact equality of stored raw action and stored mean/std. A successful completed update is source-backed evidence that these fail-closed checks passed for the 128 transitions. The saved binary rollout was **not** independently opened here.

## Fresh P01 topology and native evidence

Initial and updated metadata identify `P01_full_task_only_initial_version`, N1, no peer reset, 324/12, `phase_suffix_curriculum_implemented=false`, `prefix_request=null`, no bound prefix-policy provenance. The command's default `prefix_source=frozen_fsm` is not an active teacher in a P01 run. None of these 128 applied records contains a curriculum-start or prefix-provenance field.

- Policy source phases: P01 **1**, P02 **127**.
- First decision ends episode tick8/time0.0666667s; last ends tick1024/time8.5333333s. All decisions contain8 physics ticks, totaling **1024**.
- One ordinary P01→P02 transition, terminal=false; **0 terminal, 0 task success, 0 time-out** records. No completed task is claimed.
- Native tick summary: **1024/1024 verified**, **1024 actual-target-effect ticks**, **1023 own-phase-request-effect ticks**, **1 handoff-hold tick**. The latter distinction preserves the incoming carry/own-request boundary.
- Every record reports zero in-episode root-pose, root-velocity, force/impulse and gravity writes, with `no_in_episode_state_writes_verified=true`. These counters do not prohibit the necessary actuator target writes.

## Tracking-reference verification and temporal resolution limit

All128 decision-end native audits carry mode `previous_ack_requested_servo_reference_v1`, `tracking_reference_previous_ack_independently_verified=true`, and verified actual native targets. All endpoint references identify `previous_semantic_ack_requested_residual`; every endpoint previous-request vector has nonzero channels. For all128 endpoints, dispatch tick equals native-audit tick, previous ACK is exactly tick−1, prior sample+1 equals current mapper feedback clock, and that clock equals previous ACK write count.

First decision: compact command ticks180–187 correspond to episode ticks1–8. Its detailed endpoint context is dispatch187 / previousACK186 / previous feedback sample186 / mapper feedback187 / previous write count187, with explicit bootstrap boundary180. Last decision uses command1196–1203. Thus physical command clocks include reset/settle offsets and must not be directly equated to episode ticks. The first detailed endpoint is already seven dispatches after the first new-mode tick; it is not itself the bootstrap receipt.

The source proves the independent audit boundary: `isaac_fsm_backend.py:1937–1950` captures prior ACK/real q/mapper state **before** the actual dispatch, separately binds tracking names from the source frame, and passes that context to the auditor. `actuator_target_effect.py:76–108` recomputes and exact-compares the reference evidence, verifies independent source names, and checks exactly one mapper advance. This is stronger than accepting only the current ACK's self-reported reference.

**All128 detailed endpoints have feedback-clock modulo4 = 3, feedback_sample_tick=false, and reference_used=false for all8 channels. This is an endpoint observation only.** Feedback is eligible on modulo4=0; eligibility also depends on scheduling, unchanged/reached nominal, previous request and gain. The compact per-physics-tick list contains only episode/command ticks, source phase, verification, changed count, actual/own-request effect and handoff-hold. It does not preserve per-channel tracking-reference detail. Consequently this fixed log cannot supply the count of reference-used channels/ticks inside the8-tick decisions, nor prove sustained compensation efficacy. Native-effect counts refer to current physical target differences, not long-horizon cancellation or Cartesian task improvement.

Conclusion: actual first update, source-bound migration retention, fresh optimizer/storage/P01 topology and verified native execution are supported for this fixed boundary. It remains an early P01/P02 learning window, not a task-success result or proof of the reference's later-phase benefit. The separately reported pre-training smoke is not added to these training counts.
