# Frozen saved-policy predecessor curriculum

## Scope and evidence motivating the change

The last finalized training source is checkpoint_step_000084992.pt, 84,992 lifetime policy decisions, 629 PPO updates and 12,580 optimizer steps. Its natural-P01 deterministic evaluation on e99fde1b3e8366f0ff1d484140b82877df745f0c earned FR, FL and RR qualification/crossing/placement, but ended at P12 after 90.2 s without RL qualification. This is incomplete, not success. The fixed FSM P10 prefix and that actual saved-C P10 entry have different physical histories. See eval_84992_diagnosis.md; a common phase label is not a common physical state.

The new optional curriculum rolls in from a real natural P01 reset with an independent frozen copy of the verified saved actor. At the requested phase the trainable policy takes over in the same core/backend. No snapshot restore, teacher controller, second reset, residual/mapper/filter reset, fixed joint entrance or phase-terminal conversion is used. The first selected block starts credit at P06, including preparation before RR/RL tasks, not only an already-lifted rear state. Prefix failures are recorded and use one fresh P01 fallback; ordinary unsuccessful tasks do not block optimization.

## Production changes

- src/wlr50_clean/ppo/semantic_checkpoint_prefix_policy.py: independent deterministic actor clone, including learned state-dependent std and normalizer buffers; no optimizer ownership or source RNG/cache mutation. Frozen source checkpoint provenance and independent storage are verified once.
- src/wlr50_clean/ppo/semantic_checkpoint_prefix.py: ordinary SemanticIsaacBackend real roll-in; continuous native clocks and action/history/contact state; all-tick action execution evidence; reset-only prefix counters excluded from PPO credit. No task outcome is relabeled as full success.
- src/wlr50_clean/ppo/semantic_cli.py and scripts/run_semantic_ppo.ps1: explicit PrefixSource=checkpoint_policy; original frozen_fsm default retained. Load the source actor before binding roll-in. Initial checkpoint metadata uses the current curriculum rather than inherited source sampling flags.
- src/wlr50_clean/ppo/semantic_training.py: immutable curriculum provenance included in rollouts/checkpoints/manifests and checked before and after each physical decision, including episode resets.
- src/wlr50_clean/ppo/semantic_migration.py: recognize the new reset topology while preserving old topology validation and lifetime budgets; migration descriptions no longer imply all prefixes are FSM teachers or that unchanged reward was modified.

No action cap, reward, stage/task evaluator, nominal policy, physical limit, 324-dimensional observation ordering/preprocessing or native target audit arithmetic is changed in this revision. Frozen A files and historical data are not edited.

## Selected next real run (planned, not yet credited)

N1 / seed 1001 / P06 / checkpoint_policy / offset 0 / 2,048 additional policy decisions / rollout 128 / 5 epochs x 4 minibatches / checkpoint every 4 updates. The fixed roll-in actor is source 84,992 for the entire block; prefix decisions and ticks are separately recorded, never optimizer credit. Full natural-P01 training remains part of the continuing curriculum and formal evaluation always starts at natural P01.

An explicit NewMdpWarmStart boundary preserves compatible actor, learned std, critic, identity normalizer, training RNG, lifetime counters and original spent budgets; Adam moments are reset with configured initial LR 3e-5 and old unfinished rollout is discarded. New physical samples are collected. Subsequent same-version curriculum switches can preserve optimizer state. This is not an exact physical-state resume.

Source checkpoint SHA256: ebabc03d1d3611bbe4ab1c05feddbbe3102eedb71ae9d5924cd8f77f2826ea01. Source manifest SHA256: c39f24462b315d654c71d89ceef8e7788b0609be0b88083832f07c40b2b789a7. Original spent budgets: full_episode 33,280; phase_suffix 41,600; smoke 0. No future decisions are pre-credited.

## Verification before native execution

Root integrated CPU receipt C:/robotics_sim/wlr_robot/checkpoint_prefix_integrated_cpu.xml: 173 passed, zero failures/errors/skips, 17.283 s. It covers the new prefix/actor/CLI and original prefix, CLI, migration and initial checkpoint publication paths. Separate training/continuous-task regression receipt C:/robotics_sim/wlr_robot/checkpoint_prefix_training_regression_cpu.xml: 137 passed, zero failures/errors/skips, 11.812 s. Both used installed official RSL-RL on CPU, not simulated success evidence.

Final seam receipt C:/robotics_sim/wlr_robot/checkpoint_prefix_final_seams_fixed_cpu.xml: 9 passed (6 existing plus 3 new unique tests). Total unique passing coverage is 313, not 319. The actual RSL loop rejects provenance mutation during terminal reset before storage/update/save; another actual official update verifies 129 excluded prefix actions versus 128 credited actions and 20 optimizer steps. The new initial-publication case runs the real frozen actor builder and immutable save/reload, validating source provenance, topology and current suffix flags. The first combined seam run had 8 passes and 1 test failure from an unseeded bitwise comparison of float32 terminal return to reward. That failed receipt remains at checkpoint_prefix_final_seams_cpu.xml. The test now seeds explicitly and verifies exact official operation order (r-V)+V plus its operand-scale floating-point error bound; no production task or return rule was relaxed.

Independent read-only integration review found no blocker for the actual heteroscedastic source84992/P06 path. An unsupported older scalar-Gaussian source would currently be rejected by the helper after scene initialization rather than at preflight; it is not selected here. Native summary field names were checked against the real SemanticEpisodeEnv implementation.

## Preserved A and outstanding deliverables

The original A run remains incomplete at P10 WAIT_ENTRY, 65.3667 s, because frozen historical RR knee position/velocity entry tolerances were not met. It is not relabeled success based on Trial043. Its 989-frame H264 video decodes; no body/wheel-only task failure was measured in that A. No new A5/5 or nonzero-probe-success optimizer gate is added.

There is still no full-P01 PPO success checkpoint, successful <=200 s PPO video, matched successful comparison set or paired stability-superiority result. Future native results must be recorded separately with saved counters, actual phase sample counts and first incomplete task.

## Executed block and latest saved state

The revision was committed as a580fc2add8137ea599df558f74d34ec9c50ef97 (6 production files and 5 test files; no frozen-A or configuration changes). The selected real block finished in run `20260906T2211507063967Z_ga580fc2add81_e539358648084c70bf6b28037515480c`, lifecycle SUCCEEDED; root confirmed the launcher session closed with exit0 before starting any next Isaac process.

Actual additional work is **2,048 policy decisions / 16 PPO updates / 320 optimizer steps**, resulting in **87,040 / 645 / 12,900**. Reported training-loop wall time is 1,487.5502931 s. Phase sample counts: P06=1,971; P07=10; P08=6; P09=61; all other phases=0. Six completed episodes end in P09 BODY_COLLISION; the final seventh-episode tail is nonterminal P06. There is no RR/RL hard qualification/crossing/placement in this block. The independent complete ledger is `p06_checkpoint_prefix_block_87040.md`; the bounded first-update evidence remains separately preserved in `checkpoint_prefix_first_live_85120.md`.

The final immutable checkpoint is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000087040.pt`, SHA256 `66fdb6b0ca8be83c31566a0aa6701834d5d8159d4614b4f43251b00a51388729`; pointer-bound manifest SHA256 `ff204b1a806b771cd536f2ffc8649f40ba92cfb5731894e03c7ae3965fa2609e`. Root read the actual pointer/sidecar and verified counters, source run, a580 runtime and save_load_round_trip=true. The prefix remains frozen source84,992, distinct from the optimized actor. Spent budgets are now full_episode33,280 / phase_suffix43,648 / smoke0.

Root then launched a separate natural-P01 deterministic full evaluation of saved87,040, N1seed2001, unchanged a580: `20260906T2240147110525Z_ga580fc2add81_3846a3f7a0764d6a836c153999750ea6`. At this report update it is still running; no result, later policy decision credit or video success is prefilled. The last finalized full evaluation remains C84,992/P12 incomplete until the new run actually finishes.
