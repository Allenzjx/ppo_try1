# Continuous physical stopping progress: implementation boundary

This is a single-factor reward/progress revision after completed checkpoint 66944 (488 PPO updates / 9760 optimizer steps), its completed P10 curriculum block, and its reloaded natural-P01 evaluation. The latter failed at P09 with a verified persistent base/obstacle contact at tick 4674 / 38.95 seconds; this change does not reinterpret or claim to repair that collision.

## Changed production paths

- `src/wlr50_clean/ppo/semantic_supervisor.py`: explicit `STOP_PROGRESS_MODE=per_wheel_four_type_threshold_ratio_v1`, loader validation, three evaluator audit tuples from the same live observation, and a replacement soft stop aggregation.
- `configs/ppo_semantic_v3/stage_task_spec.yaml`: explicit revision and opt-in flag. The legacy reciprocal command flag remains a fallback only when the new flag is absent; it is not counted twice.

For each already-established positive physical tolerance T, `q(x)=T/(T+abs(x))`. The control score is the mean of four types: mean of four measured-wheel q values, mean of four applied-command q values, body-linear-norm q, and body-angular-norm q. Thresholds remain .25 rad/s, .02 rad/s, .05 m/s, and .30 rad/s. This replaces the old stop aggregation inside the existing `.2` finish allocation, not the reward family or family weights. The overall finish formula, `.15` global finish weight, activation after all placements, and terminal potential treatment remain unchanged.

The scalar function is continuous through the threshold, where q(T)=.5. That soft value does not change the hard threshold or require zero speed for success. Reducing a non-maximum wheel has a direct soft signal without changing the hard maximum test. Each wheel's isolated global-potential coefficient is `.15*.2/4/4=.001875`, not a new reward weight.

## Unchanged physical and learning contracts

The body-collision and wheel-only failure classes, independent safety aborts, historical lift/cross/placement events, final measured region and current support, all four hard stopping limits, and continuous .5-second eligibility requirement are unchanged. AIR does not become support. No nominal, mapper, action range, filter, phase transition, GAE/done, actor distribution, entropy, std, gamma, lambda, or value clipping change is included. Original A and historical artifacts are preserved.

`measured_wheel_velocity_rad_s`, `applied_wheel_command_rad_s`, and `stop_progress_wheel_order` are immutable evaluator-level tuples. They use the existing FL/FR/RL/RR order and the exact local lists already used by the hard predicate. Applied command is the canonical dispatch-ACK command, not nominal, raw PPO action, or native float32 readback. The existing native audit is separate. No extra sensor read, physics step, file write, or actor input dimension is added.

The 17-key goal features, 324 actor observations and 12 actions remain the same shape. Existing phase-progress/global-potential scalars can change in meaning/value after placement: this is explicitly a new MDP/progress revision, not a claim of bit-identical observations. Missing, wrong-order, wrong-length and nonfinite new inputs fail explicitly. Absent-mode v2 and old v3 keep the exact previous formula and omit the new payload.

## Evidence and limitations

Completed P10 ep0/ep1 P13 windows each contain 900 decision endpoints; commanded-wheel maximum is above .02 on every endpoint. Last-second measured-wheel maximum is also above .25 on all 15 endpoints of both episodes. The current clipped measured-rate terms therefore have a real sampled flat interval. One endpoint also fails final region because FR is above the existing top-gap upper bound. This revision supplies a more informative stop component; it does not prove stopping convergence or address every region/capture/collision failure.

The original PBRS formula is still `5*(.995*phi_next-phi_before)` with absorbing terminal phi=0. A local improvement may still have negative total shaping, and finite terminal discounted shaping still telescopes. Slowing outside the required final region can improve the local stop component but cannot satisfy the unchanged task. No repeated wait bonus is added.

## Version migration and next execution

The planned continuation uses the actual newest 66944 checkpoint, not 10112 or a selected older success candidate. The existing `NewMdpWarmStart` mechanism must verify and preserve actor, critic, learned state-dependent std, identity normalizer, source RNG, lifetime counts, original v3 origin, and spent budgets. It deliberately creates fresh Adam at the mechanism's explicit 3e-5 initial LR; old Adam moments and unfinished old rollout are not used. Later adaptive LR must be reported from actual logs, not assumed constant.

The next planned real course starts at a physically established P06 entry, offset 0, N1/seed1001, 4096 decisions, keeping the actual P06→P07→P08→P09→P10→P11→P12→P13 task chain. Teacher-prefix actions remain excluded. This is not a current-policy P01 success demonstration; a fresh fixed-mean natural-P01 evaluation is still required after the block. No future decisions, updates, success, or video are credited here.

At this initial implementation-note boundary, 198 existing supervisor/continuous/capture tests and 17 legacy stop tests have passed. They are overlapping portions of the later whole-suite run, not counts to add to its total. New positive/negative tests, independent diff review, complete CPU suite and actual migration/training results will be recorded at their real completion boundaries.

## Completed test/review boundary, before real continuation

The new focused suite finishes at **85 passed**, including JSON snapshot roundtrip (immutable tuples become lists and preserve the exact score). An independent read-only diff review found no blocking discrepancy with the proposal or hard/legacy contracts. The entire `tests/unit` directory then completed with **2318 passed, 9 skipped, 0 failures, 0 errors**, 2327 collected cases, in 427.59 s (JUnit 427.553 s). All nine skips are pre-existing Windows symbolic-link creation privilege limitations, not disabled physics or stop tests. The sole warning is a Torch ONNX logging deprecation. No privilege setting or environment package was changed to suppress these results.

Evidence: `C:/robotics_sim/wlr_robot/semantic_continuous_stop_all_cpu.xml`; focused new cases: `C:/robotics_sim/wlr_robot/semantic_continuous_stop_progress_20260906.xml`. Earlier subset counts overlap this full unit run and must not be added to it. This boundary credits **zero** additional policy decisions/updates. The 66944 checkpoint and its failed P01 evaluation remain the latest completed model/results until real continuation updates are actually saved.
