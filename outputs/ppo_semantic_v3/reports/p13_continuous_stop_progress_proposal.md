# P13 continuous stop progress — read-only, single-factor proposal

Status: proposed only, not implemented or selected. Written during the fixed P10 run `20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc`. Parent-reported completed boundary was global 63360 / update 460; its P13 state had region/support true, four nominal wheel commands zero, but maximum applied command about 0.363 rad/s and controlled false. This report does not count live tails or infer a new completed checkpoint. Current 4096-decision block and subsequent fresh P01 evaluation remain unchanged.

## 1. Confirmed scope of the gap

`semantic_supervisor.py:511–525` currently averages three clipped measured-rate terms and the opt-in reciprocal maximum-command term. The measured wheel maximum, body linear norm and body angular norm each give zero throughout their above-tolerance interval. Maximum-command progress is continuous outside its threshold, but changing a non-maximum wheel alone is invisible to that term. The completed 628-row historical P13 window and exact adjacent examples are preserved in `p13_stop_progress_flat_regions_readonly.md`; they establish local mathematical flat regions, not the cause of task failure.

Keep all of the following unchanged:

- `TaskEvaluator.observe`, lines 460–469: hard body linear/angular norm, maximum measured wheel speed and maximum applied wheel command checks; final region, current support, all placed history, safety and continuous 0.5 s stability requirement.
- All physical thresholds: measured wheels 0.25 rad/s, applied wheel commands 0.02 rad/s, body linear norm 0.05 m/s, body angular norm 0.30 rad/s. No new tolerance.
- Hard history, ordinary phase transitions, nominal/geometry correction, residual channels/ranges, mapper/slew, all other progress terms and reward families/weights, entropy and policy std.
- Global finish activation only after all four historical placements; no phase-specific extra reward or early whole-task success.

## 2. Minimal proposed opt-in

Add an explicit optional final flag, for example `stop_progress_semantics: per_wheel_four_type_threshold_ratio_v1`. Absent flag retains the exact existing branch, including v2 home progress and previous v3 `stop_command_progress: reciprocal_physical_stop_tolerance`. Reject unknown values. New mode requires current v3 global physical progress and physical-stable-pose semantics; validate the four existing tolerances as finite and strictly positive.

For finite magnitude x and the corresponding existing positive tolerance T:

    q_T(x) = T / (T + abs(x))
    wheel_progress   = mean(q_0.25(measured_wheel_i), i=FL,FR,RL,RR)
    command_progress = mean(q_0.02(applied_command_i), i=FL,FR,RL,RR)
    linear_progress  = q_0.05(norm(body_linear_velocity))
    angular_progress = q_0.30(norm(body_angular_velocity))
    control_progress = mean(wheel_progress, command_progress,
                            linear_progress, angular_progress)

This is four equally weighted **types**, not ten equally weighted scalar inputs. It replaces the entire existing stop/control aggregation in the new branch; do not append the old maximum-command term a second time. Preserve the finish formula `.4*history + .3*forward + .2*control + .1*settle`, its pre-success cap `.99`, and the existing `.15` global finish coefficient.

Each type therefore has coefficient `.15*.2/4 = .0075` in global phi, and each wheel within a wheel type has coefficient `.001875`, while finish is active and unsaturated. All changes stay inside the original control allocation. `q(0)=1`, `q(T)=0.5`, and decreasing any positive magnitude increases q even outside the hard tolerance. The absolute value is not differentiable at zero, but the score is continuous, bounded, and smooth on either side; this environment reward does not require differentiating through the dynamics.

This choice also rewards slowing below T toward zero, rather than declaring soft progress complete exactly at T. That is an explicit semantic change, not an accidental assertion that success now requires zero. The unchanged evaluator still completes automatically after 0.5 s within the existing tolerances, regardless of whether soft control reaches 1.

## 3. Existing physical inputs and minimal audit payload

No new sensor or backend read is needed. `TaskEvaluator.observe` already reads and finite-validates the four `WheelObservation.velocity_rad_s` and `command_rad_s` values (`semantic_supervisor.py:281–308`); it currently discards their individual values after computing maxima. Body inputs are the actual world-frame base `linear_velocity_w_m_s` and `angular_velocity_w_rad_s` (`272–273`), already stored as norms in the fixed goal features.

`sensing/sensor_reader.py:260–262,305–312` reads measured wheel velocities from `adapter.get_actual_state().full12`. The adapter takes actual articulation `joint_vel` and converts to canonical logical readback (`infrastructure/robot_adapter.py:355–377`). Wheel command comes from the dispatch ACK `drive_target_full12`, passed after the actual simulation step/readback (`ppo/isaac_fsm_backend.py:1977–1991`), not raw PPO action or nominal-only targets. These are rad/s in the existing canonical wheel convention, not linear ground speeds. Do not multiply by wheel radius or infer slip. Taking magnitudes makes native left/right sign changes irrelevant.

Be precise about evidence: `command_rad_s` is the existing ACK-bound logical applied drive command used by the hard predicate. It is not a fresh read of the native float32 setter/dispatch tensor. Preserve the existing native audit separately; do not relabel a logical double as native target readback or change the hard predicate's input definition.

Minimal new evaluator snapshot payload in the opt-in branch:

- `measured_wheel_velocity_rad_s`: immutable four-tuple, canonical FL, FR, RL, RR order.
- `applied_wheel_command_rad_s`: immutable four-tuple in the same order.
- Optional immutable `stop_progress_wheel_order` using the actual WHEEL_ORDER names, so an archived snapshot is self-describing. Existing evaluator physics_tick, simulation_time_s, source, body-norm goal features and maxima already bind the rest.

Populate these from the same finite-validated local lists used for the hard checks. Recompute/verify the maxima in focused tests. If a new-mode input is absent, wrong-length or nonfinite, fail explicitly rather than substituting zeros or using nominal/raw action. Preserve the existing invalid/terminal observation handling. Immutable tuples avoid introducing a nested mutable snapshot alias (`TaskEvaluator.snapshot:201–214`). Generic task/evaluator audit serialization can carry these fields without modifying the 17-key goal feature contract or adding per-tick I/O.

## 4. Actual historical examples and honest algebraic consequences

The following scalar examples use the actual adjacent overspeed values from the historical report. They demonstrate the proposed scalar response; the measured-wheel row is **not** a recomputation of the proposed four-wheel mean, because that table supplies only the maximum.

| Existing scalar / tolerance | Recorded decrease | Existing clipped term | Proposed scalar q change |
| --- | --- | --- | --- |
| Measured wheel maximum / 0.25 rad/s | 0.6653692722 → 0.3016962707 | 0 → 0 | 0.273114 → 0.453148 |
| Body linear norm / 0.05 m/s | 0.2055092345 → 0.0737077689 | 0 → 0 | 0.195688 → 0.404178 |
| Body angular norm / 0.30 rad/s | 0.5666096502 → 0.3461716139 | 0 → 0 | 0.346177 → 0.464273 |

At historical global 50903, the actual applied command vector was `[0.15392489696406916, -0.18488621010542575, -0.12323237071913808, -0.1339810112310154]` rad/s. Holding everything else fixed and changing only the non-maximum FL command to +0.02 leaves the old maximum-command term unchanged. The proposed command-type mean increases from 0.120531677 to 0.216783636, giving isolated global delta-phi +0.000721890 while finish is active. This is a same-state mathematical counterfactual, not an observed physical response or a positive total-reward claim. The FR hard maximum remains above tolerance, so controlled/success remain false.

## 5. Slower versus completion: tradeoff and PBRS boundary

There is a real local soft tradeoff to acknowledge: after all placements, a slow state outside the required region can score better in the control component than a necessary repositioning motion. Averaging also lets three quiet wheels partly offset one fast wheel **in soft progress only**. Neither phenomenon permits completion: region/support and hard maximum tests remain separate and unchanged. Other forward/current-capture terms remain active. This proposal supplies a more informative control component, not proof that policy optimization will balance those components correctly.

It does not add a reward for waiting or an independent stop bonus. `semantic_reward.py:179–190` still computes exactly `5*(.995*phi_after - phi_before)` once per issued action, with phi_after=0 at every true terminal. Across any finite terminal trajectory, discounted shaping telescopes to `-5*phi_start`; it does not create extra discounted trajectory return by stopping more slowly. Constant positive phi gives a small negative shaping increment, not repeated credit. Existing terminal success/failure events and time/quality costs are unchanged. Approximate critic/GAE learning, discounting of those existing costs/events, and finite optimization can still favor locally unhelpful actions; the PBRS identity is not a guarantee of learned completion.

Use the recorded physical phi for diagnostics before termination, but retain absorbing terminal zero and no bootstrap for true success/failure/deadline. Do not preserve terminal phi merely to make the plotted progress look favorable. The current 128-step PPO horizon and entropy schedule are not changed in this proposal.

## 6. Compatibility and smallest implementation boundary if later selected

Only the supervisor loader/evaluator payload/whole-task soft-progress branch, an explicit v3 stage-spec revision and focused tests are intrinsically needed. Reward implementation, backend physics, actor distribution, native audit and the observation schema file need not change. Old absent-mode and old reciprocal-only configurations retain their source behavior.

However, unchanged 324-dimensional shape is **not** unchanged observation meaning. `semantic_observation.py:236` already encodes `(phase_progress, task_progress_potential)`; the global potential changes after all placements and the P13 phase progress uses this same whole-task predicate. These existing two scalar channels may change numerically. All other observation groups, scales, clips, fixed reference and 12-action interface remain identical; never append individual wheel diagnostics to GOAL_FEATURE_KEYS or actor dimensions.

If adopted after the fixed block plus evaluation, bind this as an explicit reward/progress MDP revision using the existing v3 new-MDP continuation mechanism, not policy-distribution migration or silent exact resume. Preserve latest actor/critic/learned state-dependent std/normalizer and lifetime/spent-budget origin; fresh rollout and the explicitly recorded optimizer treatment remain that mechanism's responsibility. Old rollout rewards/values must not be reused as samples of the new potential. This report does not select or execute that migration.

## 7. Focused positive/negative tests to require in the patch, not a new training gate

1. Actual TaskEvaluator fixture with realistic wheel/body fields verifies signed four-vector order, units, clock/source binding and exact agreement with existing hard maxima. Independent wheel variations reach the new soft branch without fake nominal reconstruction.
2. Magnitudes `4T,2T,T,T/2,0` give strictly increasing q; exact threshold is continuous at 0.5, not a new zero boundary. Increasing magnitude reverses the change; sign flip and wheel permutation preserve the aggregate.
3. Reduce a non-maximum wheel while retaining the maximum: soft progress changes by the exact `.001875*delta_q`, but hard controlled stays false. Repeat for command and measured wheel independently. Hold body/other-leg terms fixed so this is not a confounded reward example.
4. Four-type weights are equal; wheel dimensions are averaged inside their type. No duplicate legacy command term, fifth family, stage/entry bonus or repeated use of the finish control term. Real SemanticReward integration matches the single PBRS formula, including a case with improving q but negative total F due to discount/other state.
5. Zero inputs maximize soft control, but missing region, support, history or safety still prevents success. Any hard rate/command violation remains a violation; a good state must satisfy the same continuous 0.5 s. Ordinary phase relabeling cannot change global phi at identical physical/history state.
6. True terminal phi remains zero/no bootstrap; hovering cannot repeatedly collect a bonus. Malformed/nonfinite inputs or nonpositive tolerances do not become zero-speed success.
7. Old v2 and old v3 configurations retain exact old behavior and payload compatibility; unknown mode is rejected. New mode preserves 324/12 shapes and all non-task-progress encoder groups; the expected changed scalar semantics are explicitly acknowledged, not asserted bit-identical.

Conclusion: there is sufficient code and historical evidence to define this narrow, testable single-factor candidate. There is not evidence that it alone solves stopping or full-task success. No production/config/test changes or Python/Isaac execution were performed for this report; implementation decision remains after the ongoing fixed block and fresh P01 evaluation.
