# Reward conflict review — timing-first repair and bounded actual continuation

## Status and attribution boundary

This report preserves the initial timing-only evidence and the later explicitly separated reward revision. The first candidate changed nominal timing only. The later body-cost allowance was committed asdbc492f and has now undergone actual finite PPO continuation; final counts and reloaded evaluation are recorded at the end and in latest_run_results.json once complete. Actor, critic, Adam, identity normalizers, HISTORY372/12 and120/15Hz were preserved; old unfinished rollout was cleared at the version boundary. Frozen-policy evaluation under a changed reward does not change that policy's action function. A control repair's benefit is not evidence of reward learning.

Source line references below describe the reviewed pre-candidate implementation. Earlier B0/C0 evidence remains immutable under `outputs/ppo_rr_video_diagnosis_v1/`. The C0 used for the numeric comparison is checkpoint166784, **not automatically the separate user-supplied filename164736 video**. No new simulation, checkpoint load, model training or production edit was performed by this review.

## Actual reward consumer

Selected `configs/ppo_fsm_reference_p09_stable_v2/reward_config.yaml:12-37`: gamma0.9985; potential weight5; success/failure events+40/-40; time cost0.02/s; family weights body0.4, contact0.2, applied smoothness0.1, residual regularization0. The existing transfer floor is0.2 for body/contact. No imitation, phase-transition or nonzero-residual bonus.

`semantic_reward.py:111-142,175-210` integrates costs at actual120Hz dt. Potential is5*(gamma*nextPhi-currentPhi), with zero nextPhi at a true terminal. Body averages normalized/clipped attitude, Euler rates and angular acceleration, all multiplied by1-0.8*f for the role version. Smoothness is the average of applied-target first/second differences; nominal/residual differences are diagnostics, not additional charged copies. Smoothness is not transfer-weighted. Constant applied targets have zero first/second differences after the transition transient; holding is not repeatedly charged a target-motion cost. Time cost and discounted constant potential remain negative during unchanged nonterminal waiting.

The runtime coefficient comes from eligible legs' role.motion_fraction (`semantic_supervisor.py:1333-1336`), not directly from phase or AIR. Eligibility follows real predecessor-placement history. The role window uses recent commanded/measured motion, support continuity and separate response maturity (`semantic_transfer_roles.py:143-146,157-210,260-261`).

## Confirmed narrow conflict and boundaries

An earned and currently useful airborne posture can outlast recent motion. However, the role fraction still multiplies recent actuated response and short-window maturity. Thus an effective carry/hold can lose the allowance before capture, even without a phase change. Static code permits full restoration when recent motion vanishes. This is a task-function versus transient-motion semantic mismatch, not proof of policy motivation.

The existing completed B0 extraction provides an actual smaller instance. In `B0_reward_0_20s.json`, decision endpoints96->104->112:

- FR Q/AIR remains true; no FR crossing/placement yet. At96->104 its top gap increases88.268014->100.828607mm, while front distance retreats-164.842781->-170.030412mm.
- Current support is FL/RR at96 and FL/RL/RR at104. Normalized load is valid throughout; at104 short support continuity is8/9.
- FR transfer/motion fraction changes1->2/9->1; body/contact weight changes0.2->0.822222->0.2. Therefore maintained large clearance does not itself preserve the allowance through the short support-response disturbance.
- The104 decision potential shaping is-0.094805705. Its existing0.1 unload share also consumes the decaying transfer_progress (`semantic_supervisor.py:1089,1108`); the separate measured front-distance retreat must not be misdescribed as no task regression.

Capture also restores the previous leg's allowance rapidly: top-contact count divided by minimum_top_samples=2, then historical placed retires that role (`semantic_transfer_roles.py:214-216,260-261`; `semantic_supervisor.py:686-688,1333`). B0 FR crosses at1486 and places at1502; endpoint1504 has f=1/6 and body weight0.866667, before the next eligible leg yields f=1 at1512. This is observed cost-window behavior, not proof that capture timing caused failure or that all capture states require a fixed delay.

Unknown load is **not** a blanket reset bug in the selected `functional_lift_edge_v2`: `semantic_transfer_roles.py:98,110-130,157-158,208-209,268-274` preserves independent verified geometry, uses None for unavailable load, and can use earned current clearance. Loss of enough independently verified supporting contacts can still lower continuation. Do not invent zero unloaded force, supporting AIR contacts or full stability from two contacts. Existing old-mode unknown-load reset tests do not prove current functional-mode behavior.

## Existing PPO comparison does not prove reward prefers failure

Reused verified122 returned decisions/976ticks/8.133333333s from `B0_reward_0_976ticks.json` and `C0_reward_0_20s_bound.json`:

- Undiscounted total: B0+0.297918504; C0+0.076725958.
- Same-step gamma0.9985 discounted prefix: B0+0.312341044; C0+0.119630638.
- Weighted attitude penalty: B0-0.038108920; C0-0.010821611. C0 is charged less for levelness, but loses more task potential and pays more applied smoothness, so the combined recorded return is lower.
- C0's120 P02 returned endpoints all have f=1 and valid normalized load. Consequently premature full attitude-cost restoration is **not observed in that C0 prefix** and cannot explain its failure on this evidence.
- At976 C0 FR gap is-13.606880mm versus B0+99.137023mm; base/CoM are also lower and angular speed is larger. More level is not more stable or closer to task success.

C0 safety-aborts at983; its last7ticks have no returned reward. No terminal penalty or complete-episode return was fabricated. B0's later FR placement is genuine, but this comparison is not a matched full-success episode comparison. These data do not provide all five requested complete trajectory classes (whole-task legal success, flattening with clearance loss, stationary stagnation, sinking, safety failure). No unsupported complete-trajectory ranking is asserted.

## Smallest subsequent candidate, only if root elects a reward revision

Keep role-observation fields unchanged. Add a narrow versioned reward-only functional allowance, using existing physical_evaluator metadata: retain the existing0.2 floor when earned lift/current usable clearance and allowed contact mode remain valid before capture, even if recent CoM speed/command motion subsides. AIR, historicalQ or a phase label alone must not qualify. Current loss of clearance, ground relapse, invalid region or safety termination removes eligibility. Existing200s deadline and negative elapsed-time cost remain. No new positive bonus, unique CoM target, historical angle or fixed1s task gate.

Capture/settle allowance may retire gradually using the existing bounded physical window and actual capture evidence; this must not delay action start or task acceptance. Test its exact necessity before combining it with the timing repair. Do not automatically discount slip/rebound or smoothness merely because the body allowance changed: separate existing consumers and report which terms were altered.

If fixing the pre-cross unload-potential decay too, reuse its existing0.1 share based on currently maintained functional carry; do not add another lift/CoM/AIR bonus. That change alters the existing potential observation scalar even with372 dimensions, so describe and verify that semantic migration explicitly. It is distinct from a reward-only consumer change.

Do not delete applied smoothness or change coefficients merely from the observed levelness tension. The evidence does not establish that smoothness forces concurrency. First compare the timing-only B/C results; any reward learning claim then needs fresh on-policy updates and reloaded evaluation from compatible checkpoint167424 or its verified successor.

## Targeted tests proposed, not executed in this review

1. Earned usable FR carry held beyond the old motion window remains protected; phase relabeling leaves the result unchanged.
2. AIR alone, staleQ, unearned low load and a sinking/ground-relapsed FR do not gain protection.
3. Unknown normalized load neither fabricates unloading/support nor clears independently valid carry geometry.
4. Real capture smoothly retires allowance, while another eligible leg can continue; no fixed1s acceptance or stop requirement.
5. Constant applied target has no duplicate nominal/residual motion charge; unchanged waiting retains negative time/discount cost and global deadline.
6. Existing terminal events, short terminal dt, no-bootstrap rules and all physical safety monitoring remain unchanged.
7. Reward-only edits preserve encoded372 observations and same-input frozen-actor outputs; updated learning is measured separately.

The existing `tests/unit/test_semantic_transfer_role_reward.py:73-86` tests fraction-to-cost arithmetic, not the functional-carry-versus-window-decay counterexample. Extend the physical producer-to-consumer test rather than changing those arithmetic expectations blindly.

## New same-N frozen pair and prepared repair (before learning)

The actual September14 B1/C1 pair at commit `a89ba82` has now completed and been exported. It used the same N/reward and checkpoint167424; no optimizer updates occurred. Both initial measured states match exactly. Every nominal channel matches at all879 shared physics ticks. C1 terminates in P02 at7.325s with RR knee final target-6.358188deg, actual-60.016800deg and tracking error-53.658612deg; B1's same-window peak absolute RR knee error is0.865149deg. All12 final channels already differ at tick1; direct same-prestate current-policy RR knee effect peaks at1.669081deg, while its final target magnitude peaks at6.649581deg. This distinguishes policy/current command/history effects from a large actual tracking departure. It does not separate the coupled whole-body loading and drive-history causes.

The107 returned C1 P02-request endpoints all retain existing transfer fraction1. Therefore the new body carry allowance would not change those returned body cost weights. The last7 safety-interrupted ticks contain physical/native evidence but no returned reward; no terminal reward is fabricated. FR still has earned Q at872 but no C/P and its gap has fallen to-15.102mm (physical terminal-16.112mm). RR remains grounded with measured load, not an invented unsupported leg. See `B1_C1_early_RR_readonly.json`.

The prepared minimal reward candidate adds only `carry_body_allowance=current_functional_carry_and_capture_settle_v1` and `capture_settle_window_s=0.5`. It extends the existing body attitude/rate/acceleration weight floor using earned current usable carry plus at least two independently verified other supports. Real captured placement retires this allowance over the existing bounded0.5s window; this is not an action wait or task-success timer. Unknown normalized load cannot invent support or erase independently verified geometry. Contact, applied smoothness, potential, event bonuses, elapsed-time cost, action ranges, observation/HISTORY and task/safety acceptance remain unchanged.

Candidate CPU tests and the actual old-trajectory offline check passed; these are not PPO updates. At old B0 endpoint104 the body transfer fraction would become1 instead of2/9; at1504 real capture-settle gives0.966667 instead of1/6; unchanged fraction1 endpoints remain bitwise unchanged in the reward calculation. All120 returned old C0 P02 endpoints are unchanged. `offline_real_carry_window_check.json` records exact physical/evaluation inputs; no actor or simulator was called by that check.

The combined production reward/compatibility patch was subsequently applied and committed as `dbc492f25c0a3f9713b8477b9ef4341159625a41`, after B2 ended.268 targeted CPU tests passed; the disabled old-reward test fixture explicitly removes both opt-in keys, and the CPU state round-trip runs with CUDA hidden. The real checkpoint167424 migration validates the timing ancestor and completed B2 same-N reference, retains full Adam/identity normalizers and clears old rollout. The P01 actual training block completed512 new decisions/4 PPO updates/80 optimizer steps, saved/reloaded checkpoint167936; effective LR remained1e-5. A P06-predecessor block is now running; its requests are not yet credited as completed updates.

An eventual improvement cannot automatically be attributed specifically to this narrow reward edit, because finite further PPO optimization also changes the weights; no matched old-reward training counterfactual is being claimed. Complete-P01 reloaded evaluation remains pending at this record.

The B2 diagnostic failure is separate from C1's early tracking departure: B2 zero-policy source RR knee-37.8deg receives mapper+10deg and existing nominal geometry-32.2deg, yielding final-60deg. Actual-60.000011deg triggers HARD_JOINT_LIMIT at6374. Geometry correction, not policy/controller bias, drives that target. Original A at its same finite source tail was already past the front plane, whereas B2 was still19.75mm before it; these are different real states. Positive inward RR policy residual remains available after the nominal-only projection, but no successful control or stability is guaranteed. See `B2_RR_knee_geometry_boundary_diagnosis.json`; no actuator limit or physical parameter was relaxed.

## Final actual continuation and reloaded P01 result

Training is finished, not waiting for an unspecified future success. The natural-P01 block contributed512 decisions/4 PPO updates/80 optimizer steps. The rear block requested512 but stopped normally at the next verified complete update after twoP09 BODY_COLLISION episodes: actual256/2/40, unconsumed256. Its third prefix was executed afresh before the last36 credited samples completed the rollout. All1344 teacher decisions/10752 teacher physics ticks are excluded. Total new learning768/6/120, original167424/1273/25460 -> saved and reloaded168192/1279/25580. Effective AdamLR stayed1e-5. See `training_total_actual.json`, `training_configuration_verified.json` and `Rear_credited_actual_control.json`.

Actual total request-phase sample counts: P01=4,P02=219,P03=19,P04=43,P05=139,P06=323,P07=2,P08=2,P09=17,P10-P13=0. All12 channels remained available; no stage transition was made an episode boundary. Two rear episodes reachedP09 and collided with the body; neither suffix is a full-policy P01 success. In the rear credited endpoints, RR knee same-prestate target effect is positive69/negative186/zero1; P09 is16/0/1, and explicit headroom clipping is zero. These are actual target effects, not measured joint displacement or proof that an alternative contact strategy is feasible.

The actual checkpoint168192 deterministic natural-P01 video run is `runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260915T0011185025896Z_gdbc492f25c0a_ccb65744dd8f48dd94fbf44d3a1285b8`. It ended at887ticks/7.391667s inP02, SAFETY_ABORT/HARD_JOINT_LIMIT. RR knee target-5.433995deg, actual-60.017312deg, error-54.583317deg. FR retains qualified lift history but has neither crossing nor placement; its terminal gap is-15.815104mm and it is150.054779mm before the front plane. RR is grounded and bearing16.158306N, not unsupported. This remains a large tracking departure with coupled policy/loading history, not a request to bend RR to-60deg. The exact single physical cause is not established by this pair.

New C and B2 share the same N and measured initial state; strict media audit allows only reward.py, reward_config.yaml, migration.py and training.py runtime differences. At common0..887ticks, B/C rollRMS=.228226/.110243rad,pitchRMS=.180749/.105296rad,world angular-speed normRMS=.145096/.163017rad/s,minimum validCoMheight=.154571/.116013m. Both latest runs fail. Lower attitude deviation does not establish better stability, and the additional8ticks compared with frozenC1 are not a task-progress improvement.

The body carry-window conflict has source, recorded-trajectory and CPU-test evidence; finite optimization and reloaded evaluation have now actually occurred, but the earlyP02 failure is not fixed. The old C1 returnedP02 costs already had full transfer protection, so it would be false to claim this narrow allowance cured that failure or proved reward was its cause. No matched old-reward retraining control, full-success improvement or superiority overA is claimed. Latest three videos are delivered even though incomplete; B1's earlier nonsafetyP09-incomplete video remains preserved as the better prior zero candidate.
