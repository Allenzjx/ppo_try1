# v8: retain signed AIR wheel capture continuity

Evidence is the sealed v7 natural-P01 source20260923T0934264919029Z_g60abc00957c0_ef405598c8954e6280e844c4c29ed041. v7 correctly continued the assist/scheduler past posttick14600, but the wheel layer still required gap>=0. Dispatch14601 read the first negative gap and selected release_slew, despite unchanged committed N=[0,0,0,0], qualified AIR/XY and measured FL/FR/RL support. FL FINAL changed+.000037812→-.014962188rad/s, exactly the existing1.8/120 slew. Actual FL became negative at14602. Gap returned positive at14604, wheel switched back14605, but slew retained negative FINAL temporarily. The last2s contain15zero-crossings/action switches and51release ticks; minimum FL FINAL-.149966, actual-.064444rad/s. All RR exact-pair force vectors remainzero, noTOP/placed. This proves the control discontinuity, not that it is the only physical cause of missing capture.

## Minimal runtime delta

semantic_rr_carry_wheel.py now uses the SAME existing RR_CAPTURE_GAP_MIN_M=-.015 lower bound as assist and scheduler. A qualified AIR attempt in that band keeps forward_floor; gap<=0 makes its numeric floor0, not a positive drive and not support credit. Existing current bearing selection, source-ACK proof, authored stop/owner priority, phase scope,1.8rad/s² slew and hard bounds stay unchanged. Real TOP, support/XY/Q loss, outside-band or phase leave still release/yield under existing rules. Earlier finite source reverse and legitimate post-capture reverse remain available. All12 policy channels/raw samples/logp preserved; the transform is declared common control, not learned policy.

No assist search/budget/target/reward/geometry/physics changes:53deg/45s absolute totals, feedback progress and tracking, sensorTOP and2-sample placement, global200s remain unchanged. No new weak-contact extension or FR/FL/RL preparation controller. Same410 observations already contain measured gap/contact and envelope state; no new hidden state/timer. Versioned exact full-state migration retains Adam/LR/Identity/RNG/ancestry and discards unfinished rollout.

## Tests and next actual work

Before fix:4new negative-band/zero-crossing tests failed, zero/outside-band controls passed. After fix:176focused wheel/dispatch/signed-capture/contact/counterfactual cases passed. These are synthetic0physics/0PPO evidence. No A5/5 or successful-probe optimizer gate.

After strict freeze/save/reload and current media CPUhelpers finish, run actualPPO naturalP01 2048new decisions/16updates, checkpoint every128, separate ancestor220544_signed_wheel_v8 branch. These numbers are planned, not completed. Actual measured phase coverage determines whether a P07 real-prefix block is additionally needed. Preserve CP221184 latest learned separately, no credit borrowing/mainpointer overwrite. Deterministic naturalP01 reload/video follows training; controller improvements must not be called new learning.
