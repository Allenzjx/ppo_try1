# Episode5: P12 genuinely starts before committed RR capture hold

Bounded read-only scope: episode5,66.5–69.2667s/tick8312 of `train_first2048_ecf205e/decisions.jsonl`. Fixed open/seek/tell snapshot267670588 bytes; no waiting for writer, no model/physics execution. Follow-up field extraction started at the previously verified complete-row byte boundary248257875 rather than rescanning earlier episodes.

**Unlike episode3, P12's new RL source lane actually starts here.** At8120 P12 source_ticks=1 despite RR local hold only0.016667s and RL still GROUND/unqualified. The source subsequently unfolds RL knee, and RL physically becomes qualified AIR before the0.5s local terminal has been committed.

| Tick / time s | RR contact / gap mm / hold s | P12 source cursor / permission | RL physical state | RL nominal hip/knee ° | RL FINAL hip/knee ° |
|---|---|---|---|---|---|
|8120 /67.6667|TOP /+0.017 /.016667|1 /active; RR bearing permits start|GROUND, not qualified,1.785N|15.4 /19.4|.083 /27.615|
|8128 /67.7333|TOP /.083333 hold|9 /active|AIR, not yet qualified|15.4 /22.6|Not separately retained in this table|
|8144 /67.8667|TOP /.216667 hold|25 /active; legal RL swing now true|AIR, current_lift_valid=true|15.4 /34.2|Not separately retained in this table|
|8176 /68.1333|TOP /−1.011 /.483333|57 /active|AIR, current_lift_valid=true|.5 /35.3|−13.234 /43.518|
|8184 /68.2|AIR /+5.848 /0|65 /holding_RL_joint_lane; wheel clock continues|GROUND, qualification revoked,11.548N|.5 /35.3|−14.251 /45.242|
|8312 /69.2667|AIR /+86.455 /0|193 /holding_RL_joint_lane; wheel clock continues|GROUND, unqualified,11.663N|.5 /35.3|−15.223 /37.997|

Source and policy both matter: at8176 RL requested residual is `[−12.484,+6.968]°`; at8184 it is `[−13.501,+8.692]°`. Thus FINAL is not a source-only causal measurement. RL source knee19.4→22.6→34.2→35.3 and hip15.4→.5 establish that a real new P12 source lane progressed, while current AIR/qualification independently establishes real RL unloading. These do not isolate P12 as the unique cause of RR loss; P09 late, residuals and body/contact response overlap.

Contact timing is not collapsed into one label:8088/8096 already show an obstacle **TOP surface** reaction near10N but `within_top_xy=false`, so qualified local TOP/hold is false. P09 late start is8100;8104/8112 endpoints are AIR. The later uninterrupted legal TOP streak begins8118 as inferred from its native count/hold, yielding3 samples at8120 and59 at8176. At8184 hold is reset. Decision endpoints do not rule out a brief uncommitted0.5s native hold between8176 and8184; no local-success terminal was committed.

At8184 the P12 wheel source changes to `[-.3,-.3,-.3,-.3] rad/s` and remains that nominal value through8312. These are **canonical nominal commands**, not measured wheel speeds or proof of wall traction. Pausing an already-started whole P12 clock would risk retaining its finite pulse and losing its authored stop; the existing independent wheel-clock/RL-joint pause must remain.

## Minimal RR-first pending-start protection

Only for the declared local RR-first route: when `local.active=true`, `P12 source_ticks==0`, and actual production `rear_dependency(...).rl_current_swing=false`, leave the new P12 lane pending and `rl_dependency_wait=true`. Do not release merely on transient RR TOP or pre-boundary hold. A true current qualified RL AIR swing retains the existing continuation exception. Once P12 has started, delegate the original wheel clock/stop and RL-joint pause unchanged. No target backfeeding or control over the12 policy channels is introduced.

This additional guard is now only in the outputs-side `staged_late_guard_v2` candidate, alongside the separate P09 late-event deferral;21 stdlib wiring tests pass. Receipts explicitly name both source revisions. It has not been deployed or physically validated, and the P09-only candidate cannot be presumed sufficient based on this episode. Current training remains untouched.
