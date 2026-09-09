# C127232 — P02 approach remains incomplete

Run: runs/ppo_transfer_roles_v1/validation/20260908T0029370739322Z_g2f27c6f5065a_1c6693d2963848b088c02e561fccdbad  
Fixed HEAD `2f27c6f5065a`; checkpoint127232; natural P01, N1, seed2001, deterministic history-policy mean, no prefix.

## Outcome and first missing task

Execution finished normally, but task success is **false**: 227 decisions /1816 ticks /15.133333 s, P02 `INCOMPLETE_CONTROLLER_BLOCKED`, phase age15 s. Evaluation optimizer updates0. Physical evaluator valid=true, hard-failure reason=null; this is measured incompletion, not collision, sensor, optimizer, I/O or video failure.

P01→P02 was a nonterminal handoff at tick16 (`transfer_ready_FR=1`). FR earned hard lift qualification at tick54, but never crossed or placed. At the deadline, `lifted_FR=clear_FR=1`, `approach_FR=.848302635`: front distance−42.924341 mm versus required minimum−5 mm, leaving37.924341 mm. FR is AIR, clearance+80.707817 mm, load0.

P01’s predecessor version already used only `load_ready_FR` (30 s), without an edge-distance predicate. This revision cannot be described as newly moving all approach work into P02.

## Actual travel and commands

| Measured window | FR front advancement | Base X advancement | CoM X advancement |
|---|---:|---:|---:|
| P02, tick16→1816 (15 s) |228.008 mm|99.453 mm|103.733 mm|
| Last5 s, tick1216→1816 |50.827 mm|49.022 mm|50.506 mm|

The final point is the closest FR approach. Maximum clearance was95.280 mm at tick800, still128.137 mm before the front plane. Forward movement continued into the deadline; this was not an absence of wheel commands or a motionless endpoint.

Of225 P02 decision boundaries,221 record all four wheel suggestions as+.3 rad/s (first at tick56). The terminal canonical wheel arithmetic is:

| Wheel | Nominal | Residual | Actual target | Measured velocity | Current load |
|---|---:|---:|---:|---:|---:|
| FL |.300000|−.259630|.040370|.106286|.427081|
| FR |.300000|−.031897|.268103|.268186|0|
| RL |.300000|−.254128|.045872|.010126|.099506|
| RR |.300000|+.214920|.514920|.444802|.473412|

Velocities/targets are rad/s; load is dimensionless. Native float32 wheel targets are `[-.040369567,+.268103153,-.045871556,+.514920235]`, reflecting left-wheel sign inversion. Measured velocities are not target receipts.

During the last5 s, mean canonical commands were [.039630,.267479,.045186,.515237]. FR had no contact throughout P02; FL/RL/RR were ground-supported over this last window. FL/RL residuals therefore substantially offset the rolling suggestion, while RR reinforces it. This arithmetic and the observed body/leg motion do **not** establish a unique cause, dynamic feasibility under another command, or a reason to relax the deadline.

Terminal observed support count3, stored geometric margin48.971 mm; this is not a contact-wrench or dynamic-stability certificate. Base linear speed.007463 m/s and angular speed.079005 rad/s. No leg has a placement event. Later phases were never exercised.

## Evidence integrity and phase quality

All1817 raw observations (including tick0) have finite data and contiguous120Hz timestamps; wheel exact-pair verification failures0, body collision flags0. All227 compact decision audit summaries conserve1816 verified/native-effect ticks,1815 own-phase ticks plus one handoff-held tick, four in-episode state-write counters all0, finite fallback0. This uses saved native summaries, not a second independent tensor-level audit.

Only P01 (16 ticks) and P02 (1800 ticks) were sampled:

| Window | Roll RMS (rad) | Pitch RMS (rad) | Angular-acceleration RMS (rad/s²) | Recorded quality score |
|---|---:|---:|---:|---:|
| P01 |.004773|.015140|2.461566|.467009|
| P02 |.173120|.145390|4.976306|.440716|
| Global C |.172356|.144755|4.959720|null|

P03–P13 metrics are **null**, not zero. Aggregate fixed score is null; quality scores are not task success.

Historical A remains P10 WAIT_ENTRY incomplete:65.366667 s, seed4001,180+64 pre-action ticks. Its global roll/pitch/angular-acceleration RMS were .139238 rad/.098914 rad/4.316558 rad/s². Current C is seed2001,180+0, a different evaluator and only P01–P02 coverage. These are **unpaired raw references**, not superiority or stability evidence; paired improvement and success-video outputs remain unset. A’s legacy commanded-wheel-speed0 is not independently validated native actuation, and its valid=true is not a new shared-evaluator pass.

Known limitation: `target_contact_reaction_possible` can miss real wall contact because it currently follows bearing-support membership. It has no hard/reward/324-observation consumer and does not explain this outcome; FR is AIR and all recorded wheel obstacle contacts are inactive.

Analysis receipt: the first streaming attempt lost its statistics because PowerShell error text preceded JSON. One explicitly authorized corrective raw/decision read succeeded; native summaries were reused without a second native-file scan. This reporting error did not affect the simulation. No Python/Torch/Isaac, production changes, tests, replay or hash recomputation occurred.
