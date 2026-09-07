# C87040 — completed natural-P01 evaluation: real P09 body collision

Final, bounded read-only diagnosis. Source: `runs/ppo_semantic_v3/validation/20260906T2240147110525Z_ga580fc2add81_3846a3f7a0764d6a836c153999750ea6`. This report reads its final manifests, decision/native ledgers and actual physical observations; it does not run a simulator, load tensors, rehash checkpoints, alter historical evidence or classify the new front-wheel-range probe.

## Outcome and complete accounting

The saved C87040 fixed deterministic mean was evaluated from natural P01, seed2001, under a580fc2. The run lifecycle is `SUCCEEDED` and root confirmed exit0/CLOSED, but the task is **not successful**: **877 decisions /7,011 physics ticks /58.425s**, terminating in P09 with `BODY_COLLISION`. The shared physical evaluator is valid and reports `TASK_FAILURE_BODY_COLLISION`, reason `central body/obstacle collision`. Optimizer updates during evaluation are0. This is an actual physical task failure, not an I/O, video, timeout or checkpoint-loading failure.

Phase counts P01–P13 are **1,195,3,1,151,515,1,1,9,0,0,0,0**. All7,011 native audit rows are contiguous and verified, with real target effect, setter/dispatch agreement and actual mapping agreement. Decision summaries agree: verified7,011, own-phase effect7,003. Root-pose/root-velocity/force-or-impulse/gravity write counts are all0. The last decision contains3 ticks, so877×8−5=7,011. All7,012 physical observations, including tick0, are contiguous and finite. The raw stream was checked in two nonoverlapping parse passes except the boundary row6920 after a report-reader wheel-key correction; no runtime evidence was changed. Actual wheel geometry uses `*_ankle` keys while contact bodies use `*_wheel` keys.

| Transition | Tick | Simulation time (s) |
|---|---:|---:|
| P01→P02 | 8 | 0.066667 |
| P02→P03 | 1568 | 13.066667 |
| P03→P04 | 1592 | 13.266667 |
| P04→P05 | 1600 | 13.333333 |
| P05→P06 | 2808 | 23.400000 |
| P06→P07 | 6928 | 57.733333 |
| P07→P08 | 6936 | 57.800000 |
| P08→P09 | 6944 | 57.866667 |
| P09 physical failure | 7011 | 58.425000 |

All eight phase transitions retain null termination, `time_outs=false` and `terminal_bootstrap_allowed=true`; the collision is terminal with bootstrapfalse. P07/P08 are each one eight-tick decision, not independent reset episodes. This full evaluation has no teacher prefix or teacher credit.

FR genuinely earns Q45/C1579/P1592; FL Q1675/C2607/P2805. **RR has no initial-clearance event and no hard Q/C/P anywhere in this evaluation.** RL has several initial-clearance events but no hard Q/C/P. Therefore this is not a case of a legally completed RR crossing being lost later. At termination `placed_RR=0.2588679554` is partial continuous progress, not a hard qualification or placement event.

## First exact body contact and actual support

There is no active exact base pair before tick7010. At tick7010 (P09 entry+66 ticks), the verified `base_link`–`/World/Obstacle` sensor pair first becomes active; at7011 (+67 ticks), the second consecutive active sample triggers the body-collision predicate.

| Raw sensor evidence | Tick7010 | Tick7011 |
|---|---:|---:|
| Force vector world (N) | [−1.790829e−7,+1.594542e−7,16.68274498] | [−4.202762e−7,−4.888666e−7,25.99365425] |
| Contact point world (m) | [0.54515606,−0.23316331,0.05080922] | [0.54584968,−0.23312132,0.04995762] |
| Consecutive active samples | 1 | 2 |
| Pair verified / active | true / true | true / true |
| Body detected / persistent | false / false | true / true |
| Geometry penetration (m) | 0 | 0 |

The source is the real IsaacLab `ContactSensor.force_matrix_w`, not a commanded contact or inferred CoM proxy. Tick7011 records active history `[true,true,false]` and the two nonzero forces. Zero recorded geometry penetration does not cancel an independently verified persistent exact contact. Contact positions are sensor aggregate points; this report does not infer a unique obstacle face, footprint or motor cause from them.

At P09 entry6944, RR is **GROUND**, force4.04331N, front distance−219.60559mm and bottom-to-top clearance−49.82634mm; it is not already lifted. FL is **AIR**, force0, front+415.83342mm and clearance+8.43823mm. FR has actual obstacle force13.26219N and RL ground force16.28987N. FL's last non-AIR tick is6938; it remains AIR for the final73 ticks6939–7011, including the entire P09 window. RR is AIR in11 of the68 entry-inclusive samples6944–7011, but never meets hard qualification.

At tick7008, before any body-pair force, FL is AIR+166.21791mm while FR obstacle force11.79874N, RL ground13.16705N and RR ground3.71237N are real. At the terminal tick, FL is AIR+169.88762mm/load0; FR is AIR+1.94193mm/load0; RR is AIR−46.14172mm/front−234.94169mm/load0. RL has just0.299468N ground force; its normalized load fraction1.0 means all *remaining wheel load*, not a large absolute support force. Historical front-leg placement is not current top support. The central-body contact itself is real at this instant.

## Body versus mass-weighted CoM, and the fixed C84992 comparison

Axes below are recorded world coordinates. RPY is derived from the recorded wxyz quaternion using standard XYZ roll/pitch/yaw; velocities are recorded, not finite-difference estimates. CoM is the valid live13-body mass-weighted articulation CoM (2.93085852265kg), not base position.

| Sample | Base xyz (m) | CoM xyz (m) | RPY (degrees) |
|---|---|---|---|
| C87040 entry6944 | [0.630419,0.016504,0.076691] | [0.621731,−0.106031,0.150820] | [0.9095,0.4059,1.7396] |
| C87040 entry+64,7008; no body contact yet | [0.670798,0.007413,0.087183] | [0.638587,−0.132187,0.153407] | [12.2380,−12.9609,0.2861] |
| C87040 entry+67,7011; terminal | [0.672388,0.007155,0.087434] | [0.639174,−0.132793,0.153641] | [12.4498,−13.4330,0.1256] |
| C84992 entry6488 | [0.620046,0.014188,0.080595] | [0.610180,−0.106801,0.155226] | [−0.2720,−0.1638,1.5803] |
| C84992 entry+72,6560; no body contact | [0.672298,0.005059,0.089109] | [0.635332,−0.131900,0.158065] | [10.3553,−14.6432,0.0434] |

From C87040 entry to termination, base moves **+41.969mm x/−9.348mm y/+10.743mm z**, while mass-weighted CoM moves **+17.443/−26.761/+2.821mm**. The body origin rising does not establish clearance of its whole oriented collider. At entry, base velocity is[+0.048642,+0.003897,+0.005226]m/s and CoM velocity[+0.042572,+0.005875,+0.038679]. At7008 they are[+0.073130,−0.016840,+0.055367] and[+0.034616,−0.045486,+0.010108]. At7011 they are[+0.031194,−0.014462,−0.002597] and[−0.009390,−0.009002,+0.034155]; the contact-time velocity change is an observation, not proof of its initiating cause.

The preserved C84992 comparison is only the fixed6488/6560 window from `validation/20260906T2132007864973Z_ge99fde1b3e83_62520cdf88e549ae9929b9916bd92336`; no new rollout is synthesized. At its entry, RR is also GROUND (2.55105N/front−218.13064mm) and FL AIR (+12.09856mm/force0). At+72, RR remains GROUND2.83595N/front−228.92666mm, FL AIR+169.71954mm, FR obstacle12.47206N and RL ground13.50204N. Old+72 base/CoM velocities are[+0.068690,−0.006922,+0.046528]/[+0.034654,−0.031456,+0.005040]m/s. C87040 ends at+67, so it has **no +72 sample**; the endpoints are deliberately not presented as synchronized identical states. New entry CoM is4.406mm lower than old entry, with different pose and outputs. Old C84992 later earns real RR Q7149/C7184/P7191 but ends P12 incomplete; it is not a successful full-task control, and these differences do not isolate a causal parameter.

## Actual requested, mapped and dispatched control

The complete terminal canonical command receipt is below. Servo columns use degrees and wheel columns rad/s; raw is the unsquashed fixed-mean network output. `mapped nominal` is the recorded same-tick native-drive nominal, not a previous-dispatch substitute. All12 controller-drive-bias values and the final geometry-adjustment vector are0. The canonical drive is the dispatched command before the frozen adapter's physical signs/standing offsets/radian conversion; it is not a measured joint pose.

| Channel | Logical nominal | Raw | Residual | Mapped nominal | Actual canonical drive |
|---|---:|---:|---:|---:|---:|
| FL hip | 49.2 | 0.433844 | 9.804693 | 41.585550 | 51.390243 |
| FL knee | −13.4 | 0.189622 | 6.745742 | −12.15 | −5.404258 |
| FR hip | 0 | 0.157828 | 3.756742 | 0 | 3.756742 |
| FR knee | 31.1 | 0.513950 | 17.028606 | 21.1 | 38.128606 |
| RL hip | 31.2 | −0.358703 | −8.257714 | 38.7 | 30.442286 |
| RL knee | 0 | −0.153597 | −5.486415 | 0 | −5.486415 |
| RR hip | 51.95 | 0.313228 | 7.280896 | 51.95 | 59.230896 |
| RR knee | 0 | 0.122086 | 4.373405 | 0 | 4.373405 |
| FL wheel | 0 | −0.347006 | −0.200230 | 0 | −0.200230 |
| FR wheel | −0.63 | 0.331745 | +0.192053 | −0.63 | −0.437947 |
| RL wheel | 0 | 0.462253 | +0.259151 | 0 | +0.259151 |
| RR wheel | 0 | 0.172231 | +0.102329 | 0 | +0.102329 |

The actual audited float32 servo setter targets are `[0.898313880,−0.093853578,0.065449774,0.667654693,−0.534449697,0.098601162,−1.039411187,−0.072573595]`rad; physical wheel setter targets are `[+0.200230420,−0.437947094,−0.259150982,+0.102328859]`rad/s. Sign differences from the canonical wheel command are the verified frozen adapter mapping, not evidence of reversed policy credit. The equal-target/mapping audit passes on all ticks, including both collision ticks.

At entry6944, four nominal wheels are+0.260335rad/s; residuals `[−0.202616,+0.219560,+0.289957,+0.046201]` yield canonical targets `[+0.057719,+0.479896,+0.550293,+0.306536]`. The6945 handoff retains exactly the previous residual (roundoff only); later physical steps continue the same mapper, not a new state reset. At7008 the finite current advisory has FRwheel−0.63 and other wheels0; final FR residual+0.192053 does **not** cancel it. This request is below the old+.6 residual cap; the old cap's structural inability to reach zero against−0.63 is separate from whether this learned raw output requests sufficient cancellation.

C84992 at+72 also has nominal FR−0.63 and actual FR−0.496693 (residual+0.133307), without a body contact there. Its same-window residual servo outputs differ too: FR knee+11.061598 versus new terminal+17.028606, RR hip+4.073811 versus+7.280896, RR knee+0.352577 versus+4.373405. This comparison does not show that negative FR control alone caused the new collision, nor that changing front caps alone will prevent it. The newly selected/committed front-range version and its separate live probe are outside this a580fc2 evaluation and receive no performance credit here.

## Attribution boundary

The first unfinished P09 task remains a genuine RR active-lift/cross/place process; it is interrupted by a separate hard body collision after67 physics ticks. No RR qualification, landing, full-task success, suffix success or stable video is manufactured from partial progress, AIR labels, initial-clearance hints, historical FL placement, or software execution success. This report supplies measured interface and collision differences for subsequent natural-P01 work, not a new entry predicate, fixed support template, historical pose/clock gate, or causal explanation of the selected range change.
