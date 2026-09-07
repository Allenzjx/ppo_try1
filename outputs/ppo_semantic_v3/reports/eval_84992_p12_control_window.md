# C84,992: fixed first-second P12 control window (read-only)

The observed RL motion is a small airborne excursion followed by ground recontact, not a qualified lift. In this window the requested RL residual is delivered without headroom clipping; nominal geometry makes **no actual target adjustment**. FL and RR remain genuinely obstacle-supported throughout. These observations do not identify one physical cause or show that the full P12 lift sequence lacks sufficient authority.

## Fixed scope and clocks

Run: `runs/ppo_semantic_v3/validation/20260906T2132007864973Z_ge99fde1b3e83_62520cdf88e549ae9929b9916bd92336`, deterministic C84,992 / e99fde1 / natural P01 / seed2001. The finalized result remains 1,353 decisions / 10,824 ticks / 90.2s, P12 `INCOMPLETE_CONTROLLER_BLOCKED`; this report does not reclassify it.

Read only raw observations **7,224–7,344** (121 state samples), native records **7,225–7,344** (120 P12 commands), and decision endpoints 7,224–7,344 (the entering P11 endpoint plus 15 P12 decisions). Thus the physical interval is exactly 60.2–61.2s. A native record at episode tick t describes the command whose resulting observation is raw t; its geometry/mapper input is the pre-step state. Native dispatch tick is t+179 in all 120 records. This is a measured clock relation for this episode, not a universal reset offset.

The earlier whole-episode diagnosis is [eval_84992_diagnosis.md](eval_84992_diagnosis.md). No new run, tensor load, checkpoint hash, optimizer operation, GPU/Python call, or production edit was performed.

## RL angles, actual target and measured excursion

Pairs below are **RL hip / knee, canonical degrees relative to captured standing pose**. `n` is the controller nominal, `m` the real mapper output before geometry/current residual, `r` the filtered policy request (also its headroom-effective value here), and `d` the final canonical target after the original final bound/slew. `q` is actual measured joint position. None of these canonical double-valued pairs is a native float32 readback.

| Episode tick | n | m | r = effective r | d | Actual q | RL clearance / front distance, mm |
|---:|---|---|---|---|---|---|
| 7,224 entry | 15.400 / 19.400 | 15.400 / 20.797 | −4.645 / −4.821 | 10.755 / 15.977 | 11.899 / 17.476 | −49.456 / −195.144, GROUND |
| 7,240 | 15.400 / 31.100 | 20.400 / 31.100 | −3.889 / −4.089 | 16.511 / 27.011 | 14.014 / 21.181 | −42.112 / −211.400, AIR |
| **7,254 AIR maximum** | 15.400 / 35.300 | 17.900 / 36.550 | −4.046 / −4.334 | 13.854 / 32.216 | 14.879 / 27.913 | **−33.887 / −232.675**, AIR |
| 7,264 | 13.200 / 35.300 | 11.950 / 36.550 | −3.949 / −4.214 | 8.001 / 32.336 | 11.810 / 30.379 | −42.207 / −236.441, AIR |
| 7,272 | 3.200 / 35.300 | 3.200 / 36.550 | −4.216 / −4.394 | −1.016 / 32.156 | 7.824 / 31.660 | −49.210 / −232.479, GROUND |
| 7,344 end | 0.500 / 35.300 | −0.750 / 36.550 | −3.661 / −4.306 | −4.411 / 32.244 | −4.796 / 32.259 | −48.581 / −207.211, AIR |

The first AIR stretch is 7,225–7,267. At 7,254 the bottom has risen **15.570mm relative to entry**, while its center has moved **37.531mm farther behind the front plane**. Hip q peaks at 15.275° at 7,248; at 7,254 hip velocity is −11.579°/s while knee velocity is +42.921°/s. At that instant the final target lies 1.026° below measured hip and 4.303° above measured knee. These are measured movement/target directions, not an inferred world-z Jacobian or a counterfactual outcome.

Nominal hip first decreases at 7,257. The first ground recontact is **7,268**, RL ground force 11.019484N, clearance −49.189mm/front −235.012mm, q=(10.169,30.991)°. That timing alone does not prove the hip suggestion caused recontact: base attitude, other joints, wheel forces, contact dynamics and target-tracking history are coupled.

Across the 120 ticks, RL is AIR94 / GROUND26 / obstacle-active0. Later short AIR stretches are 7,276–7,277 and 7,294; the last AIR stretch begins 7,297 and continues through 7,344. Initial-clearance evidence is earned at 7,236, absent again after recontact, then earned again at 7,303. All sampled hard RL Q/C/P bits remain false; both initial events are already recorded in the decision history, not promoted to hard qualification. The whole-episode report establishes that 7,254 is also the maximum AIR clearance over the complete P12; this fixed window does not rescan or reconstruct the remaining 29s.

## What the execution chain does and does not explain

- All 120 RL hip/knee filtered requests equal their post-mapper headroom-effective residuals exactly; **no servo index is headroom-clipped** in these native records. RL request ranges are hip [−4.645391,−3.535950]° and knee [−4.820659,−3.711859]°. Current P12 requested caps are ±24°/±36°: the observed requests use at most 19.36%/13.39% of those scales, not saturation. The 15 P12 decision raw means are approximately hip [−0.183014,−0.148411], knee [−0.126446,−0.103475]; the more negative 7,224 raw mean belongs to the incoming P11 decision. No stochastic sigma or sampled training action is being measured here.
- External bounded controller bias `c` is zero in both RL channels throughout. Mapper-native minus nominal ranges are hip [−1.25,+6.25]° and knee [0,+2.647435]°. This difference includes mapper slew/tracking history and must not be renamed external controller bias. At 7,240, for example, m_hip=20.4° versus n_hip=15.4°; after r_hip=−3.889°, d_hip=16.511°. This is an observable partial opposition between mapper offset and the negative policy request at that tick, not a steady-state gain model or proof of the run's failure mechanism.
- Geometry statuses are identity80, `degraded_bypass_infeasible_box_downward`13, and no geometry fields27. **All RL geometry adjustments g are exactly zero.** The first degraded interval is 7,259–7,268, then 7,277–7,278 and 7,295. The helper leaves raw m unchanged on infeasibility; therefore it did not actively cancel the RL policy or nominal in this interval, and it also did **not** provide a clearance guarantee. Eligibility uses the pre-step observation, so a status at t need not match post-step contact t.
- The actual final canonical target equals m+g+c+effective-r to numerical precision on 119/120 ticks per RL channel. Each channel has one additional final-slew-limited tick (maximum discrepancy 0.096605° hip / 0.270915° knee). The original hard/slew path still runs; these small differences are not headroom clipping.
- Native actual-vs-same-history-zero-current-policy differences are nonzero on 117/120 hip ticks and 109/120 knee ticks. Converted back by the rear sign, their ranges are about [−2.500001,0]° / [−1.999999,0]°. These are differences between **two final slewed targets sharing the same previous actual target**, not effective-residual ranges or the displacement of a separate zero-policy physical trajectory. A roughly −4° request can legitimately yield a smaller same-tick target difference under that audit definition.

The frozen adapter maps rear native radians as `radians(standing_deg − canonical_deg)`; both RL servo signs are −1. At 7,254, actual native float32 RL targets are **(−0.2449214905500412, −0.5594280362129211) rad**; same-tick zero-current-policy targets are (−0.2667381167411804, −0.5812446475028992) rad. Thus the negative canonical policy residual raises the numerical native radian target; it is incorrect to equate its sign directly with lifting or lowering in world z. Raw joint observations are converted back to canonical degrees by the sensor reader.

All **120/120** native records have verified mapping reconstruction, staged/dispatch equality, same-tick counterfactual and float32 target evidence; all four in-episode state-write sums are zero. This is control-delivery evidence, not simulation-independent physical causality.

## Wheels and actual support, not placed-history assumptions

Wheel order here is FL/FR/RL/RR; commands and measured speeds in this table are canonical rad/s.

| Tick | Nominal wheel suggestion | Final canonical command | Actual measured wheel speed |
|---:|---|---|---|
| 7,224 | −1.050 / 0 / 0 / 0 | −1.029 / −0.236 / −0.269 / +0.162 | −0.952 / −0.236 / −0.270 / +0.137 |
| 7,254 | −1.070 / 0 / 0 / 0 | −1.059 / −0.236 / −0.264 / +0.148 | −0.789 / −0.234 / −0.267 / +0.014 |
| 7,344 | −0.300 / −0.300 / −0.300 / −0.300 | −0.266 / −0.585 / −0.630 / −0.163 | −0.109 / −0.605 / −0.630 / −0.346 |

At 7,281 the new all-wheel negative nominal segment begins taking over: nominal (−1.045,−0.025,−0.025,−0.025), transitioning under the existing slew toward −0.3 each. Actual negative wheel targets and measured motion coexist with this weak RL attempt; ideal no-slip translation cannot be inferred, especially for airborne wheels. The native wheel-axis signs remain −/+ /−/+; e.g. the 7,254 physical float32 wheel targets are (+1.0592362881,−0.2359914482,+0.2642363608,+0.1478727013), not the canonical command row above.

FL and RR have verified active obstacle contact **120/120**: forces FL7.316897–16.548761N, RR6.893254–18.245841N. At the AIR maximum, FL=14.939277N and RR=16.437492N, RL=0 and FR=0. FR is AIR83 / obstacle-active37, intermittently contacting after 7,278. Support is therefore not a frozen three- or four-leg arrangement. The initial claim “no RL lift because RR was already unsupported” is contradicted in this particular window; this does not prove the available support geometry was sufficient for a successful lift.

RR center first goes behind the front at 7,281 but remains obstacle-contacting; at 7,344 it is front−12.456mm/clearance−2.997mm with13.955232N obstacle force. Historical placement is neither erased nor treated as current top-XY containment. Over entry→end base x shifts−7.941mm while COM x shifts+6.186mm and COM z+5.971mm; body and limb redistribution are significant even without assigning a unique cause.

## Finite-source and interpretation boundary

The protected Recording P12 source (steps21–24) itself begins with knee19.4→35.3°, then hip15.4→13.2→0.5° and a −0.3rad/s four-wheel segment at source time+0.466667s. Its later hip rise **1.6→31.2° is at source+2.733333–3.133333s**, followed by later carry/place segments. Those later physical states are **outside this requested first-second window**. The actual early n and wheel sequence is consistent with the continuous finite source and inherited wheel suggestion, not a frozen historical-pose reset. A knee/hip sign is not itself a world-down test.

Supported conclusions are narrow: there is a modest real RL lift-and-retreat attempt, small negative unsaturated policy requests, changing nominal/mapper targets, no active geometry correction, and continued real FL/RR support. This window does not demonstrate lost headroom, an undelivered action, absent RR support, or sufficient full-sequence target motion. It cannot distinguish a learned mean-action limitation from entry geometry, finite-source timing, contact dynamics or mapper feedback without a separately authorized comparison. No new cap, reference compensation, nominal pause, fixed entry posture or success gate is proposed here.

Source seams read: `infrastructure/command_batch.py` sign/readback mapping; `infrastructure/servo_target_mapper.py` sole advance and bounded tracking; `sensing/sensor_reader.py` canonical actual readback; `ppo/semantic_residual_adapter.py` mapper→geometry→headroom→final bound; `ppo/actuator_target_effect.py` actual/zero-current-policy native audit; `ppo/semantic_nominal_geometry.py` identity/infeasible fallback; `ppo/semantic_supervisor.py` continuous finite source; `configs/recording_motion_contract.json` P12 waypoints; v3 execution profile. No production changes were made. This report ends at the fixed 7,344 boundary.
