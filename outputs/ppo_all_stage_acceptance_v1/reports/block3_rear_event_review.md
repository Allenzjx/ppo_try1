# Block3: closed P07 rear-event review

Run `20260909T0405165532990Z_ga8b148463115_14d21d83af52454f94eb0eabd4cc01ce`, runtime `a8b1484631156bad6e6ef0c96bd79a2fbf12e308`. **Two FALL and three BODY_COLLISION outcomes; no suffix/full success, no rear crossing or placement.** The retained evidence does not establish a new execution bug or a unique physical cause. It supports keeping failure classifications distinct rather than treating every failure as absent receiver support or insufficient lift.

## Fixed scope and conservation

Ran the existing report-only `diagnose_completed_rear_windows.py` **once**, after both final manifests agreed on `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`. Input audit was 10,478,061 bytes, below the explicit 32 MiB bound. The single pass read 128 rows, global **139905–140032**, and wrote the new exclusive `block3_rear_windows` directory. Analysis thereafter used only its summary and 114 retained unique decision rows; no raw/native trajectory, later/live run, PT, hashes, production imports, tests or simulations. Extraction wall time .1461174 s is not physics throughput.

Actual phases: **P07 5 / P08 31 / P09 92**; all other phases zero. **1004 physics ticks = 128×8−20**; recorded verified/native-effect counts both 1004, own-phase 994. The ten ordinary phase handoffs account for ten non-own hold ticks. All four state-write categories are zero in all 128 rows. No nonterminal tail remains.

All five credited segments are teacher-initialized P07 suffixes at **tick 5912 / 49.266666667 s**; none starts naturally from P01 under the current policy. Manifest physical-core minus credit is **4434 decisions / 35472 ticks** of excluded prefix/reset work; this subtraction is not a claim that only five prefix executions occurred. Two inherited RL initial events per credited episode are excluded. Current-policy rear counts are RR I=9/Q=5/Q-revoked=2 and RL I=2/Q=2/Q-revoked=2; C=P=0 for both. Some intermediate initial-event windows are omitted by the bounded selector, but aggregate event counts include them.

## Five actual outcomes

All terminal physical snapshots report `valid=true`. FALL uses `SAFETY_ABORT` / `PHYSICAL_SAFETY`; BODY_COLLISION uses `TASK_FAILURE_BODY_COLLISION` / `BODY_CONTACT`. These are actual task safety failures, not controller deadline incompletions or I/O failures.

| Ep | Global interval; decisions / ticks | P07 / P08 / P09 | Terminal tick / total episode s | Current-policy qualification sequence |
|---:|---|---|---|---|
| 0 | 139905–139918; 14 / 105 | 1 / 6 / 7 | 6017 / 50.141667; FALL | RR I5998, Q6003; no subsequent recorded revocation |
| 1 | 139919–139932; 14 / 109 | 1 / 4 / 9 | 6021 / 50.175; BODY_COLLISION | RL Q5934→GROUND revoke5942; RR I5995, never Q |
| 2 | 139933–139971; 39 / 310 | 1 / 1 / 37 | 6222 / 51.85; BODY_COLLISION | RR Q6076→GROUND revoke6093; Q6167 again |
| 3 | 139972–140021; 50 / 395 | 1 / 19 / 30 | 6307 / 52.558333; FALL | RL Q5942→revoke5962; RR Q6078→revoke6213; Q6288 again |
| 4 | 140022–140032; 11 / 85 | 1 / 1 / 9 | 5997 / 49.975; BODY_COLLISION | No new RR/RL I or Q |

Event ticks above are recorded event times; accompanying geometry/control is the first observing decision-end, not invented exact event-tick state. Q is an active-attempt measurement, not a guarantee of crossing or a phase-specific imitation achievement. RL Q in ep1/3 occurred during P08 whole-body motion, not a completed P12 course.

## Contact and receiver-bearing counterexamples

At every P08→P09 boundary, RR Q/C/P is false and FL is currently AIR/load 0 despite its inherited placement history. RR is GROUND at boundaries in ep0/2/4 (loads .045657/.090427/.106088), AIR in ep1/3. P08 therefore spans **1–19 decisions**, not an unconditional one-step transition in every episode. These facts do not themselves prove a transition-rule defect.

| Ep terminal | RR front / gap (mm), Q | FL current receiver state | Other current bearing supports | Body linear / angular speed magnitudes |
|---:|---|---|---|---|
| 0 FALL | −154.277 / −25.541, true | AIR; load 0 | FR TOP .50356; RL GROUND .49644 | .18830 m/s / .58430 rad/s |
| 1 BODY | −258.569 / −47.302, false | AIR; gap +219.293 mm; load 0 | FR TOP .42091; RL GROUND .57909 | .16028 / .66650 |
| 2 BODY | −27.210 / −9.233, true | **TOP; bearing .933945 N, load .036607** | FR TOP .57390; RL GROUND .38949 | .05894 / .26732 |
| 3 FALL | −61.960 / −13.809, true | AIR; gap +51.649 mm; load 0 | FR TOP .55588; RL GROUND .44412 | .43961 / 1.16878 |
| 4 BODY | −191.889 / −45.597, false | AIR; gap +175.070 mm; load 0 | FR TOP .51010; RL GROUND .48990 | .14592 / .61100 |

RR is AIR/load 0 at all five terminals. Load fractions are current normalized bearing, not force or historical placement.

* **Bearing is not a sufficient safety guarantee:** ep2 has verified FL obstacle TOP contact and positive bearing, three supports, comparatively small speed magnitudes, yet a recorded body-contact failure. No claim that this is stable or that FL bearing caused/prevented the collision.
* **Lift height is not forward clearance:** ep3 RR at global140002/t6160 is qualified AIR with **gap +51.691 mm but front −343.124 mm**; it later reaches GROUND and revokes Q at6213. Its closest sampled front is only −61.960 mm at the final row, with gap already −13.809 mm. Thus high lift does not establish carry/cross.
* **Q can be lost despite a bearing receiver:** ep1 RL Q is first observed at5936 with diagonal receiver FR supporting load .11095 / 2.86309 N. Its revoke is observed at5944 while FR still supports .29610 / 8.57430 N. This is not explained simply by receiver AIR.
* All five RR Q first-observing rows have FL AIR/load 0. Ep2 subsequently reacquires FL TOP; that later contact must not be retroactively assigned to its Q instant. Ep2's best sampled RR gap is −6.467 mm at6192, front −45.483 mm, with FL supporting load .13188. No sample establishes front crossing.

## Control authority and actual limiting

Order: `[FL hip, FL knee, FR hip, FR knee, RL hip, RL knee, RR hip, RR knee, FL wheel, FR wheel, RL wheel, RR wheel]`. Servo values below are canonical degrees relative to standing; wheel values rad/s. N is nominal, R is projected residual before any subsequent headroom correction, F is actual canonical final target. Raw sample and Gaussian mean/std are latent quantities, not physical requests. Live measured joint fields are pre-dispatch tracking context, not exact terminal post-action q.

| Ep | Terminal FL hip N / R / F (°) | Terminal RR hip / knee F (°) | Terminal wheel F: FL / FR / RL / RR |
|---:|---|---|---|
| 0 | 49.2 / +2.046 / 52.996 | +42.664 / −20.872 | −.723 / −1.360 / −.556 / +.084 |
| 1 | 49.2 / +25.718 / 73.630 | +62.349 / −12.004 | −.197 / −.950 / −.224 / +.886 |
| 2 | 38.6 / +7.023 / 50.716 | +48.840 / −58.000 | +.150 / −.762 / +.455 / +.900 |
| 3 | 38.6 / +17.634 / 59.397 | −15.487 / −50.204 | +.025 / −.351 / +.124 / +.838 |
| 4 | 49.2 / +19.544 / 72.835 | +62.083 / −13.571 | −.368 / −.791 / +.280 / +.781 |

The earlier 14-row ep0 review already demonstrates actual negative-residual cancellation of rising FL-hip nominal, then its release; this is not an unopposable nominal action. The later outcomes also are not all the same target pattern: ep2 RR final knee hits the protected −58° command bound, while ep3 has a negative RR-hip final target and less-negative knee. Native wheel targets use the known FL/RL sign reversals, not an execution discrepancy.

Among **retained rows only**, endpoint headroom clipping appears on RR-knee index7 in **21/31 ep2 rows and 1/44 ep3 rows**, none in ep0/1/4. There are zero recorded false mapping-match or setter-dispatch equality flags. Geometry branches are not uniformly inactive: ep2 retains 7 `projected_exact_forward`, 2 `degraded_bypass_infeasible_box_downward`; ep3 retains 3 `projected_relaxed_forward`, 6 degraded-bypass rows. These are explicit computational branches, not themselves proof of incorrect geometry.

Concrete limiting example **ep2 g139970/t6216**: RR hip/knee N `[−6.9,−37.8]`, R `[+5.05654,−29.66522]`, mapped nominal `[−8.15,−39.05]`, current geometry adjustment zero. Knee headroom changes effective residual to **−18.95**, yielding candidate **−58°** inside the original −60° limit with 2° reserve. RR hip candidate is **−3.09346°**, but previous actual is **+47.58974°**, so the final single-tick dispatch is **+46.33974°** (−1.25°), not the candidate. The large 49.4332° candidate/final difference is a logged final slew effect following earlier geometric adjustment; it is not evidence of a reset or a failed native setter. At the next terminal row, the geometry branch adds `[+54.51401,−17.70892]°` to RR hip/knee before further limits. N+R alone therefore cannot describe actual authority.

All retained rows explicitly lack a live P06 rolling/tail layer (`not_applicable_no_P06_layer`). These direct-P07 suffixes retain current nominal/tracking at handoff, not a full teacher command queue. No new unconditional entry restoration is proven by the windows. Do not equate this sampling initialization with natural-P01 continuity.

## Evidence boundary

The generated physical snapshot lacks the exact body collision pair/point/force history and the precise FALL base-height/gravity/RPY threshold branch. Thus the runtime's recorded safety classification is available, but an independent contact-force/pose reconstruction is **UNAVAILABLE**. No threshold was relaxed or failure reclassified.

Unlike Gaussian moments, the retained **role diagnostic** `transfer_direction_context` does explicitly contain `mass_weighted_com_position_w_m`, `mass_weighted_com_velocity_w_m_s`, `body_angular_velocity_w_rad_s`, and finite-window displacement/reference ticks. For example ep3 terminal has reference6247/window .5s, logged CoM velocity `[+.345407,+.110234,−.051311] m/s`, angular velocity `[−1.128158,−.296226,+.074575] rad/s`; ep2 terminal reference6217/window .041667s has `[+.059750,+.011699,−.004848] m/s` and `[−.026115,+.251995,+.085311] rad/s`. These are preserved diagnostic measurements, **not reconstructed exact raw base pose, RPY, momentum, or a proof of stable support**. Different windows are not a controlled comparison. No unique failure cause, performance change, new source fix or sampling gate follows from these five attempts.
