# CP205824 deterministic RR loss: bounded native-chain diagnosis

Sealed natural-P01 run `20260922T0740416538018Z_ga802b24d78df_626f2bc2aa0e416d8c1545accf8e2585`, runtime `a802b24d78df`. Scope: actual P09 source start through first re-grounding (6392–6644), then source-stop/late-wait samples 7033/7040 and final 9840/9844. Sources are this run's matching native audit, capture-assist tick rows and policy decisions. No production/reward edits, physics, controller replay or historical scan.

## Result and first obstruction

**The RR lift/carry sequence executes, but loses clearance before crossing. It is not initially blocked.** P09 source starts at **6392 / 53.266667 s**, first completed native hip command 6393. Initial free rise is 6412; **6417 / 53.475 s is qualified lift**, not the first clearance event. RR peaks at **+57.605 mm top gap** at 6488, still **206.798 mm before the front edge**. All pending knee readiness checks pass, followed by the positive four-wheel group at local 232. RR recontacts ground at **6644 / 55.366667 s**, still 71.941 mm short of crossing.

This differs from CP203776's early knee-waypoint hold: here the authored hip return and wheel forward command actually run. Native commands are present, unchanged by a residual mask, and joints/wheels respond. The observed losing trajectory is the combined source, geometry controller and learned residual behavior. These logs do not isolate PPO as the sole cause or prove that zeroing one residual would succeed.

| Tick / s | Current RR geometry and source decision |
|---|---|
| 6417 / 53.475 | Qualified AIR, 8.585 mm free rise; top gap −42.445 mm, front −153.741 mm. FR TOP and RL ground provide two actual other supports; FL AIR is not counted as support. |
| 6472 / 53.933333 | Knee local80 permitted: current AIR gap +49.630 mm, bottom vz +0.136717 m/s. |
| 6488 / 54.066667 | Knee local96 permitted at peak gap +57.605 mm, front −206.798 mm, vz +0.004931 m/s. |
| 6512 / 54.266667 | Gap +43.090 mm while knee sequence continues; body z99.575 mm. |
| 6608 / 55.066667 | Hip return has advanced RR to front −96.088 mm, but gap already **−23.342 mm**. Current free lift still valid; source has not been paused. |
| 6624 / 55.200 | Local232 roll check passes: qualified AIR, free rise15.913 mm, gap−35.116 mm, front−80.665 mm. It does not impose an above-top static-pose gate. |
| 6625 / 55.208333 | First completed native four-wheel **+0.3 rad/s nominal** command. |
| 6644 / 55.366667 | RR GROUND 13.870 N, gap−50.348 mm, front−71.941 mm; current lift revoked. Body z109.168 mm, FL gap95.201 mm/0 N. |
| 7033 / 58.608333 | Authored four-wheel stop is present in native nominal. Residual wheels remain active. |
| 7040 / 58.666667 | First source hold, local648: `current_RR_over_top_before_late_reconfiguration`. RR is outside top XY, front−40.651 mm/gap−22.180 mm; no valid lift or actual TOP. |
| 9840 / 82.000 | Last full live decision: same local648 hold, RR ground+ambiguous obstacle pair, front−49.632 mm/gap−50.024 mm, no crossing/placement. |
| 9844 / 82.033333 | P09 `INCOMPLETE_CONTROLLER_BLOCKED`. The diagnostic changes to `no_live_physical_readiness` because the semantic task has terminated; that generic final label is **not** the first cause of the wait. |

## Same-tick RR hip/knee path

Canonical degrees. N comes from the matching native tick, not the next observation's nominal. Baseline is the recorded geometry-corrected mapper plus source-controller contribution. Current source-controller RR bias is zero in these samples; geometric corrections are not relabeled as PPO. Requested and headroom-effective PPO values match, with no final slew/clamp or assist ownership at these samples. Actual is the completed physical-step readback in the matching capture row.

| Tick | Source N | Raw mapper | Actual pre-PPO baseline | Effective PPO | Final | Actual |
|---:|---:|---:|---:|---:|---:|---:|
| 6393 | 1.600/0.000 | 1.250/0.000 | 0.000/0.000 | +13.166768/−13.361906 | 13.166768/−13.361906 | 12.984984/−13.127609 |
| 6417 | 23.800/0.000 | 18.750/0.000 | 8.750/0.000 | +13.134902/−13.330942 | 21.884902/−13.330942 | 16.241605/−13.209413 |
| 6488 | 55.600/−11.300 | 56.850/−11.300 | 46.850/−1.300 | +13.179562/−13.351336 | 60.029562/−14.651336 | 58.355816/−13.853605 |
| 6512 | 55.600/−25.100 | 56.850/−26.350 | 46.850/−16.350 | +13.199448/−13.334226 | 60.049448/−29.684226 | 59.487640/−24.235710 |
| 6608 | −2.600/−37.800 | −2.600/−39.050 | 7.400/−29.050 | +13.193640/−13.311077 | 20.593640/−42.361077 | 28.303359/−42.230956 |
| 6644 | −6.900/−37.800 | −8.150/−39.050 | 1.850/−29.050 | +13.181743/−13.328624 | 15.031743/−42.378624 | 15.681233/−42.214977 |
| 9844 | −6.900/−37.800 | −8.150/−39.050 | −8.150/−39.050 | +12.782457/−12.822956 | 4.632457/−51.872956 | 5.339381/−50.797072 |

The hip/knee baseline contains up to ±10° geometry correction; it later retires, explaining part of the post-ground final-target change. A naive `final−N` calculation would incorrectly merge that correction, mapper history and policy. RR hip returns while its positive learned residual stays near +13°, and knee residual remains near −13°; actual movement follows with transient lag. This is task-trajectory evidence, not a missing-joint write or proof of one causal angle.

## Four wheels: nominal → effective policy → final → actual

Canonical rad/s in **FL/FR/RL/RR** order. Canonical rotation signs are preserved, not assumed identical to wheel-center translation. Loaded FR/RL/RR rates may differ from targets; neither visual wheel-end motion nor ACK alone establishes traction.

| Tick | Nominal | Effective policy | Final | Actual angular velocity |
|---:|---|---|---|---|
| 6417 | [0,0,0,0] | [−.580201,+.140937,+.025741,−.129219] | same | [−.580283,+.071468,+.128176,−.130198] |
| 6624 | [0,0,0,0] | [−.577661,+.140757,+.027076,−.127291] | same | [−.577636,+.006809,+.306990,−.127336] |
| 6625 | [+.3,+.3,+.3,+.3] | [−.578105,+.140951,+.026802,−.127199] | [−.278105,+.440951,+.326802,+.172801] | [−.278124,+.276249,+.403176,+.172805] |
| 6644 | [+.3,+.3,+.3,+.3] | [−.579363,+.141384,+.026906,−.127628] | [−.279363,+.441384,+.326906,+.172372] | [−.279363,+.135911,+.459338,+.167487] |
| 9844 | [0,0,0,0] | [−.562765,+.140783,+.018980,−.121445] | same | [−.562175,+.131613,+.010694,+.008865] |

All four receive the authored forward increment at 6625. Policy reverses FL's nominal direction while modifying the other three, but FL is airborne and cannot be assigned imaginary traction. The loss of above-top clearance begins **before** that forward group, so the subsequent wheel command is not the first observed clearance loss. Final RR wheel tracking is poor while ground+obstacle constrained; delivery remains verified, not evidence of an RR mask.

## Is the held event justified?

Yes, the existing late-event condition has actual negative evidence here: RR is not over legal top XY, has no current valid lift, is below the top, and never earned crossing. The source event at local648 is the late whole-body FL/RL reconfiguration, not a forgotten RR-only lift command. Executing it solely because time passed would presume a capture-ready state not observed. This does not prove the rule globally optimal or forbid another learned whole-body route; it means this run does not demonstrate a faulty positive/negative decision by that guard.

At final9844, FR is verified TOP14.200 N and RL verified ground14.672 N; FL AIR0 N. RR has a ground+ambiguous obstacle contact with bearing verification false. The physical evaluator remains `valid=true`, `run_validity=VALID`, `physical_evidence_status=CONTACT_BEARING_UNVERIFIED`, no independent physical-safety termination. Ambiguous RR contact is not promoted to TOP, and no sensor failure is invented to explain the earlier clear AIR→ground transition.

All **253** native rows6392–6644 verify correct setter/dispatch mapping, previous-ACK reference, all-one12-channel masks, zero assist ownership, zero headroom clips and zero in-episode state writes. Capture assist is RELEASED. Boundary/source-adjacent policy histories are carried, not cleared (preceding raw samples match exactly; filtered requests differ only by float32 encoding precision). These are bounded recorded native-audit checks, not a full simulator replay.

**Final:** real initial lift but no maintained carry/crossing/placement. The first task obstruction precedes the late-source hold; no inspected implementation/mask/history/dispatch error was found. No reward, migration, runtime change or extra gate for continued genuine PPO training is proposed by this report.
