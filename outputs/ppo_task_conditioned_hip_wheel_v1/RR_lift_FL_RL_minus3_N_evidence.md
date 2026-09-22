# RR post-lift two-hip intervention: sealed diagnostic, not learned PPO

`RR_lift_FL_RL_minus3_N_20260921_01` naturally ended at tick 6872 / 57.2667 s with the original core result `SUCCESS`; the independent physical recorder reports its measured completion at 57.2583 s. Four legs have real placed events. This is an explicitly injected, finite direction diagnostic on the nominal baseline, **not** a trained PPO checkpoint or a PPO video success. Actor, critic, optimizer and normalizers are unchanged; new PPO decisions/updates are zero.

Raw source remains under `runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/RR_lift_FL_RL_minus3_N_20260921_01`. Compact measured evidence is `RR_lift_FL_RL_minus3_N_analysis.json`, computed by the output-only `rr_probe_readonly.py`. Successful B remains the previously identified 73.808333 s source; no original result is relabelled or overwritten.

## Same entry, actual two-channel intervention

The complete physical prefix is identical to B through tick 5440. RR initial lift (5428) and qualification (5434) precede the intervention and are identical to B. The first physical difference is tick 5441; the first changed dispatch is 5440→5448. This intervention cannot be credited with creating the initial RR lift.

Both selected REQUEST anchors are zero. FL hip ramps to -3 deg over 0.8 s; RL hip follows by two policy decisions / 0.1333 s, reaching -3 deg at tick 5552. Both continue through the same nominal source and mature mapper. At tick 5536, FL REQUEST/effective=-3 deg and RL=-2.89352 deg; at tick 5552 both=-3 deg, with final targets 35.6 and 25.2 deg. There is no selected-channel clipping or REQUEST/effective discrepancy, and native verification passes.

RR placement is followed for 24 decisions before release starts at tick 6376 / 53.1333 s. The 0.8 s release ends at tick 6472 / 53.9333 s; indexed follow-up ends at tick 6712 / 55.9333 s. The run then remains on the selected zero-residual nominal baseline until its physical endpoint. This is not a permanent negative-hip pose and not a recursively accumulated HISTORY offset.

## Carry and placement are preserved, but slightly slower

| Event | Post-lift diagnostic | Successful B |
|---|---:|---:|
| RR qualification | 45.2833 s, tick 5434 | Same |
| RR crossing / placed | 51.4750 s, tick 6177 | 51.2917 s, tick 6155 |
| RL qualification | 52.3167 s, tick 6278 | 52.0917 s, tick 6251 |
| RL crossing | 55.6833 s, tick 6682 | 55.4833 s, tick 6658 |
| RL placed | 56.2583 s, tick 6751 | 56.0583 s, tick 6727 |

There is no RR ground contact or current-lift revocation between qualification and placement. At 120 Hz, there are 524 AIR and 220 obstacle-pair-contact observations, with zero ground-pair contacts. The 93 decision endpoints in that event window are all current-lift-valid: 65 AIR and 28 TOP. The surface labels are 15 Hz evidence, not a claim that no brief edge state can occur between decision samples; real obstacle contact is not mislabelled as re-grounding. RR remains TOP at all 87 decision endpoints from placement through the diagnostic end, including release and the subsequent RL task. Nine of these post-placement endpoints do not mark current lift valid, but all remain physically TOP; historical placement and actual support are reported independently rather than conflated with a permanent AIR predicate.

## Measured geometric benefit and tradeoffs

The following event windows retain the complete real interval from common qualification tick 5434 through each run's RR placement, including waiting. RMS is `sqrt(time_mean((roll_rate^2 + pitch_rate^2)/2))` from wrapped 120 Hz Euler differences.

| RR qualification→placement metric | Post-lift diagnostic | Successful B |
|---|---:|---:|
| Duration, s | 6.191667 | 6.008333 |
| Body rate RMS, rad/s | 0.019326 | 0.019445 |
| Peak tilt, rad | 0.206303 | 0.210474 |
| Collider minimum world z, min / mean, mm | 90.507 / 94.511 | 86.544 / 88.195 |
| Conservative body-obstacle AABB separation, min, mm | 40.507 | 36.544 |
| RR authored mount world z, min / mean, mm | 195.010 / 197.015 | 193.333 / 195.224 |
| RR top-plane gap, max / mean, mm | 82.687 / 6.252 | 85.442 / 7.102 |
| FL verified TOP-bearing decisions | 81 / 93 | 84 / 90 |
| RR hip minimum positive hard-limit margin, deg | 88.263 | 88.271 |
| RR knee minimum negative hard-limit margin, deg | 20.526 | 20.524 |

Collider minimum rises about 3.963 mm at its minimum, and the sampled RR installation-point minimum rises 1.677 mm. This is a modest real geometric benefit, not a blanket increase in RR wheel clearance: maximum/mean wheel gap is slightly smaller and placement takes 0.1833 s longer. A single paired diagnostic with nearly equal RMS is not robust evidence of a superior learned strategy.

FL is AIR at the intervention entry; it is not presumed to support the CoM merely because the body moves toward it. For example at tick 5712, FL is truly TOP/support with 0.317 N measured bearing, body collider minimum=94.895 mm and RR mount z=196.496 mm. B at that same tick has 87.869 mm and 194.861 mm respectively. The small positive reaction confirms contact, not that greater force should be rewarded. In the interval after RR placement, FL is TOP-bearing in 85/87 decision endpoints, with two AIR endpoints during subsequent motion.

All 859 four-hip records have actual authored-frame measurements and aligned clocks; RR XYZ agrees exactly with the independent HeightDiagnostics stream. Only the comparable RR mount is compared with B, which has no four-hip stream. AABB separation is a conservative geometry bound, not exact mesh clearance; exact Cartesian feasibility remains unmeasured.

## Endpoint timing: do not claim a 16.5-second speed improvement

The diagnostic core stops on its first core terminal, whereas the published B video uses its independent physical evaluator's exact terminal tick. Task-window video has no extra post-success physical rollout: the recording observer stops immediately on common physical success. The saved B and CP185856 runtime hashes match for the task spec, execution profile, supervisor, backend, environment, physical recorder and video implementation. There is no extra mandatory home-angle or video-tail success gate to explain the difference.

The measured P13 entry is different. B first reaches the traversal event at 56.0583 s but cannot yet start the final controlled-observation window. At tick 6872 / 57.2667 s its measured maximum wheel speed is still 0.40650 rad/s, above the common 0.25 rad/s criterion, despite body linear/angular speeds of 0.02076 m/s and 0.04702 rad/s. Its one-second observation starts at 72.8083 s and its exact physical endpoint is 73.8083 s.

The post-lift diagnostic reaches traversal at 56.2583 s and immediately enters that same one-second observation. At tick 6752 its measured body speed is 0.04922 m/s (limit 0.05), angular speed 0.13527 rad/s (limit 0.30), maximum wheel speed 0.21595 rad/s (limit 0.25), and all four wheel targets are zero. It finishes the physical observation at tick 6871 / 57.2583 s and the core returns SUCCESS on the following tick. Both runs have transient speed excursions inside the fixed observation interval; the current task requires retained region and controlled conditions at the end, not uninterrupted strict-rest diagnostics. Both final `strict_recovery_quality.passed` values are false, while the common physical task result is true. No hidden strict-rest condition is added or relaxed here.

Thus the actual reason is a different measured stopping entry after the finite intervention, not omitted video padding or a faster RR crossing. Preserve both observed times, but this single manually injected diagnostic is not a formal deployment-speed, robust stability or PPO-improvement result.

Conclusion: retain this as a measured post-lift direction example with successful carry and release-to-RL continuity. Do not promote it to a fixed nominal script, an obligatory hip target or a positive imitation target. The contrasting preparation-time RL-only negative result demonstrates why entry timing, current contact and whole-body geometry must remain part of the learning problem.
