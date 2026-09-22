# RR preparation RL negative-3: sealed negative diagnostic

Run `RR_prepare_RL_minus3_N_20260921_01` naturally sealed at 60.0 s / tick 7200 / 900 diagnostic decisions. It remains P09, with no RR crossing or placement. Original physical termination reason is null and the stop is the external diagnostic budget. This is incomplete, not a full-task success or an invented collision failure. Weights, normalizers and optimizer are unchanged; PPO credit is zero. Successful zero B remains the independently verified 73.808333 s / tick 8857 success.

Data: `RR_prepare_RL_minus3_N_analysis.json`. Analysis: `rr_probe_readonly.py`. Source: `runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/RR_prepare_RL_minus3_N_20260921_01`; reference is the already identified successful B source directory ending `1894338b350241fcbbac4868c5f7f3fb/source`. No source recording, production file or active harness was changed by this analysis.

## Matching prefix and actual intervention

N, residual, final target and phase are identical to B before tick 5168. All physical full12/body observations are identical through tick 5168; the first physical difference is tick 5169, immediately after the intervention starts. The first changed dispatch is tick 5168→5176 and changes only RL-hip REQUEST and its final target. This is a matched nominal prefix, not an already divergent learned-policy entry.

The zero REQUEST anchor is ramped to -3 deg over 12 decisions / 0.8 s, held until tick 6128 / 51.0667 s, released over 0.8 s to the selected zero-residual nominal baseline, then followed to the 60 s budget. The entire ramp-plus-hold is eight seconds, not a permanent new nominal pose. All 162 indexed intervention decisions have full 12-channel permission, native verification, no selected-channel clipping and exactly matching REQUEST/effective residual. At ramp completion source N and mapped N are 6.9 deg, final target 3.9 deg and actual RL hip 5.68997 deg. At hold completion N has naturally progressed to 28.2 deg, final target 25.2 deg and actual hip 26.55934 deg. The source schedule is not held at an old joint entry.

## Initial lift did not survive into carry

| Real event | RL negative-3 preparation | Successful B |
|---|---:|---:|
| P08 entry | 43.0667 s, tick 5168 | Same |
| First initial RR free rise | 45.2917 s, tick 5435 | 45.2333 s, tick 5428 |
| First qualified RR lift | 45.3500 s, tick 5442 | 45.2833 s, tick 5434 |
| First ground contact after qualification | 45.8667 s, tick 5504 | None before placement |
| Current lift / qualification revoked | Tick 5504 | None before placement |
| RR crossing and placement | Not observed by 60 s | 51.2917 s, tick 6155 |

The negative direction produces a real qualified rise, but RR returns to ground 0.5167 s later, still during the hold. Both the 120 Hz contact sensor and physical event history identify tick 5504; this is not a 15 Hz sampling artifact. At this tick RR hip is 47.91186 deg, knee -0.12325 deg, top-plane gap -50.23194 mm and bearing reaction 25.82649 N. RR has not crossed. The trajectory cannot be used as a successful carry/placement label.

At qualification, collider minimum z is higher than B (102.776 versus 95.461 mm) and peak tilt over P08→qualification is lower (0.16032 versus 0.18863 rad), but RMS rate is slightly higher (0.09661 versus 0.09510 rad/s) and qualification is later. This local geometric improvement fails to predict the subsequent loss of support and swing space.

## Geometry and task cost over the same clock window

Both columns below use 43.0667→60.0 s, 2033 physical observations and 255 decision samples. B proceeds to later tasks after RR placement; this is an equal-time outcome comparison, not an isolated same-phase stability experiment. The rate metric is `sqrt(time_mean((roll_rate^2 + pitch_rate^2)/2))` from wrapped adjacent 120 Hz Euler differences.

| Metric | Preparation probe | Successful B |
|---|---:|---:|
| Body rate RMS, rad/s | 0.110421 | 0.089791 |
| Peak tilt, rad | 0.421147 | 0.249635 |
| Collider minimum world z, min / mean, mm | 52.878 / 68.169 | 86.544 / 134.106 |
| Body-obstacle AABB separation lower bound, min, mm | 2.878 | 36.544 |
| RR authored mount world z, min / mean, mm | 98.640 / 121.168 | 193.333 / 218.438 |
| RR current-lift-valid decisions | 7 / 255 | 211 / 255 |
| RR AIR / GROUND / TOP decisions | 12 / 243 / 0 | 70 / 28 / 157 |
| FL current TOP bearing / AIR decisions | 0 / 255 | 191 / 63 |
| RR hip positive hard-limit margin, min, deg | 74.174 | 88.271 |
| RR knee negative hard-limit margin, min, deg | 59.605 | 20.524 |

RR mount falls from 190.722 mm at tick 5448 to 123.506 mm at tick 5504, then reaches 98.640 mm at its sampled minimum. This is a real installation-point height loss, not a fabricated base-height proxy. There is no joint hard-limit saturation: unused joint angular range does not guarantee Cartesian swing space. Exact Cartesian feasibility and exact mesh-obstacle clearance remain null. The AABB quantity is a conservative Euclidean separation lower bound, not exact mesh clearance.

Between qualification and first re-grounding, collider minimum z drops 30.040 mm while mass-weighted CoM z rises 4.873 mm. Thus CoM height or base origin alone would conceal this body's geometric deterioration. The new state-quality candidate can observe the resulting small obstacle separation; this does not establish that a fixed hip-angle target should be rewarded.

## CoM toward FL is not FL load-bearing

During each run's P08→qualification window, FL has zero obstacle contacts and remains AIR in every 120 Hz sample. CoM displacement projected along the window's initial geometric direction toward FL is +31.079 mm for the intervention versus +39.912 mm for B. These are real directional displacements while FL is **not bearing**. In B's subsequent qualification→placement window, FL becomes verified TOP support in 84 of 90 decisions; RR remains currently lift-valid in all 90, with 63 AIR and 27 TOP samples. TOP contact during controlled carry is not automatically loss of valid lift. The intervention never establishes that receiving contact in the observed window.

All 900 new four-hip geometry records are available and clock-aligned; RR XYZ agrees with the independent HeightDiagnostics stream to 0 m. B has only the older RR mount stream, so no unmeasured B FL/FR/RL mount values are invented. The reference manifest includes its real terminal tick 8857, which interrupts the final 8-tick decision; using only the last returned full decision at 8856 would incorrectly mislabel that successful B.

Conclusion: this finite preparation intervention is an informative negative example. It slightly alters the initial body state, then loses RR unloading/carry, lowers the actual RR installation point and leaves the task incomplete. Preserve the result; do not promote it into nominal, a hard phase gate or a positive imitation label. The separately scheduled post-lift two-hip diagnostic addresses a different timing hypothesis and must be evaluated on its own data.
