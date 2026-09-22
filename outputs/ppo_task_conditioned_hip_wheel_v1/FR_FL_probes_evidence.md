# Sealed FR / FL direction probes: measured evidence

These are external diagnostic interventions, not learned PPO success or PPO updates. Both use deterministic CP185856 (`checkpoint_step_000185856.pt`, SHA256 `cfb5dd3f8304198b07fda3c72a01e54183a4a76e8ce8f8a98927693361155450`). Production controller/physics sources are unchanged. FR ended at 20.0 s and FL at 35.0 s by diagnostic budget, with no original task termination reason. Neither completed the full task. The later FL harness adds read-only four-hip geometry logging, so harness file hashes are not identical.

Source runs under `runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/`:

- `FR_RL_minus3_CP185856_20260921_01`
- `FL_minus3_CP185856_20260921_01`

Detailed output: `FR_FL_probes_sealed_pair.json`; reproducible read-only analysis: `probe_pair_readonly.py`. All data are real written physical observations or the same-dispatch native target ledger. No inferred historical pose or recomputed nominal target is substituted for dispatch evidence.

## FR: small single-pair benefit, slightly later capture

The FL run's entire FR prefix contains no intervention. The two runs have exactly equal full12 measurements, body position and orientation through tick 72; the first physical difference is tick 73, immediately after the FR probe begins. This makes the prefix a useful same-checkpoint comparison, not a formal learned-policy improvement result.

Metric window is natural P01 start through the first verified FR placed event. Euler rates use adjacent 120 Hz physical samples, wrapping angle differences; primary RMS is `sqrt(time_mean((roll_rate^2 + pitch_rate^2) / 2))`. Waiting time is included. Peak tilt is `hypot(roll, pitch)`.

| Metric | Finite RL negative-3 probe | No FR intervention |
|---|---:|---:|
| FR active lift / cross / placed tick | 23 / 1584 / 1596 | 23 / 1576 / 1588 |
| Event-window duration, s | 13.300000 | 13.233333 |
| Body rate RMS, rad/s | 0.1019883 | 0.1028328 |
| Peak tilt, rad | 0.2882108 | 0.2886648 |
| Collider minimum world z, min / mean, mm | 92.315 / 96.076 | 92.038 / 94.812 |
| Body-obstacle AABB separation lower bound, min, mm | 227.396 | 226.914 |
| RR mount world z, sampled min / mean, mm | 191.106 / 197.041 | 191.027 / 196.808 |
| FR top-plane gap after tick 72 and before cross, min / mean, mm | 31.710 / 87.417 | 31.710 / 90.199 |
| RR hip actual mean / max, deg | 4.43838 / 5.10463 | 4.44922 / 5.12057 |
| RR knee actual mean / min, deg | -2.96924 / -3.24236 | -2.99558 / -3.22965 |
| RR hip minimum positive hard-limit margin, deg | 129.89537 | 129.87943 |
| RR knee minimum negative hard-limit margin, deg | 56.75764 | 56.77035 |

RMS changes -0.821%, peak tilt -0.157%, capture time +0.0667 s. These are small changes in one pair, not evidence of robust superiority. There is no extra collider-height collapse in this window, RR joints do respond slightly, and hard-limit pressure is absent. RR mount samples are 15 Hz (200 versus 199 observations), not the 120 Hz rate-metric denominator.

The intervention anchors the previous filtered RL-hip REQUEST at -1.39177 deg, requests an additional -3 deg, and reaches -4.39177 deg. It ramps for 0.8 s from tick 72 / 0.6 s, holds through tick 672 / 5.6 s, then releases over 0.8 s toward the evolving live policy, with indexed follow-up through tick 1008 / 8.4 s. It is **not** a constant extra -3 deg throughout the 13.3 s FR approach. During the hold, source N=37.6 deg, same-dispatch mapped nominal=38.85 deg, final target=34.45823 deg, and actual RL hip at tick 672=34.34801 deg. The release returns to the live history-conditioned policy, not an unperturbed counterfactual trajectory.

## FL: direction reaches the actuator, but contact is still absent

FL active lift=1632, cross=2534; placed and first P06 remain null at tick 4200 / 35.0 s. No FL obstacle contact occurs during the intervention or subsequent follow-up. This is task incompletion at the diagnostic budget, not a detector-confirmed physical failure or a success.

The intervention starts at tick 2768 / 23.0667 s after the source endpoint. It anchors REQUEST=+1.302207 deg, requests -3 deg relative to that anchor, and therefore holds REQUEST=-1.697793 deg. Across all 117 indexed decisions, all 12 residual permissions are one, native verification passes, FL clipping count is zero, and maximum absolute REQUEST-versus-effective difference is zero.

| Dispatch end tick | Source N | Same-dispatch mapped N | REQUEST = effective | Final target | Post-step FL hip actual |
|---|---:|---:|---:|---:|---:|
| 2768, before intervention | 24.9000 | 22.4000 | +1.30221 | 23.70221 | 30.60076 |
| 2848, minimum gap | 22.8000 | 21.91797 | -1.59131 | 20.32665 | 21.05260 |
| 2864, ramp complete | 22.8000 | 22.16421 | -1.69779 | 20.46641 | 20.99436 |
| 3368, hold complete | 22.8000 | 22.21124 | -1.69779 | 20.51344 | 20.99098 |
| 3464, release complete | 22.8000 | 23.07703 | -0.36093 | 22.71610 | 22.47847 |
| 4200, final observation | 22.8000 | 23.45817 | +1.39244 | 24.85061 | 24.27903 |

All angles in the dispatch table are degrees. Mapped N is read directly from `policy_headroom_evidence.baseline_native_plus_controller_full12`, never inferred from an independent mapper replay.

| Interval (120 Hz ticks) | FL gap min / mean / max, mm | FL actual hip mean, deg | Obstacle-contact ticks |
|---|---:|---:|---:|
| Pre, 2744–2768 | 35.318 / 61.304 / 83.661 | 40.0546 | 0 |
| Ramp, 2769–2864 | 11.322 / 15.905 / 33.876 | 22.9382 | 0 |
| Hold, 2865–3368 | 11.682 / 14.956 / 16.154 | 21.0916 | 0 |
| Release, 3369–3464 | 15.486 / 16.845 / 19.236 | 21.5208 | 0 |
| Follow, 3465–4200 | 19.073 / 20.820 / 21.860 | 24.0219 | 0 |

Minimum gap is 11.32164 mm at tick 2848 / 23.7333 s, with FL hip=21.05260 deg and knee=-14.78850 deg. The direction is physically exercised and is not masked or clipped, but the resulting state still has air below FL. After release, the conditional policy returns to positive REQUEST and final gap is 20.0616 mm. Source N and mapper were already changing, and there is no same-entry uninjected FL counterfactual (the FR run ends at 20 s), so the full gap reduction must not be attributed solely to this intervention. P06 contact retention is unmeasured because P06 was never reached. This probe is not a sustained successful direction label.

## Four hip geometry: verified live authored frames, not invented heights

All four USD joint body0/localPos0 frames resolve to `/World/WLRRobot/base_link`. Startup world z is FL=0.211999098, FR=0.212008626, RL=0.206978463, RR=0.206988005 m. All 525 decision samples through tick 4200 contain four available mount positions and report unchanged/aligned physics clocks. RR mount XYZ agrees exactly with the independent original HeightDiagnostics stream on all matching samples (maximum error 0 m). The FR probe only has the older RR mount stream; missing other-leg measurements are not backfilled.

Collider world minimum z is actual collision geometry, not `base_z`. AABB-to-obstacle separation is a conservative Euclidean separation lower bound, not exact mesh clearance; global `minimum_z - obstacle_top_z` alone is not footprint-aware clearance. Exact mesh-obstacle clearance remains null rather than being fabricated. CoM used in the detailed JSON comes from the physical backend's live mass-weighted measurement, not a hip-height proxy.

## Existing successful zero B: shortest useful RR diagnostic horizon

From the single previously identified successful B record `runs/ppo_fl_capture_quality_v1/video_eval/prior_B/20260918T0555572376368Z_gf2e552406ea7_1894338b350241fcbbac4868c5f7f3fb/source/video_policy_decisions.jsonl`: P08 entry tick 5168 / 43.0667 s; RR initial lift tick 5428 / 45.2333 s; qualified lift tick 5434 / 45.2833 s; RR placed tick 6155 / 51.2917 s. A 60 s budget covers about 8.71 s after this B placement, 65 s about 13.71 s. These timing facts support a bounded zero-based RR diagnostic; they do not guarantee the current PPO follows the same schedule.
