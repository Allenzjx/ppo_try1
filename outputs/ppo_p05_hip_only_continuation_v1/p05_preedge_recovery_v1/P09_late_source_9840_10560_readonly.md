# Current P09: actual source reconfiguration and residual tracking

Fixed same-run window: ticks 9840–10560 (82–88 s), 721 physical samples / 91 complete decision endpoints. No policy forward, optimizer, AUX, new physics or production changes.

## Event ordering

- RR first crosses at observation **9965 / 83.041667 s**. Saved scheduler diagnostic records `late_group_start_tick=9965` with current RR front +0.642 mm and gap 21.216 mm.
- First dispatch of that late source is **9966 / 83.050 s**: FL nominal hip/knee **38.6/−13.4 → −18.5/−31.4°**, RL **28.2/0 → 15.4/19.4°**. FR and RR nominal joint requests do not change. Wheels **[.3,.3,.3,.3] → [−1.07,0,0,0]**, then FL's authored wheel pulse ends at dispatch **10182 / 84.85 s**. These are source-owner changes, not raw PPO jumps or a mask.
- A separate existing geometry-advisory transition occurs first: RR hip/knee mapped-source correction is **+10/+10°** through 9960 and absent from dispatch **9961**. The actual observation 9960 is already `within_top_xy=true` (front −4.956 mm); `capture_nominal_geometry_context` returns no correction inside that existing XY region. This code-based explanation is separate from the later true crossing and source reconfiguration.

## Four-leg same-dispatch decomposition

Pairs are hip/knee degrees. M = mature mapper output; G = geometry-corrected baseline; R = requested→effective policy residual. Final and actual are saved command and measured state, not source interpolation.

| Tick / leg | Source N | M → G | R requested → effective | Final | Actual | Headroom |
|---|---|---|---|---|---|---|
| 9872 FL | 38.600/-13.400 | 38.600/-12.150 → 38.600/-12.150 | -8.880/-30.897 → -8.880/-30.897 | 29.720/-43.047 | 29.561/-43.280 | not clipped |
| 9872 FR | 0.000/31.100 | 0.000/30.998 → 0.000/30.998 | 9.950/-50.550 → 9.950/-50.550 | 9.950/-19.552 | 10.640/-19.545 | not clipped |
| 9872 RL | 28.200/0.000 | 28.200/0.000 → 28.200/0.000 | -11.357/2.751 → -11.357/2.751 | 16.843/2.751 | 18.278/2.793 | not clipped |
| 9872 RR | -6.900/-37.800 | -8.150/-39.050 → 1.850/-29.050 | 15.490/-15.822 → 15.490/-15.822 | 17.340/-44.872 | 17.023/-44.697 | not clipped |
| 10448 FL | -18.500/-31.400 | -18.634/-41.400 → -18.634/-41.400 | -8.501/-30.839 → -8.501/-16.600 | -27.136/-58.000 | -27.647/-58.461 | FL knee clipped at reserved −58° |
| 10448 FR | 0.000/31.100 | 0.000/30.998 → 0.000/30.998 | 9.822/-49.601 → 9.822/-49.601 | 9.822/-18.603 | 10.331/-18.835 | not clipped |
| 10448 RL | 15.400/19.400 | 15.126/18.973 → 15.126/18.973 | -10.536/2.654 → -10.536/2.654 | 4.590/21.626 | 5.717/21.505 | not clipped |
| 10448 RR | -6.900/-37.800 | -7.277/-38.606 → -7.277/-38.606 | 15.511/-15.197 → 15.511/-15.197 | 8.233/-53.803 | 7.967/-53.589 | not clipped |

FL hip's large sign change is primarily the source reconfiguration; its negative residual then adds roughly −8.5°. FL knee source/mapped lowering combines with about −30.8° requested policy residual; headroom limits its effective residual to −16.6° at 10448, producing final −58° (actual −58.461°, still above the −60° hard limit). This is not a missing actuator write. FR knee residual reverses the positive source knee request; RR hip residual reverses its negative source hip, while RR knee adds further negative demand. These signs describe actual composition, not an untested claim that one joint direction must fix capture.

The final-target tracking is transiently different during reconfiguration: at 10000 FL hip actual is −0.009° versus final −13.956°, while at 10448 its error is −0.512°; all four-leg endpoint values are above. The full JSON includes onset 9965/9966, wheels, same-dispatch post-mapper additions and reserved margins. All 12 masks remained one and native setter/mapping audits passed. The FL capture assist was **RELEASED for all 721 ticks**.

## Measured body / RR response

| Tick | Body z (mm) | Body pitch (deg) | RR gap (mm) | RR front (mm) |
|---:|---:|---:|---:|---:|
| 9872 | 72.356 | -2.660 | 20.812 | -15.003 |
| 9965 | 72.080 | -2.733 | 21.216 | 0.642 |
| 9966 | 72.166 | -2.714 | 21.738 | 2.598 |
| 10000 | 83.226 | 1.556 | 64.035 | 50.579 |
| 10448 | 74.157 | 1.883 | 65.851 | 46.933 |
| 10560 | 74.818 | 1.729 | 64.498 | 45.388 |

Body z initially rises rather than steadily sinking: window range 71.947–83.615 mm. Pitch changes from about −2.66° to +1.88°, and RR gap increases from about 20.8 to 65.9 mm. These coupled actual changes accompany the geometry-correction exit, source reconfiguration and continuing residual; this read-only trace cannot isolate a single causal joint.

Height recovery is present and permitted at **91/91 endpoints**, but the configured `RL_preparation_minus3_v1` supplies only the P08 RL source preparation reduction 31.2→28.2°. Both configured post-lift recovery amounts are **0°**, so all logged recovery offsets are zero. The later P09 owner replaces that earlier RL source contribution; the height helper is neither silently disabled nor providing a new nonzero post-lift body raise.

At 10560, P09 remains incomplete: RR history crossed and current Q is valid (91/91 endpoints), but RR is AIR / no contact and **not placed**. The first remaining task is controlled RR capture/placement, not qualification or crossing. No task success or stabilization gain is claimed; this is frozen evaluation, not new PPO learning.
