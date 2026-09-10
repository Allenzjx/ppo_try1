# Completed rear-event windows

Run: `20260909T0327591294869Z_ga8b148463115_1996f4a3ba204ea7b30a7bf6b1038046`

Actual PPO interval 139521–139904: 384 decisions; 3071 physics ticks.
One audit pass; 28,036,545 bytes; 0.386s analysis (not simulation performance).

| Episode | Decisions | End phase | Terminal/reason | Windows / selected rows |
|---|---:|---|---|---:|
| 0 | 178 | P09 | True / FALL | 16 / 66 |
| 1 | 117 | P09 | True / FALL | 14 / 42 |
| 2 | 89 | P06 | False / None | 9 / 33 |

Policy rear events: `{"RL:I": 1, "RR:I": 3, "RR:Q": 2}`.

- Only current PPO credited rows/events. Inherited pre-credit events are counted separately, not learned achievements.
- Each event window is centered on its FIRST OBSERVED decision row, not an invented exact event-tick state.
- Geometry/controls are decision-end samples; reward covers that action interval; policy Gaussian moments/raw are pre-action.
- Native command physics_tick and episode physics_tick are preserved separately; no assumed equality or offset conversion.
- Extrema are over credited decision-end samples only, not all 120 Hz states. No minimum-gap/contact claims between samples.
- Native flags are recorded evidence, not independently re-executed validation; no full raw/native trajectory is read.
- old_distribution_mean/std are policy Gaussian moments, NOT encoded observations or body-pose measurements.
- Scalar body speeds, body_forward_m and world AABB do not determine RPY/gravity, signed velocity, or center of mass.
- No encoded-observation inversion. Even if optional observation-like fields appear, their schema is not assumed or decoded.
- No new success, safety reclassification, dynamics/causal proof, optimizer gate, or parameter recommendation.
