# Completed rear-event windows

Run: `20260909T0405165532990Z_ga8b148463115_14d21d83af52454f94eb0eabd4cc01ce`

Actual PPO interval 139905–140032: 128 decisions; 1004 physics ticks.
One audit pass; 10,478,061 bytes; 0.146s analysis (not simulation performance).

| Episode | Decisions | End phase | Terminal/reason | Windows / selected rows |
|---|---:|---|---|---:|
| 0 | 14 | P09 | True / FALL | 13 / 14 |
| 1 | 14 | P09 | True / BODY_COLLISION | 15 / 14 |
| 2 | 39 | P09 | True / BODY_COLLISION | 16 / 31 |
| 3 | 50 | P09 | True / FALL | 19 / 44 |
| 4 | 11 | P09 | True / BODY_COLLISION | 11 / 11 |

Policy rear events: `{"RL:I": 2, "RL:Q": 2, "RL:Q_revoked": 2, "RR:I": 9, "RR:Q": 5, "RR:Q_revoked": 2}`.

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
