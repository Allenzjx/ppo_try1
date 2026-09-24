# CP cooperative RR→RL window audit

- Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\video_eval\validation\20260924T0056424060031Z_g49eb23163a6e_c9d15196204e4b9aa94d71b1f7e309bd\source`
- Physical window: 50.06666666666667s → 85.108333s
- Result: success=False; termination=INCOMPLETE_CONTROLLER_BLOCKED
- Selected: 29 event/coarse rows; no interpolation or video processing.
- `*_ANY_SURFACE_BEARING` may be initial GROUND support; `*_TOP_BEARING` alone requires current TOP geometry/contact, support and positive force.

## First observed events

| Event | Tick | Time s | Phase |
|---|---:|---:|---|
| P01_ENTRY | 1 | 0.008333 | P01 |
| RR_ANY_SURFACE_BEARING | 1 | 0.008333 | P01 |
| RL_ANY_SURFACE_BEARING | 1 | 0.008333 | P01 |
| P02_ENTRY | 16 | 0.133333 | P02 |
| P03_ENTRY | 2688 | 22.400000 | P03 |
| P04_ENTRY | 2720 | 22.666667 | P04 |
| P05_ENTRY | 2728 | 22.733333 | P05 |
| P06_ENTRY | 4960 | 41.333333 | P06 |
| P07_ENTRY | 6008 | 50.066667 | P07 |
| P08_ENTRY | 6016 | 50.133333 | P08 |
| P09_ENTRY | 6024 | 50.200000 | P09 |
| RR_CARRY | 6024 | 50.200000 | P09 |
| RR_QUALIFIED | 6263 | 52.191667 | P09 |
| RR_CROSSED | 7002 | 58.350000 | P09 |

## Whole-window ranges

Values are descriptive extrema over the same sealed episode window. Hip-mount geometry is not contact/bearing; CoM→FR is the logged ≤0.5 s fixed-direction local projection, not total transfer or FR load.

| Quantity | Min | Max |
|---|---:|---:|
| CoM_toward_FR_local_m | -0.00725485405 | 0.0252882389 |
| FR_world_z_m | 0.206135231 | 0.23157404 |
| RL_clearance_mm | -51.5550312 | -49.91581 |
| RR_front_distance_mm | -219.20776 | 100.300673 |
| RR_gap_mm | -50.5391638 | 151.502295 |
| left_minus_right_mean_z_m | -0.0377270355 | 0.0284159327 |
| rear_minus_front_mean_z_m | -0.0431253021 | -0.00552520998 |

Full selected rows: [CP223616_cooperative_RR_RL_window_audit.json](CP223616_cooperative_RR_RL_window_audit.json)
