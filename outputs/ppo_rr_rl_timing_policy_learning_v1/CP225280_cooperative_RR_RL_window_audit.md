# CP cooperative RR→RL window audit

- Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\video_eval\validation\20260924T0214316632402Z_g49eb23163a6e_530cc3c96d834dc3ba84b0b19d8db4f1\source`
- Physical window: 56.13333333333333s → 91.166667s
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
| P03_ENTRY | 2736 | 22.800000 | P03 |
| P04_ENTRY | 2768 | 23.066667 | P04 |
| P05_ENTRY | 2776 | 23.133333 | P05 |
| P06_ENTRY | 5168 | 43.066667 | P06 |
| P07_ENTRY | 6736 | 56.133333 | P07 |
| P08_ENTRY | 6744 | 56.200000 | P08 |
| P09_ENTRY | 6752 | 56.266667 | P09 |
| RR_CARRY | 6752 | 56.266667 | P09 |
| RR_QUALIFIED | 6995 | 58.291667 | P09 |
| RR_CROSSED | 7979 | 66.491667 | P09 |

## Whole-window ranges

Values are descriptive extrema over the same sealed episode window. Hip-mount geometry is not contact/bearing; CoM→FR is the logged ≤0.5 s fixed-direction local projection, not total transfer or FR load.

| Quantity | Min | Max |
|---|---:|---:|
| CoM_toward_FR_local_m | -0.00661711427 | 0.0291354493 |
| FR_world_z_m | 0.214133169 | 0.232409754 |
| RL_clearance_mm | -51.4729259 | -49.9166523 |
| RR_front_distance_mm | -221.157875 | 100.620869 |
| RR_gap_mm | -50.3313521 | 155.531741 |
| left_minus_right_mean_z_m | -0.0394994986 | 0.029614249 |
| rear_minus_front_mean_z_m | -0.0421900498 | -0.00218932927 |

Full selected rows: [CP225280_cooperative_RR_RL_window_audit.json](CP225280_cooperative_RR_RL_window_audit.json)
