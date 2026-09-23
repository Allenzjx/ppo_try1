# Sealed v5 capture/follow evidence

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_capture_then_rl_transfer_v1\video_eval\validation\20260923T0718224682359Z_gc53119ab332f_886689c9a357416293bc8242856d0be6\source`
Physical success: `False`; 14595/120 = 121.625 s.

| Tick | Milestone | RR gap mm | Front mm | P07 CoM mm | P10 CoM mm |
|---:|---|---:|---:|---:|---:|
| 8280 | first_P07 | -51.01261 | -224.66218 | 0.0 | None |
| 8296 | first_P09 | -50.68024 | -222.70307 | 1.45673 | None |
| 8553 | RR_history_active_lift | -42.02869 | -175.73791 | 49.47627 | None |
| 9516 | first_RR_mode_BLOCKED | 27.8659 | -4.80652 | 143.63704 | None |
| 9555 | RR_history_front_edge_crossed | 27.51045 | 0.08852 | 147.21633 | None |
| 9556 | first_RR_mode_DESCEND | 27.65616 | 0.23377 | 147.32755 | None |
| 9773 | first_actual_wheel_envelope, first_actual_wheel_projection | 62.2811 | 28.21523 | 165.40574 | None |
| 10756 | first_knee_search | 47.47932 | 73.96913 | 178.65902 | None |
| 12267 | first_RR_mode_DESCEND_PROGRESS | 24.82331 | 52.52612 | 179.74037 | None |
| 13156 | first_progress_earned_reserve | 13.41626 | 50.07124 | 189.58724 | None |
| 13995 | terminal_minus_5s | 4.4836 | 33.61231 | 188.99901 | None |
| 14115 | terminal_minus_4s | 3.38117 | 31.00121 | 188.82012 | None |
| 14235 | terminal_minus_3s | 2.44921 | 28.35018 | 188.62862 | None |
| 14355 | terminal_minus_2s | 1.68549 | 25.67643 | 188.43257 | None |
| 14475 | terminal_minus_1s | 0.74363 | 22.97449 | 188.23329 | None |
| 14595 | actual_terminal_or_capture_end | 0.02696 | 20.2607 | 188.05659 | None |

Current contact/bearing is separate from placed history; actual sensor predicates are not relabelled.
Native global dispatch, episode pre-observation, and post-step ticks stay separate.
Fixed d uses true mass COM and entry receiver position, never rolling reanchored or moved-foot credit.
World-fixed diagonal progress contains forward traversal and can change interpretation with yaw; not lateral-load proof.
P10 missing means no RL-to-FR direction. Hip mount is exact height evidence or null, not upper-link origin.
No task success, policy improvement, mean attribution, GAE or learning credit is inferred.
Final video decision can be physically interrupted without step_info; no fake completion row is synthesized.
