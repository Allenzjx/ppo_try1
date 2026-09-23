# Sealed fixed CoM / RR response diagnostic

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_capture_then_rl_transfer_v1\validation\20260923T0617478341071Z_g57ae41eb4b57_c5a28ad8bd2940ef9b5ade6b5e343d29`
Outcome: `INCOMPLETE_CONTROLLER_BLOCKED`; full success `False`; duration `109.625` s.

| Post tick | Milestone | RR gap m | RR front m | P07 fixed CoM m | P10 fixed CoM m |
|---:|---|---:|---:|---:|---:|
| 8280 | first_P07_entry | -0.05101261400504103 | -0.22466218130755777 | 0.0 | None |
| 8553 | RR_history_active_lift | -0.04202869029126008 | -0.17573791401553507 | 0.049476274428976674 | None |
| 9516 | first_RR_assist_BLOCKED | 0.027865899261302965 | -0.004806515146974277 | 0.1436370423305426 | None |
| 9555 | RR_history_front_edge_crossed | 0.027510446372668793 | 8.851630520467779e-05 | 0.14721632654794234 | None |
| 9556 | first_RR_assist_DESCEND | 0.027656160447592573 | 0.00023377282452230475 | 0.14732754582972893 | None |
| 9773 | first_RR_source_finished_wheel_envelope, first_actual_wheel_projection | 0.06228109621941716 | 0.02821523291897421 | 0.16540574116186532 | None |
| 10756 | first_RR_positive_knee_search | 0.047479321373406505 | 0.07396912915539389 | 0.17865902248855145 | None |
| 12915 | terminal_minus_2s | 0.016066803612954164 | 0.053467932770963955 | 0.1888407463852761 | None |
| 13035 | terminal_minus_1s | 0.01501986741605564 | 0.05217463118862753 | 0.18957472497645558 | None |
| 13155 | terminal_or_evaluation_end, first_RR_combined_search_budget_exhausted | 0.013428596287508685 | 0.050088885854002285 | 0.1895869366332189 | None |

P07/P10 exact raw anchors and same-step wheel/servo/contact evidence are in JSON.

Fixed scheduler-entry reference, never rolling reanchored; not proof of physical transfer validity or causality.
Pre-dispatch context uses prior observed state; physical rows are after step. Native global dispatch tick is not episode tick.
Historical placed is not current bearing. Raw contact pairs and exact decision evaluator support are separate.
Decision endpoint current_legs may be absent at a non-15Hz milestone; never substituted with later state.
True hip mount and unpersisted native actuator IDs are null. Native targets are not measured angular velocity.
No actor forward, optimizer, GAE/reward credit, new physics, or task-acceptance recomputation.
Windows live Length=0 is not evidence of an empty stream; only parsed sealed lines are counted.
