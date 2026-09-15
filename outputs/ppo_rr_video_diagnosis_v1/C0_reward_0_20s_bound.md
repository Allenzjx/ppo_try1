# C0: recorded reward prefix, up to 20 s

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_fsm_reference_p09_stable_v2\video_eval\validation\20260914T0445447525570Z_g4b2c038887c4_7610e06fd2cc4efeaae4b47106f9529e\source\video_policy_decisions.jsonl`. Lifecycle: DIAGNOSTIC_FAILURE. No new simulation, checkpoint load, production changes or learning.

Read 122 returned decisions, ending at tick 976 (8.133333333 s). One subsequent row may be parsed only to close the prefix bound.

## Signed recorded totals

| Window | Decisions | Seconds | Potential | Time | Body attitude | Euler rates | Angular acceleration | Contact | Applied 1st | Applied 2nd | Terminal | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_returned_decisions_ending_by_bound | 122 | 8.133333333 | 0.293800748 | -0.162666667 | -0.010821611 | -0.000475790 | -0.005938633 | -0.000020098 | -0.014740015 | -0.022411975 | 0.000000000 | 0.076725958 |
| request_P01 | 2 | 0.133333333 | 0.130459163 | -0.002666667 | -0.000002528 | -0.000001659 | -0.000120901 | 0.000000000 | -0.001769215 | -0.001407585 | 0.000000000 | 0.124490608 |
| request_P02 | 120 | 8.000000000 | 0.163341585 | -0.160000000 | -0.010819082 | -0.000474131 | -0.005817732 | -0.000020098 | -0.012970800 | -0.021004390 | 0.000000000 | -0.047764649 |
| request_P03 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| FR_whole_body_initial_clearance_17_plus_minus_0p5s_complete_only | 9 | 0.600000000 | 0.455428174 | -0.012000000 | -0.000258397 | -0.000394572 | -0.000407933 | -0.000000085 | -0.006154279 | -0.005375330 | 0.000000000 | 0.430837579 |
| FR_qualified_measured_upward_lift_26_plus_minus_0p5s_complete_only | 10 | 0.666666667 | 0.460497811 | -0.013333333 | -0.000439066 | -0.000431266 | -0.001042441 | -0.000019466 | -0.006374069 | -0.005743130 | 0.000000000 | 0.433115041 |

Windows overlap: never sum table rows into a new episode total. P01/P02/P03 rows use the request phase; JSON identifies each handoff-spanning decision. Event windows include only fully contained decisions and list excluded boundaries. The C-to-P envelope includes padding, not an exact event-time integral.

## Event and weight evidence

- FR whole_body_initial_clearance: tick 17, 0.141666667 s, containing decision 3.
- FR first_history_Q: tick 26, 0.216666667 s, containing decision 4.
- FR qualified_measured_upward_lift: tick 26, 0.216666667 s, containing decision 4.
- all_returned_decisions_ending_by_bound: endpoint f 0.875000000–1.000000000, weight 0.200000000–0.300000000; recorded endpoint headroom clipping 0/122; FR normalized-load-invalid endpoints 0/122.
- request_P01: endpoint f 0.875000000–1.000000000, weight 0.200000000–0.300000000; recorded endpoint headroom clipping 0/2; FR normalized-load-invalid endpoints 0/2.
- request_P02: endpoint f 1.000000000–1.000000000, weight 0.200000000–0.200000000; recorded endpoint headroom clipping 0/120; FR normalized-load-invalid endpoints 0/120.
- FR_whole_body_initial_clearance_17_plus_minus_0p5s_complete_only: endpoint f 0.875000000–1.000000000, weight 0.200000000–0.300000000; recorded endpoint headroom clipping 0/9; FR normalized-load-invalid endpoints 0/9.
- FR_qualified_measured_upward_lift_26_plus_minus_0p5s_complete_only: endpoint f 0.875000000–1.000000000, weight 0.200000000–0.300000000; recorded endpoint headroom clipping 0/10; FR normalized-load-invalid endpoints 0/10.

## Verification and limits

Maximum reward reconstruction error 2.7755575615628914e-17; family-sum error 0; elapsed-tick error 0. Reward/reward_breakdown aliases are identical.

- No actor/critic/optimizer updates and no learned-motive inference.
- Transfer fraction, physical state and headroom evidence are decision endpoints, not a 120 Hz series.
- Cost components are logged 120 Hz integrated weighted/clipped costs, not raw RMS or unweighted physical rates.
- Complete-decision phase allocation is by request phase; handoff-spanning decisions are explicit.
- A physical-endpoint interrupted final step has no recorded reward; no terminal reward is fabricated.
- Motor torque/force saturation and per-channel reward clipping counts are not present and remain null.

Unreturned/missing-reward prefix intervals: [{"decision":123,"start_tick":976,"end_tick":983,"reason":"SAFETY_ABORT","logged_reward_available":false}]. No full-episode reward or full task-success claim. JSON contains exact decision membership, endpoint evidence, signed integrals, normalized weighted cost means, and boundary exclusions.
