# B0_common976: recorded reward prefix, up to 8.133333333333333 s

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_fsm_reference_p09_stable_v2\video_eval\prior_B\20260914T0422580971667Z_g4b2c038887c4_52ea704dca9c4cd19b12df41e9897e8d\source\video_policy_decisions.jsonl`. Lifecycle: DIAGNOSTIC_FAILURE. No new simulation, checkpoint load, production changes or learning.

Read 122 returned decisions, ending at tick 976 (8.133333333 s). One subsequent row may be parsed only to close the prefix bound.

## Signed recorded totals

| Window | Decisions | Seconds | Potential | Time | Body attitude | Euler rates | Angular acceleration | Contact | Applied 1st | Applied 2nd | Terminal | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_returned_decisions_ending_by_bound | 122 | 8.133333333 | 0.532875632 | -0.162666667 | -0.038108920 | -0.000542194 | -0.006501020 | -0.000011567 | -0.010348619 | -0.016778141 | 0.000000000 | 0.297918504 |
| request_P01 | 2 | 0.133333333 | 0.209739951 | -0.002666667 | -0.000003939 | -0.000009428 | -0.000123714 | 0.000000000 | -0.000494792 | -0.000147569 | 0.000000000 | 0.206293842 |
| request_P02 | 120 | 8.000000000 | 0.323135681 | -0.160000000 | -0.038104981 | -0.000532765 | -0.006377306 | -0.000011567 | -0.009853827 | -0.016630571 | 0.000000000 | 0.091624662 |
| request_P03 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| FR_whole_body_initial_clearance_15_plus_minus_0p5s_complete_only | 9 | 0.600000000 | 0.501182415 | -0.012000000 | -0.000194523 | -0.000112547 | -0.000254550 | 0.000000000 | -0.002566319 | -0.001213542 | 0.000000000 | 0.484840933 |
| FR_qualified_measured_upward_lift_24_plus_minus_0p5s_complete_only | 10 | 0.666666667 | 0.505976146 | -0.013333333 | -0.000282935 | -0.000157596 | -0.000354642 | 0.000000000 | -0.002604167 | -0.001289236 | 0.000000000 | 0.487954239 |

Windows overlap: never sum table rows into a new episode total. P01/P02/P03 rows use the request phase; JSON identifies each handoff-spanning decision. Event windows include only fully contained decisions and list excluded boundaries. The C-to-P envelope includes padding, not an exact event-time integral.

## Event and weight evidence

- FR whole_body_initial_clearance: tick 15, 0.125000000 s, containing decision 2.
- FR first_history_Q: tick 24, 0.200000000 s, containing decision 3.
- FR qualified_measured_upward_lift: tick 24, 0.200000000 s, containing decision 3.
- all_returned_decisions_ending_by_bound: endpoint f 0.222222222–1.000000000, weight 0.200000000–0.822222222; recorded endpoint headroom clipping 0/122; FR normalized-load-invalid endpoints 0/122.
- request_P01: endpoint f 0.875000000–1.000000000, weight 0.200000000–0.300000000; recorded endpoint headroom clipping 0/2; FR normalized-load-invalid endpoints 0/2.
- request_P02: endpoint f 0.222222222–1.000000000, weight 0.200000000–0.822222222; recorded endpoint headroom clipping 0/120; FR normalized-load-invalid endpoints 0/120.
- FR_whole_body_initial_clearance_15_plus_minus_0p5s_complete_only: endpoint f 0.875000000–1.000000000, weight 0.200000000–0.300000000; recorded endpoint headroom clipping 0/9; FR normalized-load-invalid endpoints 0/9.
- FR_qualified_measured_upward_lift_24_plus_minus_0p5s_complete_only: endpoint f 0.875000000–1.000000000, weight 0.200000000–0.300000000; recorded endpoint headroom clipping 0/10; FR normalized-load-invalid endpoints 0/10.

## Verification and limits

Maximum reward reconstruction error 2.7755575615628914e-17; family-sum error 0; elapsed-tick error 0. Reward/reward_breakdown aliases are identical.

- No actor/critic/optimizer updates and no learned-motive inference.
- Transfer fraction, physical state and headroom evidence are decision endpoints, not a 120 Hz series.
- Cost components are logged 120 Hz integrated weighted/clipped costs, not raw RMS or unweighted physical rates.
- Complete-decision phase allocation is by request phase; handoff-spanning decisions are explicit.
- A physical-endpoint interrupted final step has no recorded reward; no terminal reward is fabricated.
- Motor torque/force saturation and per-channel reward clipping counts are not present and remain null.

Unreturned/missing-reward prefix intervals: []. No full-episode reward or full task-success claim. JSON contains exact decision membership, endpoint evidence, signed integrals, normalized weighted cost means, and boundary exclusions.
