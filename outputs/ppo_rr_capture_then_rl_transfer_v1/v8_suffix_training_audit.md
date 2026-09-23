# V8 actual P07 nominal-prefix suffix training

Lifecycle: STOPPED_AT_VERIFIED_UPDATE_BOUNDARY (outer STOPPED_AT_VERIFIED_UPDATE_BOUNDARY).
Actual suffix additions: {'global_policy_decisions': 896, 'ppo_updates': 7, 'optimizer_steps': 140}; unconsumed requested decisions: 1152.
Actual ancestor total including the earlier512: {'global_policy_decisions': 1408, 'ppo_updates': 11, 'optimizer_steps': 220}.
Physical nominal prefix: 1290 decisions / 10320 ticks; PPO credit zero.

| Learner requested phase | Samples |
| --- | ---: |
| P01 | 0 |
| P02 | 0 |
| P03 | 0 |
| P04 | 0 |
| P05 | 0 |
| P06 | 0 |
| P07 | 2 |
| P08 | 2 |
| P09 | 431 |
| P10 | 1 |
| P11 | 1 |
| P12 | 459 |
| P13 | 0 |

Completed learner episodes: 1; trailing sampling partial: 44 learner decisions.
- Episode 0: 99.79166666666667s physical episode, 852 learner decisions, INCOMPLETE_CONTROLLER_BLOCKED, end P12, scope successful_nominal_initialized_suffix.

Nominal-prefix/suffix completion is not full natural-P01 PPO success. A sampling partial is not a task failure/success. All origins/migrations/AUX retained; main latest640 credit was not borrowed.
Verified final branch checkpoint: C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_step_000221952.pt
This bounded audit checks completed journals and checkpoint/sidecar/pointer metadata; it does not replay physics or independently recompute Gaussian likelihood.
