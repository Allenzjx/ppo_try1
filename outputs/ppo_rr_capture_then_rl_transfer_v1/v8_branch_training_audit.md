# V8 actual ancestor-branch training audit

Training lifecycle: STOPPED_AT_VERIFIED_UPDATE_BOUNDARY; outer lifecycle: STOPPED_AT_VERIFIED_UPDATE_BOUNDARY.
Actual added: {'policy_decisions': 512, 'ppo_updates': 4, 'optimizer_steps': 80}. Unconsumed requested decisions: 1536.
Final counters: {'global_policy_decisions': 221056, 'ppo_updates': 1692, 'optimizer_steps': 33840}; checkpoint: C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_step_000221056.pt.

| Requested input phase | Credited samples |
| --- | ---: |
| P01 | 4 |
| P02 | 508 |
| P03 | 0 |
| P04 | 0 |
| P05 | 0 |
| P06 | 0 |
| P07 | 0 |
| P08 | 0 |
| P09 | 0 |
| P10 | 0 |
| P11 | 0 |
| P12 | 0 |
| P13 | 0 |

Completed episodes: 1; trailing sampling partial decisions: 277.
A sampling partial is not a task outcome. Full-episode natural P01; prefix credit zero. No latest-branch640 credit borrowed.

Terminal outcomes:
- Episode 0: 15.658333333333333s, 235 decisions, INCOMPLETE_CONTROLLER_BLOCKED, end phase P02.

All inherited migration/origin/AUX objects retained exactly; Identity unchanged. Complete RNG metadata and official save/reload evidence retained. Actual LR by update is in JSON.
This bounded audit verified checkpoint bytes/sidecar/pointer and completed journal counts. It did not independently recompute Gaussian likelihood or replay physics.
