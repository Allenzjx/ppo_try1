# block10A_training_audit

PASS — actual new decisions/PPO/Adam: **512/4/80**; cumulative **216960/1660/33200**. Initialization: `full_episode`. Training lifecycle is not physical task success.

Lifecycle: `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`. Planned 1024; unconsumed 512 receive no learning credit. A legal update-boundary stop does not make a nonterminal partial episode a physical failure or success.

Phase inputs: {'P01': 4, 'P02': 508}. Frozen-prefix decisions: 0, all zero learning credit. Native-verified learner ticks: 4090; FL-assist-owned endpoints: 0. RR current-qualified/crossed-history/placed-history input counts: {'qualified_current': 0, 'crossed_history': 0, 'placed_history': 0}.

| Episode | Learner decisions | Physical seconds | Final/current stage | Actual result |
| --- | ---: | ---: | --- | --- |
| 0 | 236 | 15.683333 | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 276 | 18.400000 | P02 | sampling boundary |

All five origins and entire four-event AUX ledger carry exactly: front96/96 + RR7/8 = mixed103/104; historical separate7/8 unchanged; new AUX0. Preedge migration origin remains216448/1656/33120.

Sealed389/raw12/conditional μ/σ/logp/value/reward/done match synchronous execution records; each original sample is used5 times. CPU logp max error 3.81469727e-06. Actual actor and Adam updates verify, all12 Adam states advance80; Identity and full RNG schema/CUDA count persist and RNG advances. Actual LR values are recorded per update; final1e-05. Official save/reload is recorded true and payload/sidecar hashes verify.

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_preedge_recovery_step_000216448.pt` (05c0b58bb73c05740d9deae389b86de0f12af525b3c248f09757c36d12c9bf57). Target: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000216960.pt` (33c8a37e670513273e4b072bb41845121eacab03cbcd4efa18dbf7edb6a254d0). Run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1716180883724Z_g6ac7b553d792_bf3d5fc6bf2b4f66ab4853a81a09699a`. Read-only CPU audit; no simulation, fit, production edits or checkpoint writes. A nonterminal final episode is only a sampling-boundary partial, not an invented success/failure.
