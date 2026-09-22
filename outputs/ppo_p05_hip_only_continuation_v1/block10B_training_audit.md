# block10B_training_audit

PASS — actual new decisions/PPO/Adam: **1536/12/240**; cumulative **218496/1672/33440**. Initialization: `checkpoint_policy`. Training lifecycle is not physical task success.

Lifecycle: `SUCCEEDED`. Planned 1536; unconsumed 0 receive no learning credit. A legal update-boundary stop does not make a nonterminal partial episode a physical failure or success.

Phase inputs: {'P04': 2, 'P05': 582, 'P06': 426, 'P07': 1, 'P08': 1, 'P09': 524}. Frozen-prefix decisions: 644, all zero learning credit. Native-verified learner ticks: 12284; FL-assist-owned endpoints: 486. RR current-qualified/crossed-history/placed-history input counts: {'qualified_current': 508, 'crossed_history': 380, 'placed_history': 0}.

| Episode | Learner decisions | Physical seconds | Final/current stage | Actual result |
| --- | ---: | ---: | --- | --- |
| 0 | 1145 | 97.766667 | P09 | INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 391 | 47.533333 | P06 | sampling boundary |

All five origins and entire four-event AUX ledger carry exactly: front96/96 + RR7/8 = mixed103/104; historical separate7/8 unchanged; new AUX0. Preedge migration origin remains216448/1656/33120.

Sealed389/raw12/conditional μ/σ/logp/value/reward/done match synchronous execution records; each original sample is used5 times. CPU logp max error 7.62939453e-06. Actual actor and Adam updates verify, all12 Adam states advance240; Identity and full RNG schema/CUDA count persist and RNG advances. Actual LR values are recorded per update; final1e-05. Official save/reload is recorded true and payload/sidecar hashes verify.

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000216960.pt` (33c8a37e670513273e4b072bb41845121eacab03cbcd4efa18dbf7edb6a254d0). Target: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000218496.pt` (6f7c572aaa81ea21407a4aac13809967885cfb9119014a78151b4bf0eff9e227). Run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1727064438736Z_g6ac7b553d792_4ae7abef7e2b4f91b5656e48cd35f791`. Read-only CPU audit; no simulation, fit, production edits or checkpoint writes. A nonterminal final episode is only a sampling-boundary partial, not an invented success/failure.
