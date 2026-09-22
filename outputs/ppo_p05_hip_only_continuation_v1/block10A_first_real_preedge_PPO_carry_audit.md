# block10A_first_real_preedge_PPO_carry_audit

PASS — actual new decisions/PPO/Adam: **128/1/20**; cumulative **216576/1657/33140**. Initialization: `full_episode`. Training lifecycle is not physical task success.

Phase inputs: {'P01': 2, 'P02': 126}. Frozen-prefix decisions: 0, all zero learning credit. Native-verified learner ticks: 1024; FL-assist-owned endpoints: 0. RR current-qualified/crossed-history/placed-history input counts: {'qualified_current': 0, 'crossed_history': 0, 'placed_history': 0}.

| Episode | Learner decisions | Physical seconds | Final/current stage | Actual result |
| --- | ---: | ---: | --- | --- |
| 0 | 128 | 8.533333 | P02 | audit cutoff, training may continue |

All five origins and entire four-event AUX ledger carry exactly: front96/96 + RR7/8 = mixed103/104; historical separate7/8 unchanged; new AUX0. Preedge migration origin remains216448/1656/33120.

Sealed389/raw12/conditional μ/σ/logp/value/reward/done match synchronous execution records; each original sample is used5 times. CPU logp max error 3.81469727e-06. Actual actor and Adam updates verify, all12 Adam states advance20; Identity and full RNG schema/CUDA count persist and RNG advances. Actual LR values are recorded per update; final1e-05. Official save/reload is recorded true and payload/sidecar hashes verify.

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_preedge_recovery_step_000216448.pt` (05c0b58bb73c05740d9deae389b86de0f12af525b3c248f09757c36d12c9bf57). Target: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000216576.pt` (1226d449b30b8318123ae57b7f69d5c168435e578d094cdbda7a7955fbad78e9). Run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1716180883724Z_g6ac7b553d792_bf3d5fc6bf2b4f66ab4853a81a09699a`. Read-only CPU audit; no simulation, fit, production edits or checkpoint writes. Only the first sealed update was audited; later work is not credited here.
