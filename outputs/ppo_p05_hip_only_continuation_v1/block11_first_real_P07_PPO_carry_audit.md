# block11_first_real_P07_PPO_carry_audit

PASS — actual new decisions/PPO/Adam: **128/1/20**; cumulative **218624/1673/33460**. Initialization: `checkpoint_policy`. Training lifecycle is not physical task success.

Lifecycle: `not_audited_first_batch_only`. Planned None; unconsumed None receive no learning credit. A legal update-boundary stop does not make a nonterminal partial episode a physical failure or success.

Phase inputs: {'P07': 1, 'P08': 1, 'P09': 126}. Frozen-prefix decisions: 693, all zero learning credit. Native-verified learner ticks: 1024; FL-assist-owned endpoints: 11. RR current-qualified/crossed-history/placed-history input counts: {'qualified_current': 118, 'crossed_history': 30, 'placed_history': 0}.

| Episode | Learner decisions | Physical seconds | Final/current stage | Actual result |
| --- | ---: | ---: | --- | --- |
| 0 | 128 | 54.733333 | P09 | audit cutoff, training may continue |

All five origins and entire four-event AUX ledger carry exactly: front96/96 + RR7/8 = mixed103/104; historical separate7/8 unchanged; new AUX0. Preedge migration origin remains216448/1656/33120.

Sealed389/raw12/conditional μ/σ/logp/value/reward/done match synchronous execution records; each original sample is used5 times. CPU logp max error 3.81469727e-06. Actual actor and Adam updates verify, all12 Adam states advance20; Identity and full RNG schema/CUDA count persist and RNG advances. Actual LR values are recorded per update; final1e-05. Official save/reload is recorded true and payload/sidecar hashes verify.

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000218496.pt` (6f7c572aaa81ea21407a4aac13809967885cfb9119014a78151b4bf0eff9e227). Target: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000218624.pt` (491c05efc862641d575332c77f1f6d1fadf982528eb401926b322786cdb84758). Run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1858483265170Z_g6ac7b553d792_9351c9cdf4924033bd5f181c1a1135dc`. Read-only CPU audit; no simulation, fit, production edits or checkpoint writes. Only the first sealed update was audited; later work is not credited here.
