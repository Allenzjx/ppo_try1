# block11_training_audit

PASS — actual new decisions/PPO/Adam: **2048/16/320**; cumulative **220544/1688/33760**. Initialization: `checkpoint_policy`. Training lifecycle is not physical task success.

Lifecycle: `SUCCEEDED`. Planned 2048; unconsumed 0 receive no learning credit. A legal update-boundary stop does not make a nonterminal partial episode a physical failure or success.

Outer run lifecycle: `FAILED`. Original FAILED run manifest is retained: semantic run HEAD differs from its pinned revision. Training manifest is independently SUCCEEDED and all completed updates/checkpoint payloads are verified. Current commit f48cd2f09fe551e2c2d115c4b2fdc4d1f937467f changes only .gitignore/outputs; production Git diff is empty and every pinned runtime file still matches. The checkpoint remains a valid preserved training source, but current HEAD/new-semantic resume requires explicit contract resolution; no bypass or manifest rewrite was performed.

Phase inputs: {'P07': 4, 'P08': 4, 'P09': 2040}. Frozen-prefix decisions: 2772, all zero learning credit. Native-verified learner ticks: 16372; FL-assist-owned endpoints: 44. RR current-qualified/crossed-history/placed-history input counts: {'qualified_current': 1976, 'crossed_history': 1606, 'placed_history': 0}.

| Episode | Learner decisions | Physical seconds | Final/current stage | Actual result |
| --- | ---: | ---: | --- | --- |
| 0 | 526 | 81.233333 | P09 | INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 526 | 81.233333 | P09 | INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 526 | 81.233333 | P09 | INCOMPLETE_CONTROLLER_BLOCKED |
| 3 | 470 | 77.533333 | P09 | sampling boundary |

All five origins and entire four-event AUX ledger carry exactly: front96/96 + RR7/8 = mixed103/104; historical separate7/8 unchanged; new AUX0. Preedge migration origin remains216448/1656/33120.

Sealed389/raw12/conditional μ/σ/logp/value/reward/done match synchronous execution records; each original sample is used5 times. CPU logp max error 5.7220459e-06. Actual actor and Adam updates verify, all12 Adam states advance320; Identity and full RNG schema/CUDA count persist and RNG advances. Actual LR values are recorded per update; final1e-05. Official save/reload is recorded true and payload/sidecar hashes verify.

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000218496.pt` (6f7c572aaa81ea21407a4aac13809967885cfb9119014a78151b4bf0eff9e227). Target: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_p05_hip_only_continuation_v1\checkpoints\history\checkpoint_step_000220544.pt` (56239937cd0aaccdc8b7ea36c6266a41b3bed9df67b00f84a06806d2a6fa09b7). Run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1858483265170Z_g6ac7b553d792_9351c9cdf4924033bd5f181c1a1135dc`. Read-only CPU audit; no simulation, fit, production edits or checkpoint writes. A nonterminal final episode is only a sampling-boundary partial, not an invented success/failure.
