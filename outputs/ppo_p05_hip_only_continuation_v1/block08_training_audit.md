# Block08 sealed genuine PPO audit

Actual front progression returned in this suffix course: first episode FL placement was recorded at physics tick **3560**, with assist **WAIT**, `initialized=0` and no owned channels. All 937 learner decision endpoints in that episode remained WAIT/uninitialized/unowned; the capture initialization flag is latched within an episode, so this is not evidence of a concealed earlier assist activation. FR and FL were placed; RR crossed at tick6763 but never placed. At the P09 incomplete terminal (82.3s), RR still had **42.169373mm** top gap. This is local task progress, not full success.

**Verified real 1024 decisions /8 PPO /160 Adam**, cumulative214400/1640/32800. Training lifecycle SUCCEEDED is not task success. Final checkpoint SHA `8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9`; actual source is mean-head AUX3 `945b05e2763396c0f83c582eb85d34d778e6fb8c74b041177c3bdea9f7077cd3`. Planned1024, unconsumed0.

- Credited input counts: {'P04': 2, 'P05': 232, 'P06': 257, 'P07': 8, 'P08': 1, 'P09': 524}; all unlisted P01–P13 phases have0. Full counts are in JSON.
- Two frozen source-checkpoint prefixes: 594 actions/4752 ticks, all policy_credit=false and excluded from PPO storage. Learner physics 8188 verified ticks.
- Learner assist-owned endpoints 0; initialized endpoints 0. Per-episode modes and first actual FL placement/event tick are retained in JSON. No assist or pure-policy label is inferred merely from a stage name.
- RR actual learner input counts {'qualified_current': 502, 'crossed_history': 389, 'placed_history': 0}; endpoint counts {'qualified_current': 503, 'crossed_history': 390, 'placed_history': 0, 'TOP': 0}. Retirement inputs {'predicate_active': 389, 'workspace_share_consumed': 389, 'effective_difference': 387}, endpoints {'predicate_active': 390, 'workspace_share_consumed': 390, 'effective_difference': 388}. This is recorded physical activation, not configuration presence.
- All8 actual actor updates have finite nonzero gradients; all12 Adam states +160. LR1e-5/Identity, original3 origins/migrations and complete front AUX96/96/old7/8 ledgers preserved; AUX added0. Saved checkpoint hashes/embedded metadata and official roundtrip verify.
- Raw389/raw12/μ/σ/logp/value/reward/done match sealed storage; each sample used5 times. CPU logp error max 3.81469727e-06. Projected/assisted actions remain separate from original Gaussian samples.

| Episode | Learner decisions | Total physical seconds | Credited seconds | Status | First unfinished stage | Result |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 0 | 937 | 82.300000 | 62.433333 | ended | P09 | INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 87 | 25.533333 | 5.800000 | partial; done=false | P05 | sampling boundary |

The incomplete first episode and nonterminal collection-boundary partial retain their actual meanings. Neither the uncredited front prefix nor a later suffix can be reported as full P01 PPO success. Evidence is confined to `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1301289433343Z_g5fd88852bf20_337c83c1b0214e02b62aa98b579e2812`,8 sealed rollouts1633–1640 and actual source/final checkpoints. No other historical runs were audited. CPU-only read audit, no checkpoint/runtime/frozen-helper changes or simulator; helper exits after writing reports.
