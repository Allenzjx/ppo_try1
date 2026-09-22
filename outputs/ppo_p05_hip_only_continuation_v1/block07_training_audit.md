# Block07 sealed real PPO audit

**Verified update-boundary stop**, not successful task completion. Actual source is AUX event2 checkpoint `checkpoint_aux_detP02_step_000211968_v2.pt` (SHA `27bc7cbdd98775c4602057d7776dd3becc9eef9b42ec62f4691d2dae7bb06d55`). Final is `checkpoint_step_000213376.pt` (SHA `8e5961e3e78260f12124447bff72a7459b7323d2da8f83cc467a59ddc9290e54`).

- Actual new credit: **1408 decisions / 11 PPO updates / 220 Adam steps**; total **213376 / 1632 / 32640**. Planned 2048; **640 unconsumed decisions receive no credit**. Stop-request hash and run binding verified.
- Natural P01, no prefix. Input counts: P01=12, P02=1396, P03=0, P04=0, P05=0, P06=0, P07=0, P08=0, P09=0, P10=0, P11=0, P12=0, P13=0. Sum 1408.
- 5 ended episodes, 1 nonterminal sampling-boundary partial episode; complete-task successes 0.
- 11252 actual ticks, all compact native records verified; all 1408 detailed endpoints map correctly with all 12 residual permissions. Assist-owned endpoints 0; active-assist inputs 0. Compact tick records do not independently store masks/owners.
- FR endpoint history: qualified 756, crossed 0, placed 0; actual legal TOP 0, verified TOP bearing/support 0. RR retirement learner predicate/consumed/effective-change counts: {'predicate_active': 0, 'workspace_share_consumed': 0, 'effective_potential_difference': 0}.
- All 11 updates have finite nonzero gradients and changed actor parameters; each of 12 Adam states advanced exactly 220. LR 1e-05, Identity unchanged. Every raw12/mean/std/logp/reward/done and 389 input matches sealed storage; each real sample used 5 times. CPU logp maximum difference 5.7220459e-06.
- Both front AUX events and the whole **64/64** ledger remain exactly equal to source; old **7/8** ledger and all three origins/migration dictionaries are preserved. This block adds **0 AUX**.
- Saved RNG retains Python/NumPy/CPU and 1 CUDA states, correct seed, and exact embedded/sidecar equality. Real training advances RNG; source equality is not required. No GPU RNG restore was attempted in this CPU audit. Actual normal save/reload is recorded true.

## Episodes

| Episode | Credited decisions | Seconds | Status | First unfinished task | Result |
| --- | ---: | ---: | --- | --- | --- |
| 0 | 234 | 15.575000 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 235 | 15.658333 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 324 | 21.541667 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 3 | 234 | 15.600000 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 4 | 319 | 21.258333 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 5 | 62 | 4.133333 | partial | P02 | sampling boundary; done remains false |

Evidence: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1156115839619Z_g5fd88852bf20_a9df5b5d15aa4500a30f12933b1c36fb`; sealed rollouts 1622–1632, matching optimizer and likelihood records, synchronous physical logs, actual source/final checkpoints. The final partial trajectory is retained with `done=false`, not converted into failure or success. Earlier local successes or AUX data do not substitute for current physical outcomes. No other runs were scanned.

Auditor reuses unchanged block05 checks, extending only its lifecycle guard in memory to accept the actual verified-boundary status. No production/frozen-helper edit, optimization, GPU, simulator or checkpoint write. CPU helper exits after report creation.
