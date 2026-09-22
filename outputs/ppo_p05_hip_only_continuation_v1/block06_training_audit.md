# Block06 sealed genuine PPO audit

Training lifecycle **SUCCEEDED**, not physical task success. Source is the actual CP209920 AUX candidate (`3c45e7325210487431ebe8b5d1cbd92b480734e3d53829b04429869c4773e3e8`), not pre-AUX CP209920. Final checkpoint is `checkpoint_step_000211968.pt`, SHA256 `5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881`.

- Real new credit: **2048 decisions / 16 PPO updates / 320 Adam steps**; cumulative **211968 / 1621 / 32420**. Requested2048, unconsumed0.
- Natural P01, no prefix. Credited input counts: P01=14, P02=2034, P03=0, P04=0, P05=0, P06=0, P07=0, P08=0, P09=0, P10=0, P11=0, P12=0, P13=0. Sum2048.
- Actual physical ticks: 16366; every compact native-tick record is verified. All2048 detailed decision endpoints pass mapping/all12-mask checks; assist-owned endpoints 0, active-assist input observations 0. The compact per-tick records do not separately persist mask/assist fields, so no independent per-tick mask/owner count is invented.
- FR endpoint coverage: qualified history 1980, crossed history 0, placed history 0, current exact legal TOP 0, verified TOP bearing/support 0. History is not current contact.
- RR retirement learner inputs: predicate 0, workspace share consumed 0, nonzero actual potential change 0. Mere configured reward semantics is not physical coverage.
- Front rehearsal ledger **32 accepted/32 attempted** and earlier limited AUX **7/8** are fully equal to their source dictionaries. This PPO run adds **0 AUX**; all three original counter origins and historical migration dictionaries remain intact.
- Every update has finite nonzero gradients and changed actor parameters. All12 Adam states advance exactly320; LR 1e-05, Identity preserved, final ordinary save/reload recorded true. Exact sealed raw12/mean/std/logp/reward/done and389 input checks pass; each original sample used5 times. CPU Normal logp max difference 5.7220459e-06.

## Actual episodes

6 ended episodes and 1 collection-boundary partial episode(s); ended does not mean successful.

| Episode | Decisions | Seconds | Status | First unfinished stage | Result |
| --- | ---: | ---: | --- | --- | --- |
| 0 | 329 | 21.900000 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 326 | 21.725000 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 336 | 22.383333 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 3 | 325 | 21.641667 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 4 | 327 | 21.766667 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 5 | 325 | 21.633333 | ended | P02 | INCOMPLETE_CONTROLLER_BLOCKED |
| 6 | 80 | 5.333333 | partial | P02 | collection boundary, not terminal |

This block's current front/rear physical coverage is the table and counts above; earlier block03 local front/rear successes cannot substitute for it. The prior AUX32/32 is a lineage fact, not proof that natural deterministic P02 has recovered. The separate current deterministic evaluation remains authoritative for that question.

Evidence: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1041563379791Z_g5fd88852bf20_087b67f789494ca695e6d94de78c5623`;16 sealed rollouts1606–1621, matching likelihood ledgers/optimizer records, synchronous execution stream, completed episodes, actual source/final checkpoints. JSON records per-rollout, per-episode and all three origin details. CPU-only audit, no simulator/GPU/optimization/checkpoint writes or edits to bound v1 helpers. Helper exited after report creation.
