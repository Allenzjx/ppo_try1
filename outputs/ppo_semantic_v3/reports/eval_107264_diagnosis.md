# C107264 final — FL crossed but never captured; P05 incomplete

Run `runs/ppo_semantic_v3/validation/20260907T0242264425838Z_gf1a9bbf650b1_be40bfe9223b49a4bec2468f632992d7`, runtimef1a9bbf650b1, saved107264/N1seed2001/naturalP01/fixed deterministic mean. Root confirmed session16731 exit0/PID gone. Final lifecycleSUCCEEDED is execution completion, **not task success**.

Actual **666 decisions /5328 physics ticks /44.4s**, P05 `INCOMPLETE_CONTROLLER_BLOCKED`, stageage30s. Taskfalse, physicalvalidtrue/nullhardfailure, optimizer updates0, `window_ended_before_task_terminal=false`. This is an actual task noncompletion, not collision, I/O/codec failure or reclassification of old results.

Scope: final manifests, one complete666-row decision ledger, one whitelist scan of5328 native rows and terminal raw5328. No production/tests, hashes, PT/Python/GPU/Isaac or subsequent training were used. The master receives an append only; its existing history and block29 ledger are preserved.

## Phase clock and first missing task

Source-phase vector P01–P13 is **[1,210,4,1,450,0,0,0,0,0,0,0,0]**. P02/P03/P04/P05 enter at8/1688/1720/1728 ticks (.066667/14.066667/14.333333/14.4s). All four handoffs have terminationnull/bootstraptrue/time_outsfalse. No P06–P13 samples occur.

| Leg | Qualified lift tick | Front crossing tick | Placed tick | Terminal state / load |
|---|---:|---:|---:|---|
| FR | 50 | 1700 | 1713 | TOP / .425757948 |
| FL | 1808 | 2854 | **Absent** | AIR / 0 |
| RR | — | — | — | GROUND / .059903668 |
| RL | — | — | — | GROUND / .514338384 |

First unfinished completion value is **`placed_FL=.7`**: qualified lift and historical crossing are present, but current top geometry/capture are absent. This is not failure to earn a genuine active lift. FL's crossing history remains true after retreat, while terminal current frontdistance is **−28.900113274mm**, topgap **+27.966563972mm**, within_topXYfalse/top_geometryfalse/TOPfalse/load0. No actual FL placement occurs. RL's early initial-clearance hint at10 is not hard qualification.

At P05 decision ends, FL is AIR449 of450 samples, TOP0 and obstacle-active0. Final evaluator records3589 consecutiveAIR samples ending5328 (start1740), with0 consecutiveTOP; this is recorded history, not an independent full rawcontact scan.

The post-cross decision-end trajectory confirms that history and present geometry differ:

- Tick2880: maximum post-cross frontdistance **+17.998221944mm**, gap+83.767285021mm, AIR/0load.
- Tick2912: first sampled retreat to negative frontdistance **−1.305815150mm**, gap+27.496491941mm, stillAIR/0load.
- Tick2920: nearest absolute post-cross sampled topgap **+24.940957697mm**, front−3.040310313mm; topXY/geometrytrue but noTOP/obstaclecontact, load0.

Those are decision-end extrema only, not whole120Hz extrema. The later terminal has retreated farther and is outside topXY. A valid earlier crossing or one geometric-near-top sample does not substitute for measured capture.

## Terminal raw contact and safety

Raw5328 independently confirms FL AIR with verified inactive obstacle/ground pairs,0N; wheel geometry verified, bottomz.077966563972m. FR has a verified active obstacle pair, normal **12.345253479N**, consistent with currentTOP. RL/RR verified ground normals **14.913726807/1.736963391N**; their obstacle pairs are inactive. All wheel geometries are verified.

Support count3 (FR/RL/RR), valid, CoM projection inside with+25.354172mm margin. Raw `all_finite=true`, bodycollision detected/active/persistentfalse and penetration0. Body linear/angular speeds .026296819m/s/.063772182rad/s. These scoped terminal facts do not mean more stable control or successful FL support.

## Native completeness and interpretation

The666 decision sequence contains exactly8ticks each:5328total, no gaps or finite-observation fallback. Every decision physical snapshot is valid. Independent native scan confirms **5328 verified/actual-target-effect ticks**, setter=dispatch, mapping=dispatch and same-tick counterfactual checks alltrue. Own-phase request-effect total5324 excludes four handoff holds; four in-episode state-write totals0. Reference independent checks allpass, with0 native sequence or dispatch/previousACK/mapper-clock inconsistencies. No complex reference example or unsupported endpoint-only activity inference is repeated.

**Conclusion:** FL earns true Q/C but never gains actual capture and ends behind the front plane in AIR; P05 therefore remains incomplete. No execution-path anomaly is established. The new workspace potential is not a cause proved by this single rollout, and P06 was not entered. This is neither new-reward improvement nor a reason to change the evaluator or add a hard gate. C103168, C101376, oldA and all other outcomes remain unchanged; no full/suffix success, successfulvideo or paired stability advantage is claimed. The next P06 training is not read or precredited.
