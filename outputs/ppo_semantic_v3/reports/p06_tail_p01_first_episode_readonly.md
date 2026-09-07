# P01 first training episode under 73e9370: measured retirement, no new tail activation

## Fixed completed window

Run: `runs/ppo_semantic_v3/train/20260906T1650134938238Z_g73e937039708_7681a31f01714528b75509300dfac00f`. Runtime `73e937039708a7306b2b2c941e1f9108b8fa8b3d`, source checkpoint 68224, seed 1001, N1 natural P01 full-episode training, stochastic PPO actions. This is **not** a fixed-mean evaluation.

This report reads only episode 0: **648 issued decisions, global 68225–68872, 5,177 physics ticks, 43.1416666667 s**. Decisions 1–647 each ran eight ticks; the terminal decision ran one. Completed optimizer update **504 / global 68992** already covers this entire episode (768 actual decisions / six updates since the run's 68224 origin, including 120 decisions from episode 1). No later episode is analysed and the planned 4096 is not counted as completed work.

Result: **BODY_COLLISION in P09**, task/full-task success false. The physical evaluator is valid, with `TASK_FAILURE_BODY_COLLISION`, reason `central body/obstacle collision`. First unfinished rear task is RR lift/cross/place; no RR initial-clearance, qualified-lift, cross, or placement event exists in this episode.

The critical negative finding is that **the new finite-source tail never activated**. Its source endpoint flag is false in all 648 decision snapshots, including the terminal. P06 began at tick 2560; the episode ended at 5177, before its 3064-tick finite source could end. The live tail replacement count is zero. Therefore this trajectory cannot demonstrate benefit or harm from the newly added post-endpoint rolling extension.

## Real FL capture and rear approach

Phase decision counts P01–P13: `[1,166,3,1,149,316,1,1,10,0,0,0,0]`.

| Event | Tick | Time (s) | Physical evidence |
| --- | ---: | ---: | --- |
| FR qualified / cross / placed | 49 / 1344 / 1358 | 0.4083 / 11.2000 / 11.3167 | Recorded hard events |
| P05 entry | 1368 | 11.4000 | FL task begins |
| FL initial / qualified | 1386 / 1439 | 11.5500 / 11.9917 | Qualified gain 50.420828 mm; top clearance +1.221215 mm |
| FL cross | 2488 | 20.7333 | Recorded hard crossing |
| FL placed / P06 entry | 2560 | 21.3333 | TOP contact true, obstacle pair active, two consecutive TOP samples, load 0.279892 |
| P06 → P07 | 5088 | 42.4000 | `rear_approach=1` |
| P07 → P08 | 5096 | 42.4667 | `workspace_RR=workspace_RL=support_RR=1` |
| P08 → P09 | 5104 | 42.5333 | `workspace_RR=load_ready_RR=1` |
| Terminal | 5177 | 43.1417 | P09 age 0.608333 s; body collision |

FL placement is not just an append-only label: at tick 2560 the current leg has front +13.949994 mm, clearance +0.440267 mm, actual TOP/pair contact and support. During the later transition it leaves support: at 5088 it still supports with load 0.203381; at 5096 it is AIR, load zero. At terminal its current AIR count is 87, clearance +193.375458 mm and support false, while its legitimate earlier placement remains true. A historical placed bit must not be presented as current FL support.

## Existing measured P06 retirement worked before the finite source ended

The diagnostics explicitly describe a **current nominal suggestion before later owners and slew, not an applied target**. Their source tick matches the current evaluator snapshot; they must not be compared as if they were the command that produced that same observation.

The existing workspace lower bound is -0.220 m and its blending width is 0.005 m. This run actually reached the workspace and the layer retained its maximum measured retirement:

| Tick | RR front (mm) | RL front (mm) | Current fraction | Retained peak / wheel gain | Last dispatched nominal wheels (rad/s) |
| ---: | ---: | ---: | ---: | --- | --- |
| 2560 | -552.331 | -577.886 | 0 | 0 / 1 | `[0,0,0,0]` (incoming handoff) |
| 2568 | -552.053 | -570.521 | 0 | 0 / 1 | `[.175,.175,.175,.175]` |
| 5080 | -223.350 | -167.068 | 0 | 0 / 1 | `[.3,.3,.3,.3]` |
| 5088 | -218.987 | -164.987 | .202543 | .202543 / .797457 | `[.299973,.299973,.299973,.299973]` |
| 5096 | -205.297 | -162.372 | 1 | 1 / 0 | `[.124973,.124973,.124973,.124973]` |
| 5104 | -200.818 | -160.667 | 1 | 1 / 0 | `[0,0,0,0]` |
| 5168 | -226.930 | -155.238 | 0 | **1 / 0** | `[0,-.63,0,0]` (later FR owner) |

At 5088 the P06-only pre-owner proposal is `.3 × .7974565366 = .2392369610` per wheel. It is intentionally not identical to the last dispatched .2999729. At 5096 its own proposal is zero; the dispatched nominal still decays through .1249729 under the existing handoff/slew path. When RR retreats outside the margin at 5168, retained peak stays 1, so the old P06 rolling contribution does not restart. These are real observations of the **previously existing retirement mechanism**, not evidence of the new finite-source tail.

## Residual/native control and later ownership remain active

Logical order below is FL, FR, RL, RR. Native targets have the existing left/right physical sign conversion; they are not measured wheel speeds.

| Tick | Logical nominal wheels | Projected residual wheels | Actual logical drive wheels | Actual native float32 wheels |
| ---: | --- | --- | --- | --- |
| 5088 | `[.299973,.299973,.299973,.299973]` | `[-.125979,.076311,.212721,.092194]` | `[.173994,.376284,.512694,.392167]` | `[-.173994,.376284,-.512694,.392167]` |
| 5104 | `[0,0,0,0]` | `[-.167611,.074803,.240358,.226335]` | `[-.167611,.074803,.240358,.226335]` | `[.167611,.074803,-.240358,.226335]` |
| 5168 | `[0,-.63,0,0]` | `[-.153346,.184430,.270744,.189399]` | `[-.153346,-.445570,.270744,.189399]` | `[.153346,-.445570,-.270744,.189399]` |

Thus P06 nominal retirement is **not residual removal or all-wheel shutdown**. P07's later FR-wheel owner survives the short stage: its negative adjustment becomes visible in P09, ramping FR nominal 0 → -.2 (5136) → -.4 (5144) → -.6 (5152) → -.63 (5160). The episode does not survive long enough to verify that source's later stop. The qualified rear-carry override is not exercised, because RR never qualifies.

Finite servo suggestions also continue across the labels. For example, FL hip nominal is 22.8° at 5088, 31.55° at 5096, 37.6° at 5104, then 49.2° later in P09. RL hip progresses 6.9° →14.3° →31.2°; RR hip progresses 0° →1.6° at 5112 →52.4° at 5176. These are advisory joint targets, not proof of successful world-space RR lifting.

All P06→P07→P08→P09 transition rows are nonterminal. P07/P08 each last one decision because their actual physical completion predicates are already satisfied at the next decision boundary, not because they are episode resets. Their incoming residuals are carried without forbidden-channel drops or phase-cap clipping; recorded handoff action jumps are numerical roundoff (servo at most 7.11e-15 degrees, wheel 5.56e-17 rad/s for these three transitions).

The compact **per-physics-tick effect audit** resolves the hold duration: P07 tick 5089 is handoff-held, while 5090–5096 have own-phase request effect; P08 tick 5097 is held, 5098–5104 own-effect; P09 tick 5105 is held, 5106–5112 own-effect. The second P09 decision has 8/8 own-effect ticks. These short phases do not discard their whole PPO decision.

## Terminal evidence and observability limits

RR is ground-contacting at all inspected late P06/P07/P08/P09 decision boundaries through tick 5176. At terminal it has only one AIR sample, current clearance -45.949655 mm, front -227.467534 mm, initial=false and no qualified/cross/placed events. RL likewise has no hard qualified/cross/placed events; its repeated initial-clearance events are not rear success. At terminal FL, FR, RL and RR all report current AIR/support false, but this one sample must not be extrapolated into a sustained contact-loss interval.

The production evaluator consumes `observation.body_collision.detected` and recorded the real failure classification at tick 5177. **This training run does not retain a full 120 Hz physical observation stream.** The compact JSON has no exact central-body pair force vector, contact point, penetration depth or pair persistence history. Accordingly this report cannot independently reproduce an exact pair/two-tick-force diagnosis as was possible in full evaluation traces. It does not dismiss or relabel the detector's collision result.

All 5,177 physics ticks have verified actual native effects; 5,169 have own-phase effects (eight ordinary incoming hold ticks excluded). Every decision verifies no in-episode root pose, root velocity, force/impulse or gravity writes. Native audits preserve actual float32 setter/dispatch checks. The ordinary environment path does not call its episode reset routines at a stage change. Complete mapper/filter state is not serialized per tick here, so the evidence establishes carried residuals, measured native continuity and no resets/writes, **not** independent bitwise reconstruction of every internal filter state.

Conclusion: the first stochastic training episode genuinely captured FL and reached both rear workspaces; the existing P06 contribution retired and later owners/PPO remained active. It then failed early in RR execution with body collision. **No new-tail activation, no complete task success, and no paired causal improvement are demonstrated.** Reporting stops at this completed first episode; current training is unchanged.

Sources: first completed episode JSON; first 648 rows of `residual_and_projection_audit.jsonl` (including their compact tick-effect arrays and evaluator history); completed `optimizer_updates.jsonl` boundary 504/68992; runtime source interpretation only. Read-only PowerShell plus this report write; no Python, simulation, checkpoint rehash or production edit.
