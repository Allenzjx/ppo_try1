# P01 training episode 1: real P06 tail, incomplete rear approach

## Fixed completed sample

Run `runs/ppo_semantic_v3/train/20260906T1650134938238Z_g73e937039708_7681a31f01714528b75509300dfac00f`, runtime `73e937039708a7306b2b2c941e1f9108b8fa8b3d`, source checkpoint 68224. This is the **second stochastic training episode**, index 1, seed 1001, natural P01 start—not a fixed-mean evaluation or teacher suffix.

Only this completed episode is analysed: **global 68873–69806, 934 decisions, 7,472 ticks, 62.2666666667 s**. All decisions ran eight ticks. Completed update **511 / global 69888** covers the entire episode; its additional 82 decisions in episode 2 are excluded. The planned 4096-decision run is not reported as complete.

Result: **P06 `INCOMPLETE_CONTROLLER_BLOCKED` at stage age 40.0 s**, task success false. Physical evaluator valid true, physical termination reason null. This is an actual approach deadline, not a recorded body collision or wheel-only failure. Terminal bootstrap false, next potential zero, terminal event -40 (total terminal reward -42.271669 including potential/cost terms).

Phase counts P01–P13: `[1,180,4,1,148,600,0,0,0,0,0,0,0]`. First unfinished subtask is the current rear workspace/approach, before RR or RL lift/cross/place.

## Front capture is genuine; current support is separate

FR qualified at tick 45, crossed 1458 and placed 1473. FL qualified at 1570, crossed 2599 and placed **2665**. P05→P06 occurs at decision boundary **2672 / 22.266667 s**, with `placed_FL=1`.

At that P06 entry, FL current TOP contact and obstacle pair are active, consecutive TOP samples 9, clearance +0.277785 mm and load 0.195556. FR is also TOP; both rear legs are ground-contacting. Thus the front completion is not inferred from nominal timing alone.

During the tail window, FL is usually not supporting. At first active diagnostic (5736), FL is AIR with clearance +4.708876 mm/load 0; FR TOP load .426423, RL GROUND load .498679 and RR GROUND load .074898. At terminal:

| Leg | Current contact | Front relative to obstacle (mm) | Clearance above top (mm) | Load fraction | Historical placed |
| --- | --- | ---: | ---: | ---: | --- |
| FL | AIR | +116.822 | +24.808 | 0 | true |
| FR | TOP | +207.064 | -0.057 | .450744 | true |
| RL | GROUND | -461.729 | -49.895 | .526357 | false |
| RR | GROUND | -380.249 | -51.268 | .022900 | false |

The terminal FL current AIR count is 125; its earlier placement remains legitimate history. Neither rear leg has a qualified/crossed/placed event. Decision-end contact counts are not continuous dwell estimates: among all 600 P06 samples FL supports in 29, RR is ground-contacting in 453 and RL in 595. In the 217 post-boundary samples defined below, FL supports in 1, RR is ground-contacting in 145 and RL in all 217. These do not reconstruct unlogged contact transitions between decisions.

## First tail activation: suggestion clock versus consumed command

The last false tail diagnostic is **global 69588 / tick 5728 / 47.733333 s**. The first true diagnostic is **global 69589 / tick 5736 / 47.8 s**, P06 age **25.533333 s**:

`source_endpoint_issued=true`, `finite_source_tail_replaced=true`, `status=live_endpoint_tail`, `wheel_gain=1`, retained retirement peak 0. The first observed active diagnostic is therefore 5736, not a hand-entered 5737.

The diagnostic explicitly means **P06's local contribution before later owners and slew, not the last applied target**. It is generated from the current task observation. The next complete decision, tick **5744 / global 69590**, confirms that the actual nominal and native baseline remain nonzero after the finite endpoint. There is no full 120 Hz target trace here to independently name the first individual post-endpoint physical dispatch; compact native-effect flags alone are not full target samples.

There are **217 active decision-end diagnostic rows**, ticks 5736–7464. At terminal tick 7472 the endpoint remains true but `finite_source_tail_replaced=false`, `status=terminal_no_tail`; no new retirement is earned. The last already-consumed live nominal is still +.3, which is not a claim of an additional post-terminal dispatch.

Retirement peak is **0 throughout**. Both rear workspaces never become satisfied at a logged decision boundary, so the pre-owner P06 contribution remains `[.3,.3,.3,.3] × 1`. P07 or later owners are never entered; this episode does not test subsequent ownership priority or qualified carry override.

## Actual controls across the finite endpoint

Wheel order is FL, FR, RL, RR. Nominal/residual/drive are logical rad/s; native values include the existing physical sign mapping and float32 cast. They are targets, not measured wheel speeds.

| Tick | Tail diagnostic | Actual nominal wheels | Projected residual wheels | Actual logical drive | Actual native float32 targets |
| ---: | --- | --- | --- | --- | --- |
| 5728 | before endpoint | `[.3,.3,.3,.3]` | `[-.022515,-.252861,-.317620,.094439]` | `[.277485,.047139,-.017620,.394439]` | `[-.277485,.047139,.017620,.394439]` |
| 5736 | first active suggestion | `[.3,.3,.3,.3]` | `[.097485,-.372861,-.390883,-.025561]` | `[.397485,-.072861,-.090883,.274439]` | `[-.397485,-.072861,.090883,.274439]` |
| 5744 | active, following decision | `[.3,.3,.3,.3]` | `[.073563,-.252861,-.406820,.081680]` | `[.373563,.047139,-.106820,.381680]` | `[-.373563,.047139,.106820,.381680]` |
| 7464 | last nonterminal active | `[.3,.3,.3,.3]` | `[.098003,-.329691,-.249929,.129904]` | `[.398003,-.029691,.050071,.429904]` | `[-.398003,-.029691,-.050071,.429904]` |
| 7472 | terminal, no new tail | `[.3,.3,.3,.3]` (consumed) | `[.137640,-.281564,-.369929,.009904]` | `[.437640,.018436,-.069929,.309904]` | `[-.437640,.018436,.069929,.309904]` |

At 5744 the verified same-tick zero-residual native counterfactual is `[-.3000000119,+.3000000119,-.3000000119,+.3000000119]`, with controller wheel bias zero. This corroborates a real continuing nominal target baseline; the nonzero baseline is not just a label in a diagnostic.

For a simple **decision-end sample mean**, not a time-integrated velocity estimate:

| Window | N | Mean residual wheels | Mean actual logical drive wheels |
| --- | ---: | --- | --- |
| Before active boundary, ticks 2680–5728 | 382 | `[+.095841,-.230548,-.273841,-.028343]` | `[+.395514,+.069125,+.025832,+.271330]` |
| After active boundary, ticks 5744–7472 (includes terminal decision) | 217 | `[+.115356,-.261950,-.299427,-.034456]` | `[+.415356,+.038050,+.000573,+.265544]` |

The single boundary sample at 5736 is deliberately excluded from both means. Residuals can oppose or exceed the .3 nominal, including reversing an individual wheel; the new tail has not disabled PPO or forced motion. FR/RL command cancellation is an observed control interaction, **not proof that those components caused the incomplete approach**.

## Workspace progress before and after

The existing workspace lower bound is -220 mm. Both legs must currently meet the task, not reach their individual best positions at different times.

| Fixed sample | Tick / time | RR front (mm) | RL front (mm) | Rear approach progress |
| --- | --- | ---: | ---: | ---: |
| P06 entry | 2672 / 22.2667 s | -550.612 | -564.463 | 0 |
| Best RR before tail, also best RR of whole P06 | 5232 / 43.6000 s | **-352.695** | -479.388 | 0 |
| Best RL before tail | 5248 / 43.7333 s | -356.426 | **-477.963** | 0 |
| First active diagnostic | 5736 / 47.8000 s | -397.930 | -481.397 | 0 |
| Best RR after boundary | 6976 / 58.1333 s | **-355.341** | -464.393 | .022429 |
| Best RL / best joint goal, terminal | 7472 / 62.2667 s | -380.249 | **-461.729** | **.033084** |

Before the boundary all 382 sampled rear-approach values are zero. First positive value is tick 6464 / global 69680, .007723. The final best is only .033084, with RL still **241.729 mm** behind the lower workspace limit. Best RR occurred before the tail and remained **132.695 mm** behind that limit; the post-tail RR best does not surpass it. RL's sampled best improves by 16.234 mm between the two windows, but neither leg reaches the workspace and P06 never hands off.

This proves that the extension supplied real suggestions and preserved controllability after finite source expiry. It does **not** prove success, causal improvement, or sufficiency of that nominal. The before/after windows have different physical states and an updating stochastic policy, with no paired counterfactual trajectory.

## Audit limits and stopping point

All **7,472** ticks have verified actual native effects; **7,467** have own-phase request effects, excluding five ordinary incoming handoff ticks. All decisions verify no in-episode root-pose, root-velocity, force/impulse or gravity writes; no finite-observation fallback occurs.

This compact training log contains evaluator histories/current leg snapshots, decision-end targets and per-tick effect flags—not a full 120 Hz physical observation stream. No exact contact force vector, contact point, body-pair persistence or between-decision geometry trajectory is invented. The valid physical evaluator and null physical failure reason are retained; this remains an **incomplete P06 deadline**, not a renamed collision or task success.

Sources are the completed episode-1 record, audit rows global 68873–69806, and completed optimizer boundary 511/69888. Only PowerShell reads and this independent report were used; no Python, new simulation, hash sweep, production edit, parameter change or additional training gate. Reporting stops at episode 1.
