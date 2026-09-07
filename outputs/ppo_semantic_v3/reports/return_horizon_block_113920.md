# Finalized block 33 — return-horizon P01 training, checkpoint 113,920

Run `train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8`, frozen HEAD `f4bfe2560bfd228541f7829d830fa441054eab1d`, N1/seed1001/natural P01. Both final manifests record **SUCCEEDED execution**; the root separately confirmed process exit 0. This is full consumption of the training request, **not physical task success**.

Source `109824 /823 /16460` → final **`113920 /855 /17100`**. Actual=requested=planned **4096 decisions /32 PPO updates /640 optimizer steps**, unused0/rounding0. Recorded training-loop wall time is **1717.9151240000501 s**; do not reinterpret it as all process/reset/setup time. Original origin10112 remains; spent full53504/suffix50304/smoke0 satisfies `53504+50304+10112=113920`. The 33 finalized blocks add **103808 decisions /811 PPO updates /16220 optimizer steps** since10112/44/880.

## Scope and phase ledger

PowerShell read the finalized small manifests, completed episodes, 32 update records and five saved sidecars; one full streaming pass covered the 4096 policy records and their compact per-tick native evidence. No PT/tensor load, Python, tests, simulation, repeated checkpoint hash, or later evaluation/course was read. Initial migration details remain in [return_horizon_initial_109824.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/return_horizon_initial_109824.md); this report does not duplicate or add credit to that boundary.

The policy rows are exactly contiguous **g109825–113920**. Phase counts are attributed to the issued policy action's `phase_id`, not relabeled by its destination phase, and exactly match the final core telemetry.

| Phase | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Credited decisions | 8 | 1124 | 27 | 10 | 1727 | 1200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

All six starts are natural P01. Current sampling says `P01_full_task_only_initial_version`, prefix request is null, suffix curriculum flag is false, and there are **zero teacher decisions/ticks credited**. Episode clocks start with credited physics tick1, not a teacher handoff. Simulator settle/setup ticks are not claimed as policy credit.

## Completed episodes and optimized tail

| Episode | Global range | Decisions | Credited ticks | End time s | End phase / outcome | P01–P06 counts |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | 109825–110475 | 651 | 5208 | 43.4 | P05 INCOMPLETE | 1,195,4,1,450,0 |
| 1 | 110476–111122 | 647 | 5169 | 43.075 | P05 INCOMPLETE | 1,190,4,1,451,0 |
| 2 | 111123–112053 | 931 | 7448 | 62.066666667 | P06 INCOMPLETE | 1,172,5,4,149,600 |
| 3 | 112054–113000 | 947 | 7576 | 63.133333333 | P06 INCOMPLETE | 1,191,6,2,147,600 |
| 4 | 113001–113641 | 641 | 5128 | 42.733333333 | P05 INCOMPLETE | 2,184,4,1,450,0 |
| 5, tail | 113642–113920 | 279 | 2232 | 18.6 | P05, **nonterminal** | 2,192,4,1,80,0 |

Five completed records reconcile to3817 decisions; plus the279-decision tail equals4096. All completed reasons are `INCOMPLETE_CONTROLLER_BLOCKED`, physicalvalid=true and physical termination reason=null. P05 deadline ages are30/30.008333333/30s in episodes0/1/4; P06 ages are40s in episodes2/3. The tail has no termination reason, P05 age5.333333333s, and is already included in complete PPO updates—not a sixth failure or a success.

The only short policy interval is episode1/g111122: **1 physics tick**, true P05 deadline terminal. It exactly explains `32761=4096×8−7`; all other intervals have8 ticks.

## Physical accomplishments versus missing tasks

Q/C/P below are the recorded hard qualification/front crossing/placement event ticks within each natural episode. They are current-policy history, not teacher credit. The whole compact stream records **no hard RR or RL Q/C/P in any episode**; low or brief AIR states are not promoted to qualification. No P07–P13 was sampled.

| Episode | FR Q / C / P | FL Q / C / P |
| --- | --- | --- |
| 0 | 48 /1596 /1597 | 1677 /— /— |
| 1 | 63 /1536 /1556 | 1637 /2725 /— |
| 2 | 66 /1394 /1419 | 1535 /2428 /2643 |
| 3 | 55 /1538 /1579 | 1752 /2707 /2774 |
| 4 | 67 /1505 /1515 | 1624 /3300 /— |
| 5, tail | 68 /1549 /1577 | 1679 /— /— |

- **Episode0 P05:** FL has qualification but no crossing or placement. Terminal FL is AIR/load0/front−70.140627mm/gap−26.042777mm; `placed_FL=.573753903`. Thus crossing and real capture are still missing, not merely a historical joint endpoint.
- **Episodes1/4 P05:** FL has Q+C but no P; both terminal FL are AIR/load0, at front−28.561320/−29.994633mm and gap+27.031337/+26.913320mm. `placed_FL=.7` is partial progress, not current contact or capture.
- **Episodes2/3 P06:** both front legs truly placed earlier, and P06 began at2648/2776. The terminal completion is `rear_approach=0`; RR/RL remain well behind the obstacle: episode2 fronts−658.167955/−485.278592mm, episode3−509.641903/−361.790773mm. RR is currently AIR but below top (gaps−47.810080/−50.166609mm, load0), RL is GROUND (loads.542885388/.527029817). Those AIR states never earned hard qualification. FL has lost current contact in both episodes: AIR/load0, gaps+35.385912/+87.175106mm. Episode2 FR is now GROUND, while episode3 FR is TOP/load.472970183. Historical front placement is not maintained platform support.
- **Tail:** FL is Q only, still AIR/load0/front−88.378903mm/gap−7.703217mm; `placed_FL=.545336920`. FR is TOP/load.431722599; RL/RR are GROUND/load.493740780/.074536621. This is an unfinished P05 attempt, not an expired task.

There is **no full/suffix success**. Absence of a hard collision in these five incomplete episodes does not establish stability improvement or a causal effect of the return-profile change.

## Native execution, clocks and terminal masks

All **32761 compact native ticks** record verified/effect=true, and their per-interval counts equal the summaries and policy intervals. Own-phase request effect totals**32735**, with exactly**26 handoff-hold ticks** accounting for the difference. Every endpoint has verified same-tick counterfactual/mapping/float32 setter evidence and matches the stored raw policy request. Four in-episode state-write totals—root pose, root velocity, force/impulse and gravity—are each**0**, with every no-state-write verification true.

There are **26 P01→…→P06 phase changes**, all nonterminal/bootstrap=true; none is treated as episode completion. All4091 nonterminal policy rows permit bootstrap. The five true task terminals have bootstrap=false and `potential_after=0`. `time_outs=false` throughout; no terminal-observation finite fallback occurred. This checks the stored masks and reward receipts, not an independent PT/GAE tensor replay.

No discrepancy was found in the sequential global decisions, episode-local physics clock, `sim_time_s=tick/120`, compact command/episode tick relationship, native summary counts, or raw binding. All recorded physical evaluator snapshots are valid/null hard failure; the selected action/mean/std/value/log-probability/reward/drive vectors are finite. No teacher-storage flag was true. These are bounded execution-evidence checks, not a new physical-control-success gate or an unlogged contact/geometry inference.

## Updates, saves and final receipt

All**32 updates824–855** advance exactly128 decisions and20 optimizer steps. The source→first, all31 adjacent, and last→result actor-fingerprint comparisons match. Every update reports a changed actor and finite nonzero gradients; checked loss/gradient/entropy/KL/clip/LR scalars are finite. Recorded gradient-norm range is1.000596886–1.414213591; value-loss range.000631307586–397.232115173. Recorded update-end learning rates are1e−5,1.5e−5 and2.25e−5 under the adaptive schedule; these are not inferred per-minibatch constants.

Saved **109952,110848,111872,112896,113920** sidecars all record roundtrip=true, match the corresponding update's actor, retain the source identity-normalizer fingerprint, and name this run. All record gamma.9985/lambda.99, rollout128 and `v3_gamma_09985_lambda_099_v1`. Their budget progression ends atfull53504/suffix50304, origin10112; the initial's fresh-Adam boundary is not a budget reset.

The final pointer names `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000113920.pt`; recorded checkpoint SHA **`54bbc7de91a4d7a7461c52b2165abf6286127918522cc52606a96bb140499a6a`** equals the final sidecar field. The pointer's sidecar SHA is `4380bccdbaa9bb55dcfdc90700cba9f5988be178cc3e12567f0b7e920a853bee`. Final actor fingerprint is `fb377ac16b8ec08d5592bbd7e70b3f766ca251abc9f1453aa72db0bf54cd1253`. These are read receipts, not newly computed PT hashes.

Latest previously completed full-P01 evaluation remains C109824/P05 incomplete for this report's scope. No C113920 or next-course result was inspected, inferred or precredited. Historical failures and the first128/initial reports remain intact.
