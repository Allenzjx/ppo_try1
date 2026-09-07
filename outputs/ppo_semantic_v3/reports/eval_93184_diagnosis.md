# C93,184 — final natural-P01 evaluation diagnosis

Final execution: SUCCEEDED / exit0 (root verified process closure). **Task not successful: P05 INCOMPLETE_CONTROLLER_BLOCKED**, not a physical hard failure or an I/O/video error. Actual **657 decisions / 5,256 physics ticks / 43.8 s**, P05 age30s, optimizer updates0, `window_ended_before_task_terminal=false`.

Source: `runs/ppo_semantic_v3/validation/20260907T0009333548816Z_g2677995544c9_869bd7c54b5f4b688d9582b8a4e507ae`. Runtime2677995544c9, N1, seed2001, deterministic fixed mean, naturalP01, saved `checkpoints/history/checkpoint_step_000093184.pt`. No teacher prefix or policy optimization is credited to this evaluation.

## Scope and ledger

Read the final manifests, parsed the complete657-row decision ledger once, and inspected only raw ticks4260–4268 plus final raw5256. This is not a second complete120Hz geometry/contact scan. Native totals below are summed from each decision's recorded120Hz audit summary, not independently replayed physics. No tensors, Python, hashes, production changes or later training were used.

| Start phase | Decisions | Transition at decision end |
|---|---:|---|
| P01 | 1 | P02 at tick8 / .066667s |
| P02 | 201 | P03 at tick1616 / 13.466667s |
| P03 | 4 | P04 at tick1648 / 13.733333s |
| P04 | 1 | P05 at tick1656 / 13.8s |
| P05 | 450 | Incomplete at tick5256 / 43.8s |
| P06–P13 | 0 | Not visited |

All657 decision/tick sequences are continuous, all intervals contain8ticks. Summed native verified/effect ticks=5256/5256, own-phase request effect=5252; the four transition hold ticks are not own-phase credit. All four in-episode state-write counters are0, with every decision's no-write verification true. There are0 invalid physical-evaluator decision snapshots and0 terminal finite-observation fallbacks. Final physical evaluation is valid, `success=false`, hard `termination_reason=null`.

## First missing task: genuine FL capture

| Leg | Qualified active lift | Front crossing | Placed | Terminal physical support |
|---|---:|---:|---:|---|
| FR | tick45 | 1627 | 1641 | TOP; load.421856345 |
| FL | 1731 | 2763 | **Absent** | AIR; load0 |
| RL | Absent | Absent | Absent | GROUND; load.498455901 |
| RR | Absent | Absent | Absent | GROUND; load.079687754 |

The first unfinished P05 completion value is `placed_FL=.85`, not a missed historical joint/clock entry. The current predicate gives .35 qualified lift + .35 crossing + .15 top-XY/height geometry, but0 sustained actual TOP-contact contribution. FL has crossed legally; this is **missing placement/capture**, not an unqualified wheel-only crossing. Endpoint history contains no FL placed event. The initial RL clearance event at tick13 is not a hard RL qualification.

Among450 P05 decision-end snapshots, FL is AIR449 times, TOP0 and obstacle-pair-active0. The first P05 sample precedes FL's later lift. Final evaluator records3587 consecutive FL AIR samples (ending5256, corresponding to start1670), zero consecutiveTOP samples and `within_top_xy=true`. That continuous streak is the recorded evaluator history; this report did not independently parse all3587 raw rows.

**Decision-end nearest top gap after FL qualification/crossing and within topXY:** decision533/tick4264/35.533333s, clearance **+1.549241515mm**, front **+18.195870946mm**, AIR/load0/TOPfalse/obstaclefalse. This is the minimum absolute clearance over those eligible decision endpoints, not a claim about the uninspected whole120Hz trajectory.

**Small raw confirmation:** ticks4260–4268 have gaps+2.191049,+2.079104,+1.884451,+1.708975,+1.549242,+1.664547,+1.862881,+2.037255,+2.201242mm. Their local minimum is again4264. All9 samples are finite, geometry-verified AIR, with verified but inactive obstacle/ground pairs and0N force. Thus this closest sampled approach did not actually touch down. Other raw intervals were not searched for a global minimum.

## Terminal measurements and interpretation

At raw5256, FL bottom z=.052410625489m versus obstacle top=.05m: **+2.410625489mm** clearance, front+17.634275983mm, AIR, obstacle/ground forces0. The obstacle pair retains a non-null contact-point field despite active=false/zero force history; that field alone is not evidence of current contact. FR has a verified active obstacle pair, normal force12.094018331N (z12.078217507N); RL/RR verified ground normal forces14.290018082/2.284533978N. Actual support bodies are FR/RL/RR, count3, support valid and CoM projection inside (signed margin+20.423134mm). This does not supply the missing FL support.

Final raw `all_finite=true`, body collision detected/active/persistent=false and geometric penetration0; the selected9-row window is also finite. These scoped checks do not assert a separate all-raw audit. Physical body linear/angular speed=.024616589m/s/.054987153rad/s. Final wheel commands in FL/FR/RL/RR order are [−.068949521,−.038235437,+.010788331,+.066480027]rad/s; measured wheel speeds are [−.069142111,−.198440328,−.154643491,+.106330395]rad/s. Commands and measured motion are distinct evidence.

No P06 layer, finite P06 tail, rear geometry continuation or P06–13 front-wheel1.2 range was exercised. Those later-stage mechanisms cannot explain this run's P05 endpoint by direct execution. This report establishes missing actual FL capture, not why the learned policy produced it; it neither proves forgetting nor attributes a regression to one training change. All earlier C89,088/P06 and other A/B/C outcomes remain intact. No full/suffix success, paired stability improvement or successful video is claimed.
