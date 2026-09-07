# C101,376 — final natural-P01 P06 workspace noncompletion

Run `runs/ppo_semantic_v3/validation/20260907T0108106916731Z_g2677995544c9_3cfbdf407efc46619d280462e8ac47a4`, frozen runtime2677995544c974c03d8b1e41d77375b45e323c9e, saved checkpoint101376, N1seed2001, naturalP01 deterministic fixed mean. Root confirms session43101 exit0; final lifecycleSUCCEEDED means execution completed, **not task success**.

Actual **966 decisions /7728 physics ticks /64.4s**, P06 age40s, `INCOMPLETE_CONTROLLER_BLOCKED`, taskfalse, optimizer updates0, `window_ended_before_task_terminal=false`. Physical evaluation is valid with null hard-failure reason. This is incomplete task execution, not a collision, interface, codec or file-writing failure.

Scope: final manifests, one complete966-row decision-ledger pass, terminal raw7728, and the historical2677995 task-spec content for the actual workspace bound. No complete raw-geometry rescan, hashes, PT/GPU/Python/Isaac, later production revision or unfinished test conclusions were used.

## Phase ledger and first missing task

| Source phase | Decisions | End transition |
|---|---:|---|
| P01 | 1 | P02 at8/.066667s |
| P02 | 210 | P03 at1688/14.066667s |
| P03 | 4 | P04 at1720/14.333333s |
| P04 | 1 | P05 at1728/14.4s |
| P05 | 150 | P06 at2928/24.4s |
| P06 | 600 | Incomplete at7728/64.4s |
| P07–P13 | 0 each | Not visited |

The exact first missing completion predicate is **`rear_approach=0`**, the minimum of current RL/RR workspace progress. Historical2677995 config uses workspace front-distance interval **[−.22,+.06]m**. Both rears remain within lateral span but far behind the front-distance lower bound; this is measured task geometry, not a fixed joint/velocity/clock entry template.

| Rear leg | First P06 decision-end front(mm), tick2936 | Closest P06 decision-end front(mm), independently selected | Terminal front(mm) | Terminal deficit to−220mm |
|---|---:|---:|---:|---:|
| RR | −532.472256 | −492.873205 | −823.927995 | **603.927995mm** |
| RL | −541.348366 | −530.957533 | −855.566439 | **635.566439mm** |

All600 P06 decision-end `rear_approach` values are0. The independently closest RR/RL positions still miss by272.873205/310.957533mm; these minima are sampled decision endpoints, not a claim about unparsed120Hz extrema or a simultaneous pair configuration. The recorded body-forward field changes from−.238605406m at2936 to−.567992438m at7728 (−.329387031m); this is measured body progression, not inferred CoM or proof of one actuator's causal contribution.

## Actual Q/C/P and present contact

| Leg | Qualified lift tick | Front-cross tick | Placed tick | Terminal current state / load |
|---|---:|---:|---:|---|
| FR | 47 | 1699 | 1717 | GROUND / .443921832 |
| FL | 1813 | 2828 | 2922 | AIR / 0 |
| RR | — | — | — | GROUND / .062775857 |
| RL | — | — | — | GROUND / .493302311 |

Both front placements are real current-episode history, unlike C93184's missing FL placement; they are not teacher events. But neither front remains TOP at the terminal. The lone RL initial-clearance hint at9 is not a hard qualification; no rear hardQ/C/P occurs. At first P06 decision end2936, both fronts are TOP (FL load.475372256, FR .043989500). Across P06 decision ends, FL is AIR587/TOP9 out of600, FR TOP166. These sampled counts do not assert complete raw contact continuity.

Terminal raw7728 independently confirms FL AIR, both exact ground/obstacle pairs verified but inactive/0N. FR/RL/RR are GROUND, verified active normal forces **12.796389580 /14.219820023 /1.809562564N**; all obstacle pairs are inactive/0N. FL gap+13.695089mm/front−340.180662mm, FR gap−50.216486mm/front−260.284540mm. Historical placed bits therefore do not establish current obstacle-top support.

Raw support is valid with3 bodies (FR/RL/RR), CoM projection inside and signed margin+15.124766mm. Raw `all_finite=true`; body collision detected/active/persistent=false, penetration0. Body linear/angular speed=.020172026m/s/.036539625rad/s. These are scoped terminal safety facts, not evidence that this policy is more stable than another failed or longer-reaching rollout.

## Real command path and native integrity

Terminal canonical wheel arithmetic FL/FR/RL/RR (rad/s):

| Wheel | Nominal | Applied residual | Actual canonical drive command |
|---|---:|---:|---:|
| FL | +.3 | −.447268688 | −.147268688 |
| FR | +.3 | **−.697472758** | −.397472758 |
| RL | +.3 | −.313221581 | −.013221581 |
| RR | +.3 | +.237757316 | +.537757316 |

FR's actual residual magnitude exceeds the old.6 cap; this proves real use of the current front range at that endpoint, not just a large nominal+residual command. It does not establish that the range change caused noncompletion. Measured wheel speeds are separately [−.146574244,−.231856510,−.120191328,+.602791786]rad/s.

The finite P06-source extension is present:217 P06 decision-end snapshots report `live_endpoint_tail`. Terminal diagnostics stop offering a new tail action (`terminal_no_tail`) while the final already-dispatched nominal remains+.3 on all wheels; retirement peak remains0/gain1. Thus the recorded endpoint is not an unexplained loss of nominal at the source clock. P07/rear-carry/P13 behavior was not exercised; no conclusion about it is inferred.

Every decision/tick interval is continuous8ticks. Recorded per-decision120Hz summaries total **7728 verified/native-effect ticks**, own-phase effects7723 (five transition holds excluded), four in-episode state-write counters0. All audit/no-write flags true, no invalid physical snapshots, no finite-observation fallback. Native totals are aggregated recorded evidence, not an independent simulator replay. No execution-path anomaly is demonstrated by this audit; it documents a genuine lack of measured workspace progress with changed current support.

**Conclusion:** first missing task is P06 rear workspace approach, with both rears substantially behind the actual bound. This is not whole-task success, a causal regression proof, or evidence for imposing a new entry gate. C93184/P05 incomplete, older A outcomes and all training histories remain preserved. The separately proposed tracking-reference production revision is outside this evaluated runtime and receives no test or training credit here.
