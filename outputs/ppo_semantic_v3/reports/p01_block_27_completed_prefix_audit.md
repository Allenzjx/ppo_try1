# Block27 — bounded audit of completed episodes0–5

Run `runs/ppo_semantic_v3/train/20260907T0017590892338Z_g2677995544c9_bf91484d9ac74584b18ac862afe2f22c`, runtime2677995544c9, N1seed1001, ordinary resume from93,184/693/13,860. Started arguments explicitly select naturalP01/full_episode and `new_mdp_warm_start=false`. The title's “completed prefix” means the first six completed episodes of this live block, **not a reset teacher/checkpoint prefix**. Each episode starts policy decisions at P01/tick8. The unused `prefix_source=frozen_fsm` default in arguments does not turn these full episodes into teacher roll-ins.

Scope is exactly the first6 completed-episode receipts and the first **5,784 policy rows, global93,185–98,968**. The bounded policy scan stopped after that last row; no episode6/later policy rows, optimizer logs, current checkpoint or final block manifest were read. This is not a final block ledger, an optimized-checkpoint count update or a master-report update. Production/config/tests/history remain unchanged; no Python, PT, hashes, GPU or Isaac process was started.

## Verified episode outcomes

| Episode | Policy global interval | Decisions | Episode physics ticks | End time(s) / phase | Actual outcome |
|---|---|---:|---:|---|---|
| 0 | 93185–94138 | 954 | 1–7632 | 63.6 / P06 | INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 94139–95079 | 941 | 1–7528 | 62.733333 / P06 | INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 95080–96038 | 959 | 1–7672 | 63.933333 / P06 | INCOMPLETE_CONTROLLER_BLOCKED |
| 3 | 96039–96905 | 867 | 1–6936 | 57.8 / P09 | BODY_COLLISION |
| 4 | 96906–98156 | 1251 | 1–10008 | 83.4 / P09 | INCOMPLETE_CONTROLLER_BLOCKED |
| 5 | 98157–98968 | 812 | 1–6496 | 54.133333 / P09 | BODY_COLLISION |

All6 terminal physical evaluations are **valid**. Episodes0/1/2/4 have null physical hard-failure reason; episodes3/5 record `TASK_FAILURE_BODY_COLLISION` / central body–obstacle collision. Physical validity means the evaluation is usable, not that collision is acceptable. All6 task/full-task-success flags arefalse. Compact training receipts do not expose a full independent raw body-pair force/contact-point trajectory, so no force magnitude or collision mechanism is invented here.

## Actual phase sampling

Counts use the source phase under which each policy decision was issued, not terminal labels.

| Episode | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10–P13 each |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 198 | 4 | 1 | 150 | 600 | 0 | 0 | 0 | 0 |
| 1 | 1 | 188 | 4 | 1 | 147 | 600 | 0 | 0 | 0 | 0 |
| 2 | 1 | 203 | 4 | 1 | 150 | 600 | 0 | 0 | 0 | 0 |
| 3 | 1 | 178 | 4 | 1 | 149 | 521 | 1 | 2 | 10 | 0 |
| 4 | 1 | 194 | 4 | 1 | 150 | 445 | 5 | 1 | 450 | 0 |
| 5 | 1 | 190 | 4 | 1 | 148 | 454 | 3 | 1 | 10 | 0 |
| Total | 6 | 1151 | 24 | 6 | 894 | 3220 | 9 | 4 | 470 | 0 |

P06 first incomplete values `rear_approach` are0/0/.103842277 in episodes0/1/2. Terminal RR/RL front distances are respectively −745.039/−651.463mm, −691.574/−650.491mm, and −441.820/−444.039mm: both are behind the existing −220mm lower workspace bound. They are task-space noncompletion, not a historical pose gate. Episodes3/4/5 do pass P06 at ticks6832/6360/6384, then enter P09 at6856/6408/6416. P09 first missing completion remains actual RR placement; collision interrupts episodes3/5 after10 P09 decisions each, while episode4 exhausts450 P09 decisions without placement.

## Q/C/P events versus current support

All ticks below are actual within-episode120Hz history events; Q=qualified active lift, C=front crossing, P=placed.

| Episode | FR Q/C/P | FL Q/C/P | RR Q/C/P | RL Q/C/P |
|---|---|---|---|---|
| 0 | 45/1595/1617 | 1710/2735/2832 | —/—/— | —/—/— |
| 1 | 59/1520/1542 | 1630/2665/2728 | —/—/— | —/—/— |
| 2 | 46/1636/1657 | 1765/2774/2868 | —/—/— | —/—/— |
| 3 | 51/1462/1463 | 1548/2581/2657 | —/—/— | —/—/— |
| 4 | 48/1571/1589 | 1691/2401/2795 | **6998/—/—** | —/—/— |
| 5 | 39/1542/1553 | 1647/2604/2746 | —/—/— | —/—/— |

The only RR qualification is the real `qualified_measured_upward_lift` event at episode4/tick6998. Its current crossing eligibility is explicitly revoked by `qualification_revoked_ground_before_cross` at9033. Terminal RR is GROUND/load.036429851, front−450.187777mm, gap−47.778136mm. The retained Q event records an earlier genuine lift, not successful transfer or still-valid qualification. No RR crosses or places, and no RL hardQ/C/P occurs in these six episodes. Initial-clearance hints are not counted as Q.

Front placement is genuinely recorded in all six natural-P01 policy episodes, with no teacher credit substitution. Nevertheless, **terminal FL is AIR/load0 in all6**. Terminal FR is GROUND in episodes0/1, TOP in2/3/5; in episode4 FR has load.519152163 but TOPfalse/GROUNDfalse, front−15.073117mm (contact outside the qualifying top region is not renamed TOP). Thus historical placed bits do not establish current front support or whole-task success. In episode3 RR remains GROUND/load.015435121, noQ; episode5 RR is AIR/load0 but gap−46.166322mm/front−145.028141mm, also noQ. Neither collision episode is a case of completed RR placement later being lost.

## Execution audit and bounded conclusion

All5,784 global/episode decision and tick intervals are continuous, each8ticks; exactly6 terminals agree with the completed ledger. Summed per-decision native audit verifies/effects **46,272/46,272ticks**, own-phase request effects46,233. All four in-episode state-write totals are0, every no-write/audit verification true, no finite-observation fallback and no invalid physical-evaluator snapshots. Recorded12-dimensional mean/std/raw samples, old log probabilities and values are finite; std is positive. All39 phase changes are nonterminal, bootstrap-allowed and not timeouts, so no phase-boundary terminal/GAE cutoff is observed.

**No execution anomaly supporting an immediate production change was found within this scope.** The evidence is four valid task incompletions and two genuine logged physical collision failures during stochastic exploration, with a single unsuccessful RR lift attempt. Native audit success does not prove every controller choice optimal, and the compact logs do not prove collision causality or sensor correctness beyond their recorded checks. No new hard entry gate, cap/reward change, full/suffix success, or paired stability claim follows from this audit. The live block's final budget/counters and later episodes remain deliberately unreported; the master stays at26 finalized blocks.
