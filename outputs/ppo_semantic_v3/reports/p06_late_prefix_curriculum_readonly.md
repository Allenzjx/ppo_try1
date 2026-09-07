# Later P06 predecessor windows: measured availability, not a new run

This is a read-only curriculum choice note while the 73e9370 P01 training block continues. It does not change that block, choose an already-airborne snapshot, run a teacher replay, or count future PPO samples. The first 2048 completed decisions of the running block (globals68225–70272, update514) have phase counts `[3,523,11,3,445,1051,1,1,10,0,0,0,0]`; P06 dominates the credited rear samples while P10–P13 remain unvisited in this fixed half-block window.

The independently inspected source is the **first real teacher prefix only** from completed P10 run `20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc`. Its actual handoff transition history records:

| Semantic boundary | Physics tick | Time s |
| --- | ---: | ---: |
| P06 begins | 3584 | 29.866667 |
| P07 begins | 5952 | 49.600000 |
| P08 begins | 5960 | 49.666667 |
| P09 begins | 6088 | 50.733333 |
| P10 begins | 7584 | 63.200000 |

The prefix decision rows corroborate P06=296, P07=1, P08=16 and P09=187. Existing offset160 would correspond to tick4864 /40.533333s, leaving136 decisions before the historical P07 boundary; offset240 corresponds to5504 /45.866667s, leaving56. Both actual rows are P06→P06, native verified, with canonical four-wheel targets+.3. These are historical availability facts, not a prediction that a changed PPO trajectory will reach P07 after that many steps.

RR's recorded first initial-clearance event occurs later at6901; qualification/cross/placement are6938/7109/7579. Thus neither candidate is a teacher-qualified RR suffix. However these compact intermediate prefix rows do **not** retain current RR contact, gap, load or CoM. No event before the candidates does not prove absence of subthreshold/transient AIR, and a near-zero joint target is not measured q or contact. A future real handoff's full current observation must be reported rather than replacing that evidence with this historical inference.

The production prefix checks live semantic phase after observing the current tick and before granting the offset. At offset296 the source trajectory is already P07 and would take its existing explicit fresh-P01 fallback; it is not a suitable target. Offset295 leaves only one decision before that boundary and takeover can consume it. The bounded160–240 range retains a more meaningful predecessor interval and does not require changing phase labels, hard entry tests, body pose, sensors or ordinary transition continuity.

If a later block uses one of these windows, retain its actual `start.actual_phase`, `requested_phase_still_active_at_credit`, teacher-excluded decisions/ticks, real RR support/gap/history, and any legitimate fallback. Only new post-takeover on-policy samples count as PPO. A teacher prefix is not current-policy P01 success, and a suffix result is not full-task success. The actual current P01 block and subsequent saved-model P01 evaluation finish before selecting the next course; no future count, optimizer reset or checkpoint is credited here.
