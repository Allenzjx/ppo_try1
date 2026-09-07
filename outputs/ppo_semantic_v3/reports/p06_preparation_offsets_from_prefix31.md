# P06 preparation offsets — measured availability within block31's first teacher prefix

Read-only scope: `train/20260907T0313083674401Z_gf1a9bbf650b1_98167177c0ff4cd6b15dd57e491fa8c2`, runtime `f1a9bbf650b1b8f80d5169e09173fcfc68797d99`, N1/seed1001/frozen_fsm/requestedP07/offset0. Read its started arguments and `prefix_evidence.jsonl` only through the **first** `policy_credit_start`:747 records comprising744 teacher decisions plus start/result/credit-start. No later prefix, active PPO stream, checkpoint/hash, Python/PT/GPU/Isaac, production/config/test or master-report change.

The actual P07 prefix was accepted without a miss:744 decisions/5952 physics ticks, P07 handoff at49.6s, remaining150.4s. All read prefix records have policy_credit=false. This was not a P06-offset100/160/200 training experiment.

## Actual candidate records

The record ending tick3584 transitions P05→P06 at29.8666666667s. The requested candidate arithmetic is therefore `3584 + 8 × TeacherOffsetDecisions`.

| Hypothetical P06 offset | Actual recorded tick | Actual time (s) | Time since P06 entry (s) | Source/end phase | Recorded task termination |
|---|---:|---:|---:|---|---|
| 100 | 4384 | 36.5333333333 | 6.6666666667 | P06/P06 | null |
| 160 | 4864 | 40.5333333333 | 10.6666666667 | P06/P06 | null |
| 200 | 5184 | 43.2 | 13.3333333333 | P06/P06 | null |

All three rows contain8 executed/verified ticks, all_ticks_verified=true and no_in_episode_state_writes_verified=true. Raw and projected policy residuals are allzero, policy_credit=false. Their actual canonical drive-target vectors are **exactly equal**:

`[21.55, -12.15, -0.7996284021635888, 45.347052073602775, 5.65, 0.6421599783343184, -1.3499030061922759, 0.9543498893273243, .3, .3, .3, .3]`

Servo entries are canonical degrees and wheels canonical rad/s; these are target commands, not measured joint positions or wheel translation. The same vector is already recorded at the3584 boundary. Zero current-residual native-effect count is expected during zero-raw teacher initialization; it does not mean the robot stopped moving.

## Essential observation limit: candidate geometry/contact cannot be recovered

The candidate `reset_only_prefix_decision` records persist exactly: kind, policy_credit, source/end phase, physics tick/count/time, raw policy action, projected residual, actual drive target, native-audit summary, no-state-write verification, and termination reason. They **do not persist** semantic_task/current_legs, per-leg ground/AIR/TOP flags, measured load, Q/C/P histories, wheel-front geometry, or full raw observations.

Consequently the following requested measurements at4384/4864/5184 are **unavailable**, not zero:

- RR/RL front distance, ground/AIR state, load and hard Q/C/P;
- FL current support/contact/load;
- any difference in actual body motion or support among these candidates.

The full sensor snapshots available in this first prefix are its initialtick0 and final credit-starttick5952, not the three intermediate candidates. At5952 only, the actual handoff evaluator reports FL TOP/support=true/load.199119297/front+.370607260m; RR GROUND/support=true/load.247819573/front−.205028680m; RL GROUND/support=true/load.303504492/front−.219366219m. Rear initial-clearance fields are false at that later snapshot. These later values must **not** be relabelled as candidate values, interpolated backward, or used to assert intermediate Q/C/P. Likewise, the mere P06 label cannot establish per-leg AIR/load/history.

Thus the data proves different elapsed initialization times with the same sampled teacher targets and nonterminal P06 labels. It does not establish which candidate has better rear workspace or support. No missing measurement is reconstructed from command sign, constant target, historical A pose or another run.

## Existing offset semantics and continuity

`semantic_prefix.py:174–202` observes the same live shadow supervisor first and latches the target phase's first observed physical tick. The handoff condition requires: current target phase, valid physical evaluator, tick divisible by8, no task termination, and elapsed target ticks at least `offset×8`. Leaving the target before the offset produces an explicit miss; no phase or pose is forced. Here all three sampled ticks satisfy the observed target-phase/time part of that proposed P06 condition. P06 history/entry predicates remain the production physical rules.

At handoff, `SemanticControllerAdapter.from_live_prefix` receives the same supervisor and the actual previous dispatch nominal/tracking; the teacher object is released rather than advanced/forced through an old gate. The physical core, mapper, reader/contact, observation/reward and residual/bridge histories continue; no old pose is restored. Teacher raw residual iszero throughout reset-only preparation. A new semantic nominal provider is seeded from that real receipt, not restored from a historical full-state snapshot.

If a teacher controller bias is still present, the existing bounded takeover retires it and requires the actual previous dispatch to be bias-free before READY. Therefore4384/4864/5184 are requested **handoff-eligibility** times on the observed teacher prefix, not proof that a separately executed P06-offset course would open PPO credit at exactly that tick. The three candidate rows do not persist the complete bias/ACK needed to establish that finer point. Prefix time and any takeover consume the same200s physical horizon; no successful handoff invokes core.reset (`semantic_prefix.py:315–388`). A legitimate miss has the existing single freshP01 fallback, not retry-until-success.

## Minimal follow-up choice, not a new gate

All three offsets are supported existing options and were observed before the teacher's P07 handoff; no unique pose, contact template, extra dwell condition or implementation is necessary. These logs provide no defensible geometry-based ranking of100/160/200. If a later P06-offset course is selected, the smallest informative check is to use **one** existing offset (for example160 as a middle sampling choice, not a validated optimum) and inspect its already-written actual `policy_credit_start` sensor/evaluator receipt alongside the subsequent ordinary PPO samples. That supplies true entry front distances, support and histories without a separate probe or new instrumentation gate.

No such offset course was launched by this task. The current training/evaluation/P06/P10 ordering remains parent-controlled and unchanged; this report is only fixed-prefix evidence for later sampling selection.
