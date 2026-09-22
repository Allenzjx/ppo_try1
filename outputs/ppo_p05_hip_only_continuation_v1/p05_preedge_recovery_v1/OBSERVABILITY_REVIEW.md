# P05 pre-edge recovery: bounded observability review

Read-only review of production `336b7c56d2f048d44c866b2357d33e1eb1f21dce` and this directory's **unapplied** `production.patch`. No migration, simulation, optimization, or production edit was performed for this review.

## Conclusion

**The 389-vector is not claimed to be a complete Markov state.** It exposes the current recovery nominal advice, physical/task summaries, action history, and mapper state, but does not individually encode all source scheduler internals or full contact classification.

The new gate adds no independent latch, resettable clock, or mutable recovery timer. Its 30–40 s window uses existing stage age. The provider evaluates the new suggestion before constructing the frame; `SemanticObservationBuilder` then encodes the resulting current full12 nominal. Therefore all newly changed nominal values are visible to both actor and critic, including the four wheel values. This is visibility of **current advice**, not a proof that every future scheduler transition can be inferred from one observation. Migration correctly declares `same_mdp_claimed=false`.

## Existing observation mapping

Indices below are zero-based and inclusive. Values use the schema's fixed scales.

| Evidence | Indices / limitation |
| --- | --- |
| P05 phase indicator | 4 |
| Simulation time, remaining task time, stage age | 18, 19, 20; each divided by 200 |
| FL conservative gap and front distance | 21, 22; metres |
| Evaluator bearing-based leg load fractions | 23 / 26 / 29 / 32, FL / FR / RL / RR |
| Evaluator support count | 33 |
| Wheel geometry | 99–110: per-wheel bottom-front x, bottom-back x, bottom-top z; no wheel-centre y field |
| Raw pair normal-force magnitudes | 123–130, ground/obstacle per leg; not full bearing-force vectors |
| Exact ground/obstacle pair-active flags | 131–138; FL is 131/132 |
| Aggregate raw support diagnostics | 143–145: count, valid, margin |
| FL qualified lift / crossing / placement; FR placement | 146 / 150 / 154; 155 |
| Current nominal full12 | 171–182; wheel values 179–182 divided by 2.1 |
| Previous-dispatch mapped nominal full12 | 183–194 |
| Previous raw, residual, prior residual, applied, prior applied | 195–254 |
| Previous nominal full12 | 255–266; wheel values 263–266 divided by 2.1 |
| Mapper feedback phase | 323; modulo-8 feedback phase, not source owner/clock |
| Transfer-role summaries | 324–371; not source scheduler or exact per-leg contact-classification bits |
| Capture assist state | 372–383 |
| FL pending / allowed continuation / scheduler advanced / local warning / pending elapsed | 384 / 385 / 386 / 387 / 388; not source endpoint or fresh wheel-owner flags |

`source_endpoint_issued`, the prior observation tick, per-layer source ticks, atomic groups, and fresh wheel-owner identity are present in provider state/diagnostics but are **not independently encoded**. Likewise, `bearing_verified`, exact TOP-surface classification, and lateral-validity/centre-y evidence are not individually encoded by the listed summaries. These are existing interface limitations; the review does not claim equivalent reconstruction from aggregate values.

## Actual supported prefix path

For natural P01 and `checkpoint_policy` / `successful_nominal` phase-suffix training, `semantic_cli.py` selects the ordinary `SemanticIsaacBackend` and `CheckpointPolicyPrefixRslAdapter`. `_CheckpointPolicyCreditCore._roll_in()` repeatedly calls the **same** `core.step(raw)` and only opens the PPO-credit boundary at the requested entry. It neither clones/replaces the controller nor calls `NominalMotionProvider.from_handoff`; source clocks, action history and mapper state continue.

P01–P05 source clocks are not paused by `_sequence_permission`; their fixed source tails end before the new P05 age-30 window. P05's last atomic event is approximately 9.07 s and its source tail approximately 9.73 s, with normal time scale 1. Contiguous physical ticks are already enforced by the evaluator/backend. Thus in these supported continuous paths the source-endpoint/prior-tick/fresh-owner checks are chiefly integrity guards, not a newly introduced hidden recovery latch. No new erroneous trigger or perpetual exclusion was identified for those paths by this code review; this is not a physical validation result.

The legacy `frozen_fsm` prefix is different: `PrefixSemanticIsaacBackend` may call `SemanticControllerAdapter.from_live_prefix()` → `NominalMotionProvider.from_handoff()`, which restarts the current source phase while retaining supervisor stage age. A hypothetical P05 handoff at age 35 s would not finish a restarted roughly 9.73 s source before the age-40 recovery window expires. This is a code-derived limitation, not an observed run. Continue using the existing continuous checkpoint/nominal prefix paths; do not silently extend this timing claim to a late legacy handoff.

Minimal recommendation: describe observability as **current nominal advice visible, source/contact internals only partially represented**. No new dimensions, observation re-encoding, tests, or additional control mechanism are proposed by this review.
