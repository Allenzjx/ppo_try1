# Unapplied local-capture lineage predicate v2

Status: **isolated preparation only**. Production/local training remains `ecf205e`; no Torch/PXR or simulator was imported. This candidate is independent of the separate P09 late-scheduling experiment.

## Correct state, including the actual positive case

Use the physical evaluator's `RR.lift_established`, not current AIR/swing validity and not irreversible active-lift history. `_observe_functional_rr()` clears this current-attempt latch on **every GROUND** observation and re-earns it from fresh measured unsupported AIR rise, whole-body actuation and existing body/support checks. At TOP it can remain true even when `current_lift_valid` becomes false because that latter field also depends on current other-support/geometry availability.

The candidate defines the explicit bit:

`current_attempt_capture_eligible = live AND lift_established AND NOT ground_contact AND motion_continuation_allowed`.

It gates accumulated qualified capture hold, local success, and the RR task's `real_contact` potential, but does not clear the active policy gate. An unqualified TOP/bearing observation has zero **task contact potential**, not a fabricated zero contact measurement. Current TOP/bearing metrics and their two observation fields stay raw physical facts; absence of qualification is **unfinished**, not automatically a wheel-only violation. Missing `lift_established` is rejected rather than filled from old history. Legal geometry/gap potentials and all reward coefficients remain unchanged.

## Current recorded success audit (one fixed snapshot)

`train_first2048_ecf205e/decisions.jsonl`: opened/seek-END/tell snapshot248257875 bytes,5965 complete rows,1746 actual PPO rows. Exactly **one** recorded local-success row: tick9496/79.133333s,0.5s hold,61 TOP samples,13.2608976N; `lift_established=true`, `active_attempt=true`, `current_lift_valid=false`, `motion_continuation_allowed=true`, current TOP/bearing verified. It remains accepted by this candidate. No other success was present in this fixed snapshot; this is not a statement about later appended rows. The observed episode4 ground return was not a local success.

## One additional observation is required

The original422 schema encodes `active_lift_history` at indices146–149, but the RR slot149 is explicitly overwritten with **current_lift_valid** (`semantic_supervisor.py:1915–1923`). It does not expose `lift_established`. The7 RR context fields at403–409 and9 source-role fields at410–418 do not expose it either; they contain derived carry/reachability/contact/bearing/role/timing fields. The additional439-owner tail is explicitly17 zero fields in this route. None uniquely identifies the retained same-attempt establishment at TOP.

Therefore append **one Boolean local field** at index447, preserving every original prior input: `439 + 9 = 448`. The new field is the complete `current_attempt_capture_eligible` predicate, so the additional continuation/live condition is not hidden from actor/critic. The existing8 local fields retain order and meanings; `capture_hold_progress` remains index446 and is now progress only for eligible capture. The staged task API names are deliberately `obs9()` and snapshot `obs9`, not a misleading9-element `obs8()`.

`local_task.patch` changes only the task module. It is **not deployable alone**: root must version/adapt route encoding, actor/critic dimension checks, request receipts, metadata/schema and compatible checkpoint migration together. Keep frozen prior439 byte-identical; preserve old local/critic input columns and initialize only the new input column to zero, with corresponding Adam moments expanded by zero while all existing moments/steps are retained. Preserve LR/RNG and use a fresh new-semantic rollout after the normal saved update boundary. This document does not implement that migration.

## Directed tests

`C:\Program Files\Python313\python.exe outputs/ppo_rr_capture_first_cp225280_v1/staged_capture_lineage_v2/test_capture_lineage.py`

**9 passed**: normal AIR→TOP hold; real-positive-compatible TOP with `current_lift_valid=false` but retained establishment; TOP→GROUND→unqualified TOP rejects success and retains gate; GROUND→fresh qualified AIR→TOP succeeds; missing lineage rejected; continuation unavailable resets hold without clearing gate; lack of qualification does not fabricate pure-wheel failure; original8 local columns preserved for the valid path; unqualified contact potential is zero while sensor facts stay true and the full valid AIR→TOP→successful-hold reward path is exactly unchanged from v1.

Tests use synthetic observations and the existing fixture, not a simulated wall ascent. `git apply --check` passed for the patch, but it was not applied. Separate future evaluator integration must test post-cross powered-wall retry detection using positive physical evidence. No actor, optimizer, checkpoint, active rollout, source schedule or reward coefficient was changed here. The staged contact-potential qualification is explicitly a task-reward correction, not a relaxation or rewriting of sensor contact facts.
