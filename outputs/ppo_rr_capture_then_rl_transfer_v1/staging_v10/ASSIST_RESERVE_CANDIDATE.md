# v10 staged reserve permission — not applied or tested

Only static reads and these staging files were written while v9 Isaac sampling remained active. No production file, profile, source event, reward, sensor, checkpoint, process or rollout changed. No Python/Torch/test/physics/actor/PPO was executed for this review.

## Established code mechanism and evidence boundary

Production `semantic_rr_capture_assist.py` requires the 25 mm upper bound in **three** places: earning progress mode6 after a measured ≥0.2 mm peak-to-current decrease, retaining mode6 on subsequent valid/tracked samples, and validating its public snapshot. The same upper bound is **not** present in the initial reachable/ready permission or `rr_capture_transfer_context`'s live-descent recovery. This mixes a task contact-region band with permission to use already finite descent reserve.

Root supplied a sealed v9 first-episode row: global222486 / 78.6 s, public pre-peak .030055098814890588 m and current gap .02972366842018978 m, a 0.3314303947 mm decrease; the local window reset but after-mode stayed1, travel39.84166666666554°. This is a concrete progress-positive, above25 mm case. At the terminal row the .030255300658→.0301154162 pair decreased only .1399 mm: **that row alone cannot earn new credit**. A natural rerun under the revised law may earn credit at the earlier progress sample and retain it through40°, not retrospectively alter the blocked terminal snapshot.

The supplied hip/knee tracking errors are about .296°/.084°, below the existing3° cap; three other supports and current Q/cross/XY/valid permit search if verified on the relevant same tick. This file does not independently read the raw trajectory or assert the whole terminal context is also the complete78.6 s context. The corresponding staged test labels its other values as a synthetic valid fixture, not a full replay.

## Narrow candidate

`assist_production.patch` removes only the three **upper** reserve-gap restrictions; keeps current gap/window lower bound−.015 m; updates explicit feedback/search metadata and reason10 wording to `fresh_capture_progress_required` (numeric code unchanged). The initial40° law, current valid/XY/Q/cross/AIR/no-contact/two-other-supports/3° tracking gates, 2 active-second/0.2 mm peak-progress renewal, exact hip→knee boundary and limits are otherwise unchanged.

- Existing12° reserve remains52°/44 s maximum. The existing last1°/1 s still needs public peak≤1 mm and gives absolute53°/45 s. No budget reset, larger travel, new clock, timeout or hard-limit change.
- Any real contact still stops descent. TOP/bearing/place are still sensor/evaluator facts; no AIR sample gets support credit. Captured-follow and finite retirement are unchanged.
- Removing the upper ceiling permits reserve at **any positive gap** satisfying current geometry/support/progress checks, not just30 mm. This is the explicit behavioral change, not a claim of global kinematic feasibility. If gap is flat/rising, tracking/support is lost or total budget is spent, old stop rules remain.
- `rr_capture_transfer_context` already accepts qualified legal AIR gap≥−.015 and allows local continuation only with live DESCEND/DESCEND_PROGRESS. Supervisor additionally requires adjacent committed feedback; global200 s and physical safety win. No context/scheduler/sensor task-band change is needed.
- All14 public state fields/scales and410 observation remain, including mode6 and public peak; therefore no hidden new state. New control semantics still require an explicit same410 migration, fresh rollout, preserved full learned weights/Adam/LR/Identity/RNG/lineage, and official save/reload from the **actually sealed new checkpoint**, not a terminal-physics-state edit.

## Integration and focused verification

Staged normal-import file `test_semantic_rr_capture_progress_reserve_v10.py` includes the measured gap pair, insufficient terminal credit, continuous boundary crossing, above-band valid and invalid gates, unchanged−15 mm floor, two-second stall, contact/no recharge,52/44 vs53/45 split,2° joint reserve, snapshot replay and scheduler-context non-contact facts. It has **not run**. Root should apply only after the current training update seals, then run it with existing contact/dispatch/current-support/global200s safety tests.

Existing `test_semantic_rr_capture_progress_incremental.py:test_near_top_gate_rejects_far_gap_even_after_true_drop` deliberately asserts the old ceiling and needs a versioned replacement, not deletion of physical negative cases. Exact old feedback/reason strings in tests need explicit v10 updates; many older knee/peak tests also already assert historical v4 revision. Do not mechanically treat stale numeric40° assumptions under continuously descending >25 mm samples as unchanged behavior. Historical migration validators/receipts must remain intact; new same410 migration handles v10. `semantic_backend.py` requires the execution profile's feedback revision to match the new literal.

Proposed constants shared with migration author:

```text
RR_CAPTURE_FEEDBACK_REVISION = progress_earned_capture_reserve_incremental_v10
RR_CAPTURE_SEARCH_SEMANTICS = hip20_knee20_progress_earned_current_XY_support_tracking_gap_ge_minus15mm_knee12_then_1deg_public_peak_gap_le1mm_total53_exposure45_no_recharge_AIR_not_contact_sensor_TOP_unchanged_captured_issued_N_request_deltas_RL_current_TOP_retirement
```

No claim that the revised reserve will actually produce TOP, retain it through P12 or complete RL. This fixes a demonstrated permission exclusion while retaining finite physical-feedback stopping criteria.
