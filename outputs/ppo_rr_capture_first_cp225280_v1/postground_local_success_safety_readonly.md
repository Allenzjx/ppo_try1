# Post-GROUND RR capture success: narrow safety semantics review

**Confirmed predicate gap; not an observed false physical success.** The reported episode4 has returned to GROUND but has not subsequently regained TOP in the evidence supplied for this check. No production/config edits, Torch/PXR imports or simulation were performed.

1. **Current lift revocation works.** In the selected functional RR evaluator, every real GROUND observation clears the current attempt, joint-demand accumulator, sample window and initial clearance. `_observe_functional_rr()` also clears `_rr_established`; a fresh unsupported AIR rise with measured whole-body actuation must re-establish it. `current_lift_valid` consequently becomes false. Completed `front_edge_crossed`/`placed` remain historical, as intended. References: `semantic_supervisor.py:837–860`, `1168–1179`, `1198–1218`.

2. **The pure-wheel terminal is not complete for a post-cross retry.** Current powered FRONT_WALL ascent is accumulated, but its actual failure check is inside `distance >= 0 and not history.front_edge_crossed[leg]`. With an earlier completed crossing, that branch is skipped even after GROUND revoked the current attempt. The semantic backend only latches the semantic controller's `TASK_FAILURE_WHEEL_ONLY_CLIMB`; the legacy wheel-only guard is retained as a diagnostic, not an independent terminal. Other safety guards remain enabled, but do not fill this specific qualification gap. References: `semantic_supervisor.py:940–1009`; `semantic_backend.py:434–470`.

3. **Local success can reuse the old crossing/placement without new lift.** `extract_rr_metrics()` validates current sensor TOP and bearing and exposes `current_lift_valid`, but `RRCaptureLocalTask.local_success` only requires the latched active gate, current bearing, historical placed/crossed, TOP sample count and0.5s hold. It does not require current-attempt lift establishment. GROUND resets hold/potential, not the active gate, which is correct; the missing safeguard is success eligibility after the reset, not gate deactivation. References: `semantic_rr_capture_local_task.py:115–155`, `198–214`.

## One directed counterexample

Using only the existing stdlib unit fixture and current production local-task class:

`qualified AIR → short real-style TOP/placed history → GROUND with revoked lift → 61 consecutive TOP/bearing observations, with current_lift_valid=false, active_attempt=false, lift_established=false throughout the renewed contact`.

Observed arithmetic result: after GROUND, active=true/hold=0/local_success=false; after the61 renewed TOP observations, active=true/hold=0.5/local_success=true. This reproduces the **local predicate** vulnerability with synthetic inputs. It does not prove that these states occurred in Isaac, nor test the physical feasibility of a wheel-driven wall ascent.

The directed integration negative case should additionally exercise the actual evaluator: begin with earned crossing/placement, return to GROUND, provide verified powered FRONT_WALL ascent without fresh unsupported AIR lift or active joint response, then verified TOP hold. Expected: no local success based on the old attempt. A positive counterpart should allow a genuinely re-established whole-body AIR lift followed by contact/hold, while retaining the active policy gate and original completed-event timestamps.

## Minimal correction direction, not applied

Keep the local active gate latched. Require a **fresh, same-attempt lift-established qualification after any ground return** for capture hold/success; the existing evaluator's `lift_established` latch already has the correct GROUND revocation/re-earning lifecycle. `active_attempt` alone represents initial clearance, not necessarily qualified lift. Do not blindly require `current_lift_valid` at every TOP sample: that field additionally depends on current other-support/geometry evidence and could reject a legitimate subsequent support transfer. Expose the chosen current-attempt state explicitly if needed rather than hide a new success latch.

Separately test the existing positive-evidence powered-wall failure detector for post-cross retries. Missing fresh qualification must mean “unfinished,” not automatically “pure-wheel failure”; the latter still requires measured causal wheel/FRONT_WALL evidence. No source timing, reward, exploration, checkpoint or running rollout was changed by this review.
