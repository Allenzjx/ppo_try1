# RR support-wheel projection: implemented, not physically validated

Only the new production module `semantic_rr_carry_wheel.py` and its new unit test were authored by this subtask. Backend, adapter, independent audit and profile wiring belong to the root task. No simulator, actor forward, fitting, checkpoint or commit was performed here.

`MODE=rr_capture_support_forward_projection_v1`. The current implementation supersedes the earlier design note's 3 rad/s² choice and any suggestion that depth gain zero disables the floor: the applied slew is the existing **1.8 rad/s²**; in the qualified capture window, desired floor remains nonnegative when gain is zero.

## Integration contract

- Call `build_rr_carry_wheel_context` after the unique source evaluation and before the actuator write. Supply current task/observation/source frame, actual provider, physical dispatch tick, real pre-dispatch ACK and independent pre-write count. `mode` defaults to `off`; opt in explicitly. Actual residual wheel rate can be passed and must equal the existing 1.8.
- Call `project_rr_carry_wheels` on the unchanged-hard-bounded pre-envelope candidate Full12, with the independently read pre-dispatch FINAL wheel4 and 1/120 s. It returns `output_full12`; all eight servos and RR wheel pass through unchanged.
- Commit its receipt as `ACK['rr_carry_wheel_evidence']` **even when unarmed**. Next tick checks the prior committed source evidence, adjacent source/dispatch clocks, write count, one write, prior nominal request wheels zero, source endpoint and final authored stop. Missing evidence conservatively bypasses. This is issued-source + actual ACK sequencing under the one-write/one-step contract, not a claim that measured wheels stopped.
- Receipt keys: `context`, `previous_final_wheel_rad_s`, `candidate_full12`, `desired_before_slew_full12`, `output_full12`, `selected_indices`, `envelope_active`, `actual_projection_changed`, `controller_delta_full12`, `source_evidence`. `context['selected_full12_indices']` identifies the selection before projection.
- X409 reports `context['envelope_active']`: the current floor/release envelope is armed, **not** that output changed or traction occurred. Previous FINAL four-wheel values already exist in action-history observations. Full contact classes and source internals are not individually encoded; do not claim the complete 410-vector is proven Markov.
- Independent audit must replay actual-current-policy and zero-current-policy candidates through this same function with one identical context and independent previous FINAL. Do not read the new receipt and call it the previous target; do not subtract a separately advanced mapper result.

## Scope and exits

Qualified current RR AIR/Q/cross + current legal XY/lateral/nonnegative gap + FL and another verified support, after committed P09 finite-source stop: only currently bearing FL/FR/RL receive desired `max(candidate, .3*w)`. `w=clip((.06-depth)/(.06-.005))*clip(gap/.015)` uses existing spec scales. The first -0.93 rad/s FL target becomes -0.915, not an instantaneous nonnegative jump. AIR wheels are never called tractive supports.

Current TOP contact receives **no zero or positive floor**. While still P09 with source/support/safety eligibility, TOP or loss of current Q/XY/lateral gives only bounded return toward original candidate. GROUND, wall, missing authoritative validity, fresh/unfinished source ownership, another phase, or safety terminal bypasses the envelope and yields immediately. No local timeout or phase completion is extended by this module.

## Verification

45 focused CPU tests passed. Fixtures use the real compact P09 MotionExecutor endpoint and following source sample plus explicitly synthetic sensor/ACK values. Tests cover source-proof negatives, default-off identity, current-support selection, taper/floor/slew, TOP/release, source stop precedence, identical counterfactual replay, unchanged servos/RR wheel, finite units and unchanged hard bounds. This is not measured advancement/capture success; new-version physical diagnosis remains necessary.
