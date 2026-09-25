# RL EDGE / issued-owner focused verification

2026-09-24. CPU tests only; no Isaac, optimizer update, physical recovery or new success claim.

## Changed interface

`semantic_rear_policy_timing.py::rear_dependency` adds opt-in
`rr_rl_edge_recovery_v4`. `rl_current_swing` remains AIR-only.
`rl_edge_recovery_permitted` requires the real evaluator's current, same-attempt
lift qualification, no ground, exact current obstacle reaction, and
FRONT_WALL/OBSTACLE_AMBIGUOUS rather than TOP. `rl_motion_continuation_permitted`
is AIR or this qualified EDGE continuation. The existing public
`rl_swing_capture` is a task-work flag, not a claim of AIR, contact support or
success. Old v1/v2/v3 behavior is retained; v4 inherits current recapture logic.

Ground revokes the current attempt even if historical lift/cross/placed remains.
A later AIR or wall contact cannot reuse those historical events; a newly
measured qualifying lift can restore permission. No new contact/lift latch,
pose target, force or placement event is fabricated in this module.

## Contact consumers checked

| Actual consumer | Scope / result |
|---|---|
| `BodyCollisionDetector.evaluate` (`sensing/body_collision_detector.py:34`) | Uses the central body / obstacle exact pair and corroboration. Leg/wheel contact is not this detector's body collision. |
| `physical_contact_surface` (`semantic_physical_sensing.py:197`) | Validated force/contact evidence distinguishes wall, ambiguous obstacle and TOP. Reaction is not automatically verified bearing. |
| `TaskEvaluator.observe` (`semantic_supervisor.py:741`) | Body collision and independent safety still terminate. Current RL ground revokes qualification. Wheel-only failure is a measured wall-ascent process without required lift, not one wheel-contact or reverse-wheel sample. |
| `rear_dependency` / `public_timing` (`semantic_rear_policy_timing.py:29,83`) | AIR-only downstream continuation was the interface restriction repaired here. Qualified EDGE is separate from RR-bearing permission. |
| `NominalMotionProvider._sequence_permission` (`semantic_supervisor.py:2330`) | Strong P12 source-lane permission remains current RR support or actual ongoing RL AIR; EDGE alone does not release that lane. Once started, source wheel pulse/stop clock continues while RL joint clock pauses without catch-up. |
| `_rr_late_reconfiguration_ready` (`semantic_supervisor.py:2537`) | Current bearing remains the late-group start gate. A started P09 group is not retroactively un-emitted; new issued-owner suspension separately stops pursuit of its old FL/RL servo goals. |
| `_terminal_reason` (`semantic_env.py:48`) | Consumes explicit physical safety/task failure or success, not a generic RL contact flag. |
| `EndpointObserver.__call__` / `capture_semantic_video` (`semantic_video.py:522,733`) | Video ends on real evaluator terminal/success or stated video bound; no separate RL EDGE contact stop was found here. |

This change therefore does **not** claim removal of a previously existing
generic “RL contact => done” condition. The concrete fix is downstream motion
permission plus suspension of already-issued support-dependent targets.

## Issued-owner integration and tests

Root-owned production path was independently exercised using the real
`SemanticIsaacBackend._atomic_apply`, residual adapter, mapper, tensor-backed
fake robot setters and `actuator_target_effect` native audit. This is actual
command plumbing / tensor readback, not floating-body physics.

- Constant policy request holds previous committed FINAL while old N continues.
  Changing that request produces both positive and negative bounded responses.
- New nondependent owner / current RR bearing releases suspension. Current RL
  AIR releases only RL channels, not FL's RR-dependent strong target. EDGE
  retains independent policy adjustment but does not release strong source goals.
- Explicit source wheel stop remains zero; no wheel channel is owned by this
  servo suspension. One mapper/actuator dispatch path and one write remain.
- Previous FINAL and previous independent policy request are separate. An
  adjacent committed ACK is required; independent backend pre-dispatch snapshot
  catches ACK-only state/context/previous-request/previous-final tampering.
- P09 late owner remains identified after loss; group is not replayed. P12
  already-issued RL owner survives its cursor wait, source stop is consumed once,
  and resumption advances one tick rather than catching up elapsed wall time.

During review two issues were reported and fixed by root: a global RL-AIR
exception had released FL as well as RL, and owner provenance lacked an
independent pre-dispatch receipt / explicit adjacent tick check. Regression also
found an undefined `LIVE_SWING_MODE` reference in the updated supervisor;
root corrected it to the active spec mode.

Final targeted run: **92 passed in 5.10 s**, base Python,
`CUDA_VISIBLE_DEVICES=-1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, repo `src` on
PYTHONPATH. Included new owner integration 21 cases, EDGE 27 cases, prior v3/v2
behavior, two source-stop/cursor cases and two central-body-contact safety cases.
Historical v1/v2/v3 test setup alone disables unrelated newer v5 collider
proxies; requested EDGE-mode fixtures keep current requirements.

No additional blocker was observed within this bounded scope. Physical contact
recovery, tracking under load, sustained support and task completion still require
the next actual frozen-version simulation; these CPU results do not establish them.
