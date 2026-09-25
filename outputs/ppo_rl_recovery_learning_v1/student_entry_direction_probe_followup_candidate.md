# Student-entry probe follow-up: one bounded candidate

Status: **draft only**.  This does not alter the formal task, production
controller, checkpoint, or training credit, and it has not been executed.

## Why v2 was too short

The sealed v2 probe armed with almost no longitudinal margin: its trigger row
had RR front distance `+0.578 mm` and gap `58.988 mm`.  It applied only three
15 Hz decisions (`0.200 s`, ticks 7984--8008), then released solely because
front distance was `-0.139 mm`.  RR was still qualified AIR, within top XY and
lateral span, FL/FR/RL were real supports, gap was `63.455 mm`, and there was
no RR TOP load.  Thus the release was truthful under the old exact-zero gate,
but too early to test a whole-body response.

The same sealed episode later supplied a better *observed* entry, without
interpolation: at decision-start tick 8264 / `68.866667 s`, RR was qualified
AIR with front distance `+51.709 mm`, gap `15.168 mm`, within both XY gates,
and FL/FR/RL recorded as supports.  This is evidence that waiting for real XY
margin is feasible; it is not evidence that the proposed intervention works.

There was also a material wheel confound during ACTIVE.  The student FL-wheel
raw request remained about `-1.204..-1.199`, while mapper nominal N was
`+0.300 rad/s`; executed FINAL stayed about `-0.702..-0.700 rad/s` and measured
speed was negative on the first/last selected rows (`-0.932`, `-0.879 rad/s`).
Leaving that learned residual untouched does not test nominal forward motion.

## The single candidate

Create a separate outputs-only follow-up runner; keep the existing v2 runner
and evidence immutable.

**One-shot entry (decision-start evidence only):** wait in P09 until all old
qualification, AIR/no-contact, evaluator-valid, current-lift, history,
lateral-span, FL-support, and minimum-other-support checks pass, and additionally
require:

- `within_top_xy == true` and finite `front_distance_m >= +0.010 m`;
- finite `clearance_m >= +0.010 m`.

These are diagnostic arming margins, not new task success thresholds.

**Absolute raw candidate while ACTIVE:** FL knee `+0.25`, FR knee `+0.10`,
RR hip `-0.20`, and FL wheel `0.00`.  The first three values are unchanged
from v2.  FL-wheel zero is a residual-neutral isolation: mapper N, projector,
caps, slew, owner projection, and the single native write remain authoritative.
It does **not** assert that FINAL or measured wheel speed instantly equals
`+0.300 rad/s`; raw, mapped N, ACK FINAL, and measured speed must all be logged.
RR knee and the other seven channels remain the current student's request.

**ACTIVE release:** keep the existing five-second overall limit and one-shot
semantics.  Preserve contact-specific classification first, then immediately
release on any of the following:

1. RR ground return, RR TOP/other real contact, or loss of AIR;
2. task terminal/body collision, invalid evaluator, or phase leaving P09;
3. loss of current RR lift qualification, lateral/top-XY membership, positive
   gap, FL real support, or the required number of other real supports;
4. nonfinite geometry, or `front_distance_m < -0.002 m`.

Only `front_distance_m` receives hysteresis: if it is in
`[-0.002 m, 0)`, remain ACTIVE for at most `0.5 s` continuously, reset that
timer once it is nonnegative, and release if the grace expires.  No other
eligibility or safety blocker gets a grace period.  The runner must record the
first negative-front tick, recovery/release tick, every blocker, and the exact
contact/body-collision reason.

This single candidate should produce enough decisions to observe joint/wheel,
support, RR gap/front, and COM response when geometry remains safe, while a
real ground return, collision, support loss, or genuine XY departure still
ends it immediately.  It remains a zero-PPO/zero-AUX directional diagnostic,
not a controller proposal, success label, or training target.

Evidence authority: `student_entry_direction_probe_v2_analysis_ACK_FINAL.json`
(executed FINAL only from `atomic_ACK.drive_target_full12`; mapper N only from
`native_readback_audit.native_drive_target_full12`).
