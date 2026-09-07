# Candidate 02 — bounded desired active-reference correction

Report-only prototype, not an approved control change. Production, configuration,
the frozen mapper, and the repository test suite were not edited. Candidate 01
and `cpu_trial_01.json` are preserved unchanged.

## Exact mathematical scope

For each currently scheduled servo with nonzero previous effective residual,
use the current clamped canonical nominal `n`, actual measured physical q, original
standing/sign mapping, and original gain K / tracking limit C:

```
e0 = (standing + sign*n - degrees(q_actual)) / sign
c0 = clip(K*e0, -C, C)
c1 = clip(K*(e0 + previous_effective), -C, C)
allowed = [max(-C, min(c0, 0, L-n)), min(C, max(c0, 0, U-n))]
bounded_c1 = clip(c1, allowed)
```

L/U are the original hard limits inset by the existing 2 degrees. This is the
convex hull of the zero-anchored reserved-band correction interval and c0,
intersected with the original tracking correction limit. It forbids expanding
the desired correction beyond this hull; it does not forbid the original
current-load correction c0 from already lying outside the reserved band.

If c1 is not clipped, candidate 01's computational reference is retained exactly.
Only when it is clipped, invert the frozen feedback arithmetic:

```
q_computational = radians(standing + sign*n - sign*bounded_c1/K)
```

No history / zero gain / inactive channel passes the original q unchanged.
Ended-tracking retirement sees actual q because ended and currently scheduled
names are disjoint. There is one call to the original mapper, no copy of its
advance loop, no private state mutation, no second mapper, and no sensor write.
The original previous compensation, feedback sampling, nominal slew, tracking
restart, final 1.25-degree slew, and hard limits are retained.

**c0 is a same-current-measurement comparator with shared mapper history, not a
separate B trajectory.** This desired-correction bound is not an instantaneous
clamp on old compensation. An already outside-band compensation still follows
the original sample clock/slew on withdrawal. It is also not a guarantee that
subsequent geometry or controller-bias composition stays inside the reserved
band: this isolated prototype only bounds the mapper reference component.

## Executed bounded CPU receipt

Root explicitly authorized one pure CPU invocation while the separate real
training continued. `CUDA_VISIBLE_DEVICES` was empty, `PYTHONPATH` was the project
src directory, and Python used `-B -P`. The imported guard rejects Torch, Isaac,
and Omni modules. No simulator, optimizer, actor, CUDA device, or RobotAdapter
instance was constructed. Only original pure arithmetic functions were reused.

- Files: `bounded_active_reference_candidate.py`,
  `validate_bounded_reference_candidate.py`, `cpu_variant02_trial01.json`, and
  its `.stderr.txt` / `.stdout.txt`.
- 10 tests passed in 0.435 seconds; whole PowerShell invocation 3.447 seconds.
- Owned PID 182220 exited 0; subsequent read-only process check found it absent.
- The JSON records hashes of both candidate versions and their harnesses.

### Counterexample response

For blocked hip q=nominal=+130 degrees and request +3.7 degrees, candidate 01 had
476/960 ticks with native baseline beyond the reserved +133-degree boundary.
Candidate 02 had **0/960**, with maximum excess 0 and last native/final +133.
The mirrored -130/-3.7 case also had 0/960 outside-band ticks and maximum excess
0. Both had 484 clipped-reference ticks. Maximum inverse reconstruction error
was 0 on the upper example and `1.1368683772161603e-13` degrees on the lower.
Tiny nonzero residuals from subtraction remain explicitly visible; no epsilon
threshold was added to silently discard them.

The tests also verify that an original load error requiring native ±135 is not
relocated to ±133, and that prior outside-band compensation is not instantly
erased. Thus the prototype does not claim a new global reserve invariant.

### Far-from-boundary comparison

All five held trials had zero desired-reference clips and reproduced candidate
01's recorded offset statistics exactly:

| Held request | Original mean offset | Candidate 01 = 02 mean | Candidate peak-to-peak |
|---|---:|---:|---:|
| FL hip +20 deg | +10 | +19.8124845321 | 1.25 deg |
| FL hip -20 deg | -10 | -20.1337890625 | 1.25 deg |
| FL hip +3.7 deg | +0.575 | +3.5055572510 | 1.25 deg |
| RR knee +20 deg | +10 | +20 | 0 deg |
| RR knee -20 deg | -10 | -19.8818359305 | 1.25 deg |

The plant is still the deliberately idealized float64 scalar model: free q
follows the final target immediately on the next tick; blocked q does not move.
The remaining oscillation is not cured. No claim follows about actual robot
contact dynamics, stability, control speed, task success, or PPO learning.

Coverage includes upper/lower blockers, all eight servo signs and clamped nominal,
outside-band original c0 allowance, knee nominal -58 with rejected previous
request (effective zero), exact no-clip equivalence, zero gain/history/inactive,
ended-tracking retirement, current-zero/previous-nonzero withdrawal, nominal
change/restart/clock, prior outside compensation, and invalid configuration.
No ACK/reset generation/geometry/native-buffer audit integration is implemented.

## Exact rerun command (requires root permission; new output name)

```powershell
$previousCuda=$env:CUDA_VISIBLE_DEVICES
$previousPythonPath=$env:PYTHONPATH
try {
  $env:CUDA_VISIBLE_DEVICES=''
  $env:PYTHONPATH='C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\src'
  & 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -B -P 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\tracking_reference_candidate\validate_bounded_reference_candidate.py' --output 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\tracking_reference_candidate\cpu_variant02_trial02.json'
} finally {
  $env:CUDA_VISIBLE_DEVICES=$previousCuda
  $env:PYTHONPATH=$previousPythonPath
}
```

The actual run was wrapped in an owned-process 10-second timeout with hidden
window and redirected output. No further invocation or production selection is
authorized by this report.
