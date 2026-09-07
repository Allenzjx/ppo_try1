# Unwired active-tracking reference candidate — bounded CPU trial completed

Prepared while real Isaac evaluation was active. Subsequently the root explicitly
authorized one under-10-second pure CPU exception with CUDA hidden. No GPU, Isaac,
production edit, or production integration was executed by this task.
Source at preparation: `e99fde1b3e8366f0ff1d484140b82877df745f0c`.

`active_reference_candidate.py` is pure: it accepts actual physical joint radians,
the caller-declared previous effective servo residual, and current scheduled
tracking names. It returns a distinctly labelled computational reference,
`q_actual - sign*radians(previous_effective)`, only for scheduled channels.
Standing cancels; rear signs are negative. It changes no observation/tensor,
mapper state, history, nominal, wheel, or physical target. Zero reference preserves
the original float values, including signed zero. It does not validate that a
claimed previous residual actually belongs to the immediately preceding ACK.

The frozen mapper uses measurement only for active feedback and ended-tracking
retirement; ended names cannot be currently scheduled. Thus the candidate leaves
retirement's measured input real. The harness calls the existing mapper once;
there is no copied state machine, second advance, monkeypatch or corrective write.
The helper can have no control effect while nominal slew is unfinished or between
feedback samples. A one-tick reference withdrawal does not erase accumulated
mapper compensation.

## Deliberately limited experiments

The harness uses actual frozen gain 8 / four-tick feedback / compensation cap 10,
current pure headroom, and the original final hard/slew function. Requested offsets
ramp at the configured 0.5 degrees/tick. Free toy joints exactly follow each final
target on the next tick; blocked joints do not move. These are float64 scalar
models, not robot dynamics, actuator response, float32 readback, or contact physics.

It compares held ±20 degrees on FL hip and RR knee, and +3.7 degrees on FL hip,
reporting last-240-tick mean/min/max, oscillation range/std, and compensation.
Tests assert mathematical invariants, not candidate improvement. Nonzero standing
and rear sign conversion are exercised. Tests also cover zero equivalence, active
versus ended retirement, clipped-to-zero request, nominal change/restart, one
feedback-clock advance, current-zero/previous-nonzero, and invalid inputs.

An explicit blocked-contact counterexample uses hip nominal/actual 130 degrees
and requested +3.7 degrees. It checks whether previous-effective reference can
push mapper baseline past the reserved 133-degree boundary while the original
135-degree hard limit remains enforced. This expected limitation is not hidden,
and its test is not a reason to approve production. A separate counterexample
shows a sustained 20-degree target can have only a 1.25-degree same-history
actual-minus-zero-policy target delta: incremental audit effect is not the full
held target reference.

No ACK/reset generation, previous-tick binding, actor observation changes,
geometry context, bridge, real dispatch, or native audit integration exists here.
No new policy-reference-specific reserve projection is implemented. Selecting
that behavior would require a separate explicit design/MDP decision.

## Exact command — only after the root clears the live barrier

From PowerShell (no need to start Isaac or import Torch):

```powershell
$previousCuda=$env:CUDA_VISIBLE_DEVICES
$previousPythonPath=$env:PYTHONPATH
try {
  $env:CUDA_VISIBLE_DEVICES=''
  $env:PYTHONPATH='C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\src'
  & 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -B -P 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\tracking_reference_candidate\validate_tracking_reference_candidate.py' --output 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\tracking_reference_candidate\cpu_trial_02.json'
} finally {
  $env:CUDA_VISIBLE_DEVICES=$previousCuda
  $env:PYTHONPATH=$previousPythonPath
}
```

The output is exclusive-create and records hashes of the actual source files
used. A rerun needs a new output name. Root chooses whether/when to execute again;
the exceptional permission for the first bounded CPU invocation is not permission
for arbitrary Python or another simulator alongside the live evaluation.

## Actual CPU receipt (not physical validation)

`cpu_trial_01.json` and its `.stderr.txt` contain the first actual result:
11 tests passed in 1.679 seconds; entire PowerShell invocation 3.896 seconds.
Owned CPU PID 163044 exited 0 and was then absent. CUDA was hidden, and the module
guard found no Torch, Isaac Lab, Isaac Sim, or Omni import. No optimizer ran.

The last 240 ticks of each 960-tick free-plant experiment gave:

| Servo / held request | Original mean offset | Candidate mean offset | Original / candidate peak-to-peak |
|---|---:|---:|---:|
| FL hip +20 deg | +10.000000 | +19.812485 | 0 / 1.25 deg |
| FL hip -20 deg | -10.000000 | -20.133789 | 0 / 1.25 deg |
| FL hip +3.7 deg | +0.575000 | +3.505557 | 1.25 / 1.25 deg |
| RR knee +20 deg | +10.000000 | +20.000000 | 0 / 0 deg |
| RR knee -20 deg | -10.000000 | -19.881836 | 0 / 1.25 deg |

Thus even ideal instantaneous target tracking exhibits candidate oscillation in
four cases; the prototype is not an exact offset/no-oscillation solution. These
direction/channel differences were not separately isolated from initial timing
and floating-point conversion, so no rounding-only causal claim is made.
This output alone does not establish why a real robot fails, whether the mapping
is stable with contact, or whether any task outcome would improve.

The blocked-contact limitation was also reproduced, not merely documented:
actual q stayed 130 deg in both branches; original native stayed 130 and final
stayed at 133. Candidate native/final first reached 133.75 at feedback tick 252,
despite current effective residual then being zero. Of 960 scored ticks, 476 had
candidate baseline outside the reserved 133-degree band. Original hard 135 degrees
and final 1.25-degree/tick slew remained respected. No reserve remedy was added.

The candidate remains unapproved and unwired. There is no claim of an actual
control benefit, production readiness, or permission to change the active MDP.
