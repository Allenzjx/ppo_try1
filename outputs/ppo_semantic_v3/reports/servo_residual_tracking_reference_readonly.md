# Persistent servo residual versus frozen nominal tracking — read-only

## Finding and scope

**There is a real structural reduction of low-frequency residual authority on
servos whose nominal tracking remains active.** The mapper compares the real
measured joint with the nominal target, without subtracting or adding the
independent policy residual to its feedback reference. A sustained policy-driven
deviation can therefore produce an opposing mapper correction on later samples.
This is not an all-channel disable, an instantaneous cancellation, an integrator
that necessarily cancels everything, or proof of the cause of FL non-capture.
The completed C76,416 trace gives direct joint-space evidence of this interaction;
it does not identify a Cartesian/contact causal effect or select a unique fix.

No Python, Isaac, GPU, optimizer, production/configuration/test edits, checkpoint
hashing, or changes to the ongoing P01 training were performed. The fixed data
window is the final 120 completed decisions of C76,416: decision endpoints
4,232–5,184, their 960 physical ticks **4,225–5,184**, and the single preceding
raw observation at4,224 for input-state alignment. This is not the current run.

## Exact control dependency

- `semantic_residual_adapter.py:83–86` reads actual `joint_pos` and calls the
  unique `servo_target_mapper.advance(logical_applied.servo_deg, measured,
  tracking_servo_names=...)`. Its nominal input and measured feedback have no
  residual adjustment. Optional geometry and policy headroom occur afterward;
  the effective `controller + residual` reaches the original final-drive
  clamp/slew at126–134. There is no policy-aware reference, subtraction from
  measured joints, or tracking-bias cancellation in this path.
- `servo_target_mapper.py:174–210` enables feedback only for channels both
  `_nominal_reached` and `_tracking_active`. At a sample, physical reference is
  `standing_pose + command_sign * requested_nominal`; its difference from actual
  physical angle is divided by the sign before correction. In canonical logical
  degrees this is **8 × (nominal − measured_joint)**, limited to±10 degrees,
  then changed by at most1.25 degrees per feedback sample (`245–267`). It is
  proportional feedback with bounded slew, not accumulated integral error.
- The mapper's own persistent counter advances once per `advance`; feedback is
  sampled at counter modulo4=0 (`179–181`), nominally30 Hz within120 Hz physics.
  This is not a new15 Hz PPO sample clock and is not reset by an ordinary phase
  change. At that sampling rate the compensation can change at most1.25 degrees
  per4 ticks. The distinct final-drive slew remains1.25 degrees every physics
  tick (150 degrees/s), with unchanged hard joint limits.
- **0.75 degrees is not a tracking deadband.** The constant is only used in the
  segment-end stale-bias retirement condition (`144–156`): when nominal has
  converged and a tracking segment ends, a correction exceeding2 degrees can
  retire. Smaller carry-over corrections remain. Active proportional feedback
  has no corresponding0.75-degree zero-error band.
- A changed nominal (>1e−9) resets nominal-reached/compensation; the mapper first
  slews toward nominal. Channels no longer scheduled for tracking do not keep
  recomputing a proportional correction, although a stored bias may persist.
  Wheels are not handled by this servo feedback law.

Tracking is an authored/current-owner property, not the PPO phase mask.
`MotionExecutor.tick` derives names from the current waypoint's atomic servo
channels (`fsm/motion_executor.py:249–252`). The semantic layer composer updates
tracking only for channels touched by each owner and gives later owners priority
(`semantic_supervisor.py:971–989`). In the fixed P05 tail, all source layers have
reached their finite endpoints: P05's final waypoint9.733333s tracks **FL hip
only**, whereas its preceding knee waypoint is9.4s. Last-owner reconstruction
therefore keeps FL hip, FR knee (P02), and RL hip (P04) active, **not FL knee**.
The saved compact native audit omits the raw `servo_tracking_active` receipt;
this active-set statement is source/owner reconstruction, corroborated by the
different observed per-tick native-target behavior below, not a falsely quoted
saved active flag.

## Fixed-window physical and command evidence

All angles in this table are canonical logical degrees; `sensor_reader.py:294–302`
records joint positions from actual full12 readback, separately from command.
Controller post-mapper bias is exactly0 on both FL channels for all960 ticks.

| Quantity, 960-tick mean | FL hip | FL knee |
|---|---:|---:|
| Nominal |22.800000|−13.400000|
| Mapper native |19.554718476|−12.150000000|
| Native minus nominal |−3.245281524|+1.250000000|
| Requested policy residual |+3.699374794|−0.790312444|
| Actual dispatched command from raw readback |23.253971114|−12.940312444|
| Actual measured joint |23.203499133|−12.942638007|
| Dispatched command minus nominal |+0.453971114|+0.459687556|

The hip residual is narrowly3.684318–3.715609 degrees, while its native target
alternates between **20.179718476 and18.929718476**, four ticks at each value.
Across959 adjacent transitions there are120 negative1.25-degree steps,119
positive1.25-degree steps, and720 unchanged ticks. The final-drive target has
the corresponding oscillation (22.614036–23.895328 degrees); the actual hip
position remains23.089306–23.314514. Thus a roughly+3.7-degree policy request
does not behave as a sustained+3.7-degree offset from nominal in this window.

Two correctly time-aligned feedback samples illustrate the mechanism. The raw
joint observation at`t−1` is the pre-dispatch feedback input, not the observation
after the command at`t`:

| Episode dispatch tick | Pre-tick measured hip | Previous native bias | Gain8 desired bias | Bias after original ±1.25 sample slew | Actual native | Actual target |
|---|---:|---:|---:|---:|---:|---:|
|4,229|23.296243456|−2.620281524|−3.969947645|−3.870281524|18.929718476|22.619267947|
|4,233|23.092299130|−3.870281524|−2.338393037|−2.620281524|20.179718476|23.869097330|

These match the actual mapper result using the nominal-only reference and
pre-tick measured angle. Dispatch physical-command ticks are4,408 and4,412;
their four-tick spacing agrees with the persistent mapper sampling. Neither
correction is at its±10-degree cap. The hip retains a real nonzero influence,
but the active loop repeatedly opposes sustained deviation from nominal.

**Do not infer the same active mechanism for the knee.** Its native target is
constant−12.15 in all960 ticks: the stored+1.25-degree correction persists after
its tracking segment ends. Its−0.7903 residual then remains almost one-for-one
relative to that native target. The similar final+0.46-degree offset from nominal
on the two joints is not proof that both are being continuously regulated in
the same way.

There is also a logging alias: every15 Hz decision endpoint falls on the low
half of the hip's8-tick cycle. The 120 decision-end native mean is18.929718476
and target mean22.629093270, whereas the full120 Hz means are19.554718476 and
23.253971114. Endpoint-only averages must not be described as time-weighted
physics averages.

## Interpretation and limits

In an idealized settled, unsaturated, load-free joint model with actual joint
equal to drive target, the code's relation is
`q = n + 8(n−q) + r`, giving `q−n = r/9`. This explains why strong attenuation
is structurally possible; it is **not a fitted plant model or a physical
guarantee**. The measured hip mean offset+.4035 versus residual+3.6994 is
consistent with attenuation, but sampled slew, oscillation, actuator dynamics,
loads and body coupling prevent treating that simple equation as exact here.
Once compensation saturates, during changing/unreached nominal, or with
tracking inactive, that reduction is different or absent. Larger residuals are
not all necessarily canceled; bounded±10-degree compensation cannot absorb
arbitrary requests.

The same-tick native effect audit correctly removes only the current residual
while retaining the already-evolved mapper and previous final-drive history.
It therefore proves current-command influence, **not** sustained authority
relative to a counterfactual history without past residuals. Passing native
effect and observing longer-term attenuation are compatible statements.

C76,416 remains P05 incomplete: FL legitimately qualified/crossed but stayed
AIR without obstacle force or placement. This inspection does not show that a
particular hip/knee correction would yield touchdown, that sensor evidence is
wrong, or that Cartesian control is impossible. It establishes a material
active-feedback/reference interaction worth separating from headroom clipping,
not a unique repair or a reason to interrupt optimization. No feedback mode,
gain, nominal, action cap, reward, or success criterion was changed or selected.
