# Post-placement actual wheel effort — report-only objective variants

Status: **not selected or integrated into production**. This task changed only
new report candidate files and their receipts. It did not modify production,
configuration, the repository test suite, previous runs, or the separate tracking
reference/oscillation candidates. No Torch, GPU, Isaac, optimizer, or robot policy
was run. Root explicitly authorized bounded standard-library CPU checks while
the existing real P10 training continued.

Motivation was root-provided first-episode evidence: all four hard placed-history
flags in 120/120 samples, current region 110/120, support 120/120, zero nominal /
native / controller wheel commands, and both actual-command and measured-wheel
stop conditions failing 120/120. This candidate task did not rescan or independently
certify that real window. Its experiments below are mathematical cases, not new
physical evidence and not assertions of achievable task success.

## Exact proposed objective

S is the existing applied Full12 first/second-difference smoothness cost, averaged
exactly as current `smoothness_components: applied_only`; S is in [0,1]. It includes
the eight servo channels and four wheel channels, using existing 60 deg/s and
1.8 rad/s² scales, and the existing second-difference normalization.

E is the mean over all four **actual filtered canonical wheel commands** of
`abs(command)/2.0943951023931953`, the original wheel hard limit. These are neither
raw/mean actor outputs nor a float32 native readback claim. The candidate validates
finite commands and rejects actual commands beyond the original hard bound (apart
from the explicit 1e-12 input tolerance). No limit is enlarged.

The gate is the conjunction of the four existing hard `history.placed` flags.
They must come from the actual evaluator in any future implementation. The pure
helper cannot authenticate their provenance. Current region, current contact,
phase ID, nominal pose, residual magnitude, body velocity, and a success label
are not inputs. In particular, leaving the platform does not turn the gate off.

```
Before all four placed: C = S                       (both variants, exact old)
Variant 01 afterward:  C = 0.5*S + 0.5*E
Variant 02 afterward:  C = S + 0.5*(1-S)*E
Family reward:        -0.1 * C * actual_dt_seconds
```

Variant 01 is preserved in `wheel_stop_effort_candidate.py`. It halves the old
derivative penalty after the gate, including servo smoothness; at E=0 it can
increase reward by reducing an existing cost. Root requested variant 02 to avoid
that reduction in the per-state cost value.

Variant 02 is `wheel_stop_effort_variant02.py`. For valid S/E:

```
C-S = 0.5*(1-S)*E >= 0
C-0.5*E = S*(1-0.5*E) >= 0
C <= 1
dC/dS = 1-0.5*E >= 0.5
dC/dE = 0.5*(1-S) >= 0
```

This preserves the **cost floor** C>=S, not every original marginal derivative
weight: dC/dS can be 0.5 rather than 1. At S=1, no additional E gradient remains
because the family ceiling is exhausted. At S=0, both variants are identical.
The existing 0.1 family weight, five-family structure, time cost 0.02/s,
success/failure event magnitudes 40, and hard task predicates remain unchanged in
the proposed arithmetic. Production has not been changed to implement it.

This is a new running-command quality objective, **not a PBRS bug fix, a potential
reparameterization, or a policy-invariant reward change**. Any future adoption
requires an explicit versioned objective decision; this prototype supplies no
migration, actual history wiring, audit, or training authorization.

## CPU verification and preserved receipts

All execution used standard library only, `CUDA_VISIBLE_DEVICES=''`, explicit
project `PYTHONPATH`, and `-B -P`, with an owned-process 10-second timeout and hidden
window. The harness AST-compiles only the original pure `_square_cost` function
and `failure_avoidance_bound` getter: it does not import the production reward,
observation, YAML, Torch, or simulator modules. Numeric configuration values are
read as text and matched to the current applied-only premise.

- Original `cpu_trial_01`: 12 tests / 1 error. The harness incorrectly looked for
  an annotated assignment for the original wheel limit; production uses an
  ordinary assignment. This failed receipt is preserved.
- Original `cpu_trial_02`: 12/12 passed after fixing only that report-harness AST
  lookup, not the constant or candidate formula.
- Original final `cpu_trial_03`: 14/14 passed in 0.035 s. Two added tests explicitly
  preserve delayed-placement and long-discount counterexamples. PID 147020 exited 0.
- Variant 02 `cpu_variant02_trial01`: **11/11 passed in 0.024 s**, whole PowerShell
  command 0.537 s; PID 123372 exited 0. These are separate candidate suites, not
  additional production test counts.

Receipts include source hashes. All earlier candidate files/receipts were left
unchanged when creating variant 02. Tests cover original derivative matching,
gate opt-out, all four mean and sign/permutation symmetry, actual rather than
residual commands, assumed stationary body with opposing wheel commands,
phase/region independence, zero after nominal/residual cancellation, magnitude
monotonicity, derivative monotonicity, costs/weight/ceiling, dt integration,
invalid evidence, and the following limitations.

## Tradeoffs and counterexamples retained

1. **Necessary motion still costs effort.** Once the hard history gate is true,
   remaining forward positioning or recovery from a platform departure is taxed.
   No fixed support combination, reference pose, or new hard goal is introduced,
   but that does not make the quality cost neutral to needed movement.
2. **No region escape, but a pre-history incentive remains.** At otherwise equal
   cost and fixed terminal time/event, delaying the fourth placed flag can avoid
   this running cost. If complete-episode PBRS telescopes equally, the 40 event
   and unchanged bound do not exclude that incentive. This is a mathematical
   possibility, not a demonstrated physically feasible policy exploit.
3. **Effort is not actual travel or energy.** Integrating absolute canonical wheel
   command over dt resembles unsigned angular-distance effort under ideal rolling
   assumptions. It neither measures work nor proves travel in contact/slip. Equal
   commanded integrals have equal undiscounted E cost when S/gating agree: four
   wheels at 0.2 rad/s for 10 s or 0.02 rad/s for 100 s both give 2 rad/wheel and
   add 0.04774648 reward-cost units with S=0. Time cost still differs by duration.
4. **Smoothness saturation remains.** At S=1, variant 02 adds no wheel-effort
   distinction. It fixes variant 01's reduced cost value, not all competing
   objectives or their gradients.

The existing failure-avoidance bound is unchanged:

```
((0.4 + 0.2 + 0.1 + 0.0 + 0.02) / 15) / (1-0.995) + 5 = 14.6 < 40
```

That is the same bounded avoidable-cost comparison, not a theorem that every
eventually successful long trajectory outranks every delayed failure.

### Explicit assumed-event arithmetic, not physical trajectories

The table assumes zero body/contact/derivative costs (S=0), initial and terminal
potential zero for discounted telescoping, zero other wheel effort, and externally
earned placement history before the listed command segments. Endpoint ramp costs
are omitted. The +40/-40 terminal events are **assumptions**, not values produced
by a fabricated evaluator success/validity flag. No intermediate task history,
physical trajectory, or test policy is synthesized.

Reasonable completion assumes 80 s total and commands 0.2 rad/s on each wheel
during seconds 60–70. Ultraslow completion assumes 180 s total and 0.02 rad/s
during seconds 60–160. Idle deadline is 200 s; early failure is 20 s; both assume
zero command effort. S=0 makes this table apply equally to variants 01 and 02.

| Assumed event | Non-PBRS undiscounted ledger | Discounted ledger, gamma=.995/decision |
|---|---:|---:|
| Completion at 80 s | +38.352253517 | -0.168225294 |
| Ultraslow completion at 180 s | +36.352253517 | -0.266682925 |
| Idle deadline failure at 200 s | -44.000000000 | -0.266678432 |
| Early failure at 20 s | -40.400000000 | -9.143758590 |

The reasonable-duration assumed completion wins these comparisons. However,
under exactly these assumptions, the ultraslow completion is approximately
4.49e-6 worse than idle deadline failure in discounted return. That limitation
was tested and retained rather than hidden or dismissed using the 14.6 bound.

## Exact variant 02 rerun command (root permission required)

```powershell
$previousCuda=$env:CUDA_VISIBLE_DEVICES
$previousPythonPath=$env:PYTHONPATH
try {
  $env:CUDA_VISIBLE_DEVICES=''
  $env:PYTHONPATH='C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\src'
  & 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -B -P 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\wheel_stop_effort_candidate\validate_wheel_stop_effort_variant02.py' --output 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\wheel_stop_effort_candidate\cpu_variant02_trial02.json'
} finally {
  $env:CUDA_VISIBLE_DEVICES=$previousCuda
  $env:PYTHONPATH=$previousPythonPath
}
```

Output is exclusive-create. No further execution or production change is
authorized by this report. Selection remains with root after the real block/eval.
