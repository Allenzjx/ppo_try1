# Post-placement wheel-command quality: bounded read-only review

2026-09-06. No Python/Torch/CUDA/Isaac execution, production/config/test edits,
optimizer work or rollout scan was performed for this review. Source reads and
the existing candidate receipt are the evidence. Active training is unchanged.

## Decision and current definition

**Do not adopt the original convex blend as a pure stop-cost addition.** Root
has explicitly declined that original variant. It lowers the pre-existing
derivative cost after all four historical placement flags. The proposed
variant02 below removes this specific problem, but is still an unselected
reward-objective change with the other limitations listed here.

Current production uses `S = (first_difference_cost + second_difference_cost)/2`
over all twelve **actual dispatched** canonical targets, each normalized and
bounded, and integrates `-0.1*S*dt` at the actual 120 Hz intervals. Separate
nominal/residual differences are diagnostic only; residual magnitude
regularization is disabled. Sources: `semantic_reward.py:158–175`,
`reward_config.yaml:4,22,33,38`. `semantic_env.py:159–180` obtains these targets
from the completed frame's `drive_target_full12`, not the actor raw sample,
pre-mapper nominal or measured wheel speed.

Candidate `wheel_stop_effort_candidate.py:54–69` defines
`E = mean_i(abs(actual_wheel_command_i)/H)`, `H=2.0943951023931953 rad/s`, bounded
to [0,1]. H is the original physical wheel limit, not the .6 residual allowance
or the .02 hard stopping tolerance. Wheel order is FL, FR, RL, RR. It uses mean
of absolute values, so opposite commands cannot cancel to fake low activity.
No sign/units error was found in this expression. “Command activity” is precise;
it is not torque, electrical power, mechanical energy, or measured effort.

Before all four hard `placed_history` values are true, the numeric S formula is
the old one. Historical completion is not current support, position retention,
safe stopping, or task success: production independently requires current
region, measured/body rates, actual command limits, current TOP support and
the existing continuous settling interval (`semantic_supervisor.py:524–535`).

## Original variant: concrete reason to reject its “addition only” claim

Original `C=.5*S+.5*E`, after all placed, gives `C-S=.5*(E-S)`. This can be
negative; the change affects all eight servo channels as well as the wheels.

| Isolated same-state/command condition | Old S | Original candidate C | Meaning |
| --- | ---: | ---: | --- |
| Constant nonzero wheel targets | 0 | .5E | Fills the old constant-command cost floor |
| Servo target changes, wheels zero | S>0 | .5S | Halves old servo derivative punishment |
| Necessary final forward/recovery commands | any | .5S+.5E | Costs motion even when needed to finish |
| Measured coasting, all commanded wheels zero | 0 | 0 | Does not itself penalize measured motion |

The second row is already a positive `reward_delta` counterexample in the
candidate harness; it is an objective tradeoff, not a floating-point bug.
Adding wheel activity under the unchanged label `control_smoothness` also
changes that family's meaning. The ID can remain for compatibility, but its
signal ownership must describe applied derivative **and command magnitude**.
It must not be presented as secretly enabling the disabled projected-residual
regularizer or as unchanged physical MDP reward.

## Variant02 requested by root: algebra only, not implemented here

After the same all-placed gate, proposed `C=S+.5*(1-S)*E`; before it, C=S.
For S,E in [0,1]:

- `C-S=.5*(1-S)*E >= 0`: the old cost never goes down.
- `C <= .5+.5*S <= 1`: the same family ceiling is retained.
- `dC/dS=1-.5E >= .5`: increasing S alone cannot lower C.
- `dC/dE=.5*(1-S) >= 0`: command activity remains monotonic for fixed S.

This preserves the old **cost lower bound**, not its exact marginal slope.
At S=1, wheel activity has no additional cost signal; the family is saturated.
At S=0 it has the same constant-command cost as the original candidate. This
is an explicit bounded-budget compromise, not a proof of physical efficacy.

## Risks neither variant removes

1. **Necessary movement is charged.** All-placed history stays true if a leg
   later retreats or support is lost; recovery or forward motion still incurs
   E. Ordinary phase changes, nominal cancellation, or leaving the region
   cannot deactivate the history gate. This prevents toggling current region
   to avoid the term, but does not establish that every penalized movement is
   unnecessary. Zero command may also leave the robot coasting; existing
   measured-rate and support criteria remain essential.

2. **Delaying the fourth placement can avoid this running cost.** In a
   hypothetical pair with the same initial potential, same true-terminal time
   and failure event, and otherwise equal costs, one path which establishes
   the fourth placement earlier then drives pays E longer. The other can
   avoid that added term by delaying placement. That is a mathematical
   objective incentive, not an observed learned exploit or a claim that all
   such physical trajectories exist. The irreversible gate prevents clearing
   earned history afterwards; it does not remove the pre-earning incentive.

3. **Absolute command cost is not PBRS.** Original shaping remains
   `5*(.995*phi_next-phi_before)` with true-terminal phi_next=0, one discount
   per issued decision and no terminal bootstrap (`semantic_reward.py:179–190`).
   Its discounted sum telescopes to `-5*phi_initial` on a finite true-terminal
   trajectory. Adding integrated E changes the occupancy/action objective;
   the unchanged hard goal and correct PBRS do not make this reward change
   policy-invariant. It is neither a new event bonus nor double PBRS, but its
   deliberate overlap with physical stop progress must be disclosed.

4. **The failure bound is not a global success-ordering theorem.** Keeping
   costs <=1 preserves the loader's 14.6 bound and its check `40>14.6`. This
   does not compare every differently timed success/failure trajectory under
   gamma=.995. The already written `cpu_trial_02.json` assumed-event ledger
   has 180-second very slow completion return -0.26668292470637595 versus
   200-second idle failure -0.2666784315799519 (idle higher by about 4.49e-6).
   These deliberately simplified ledgers assume zero body/contact/derivative
   costs and chosen terminal labels; neither is a robot evaluation. They
   refute a blanket ranking claim, not identify the current training cause.

Magnitude is modest but not provably irrelevant: with S=0 and all four wheel
commands of magnitude .4 rad/s, either version adds about -.0095493 reward/s,
or -.00063662 per full 15 Hz decision. At .02 rad/s it adds -.000477465/s;
hard-valid nonzero stopping commands are still charged slightly. These are
algebraic command examples, not predicted Cartesian motion or success odds.

## Acceptance boundaries for any later explicit choice

Retaining variant02 as a report-only candidate is reasonable. Integration is
not approved by this review; root must explicitly accept a changed running
quality objective, including necessary-motion/delayed-placement risks. If the
requirement instead means “unchanged objective”, “never penalize necessary
movement”, or “guaranteed success preferred under all timings”, reject both.

A minimal later integration must bind the exact physical hard bound and
actual same-tick post-dispatch commands; preserve old/default mode and all
pre-all-placed costs; never fabricate or clear hard placed history; retain
actual dt, original PBRS/terminal/safety/success standards and all PPO action
channels; and record old S, E, gate, resulting C, weighted reward and delta.
Describe the broadened family honestly and bind an explicit reward revision,
not a policy-only or exact-resume-equivalent change. These are semantic
implementation conditions, not new robot success/pretraining gates.

Candidate receipt read: trial01 had 12 tests with one error; trial02 records
12/12 passing and `production_wiring=false`, `simulated_robot=false`. The owner
is extending its tests for the limitations above; this review did not run or
aggregate them. Formula-level tests do not establish behavior improvement.
