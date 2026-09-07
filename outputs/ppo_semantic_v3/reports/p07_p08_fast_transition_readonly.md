# C82,560 P06→P09 rapid transitions — predicate and nominal semantics

Bounded source/report review only. No new training or evaluation stream was read; no code, configuration, test, gate or count was changed. Sources are the completed `eval_82560_diagnosis.md`, current `semantic_supervisor.py`, `semantic_env.py` and v3 task specification.

## What actually passed

The completed evaluation records P06→P07 at6,016/50.133333s, P07→P08 at6,024/50.2s and P08→P09 at6,032/50.266667s. P07 and P08 each receive one real eight-tick policy interval. P09 ends75ticks later in a verified body collision, with no RR initial/qualified/cross/place event. These are not three transitions credited on one unchanged observation.

| Stage completed | Actual completion predicates | Meaning at the completion observation |
|---|---|---|
| P06 |rear_approach=1 |Both workspace_RR and workspace_RL=1 |
| P07 |workspace_RR=1, workspace_RL=1, support_RR=1 |Both rear wheels are within the task's longitudinal/lateral operating region, with at least two other measured supports for RR |
| P08 |workspace_RR=1, load_ready_RR=1 |RR remains in its operating region, has load fraction≤0.20, and has at least two other measured supports |

The v3 spec (`stage_task_spec.yaml:133–166`) uses `[physical_valid, placed_FR, placed_FL]` as the valid-start conditions of P06/P07/P08/P09. Here placed_FR/FL are earned history, not a requirement to freeze those wheels in contact. The completed run has real FR/FL qualification, crossing and placement; their histories are not invented by advancing a label.

Exact predicate definitions (`semantic_supervisor.py:619–659`):

- `rear_approach = min(workspace_RL, workspace_RR)`. Workspace=1 requires current wheel-center front distance in **[−0.22,+0.06]m** and within the measured lateral span. It is an operating corridor, not front crossing, bottom clearance, joint posture or a proof of kinematic reachability.
- `support_RR` counts the **other** legs with measured `support=true`, requiring at least2. Each support uses active, verified ground/obstacle pairs with summed nonnegative normal force≥0.2N. Neither a named support-leg template nor a TOP-only support condition is imposed. This is actual contact presence, not a load-bearing/stability-margin certificate.
- If other support is absent, `load_ready_RR=0`. Otherwise it is `clip((1−RR_load_fraction)/(1−0.20))`; completion at1 therefore requires RR load fraction≤0.20. The fraction is RR measured active-pair force divided by the summed four-wheel force, not a commanded unloading value or a comparison with a historical force trace.

Thus **P08 does require measured relative unloading at that instant**, but does **not** require zero force, AIR, upward velocity, initial clearance, hard lift qualification, a fixed body pose or stillness. A genuinely contacting RR carrying15% of total wheel load, with two other genuine supports and valid workspace, can correctly pass this predicate while still on the ground. That is a static semantic counterexample to “P08 complete means RR already lifted,” not evidence of a bad sensor or a fabricated pass in this run.

## Why one-decision phases are possible

`observe_and_update` (`semantic_supervisor.py:748–801`) consumes every actual physical observation, but advances at most one phase and only on `physics_tick % 8 == 0`, with current entry valid and every completion value≥1. It has no minimum action duration, waypoint-end requirement or settling wait. The next phase is evaluated immediately for observation/progress but cannot also advance in that call.

P07 deliberately repeats the two workspace predicates that just satisfied P06, adding other-support availability. If these conditions persist, P07 can finish at the next decision boundary. P08 then adds relative unloading while retaining RR workspace. Its successful evaluation is still a current physical check, not an unconditional skip.

Completed predecessor conditions are **not invariants** that freeze the subsequent motion. P08 no longer demands RL workspace in its completion set. P09 start requires physical validity and actual front-placement history, not that RR workspace or unloading stay satisfied at every later tick. P09 owns the active lift/cross/place task and continues whole-body residual exploration. Hard contact/qualification/crossing/placement rules, collision/fall checks and finite task deadlines remain independent; a P08 completion does not award RR lift history or excuse an invalid crossing.

This distinction matters in C82,560: FL had real P06 support (13.070292N at6,016), then its final AIR stretch starts6,023. P07 still reports support_RR=1 and P08 reports load_ready_RR=1; neither predicate means that FL specifically must be the supporting leg. The source definition and completed report do not license inferring a fixed contact template, sustained support margin, or an already airborne RR from those scalar passes. The later P09 raw summary has60 RR ground-active and15 AIR samples, all without qualification.

## Fast task handoff is not unconditional cancellation of the prior

Current v3 opts into continuous channel inheritance. `NominalMotionProvider._continuous_advisory` (`semantic_supervisor.py:923–986`) retains predecessor motion layers, advances each live layer, and assigns only channels it has actually changed. Newer changed-channel owners take precedence; therefore older suggestions continue, but **every old preparation movement is not guaranteed exclusive execution to its original endpoint**. A quick semantic transition must not be reported as completion of all old P07/P08 motor work.

Nominal state is persistent, held on the handoff tick and subsequently slewed at150deg/s for servos and3rad/s² for wheels (`evaluate`, lines1,006–1,033). The environment bridge carries feasible residual history rather than resetting it to zero. All12 residual channels remain open under the same rate/hard-safety bounds. There is no restored absolute entry posture, knee-sign pause, source endpoint veto or requirement to replay a historical load-transfer duration. The conditional P09 carry wheel prior requires already qualified lift plus current clearance and behind-front geometry; its availability does not make that lift a P08 completion requirement.

## Bug versus design choice

The inspected conditions are related to the downstream rear task: workspace, alternative contacts and relative unloading. They are **coarse permissive readiness conditions**, not a prediction that the subsequent lift will be safe or successful. P07's workspace duplication and the absence of dwell, sustained balance, AIR or hard-lift requirements explain the eight-tick passes. Relative load≤20% is weaker than “RR completely unloaded,” and two0.2N contacts are weaker than a stability certificate; these are explicit limitations of the selected predicates, not newly discovered arithmetic errors.

No source/report mismatch establishes a transition bug here. The confirmed later body collision establishes task failure, not that a stricter phase gate would prevent it. Demanding prior RR AIR/qualification, fixed joints, named support legs or static stability before P09 would change the task/entry semantics and could prevent P09 from producing the very lift it is intended to learn. This review neither selects nor implements such a change. Future interpretation should say **“P08 current relative-load readiness passed”**, not “rear lift/preparation is fully completed.”
