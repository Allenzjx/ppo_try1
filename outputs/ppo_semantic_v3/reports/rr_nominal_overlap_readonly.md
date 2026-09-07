# C16768 RR nominal overlap — read-only diagnosis

This report does not change the running optimizer, task predicates, runtime, or prior evidence. Only PowerShell reads of completed runs and production source were used; no Python or Isaac was launched. Current source has no `semantic_prior.py`: the production prior is `NominalMotionProvider` in `src/wlr50_clean/ppo/semantic_supervisor.py`.

## Finding

**The proposed specific failure, a newer P07/P08/P09 layer overwriting an unfinished RR hip/knee lift from an earlier phase, is not present in this C16768 trajectory.** P07 changes FL hip, FR knee, and FR wheel; P08 changes RL hip; the RR hip/knee suggestion begins in P09. No P10 layer is ever created in this failed episode. P09 actually issues its RR hip ascent to 55.6 degrees, knee sequence to -37.8 degrees, and subsequent hip decrease. Removing the earlier knee-sign pause did restore these commands.

**There is a different, concrete scheduling problem: unfinished P06 rolling and P07 preparation run concurrently with the P09 lift, and unchanged zero wheel values in the newer suggestions do not cancel inherited rolling.** This is observable at the native target boundary, not just an interpretation of phase names. It is a plausible adverse prior for this entry, but the existing unpaired A/C trajectories do not establish that it is the sole cause of RR failure.

## Exact production mechanism

- `semantic_supervisor.py:615–621`: a new layer starts immediately on each semantic stage change, with its independent motion clock at zero and `last` initialized from the source phase's start vector.
- `semantic_supervisor.py:624–640`: every old layer continues ticking. A channel joins its sticky `touched` set only when a source value changes; layers are composed oldest to newest. A newer touched channel takes precedence forever, but an unchanged source value has no ownership. These two rules are distinct.
- `fsm/motion_executor.py:224–252`: a layer emits the source waypoint Full12 and tracking schedule. Its `initial_full12` does not turn the source into relative deltas or delay it until predecessor suggestions finish. The outer provider applies the existing 150 deg/s and 3 rad/s² slew, with one same-target handoff sample (`semantic_supervisor.py:665–690`).
- The compact source (`configs/recording_motion_contract.json`, P06/P07/P08/P09 at lines 10746/12372/14681/16399) gives P06 four wheels +0.3 rad/s until its finite 25.533333 s stop event. P07 starts with wheel zeros, but those equal its source start vector; it only newly touches FR wheel at +0.333333 s, requesting -0.63, and later zero. P08 never changes wheels. P09 does not touch wheels until its +1.933333 s rolling event. Consequently FL/RL/RR inherit +0.3 through the initial RR lift, while FR performs the still-running P07 reverse segment.
- P07/P08 do not touch RR hip/knee at all. P09 does not touch FL/RL support channels until its +5.4 s later segment. Therefore the early support suggestions really do continue; they are not all silently restored to P09 historical anchors. The issue is their concurrency and locomotion ownership, not a blanket failure of carry.

## Actual C16768 evidence

Source: `runs/ppo_semantic_v3/validation/20260906T0454524190796Z_ga475bad8f9a8_dc53cb90c56147619d5c3b654f8211ab`, deterministic checkpoint 16768, seed 2001, runtime a475bad8f9a8. Files used: `stage_transition_evidence.jsonl`, `residual_and_projection_audit.jsonl`, `native_tick_audit.jsonl`, and `physical_observations.jsonl`.

Transition events are P06→P07 at tick 5240 / 43.666667 s, P07→P08 at 5248 / 43.733333 s, P08→P09 at 5256 / 43.8 s. Actual source-phase dispatch begins on the following tick; this explains the end-of-decision versus next-source phase labels. P07 and P08 each occupy eight physics ticks, but their suggestion clocks continue afterward. P06 occupied 21 s, less than its 25.533333 s suggestion tail.

All following entries are recorded logical nominal values, not actual joint angles. Servo pairs are degrees; wheel order is FL, FR, RL, RR in rad/s.

| End tick / seconds | Source phase | FL hip/knee | RL hip/knee | RR hip/knee | Wheels |
|---|---|---|---|---|---|
| 5240 / 43.666667 | P06 | 22.8 / -13.4 | 6.9 / 0 | 0 / 0 | .3, .3, .3, .3 |
| 5248 / 43.733333 | P07 | 31.55 / -13.4 | 6.9 / 0 | 0 / 0 | .3, .3, .3, .3 |
| 5256 / 43.8 | P08 | 37.6 / -13.4 | 14.3 / 0 | 0 / 0 | .3, .3, .3, .3 |
| 5264 / 43.866667 | P09 | 46.1 / -13.4 | 18.5 / 0 | 1.6 / 0 | .3, .3, .3, .3 |
| 5280 / 44.0 | P09 | 49.2 / -13.4 | 22.8 / 0 | 17.5 / 0 | .3, .3, .3, .3 |
| 5336 / 44.466667 | P09 | 49.2 / -13.4 | 31.2 / 0 | 55.6 / 0 | .3, -.63, .3, .3 |
| 5392 / 44.933333 | P09 | 49.2 / -13.4 | 31.2 / 0 | 55.6 / -37.8 | .3, -.63, .3, .3 |
| 5400 / 45.0 | P09 | 49.2 / -13.4 | 31.2 / 0 | 49.2 / -37.8 | .3, -.63, .3, .3 |
| 5448 / 45.4 | P09 | 38.6 / -13.4 | 31.2 / 0 | 13.2 / -37.8 | .3, 0, .3, .3 |

The P09 first dispatch tick 5257 preserves the prior target (RR 0/0); by tick 5264 its own RR suggestion is active. At 5336 the RR request is not clipped away or replaced by an older layer. The policy adds approximately -2.0065 degrees hip and -7.0914 degrees knee at that decision, which is another real difference from zero-residual A, not evidence of layer preemption.

Native audit confirms the rolling baseline. At tick 5264, verified actual wheel targets are `[-.34806010, .32694960, -.28012836, .36398008]`; same-tick zero-residual counterfactual targets are `[-.30000001, .30000001, -.30000001, .30000001]`. At tick 5336 they are `[-.34482586, -.60143554, -.28797093, .36460325]` versus `[-.30000001, -.629999995, -.30000001, .30000001]`. Native signs include the existing side mapping; negative left native velocity is not evidence of opposite logical travel. These counterfactuals establish the current prior's actuator contribution, not equivalence with a separately initialized A trajectory.

At P09 entry (tick 5256), actual RR wheel bottom is -0.145912 mm relative to ground, with ground contact; FL/RL actual hips are 30.069233/8.207641 degrees. At tick 5336, actual RR bottom is only about +1.16 mm despite actual RR hip 49.41 degrees; at tick 5392 it is about +2.00 mm with actual RR knee -37.48 degrees. The run never earns RR initial/qualified lift. It later hits the obstacle and fails wheel-only at 49.383333 s. Initial clearance is not success; AIR contact classification alone is not qualified lift.

## Real A comparison, with provenance limitation

The complete same-seed historical A source is `runs/ppo_phase_v1/baseline-fsm-eval/20260905T010448442021Z_g36a0d57eb96a_c5dc85cd3cb31_s2001_n1_baseline-fsm-eval-fresh-process/episode_000_seed_2001`. Its old episode summary records SUCCESS at 107.866667 s. This is used only as an observed command/geometry comparison, **not claimed to be a current-v3 unified evaluation or a required pose/timing template**. Its observation projection has verified wheel-bottom geometry and exact-pair normal forces, but lacks current semantic history, full wheel-center/front geometry, and the new native audit.

Actual A command onsets:

- P07 first command at 55.408333 s; all four wheels are already zero. FL hip rises to 49.2; FR wheel -0.63 starts at 55.741667 and stops at 56.741667. FL hip subsequently reaches the finite 38.6 suggestion at 57.075 s.
- P08 first command at 57.141667 s; RL hip suggestions progress 14.3→31.2, the last at 57.541667 s. All wheels stay zero. RR own hip/knee nominal remains 0/0.
- Before RR's own hip command, at tick 6912 / 57.6 s, measured RR bottom is already +6.614 mm, with ground and obstacle pair normal forces both zero. Actual FL/RL hips are approximately 37.32/28.86 degrees. Thus observed whole-body preparation precedes RR hip actuation in this A, rather than being inferred from the RR joint alone.
- P09 RR hip starts 1.6 at 57.608333 s and reaches 55.6 at 58.208333 s; knee starts -4.9 at 58.275 s. All four wheels remain zero throughout this lift/carry portion, until 59.541667 s. At 58.208333 s actual RR bottom is +162.678 mm, with both pair normal forces zero.

This contrasts with C's rolling wheels, overlapping FR reverse/support change, residual knee offset, and not-yet-lifted entry. It does not identify the unique necessary support posture or prove which of these differences dominates dynamics. A source angles/durations must not become new stage, success, or dwell gates.

## Smallest justified continuous-prior correction to consider after the fixed run

The narrowest evidence-backed target is **retiring/blending the completed approach locomotion suggestion**, rather than modifying RR success or requiring historical FL/RL joint endpoints. Channel inheritance should distinguish an unfinished support-shaping suggestion worth carrying from an obsolete rolling suggestion that survives only because newer Full12 zeros never acquired ownership.

A minimal task-conditioned proposal is to smoothly reduce the P06 approach-wheel prior toward zero as its existing measured approach goal is satisfied, retaining normal wheel slew and residual authority. The weight must use existing current geometry/task features, not elapsed source time, a fixed entry pose, or an AIR prerequisite that prevents attempting lift. Keep the finite support suggestions and all 12 residual channels available; do not wait for the old A endpoint or reset the mapper/history. If support suggestions are later blended, use current physical task progress for their weights and record them, not an implicit queue that recreates historical phase dwell.

This is a proposed one-factor prior change, not implemented or proven successful here. It avoids making PPO first cancel three inherited +0.3 rad/s wheel commands merely to explore a low-translation lift. PPO can already override them within the current range; that capability is not evidence that the inherited nominal is desirable. Do not additionally alter caps, RR geometry criteria, rewards, or success conditions in the same comparison. Training remains allowed while this diagnosis is reviewed.

For a later focused regression, use real P06→P07→P08→P09 early transitions and assert (a) inherited rolling is task-conditionally blended, (b) RR/support suggestions and residual channels remain live, (c) no stage/episode done or history reset is introduced, and (d) no geometric success predicate is weakened. A current-run native target trace is needed to quantify the physical effect; this report claims no unperformed experiment.

## Precise one-factor proposal (not implemented)

### Fade after entering the workspace, not while approaching its boundary

Do **not** multiply by `1 - predicate('rear_approach')`. The existing workspace predicate (`semantic_supervisor.py:459–463`) approaches one continuously from outside the workspace. That choice makes a distance-to-go velocity tend to zero at the very boundary it must reach. It also conflates lateral invalidity and overshoot with a request to roll forward.

For an explicitly opted-in revision, define the P06 rolling layer's retirement from the existing rear geometry, using no new task threshold:

- `L = geometry.workspace_min_m = -0.22 m`.
- `E = geometry.xy_measurement_tolerance_m = 0.005 m`; require `0 < E < workspace_max_m - L` in configuration validation. This reuses an existing measurement-scale distance as a **prior blending width**, not as a new task-entry/completion condition.
- `x = min(RL.front_distance_m, RR.front_distance_m)` from the same current validated `physical_evaluator.current_legs` snapshot. Both `within_lateral_span` fields must actually be true for this tick to contribute retirement. A valid false lateral flag freezes retirement; it does not falsely report a reached rear workspace. This change is not a lateral steering controller.
- If both lateral flags are true, measured retirement is `r_now = clip((x - L) / E, 0, 1)`. Otherwise it is zero. The farther-behind rear wheel determines the fraction; one far-ahead leg cannot retire the approach while its peer remains behind.
- A real P06 layer starts with `r_peak = 0`, then updates `r_peak = max(r_peak, r_now)` on valid observations; its wheel gain is `w = 1 - r_peak`. Record the input distances, `r_now`, `r_peak`, and `w` as nominal-provider diagnostics. Do not derive the peak from a phase label, a target joint angle, or future teacher labels.

For a newly created layer, the resulting values are:

| Minimum rear distance x | Measured retirement | P06 gain w | P06 .3 rad/s contribution |
|---|---|---|---|
| -.30 m or -.221 m | 0 | 1 | .3 |
| -.220 m (workspace boundary) | 0 | 1 | .3 |
| -.219 m | .2 | .8 | .24 |
| -.2175 m | .5 | .5 | .15 |
| -.215 m or farther forward | 1 | 0 | 0 |

Thus this modification supplies the full existing approach suggestion up to and at the lower workspace boundary. It cannot create the `1-progress` asymptotic failure **before workspace entry**. The target can asymptotically approach the interior end of the blend in an ideal kinematic model, but the existing rear-workspace task has already become eligible; there is no new requirement to reach -.215 m, wait for zero wheel speed, or finish a source sequence before transitioning. This is not a claim of guaranteed real reachability: traction, residual action, the unchanged finite P06 tail, and other physical limitations still matter.

The monotone peak is important: an RR lift often swings its wheel rearward. Using only current `x` would reactivate the old P06 rolling suggestion precisely during that lift. The peak is one new **nominal scheduling state**, not a physical lift qualification or reward state; it survives ordinary stage changes and is discarded with the provider on a genuine episode reset. It must be declared in the new-MDP revision and diagnostics, never hidden as a fabricated semantic history bit. No actor dimension need be claimed or silently changed: current/previous logical nominal and mapped nominal already expose the executed target effect, while this additional scheduler memory must be acknowledged explicitly. If adding any new scheduler memory is rejected, do not silently substitute a stateless rule: the narrower alternative is to retire on already-recorded P06 physical completion with existing slew, but that alternative has a discontinuous target gain and is a different proposal.

### Exact insertion point and ownership preservation

Calculate and validate the tick's retirement inputs **before advancing any source motion clocks**. In `_continuous_advisory`, between obtaining each layer's source sample and assigning its own touched channels (`semantic_supervisor.py:631–635`), use a local contribution vector:

1. Preserve `sample.full12`, `layer.last`, tracking, and `layer.touched` calculations in their current source coordinates. Do not mutate the immutable sample or make gain changes count as new source channel touches.
2. Only when `layer['stage'] == 'P06'`, multiply that layer's indices 8–11 by its gain. Leave its servo values unchanged. The present P06 source owns only wheels; no broad whole-vector multiplier is needed.
3. Write those values only through the layer's existing touched-channel loop. All newer layers retain their existing overwrite order. In particular P07's FR -0.63/zero wins over the scaled P06 FR contribution, and P09's subsequent four-wheel .3 segment wins over all of it.
4. Leave the final qualified-lift/current-clearance carry override (`semantic_supervisor.py:641–647`) after all layers, unchanged. A fully retired P06 layer cannot suppress that existing P09/P12 carry suggestion.
5. Leave outer nominal slew, transition hold, mapping/feedback, actuator audit, and the entire residual projector intact. The proposed gain scales only P06's contribution; it is **not** a new-stage all-wheel zero command and it is **not** a multiplier on the final combined target.

The old finite P06 stop sample remains zero regardless of gain. No extra source duration, synthetic rolling event, source replay, or attempt to guarantee arrival is introduced. P01/P02 assist, P07 support/frontal wheel adjustment, P08 RL suggestion, and all P09 servo events remain byte-for-byte unmodified inputs to composition.

### Invalid input and reset/prefix semantics

When opt-in is enabled and a real P06 layer exists (or will be created this tick), require the existing evaluator mapping, `valid is True`, matching current observation provenance, finite non-boolean RL/RR distances, explicit boolean lateral flags, and finite positive configuration width. Missing, malformed, or nonfinite metadata raises the established `SemanticObservationError` before layer clocks/retirement/nominal advance. It must not default missing distances to zero, assume successful retirement, or silently use gain one and keep driving. A known task/safety termination remains on the existing terminal path; this helper must not invent a new success, continue advancing suggestions, or claim that throwing an exception itself has physically stopped the robot. Existing termination/exception handling owns that lifecycle.

An explicit valid lateral flag of false is not a schema error; it earns no additional retirement. A temporarily retreating wheel also cannot reduce earned retirement. These are distinct from invalid/missing data. The chosen minimum-distance rule is only a forward approach prior: if one wheel is already beyond the workspace while its peer is far behind, it does not solve that geometry or change the evaluator's unchanged failure/continuation criteria.

`from_handoff` (`semantic_supervisor.py:591–605`) creates a provider with the physically executed nominal/tracking but no reconstructed historical layers. Preserve that meaning:

- A suffix beginning at P06 creates a real P06 layer when evaluated, so its gain is derived from then-current measured geometry.
- A suffix beginning at P07/P08/P09 without a P06 layer gets **no** P06 gain, no synthetic P06 layer, no reconstructed peak, and no subtraction from the teacher's retained wheel targets. Do not claim this fixes arbitrary inherited teacher wheel values whose contribution ownership was not retained.
- A genuine same-provider continuation through P06 retains its real peak across subsequent phases. An already-running provider must not be destroyed/rebuilt at those transitions.
- Teacher preparation remains excluded from PPO credit; no future semantic history is imported to seed retirement. Suffix/full-task labels and 200 s total clock remain unchanged.

### Focused assertions for a later implementation

No tests were executed for this proposal. The smallest useful CPU coverage would assert:

1. Using real `TaskEvaluator` rear geometry, the table above holds with fresh providers; continuity holds on both sides of L and L+E. Just-outside-workspace and exactly-at-boundary samples still retain full P06 rolling, and the workspace predicate itself is unchanged.
2. With asymmetric rear x values, the lagging wheel governs. Valid lateral false never earns retirement. NaN/inf/bool/missing x, missing evaluator, and invalid flags fail before any source clock/nominal/peak mutation; missing data must not pass by a default.
3. A monotonic measured advance retires P06, a later RR rearward swing does not reactivate it, and a true fresh reset restores the initial unretired state. Merely changing ordinary stage labels with fixed geometry cannot create retirement.
4. In a real early P06→P07→P08→P09 provider sequence, zero-gain P06 leaves P07's newly touched FR -0.63/zero intact; other-wheel contribution reflects P06 only until a newer layer owns it. P09's own .3 event and the existing qualified-lift carry override both still produce the original four-wheel suggestion.
5. RR hip/knee, FL/RL support suggestions, tracking schedule, source endpoint timing, and phase transitions match the unmodified provider for the same supplied semantic snapshots. A gain transition respects the existing per-tick final wheel slew (3/120 = .025 rad/s), preserves the transition's held sample, and does not bypass the projector.
6. Real nonzero wheel residuals still reach the actual native adapter/audit when P06's nominal gain is zero; no test should equate zero nominal with zero applied/native action. Keep the existing side-sign mapping and native counterfactual comparison.
7. `from_handoff` at P09 keeps the actual incoming Full12/tracking and has no P06 retirement state, while a genuine P06 core continuation retains it. No duplicate prefix credit, episode reset, stage-done, new observation dimension, or fabricated task history appears.
8. At the existing P06 finite zero tail, scaled output stays zero; the patch never prolongs old rolling. Old v2 and any v3 config without the explicit opt-in retain their previous provider output exactly.

These are regression checks for one scoped prior change, not additional optimizer-start gates. The current fixed training block continues unchanged; implementation and a new-MDP continuation remain a later root decision.

### Root selection and smallest diagnostic transport

Root selected the continuous inside-workspace blend plus measured per-P06-layer peak above. This remains design-only until an explicit post-run implementation instruction. There is no unique joint-angle target, designated support-group requirement, AIR prerequisite, or dwell gate in this rule. Physical support/termination evaluation remains unchanged.

The smallest diagnostic object can be a copied, non-mutating snapshot named `nominal_provider_diagnostics.p06_rolling_retirement` with: `schema`, `enabled`, `layer_present`, `source_observation_tick`, `source_sim_time_s`, `front_distance_rl_m`, `front_distance_rr_m`, `lateral_valid`, `measured_fraction`, `peak_fraction`, `wheel_gain`, and `origin='current_live_P06_layer'`. With no layer, the diagnostic must explicitly say `layer_present=false` and omit/null the unmeasured numeric fields; it must not report an invented peak of one. The incoming-handoff case should identify `not_applicable_no_P06_layer`. Store plain copied values, not a mutable layer or a controller object. This is audit data, not a new policy group or a lift-history bit.

Minimal existing transport seams:

1. Expose the provider's copied diagnostic through `SemanticControllerAdapter.task_snapshot` (`semantic_supervisor.py:738`) and the local task dictionary used in `ControllerFrame.feedback` **after** `nominal_provider.evaluate` (`:745–758`). Decorate a copy; do not mutate the supervisor's task result, physical potential, or completed history. The pre-evaluation task must not accidentally carry the previous tick's peak.
2. `SemanticIsaacBackend` already copies `controller.task_snapshot` to `AuthoritativeFrame.info['semantic_task']` (`semantic_backend.py:277–280`), so no new backend mapping or observation field is necessary. `semantic_observation.semantic_task` validates the known schema and returns the dictionary (`semantic_observation.py:64–82`); additional diagnostic keys are not encoded as actor groups. The schema remains exactly 324 dimensions.
3. `SemanticEpisodeEnv` already copies the final-frame task to `step.info['semantic_task']` (`semantic_env.py:225`); N1 and prefix adapters preserve the info mapping, and training serializes it under `applied_audit` (`semantic_training.py:573–579`). This preserves **decision-boundary scheduling diagnostics** with no runner/CLI schema change. The source tick in the diagnostic must make clear that this final-frame object accompanies the next nominal proposal, not all eight previously executed commands.
4. For an exact executed-target association, `PhysicalEvaluationRecorder.tick` already receives `before, after, projection` and writes `native_tick_audit.jsonl` (`semantic_legacy_evaluation.py:99–103`). A single additive field copying `before.info['semantic_task']['nominal_provider_diagnostics']` into that native row is the narrow 120 Hz evaluation seam. Do **not** take `after`'s diagnostic: the row's actual command/projection belongs to `before`. This field is separate from the validated native-effect audit schema and must not change its verification result.
5. If the first training continuation needs a same-executed-tick summary rather than only boundary diagnostics, the minimal extra env field is the **last actually applied source** diagnostic copied inside the existing physics loop before `self.frame` advances, plus its applied episode tick. Label it `last_applied_nominal_provider_diagnostics`; do not mislabel final-frame metadata as that source. A bounded min/max gain and retired-tick count over the already executed interval is optional; no full raw observation/history dump or framework is required. This small env addition is separate from the presently selected provider/config scope and needs root ownership coordination when implementing.

Useful transport assertions: the diagnostic source observation tick binds to the command-source frame; the recorder writes that same diagnostic beside the corresponding actual native audit; its value is not retroactively changed by the next tick; adding/removing only diagnostic keys leaves the encoded observation tensor bit-identical; prefix and N1 serialization preserve the declared no-layer/real-layer distinction. Do not call a boundary-only log a full 120 Hz training trace.
