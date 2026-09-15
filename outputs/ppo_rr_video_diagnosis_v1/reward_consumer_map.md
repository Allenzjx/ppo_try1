# Reward consumer map — static review, before completed B0/C0 analysis

Scope: read-only production review at the root-reported HEAD `4b2c038887c4109c639eea720f9e60de2c6d8d93`, using the existing `ppo_fsm_reference_p09_stable_v2` configuration. No active video logs, checkpoint tensors, simulation, tests, or counterfactual learning were read/run here. This report is not evidence of the policy's learned motive or of a completed physical trajectory.

## Exact signed consumers

The selected reward configuration is `configs/ppo_fsm_reference_p09_stable_v2/reward_config.yaml`, revision `role_transfer_body_motion_applied_only_v1`. Let `dt=1/120`, `f=physical_transfer_fraction`, and `w=1-0.8*f`, so `w` ranges from 1 to 0.2, never zero. `sq` below is the per-channel clipped squared normalized error, averaged across the named channels and capped at 1.

| Consumer | Actual reward contribution |
| --- | --- |
| Progress, once per returned policy decision | `5*(0.9985*Phi_after-Phi_before)` |
| Terminal event | `+40` for actual SUCCESS; `-40` for any other non-null task termination, including incomplete |
| Elapsed time | `-0.02*sum(dt)` |
| Body level/attitude | `-(0.4/3)*sum(dt*w*sq(roll,pitch; 0.5 rad))` |
| Euler roll/pitch rates | `-(0.4/3)*sum(dt*w*sq(rates; 2 rad/s))` |
| Body angular acceleration | `-(0.4/3)*sum(dt*w*sq(acceleration xyz; 20 rad/s^2))` |
| Contact quality | `-0.2*sum(dt*w*C)`, with `C` the four-wheel average of `(impact*(0.5+0.5*load)+confirmed_rebound+slip)/3` |
| Applied first difference | `-0.05*sum(dt*sq((drive_t-drive_previous)/dt; servo 60 deg/s, wheel 1.8 rad/s^2))` |
| Applied second difference | `-0.05*sum(dt*sq((drive_t-2*drive_previous+drive_previous_previous)/dt^2; preceding scales*120))` |
| Residual magnitude | Exactly zero: regularization disabled and family weight 0 |

The body terms and contact quality are integrated over all actually executed 120 Hz samples, not multiplied by the final decision-end `f`. Smoothness is **not** discounted by `f`, by a phase label, or by lift status. Nominal and residual first differences are logged diagnostics only under `smoothness_components: applied_only`; they are not additional costs. There is no explicit imitation reward, phase-transition bonus, nonzero-residual bonus, or AIR penalty. Potential is global physical progress/history, not a phase-local origin; ordinary phase transitions do not zero potential or terminate the return. Terminal `Phi_after` is zero.

Contact details: impact requires a contact onset and previous downward speed above 0.05 m/s; slip requires contact and speed above 0.15 m/s. Rebound requires upward departure above 0.15 m/s **and** a touchdown within 0.25 s with cumulative whole-body dispatched command excursion at most 0.1 equivalent degrees. Actuated or ambiguous departures are not charged as rebound. Chatter remains diagnostic, not a current cost. Contact metrics use verified ground/obstacle pair reactions; this is distinct from the transfer tracker's normalized bearing-load validity.

Body attitude comes from raw base quaternion multiplied by the fixed chassis-axis transform, not per-episode leveling. Euler rates use wrapped consecutive-angle differences at actual tick time; angular acceleration uses consecutive body-frame angular velocities. Initial derivatives are explicitly zero before a prior observation exists, not a statement that the physical robot was stationary.

Sources: `semantic_reward.py:119–207`, `semantic_observation.py:240–274,286–307,357–363`, selected reward YAML.

## Where P01–P03 attenuation comes from

`TaskStageSupervisor` first computes the old physical-load fraction, then **overwrites it** with the role-tracker fraction in this selected configuration (`semantic_supervisor.py:1320–1336`). The active-leg context is diagnostic/observable, but reward eligibility is selected by actual placement history: FR has no predecessor; FL requires placed FR; RR requires placed FR/FL; RL requires placed FR/FL/RR. Thus before actual FR placement, only the FR role can supply `f`, regardless of whether the phase is P01, P02, or P03. Once FR is placed, FR is ineligible and the eligible unfinished FL role takes over. This may alter `f` without any forbidden phase-boundary reset.

For each eligible leg, `TransferRoleTracker` uses a sliding 0.5 s physical window with minimum evidence 1/15 s. Direction points toward that leg's diagonal receiver at the start of the current physical window (FR toward RL). Motion scales are 1 mm displacement, 5 mm/s directional velocity, and 0.02 normalized load change. The role window is episode-local and survives phase handoffs.

- `measured_transfer = max(directional CoM progress, measured load-drop progress, initial)`.
- For **FR**, `initial = current.initial_clearance AND current.air`. This is not the Q bit. For RR only, the current functional-lift-valid bit replaces this expression.
- `preparation` also admits measured receiver-space contraction. `transfer_progress` multiplies measured-transfer response by independently matured response/support continuity and unload progress.
- Both response paths require current actuation evidence and at least two observed other supports, with short-window continuity. Demand can come from command motion, tracking error, or wheel demand, but must have actual body/joint response or evaluator whole-body actuation evidence.
- `motion_fraction = clip(max(preparation,transfer_progress)*actuated*(1-capture))` while the leg is not placed; it is zero after placement. `capture` is consecutive current top samples divided by the existing two-sample threshold, and is applied only after actual crossing.

Therefore FR I+AIR can **increase** attenuation (reduce body/contact penalties); it does not impose a switch back to level. Q alone does not directly restore the penalty. AIR without I has no direct unit credit, though physical directional/workspace evidence can still support attenuation. When AIR becomes edge/TOP contact before full placement, the FR `I AND AIR` route disappears; directional/workspace/load routes may retain activity. After crossing, actual top samples retire attenuation toward full cost. History Q is not a substitute for present support, AIR, or role response.

Evidence can fade even while transfer/clearance remains unfinished: the sliding displacement/contraction drops, measured forward velocity diminishes, current actuation evidence ends, two-other-support evidence is lost, or capture starts. Then `f` decreases and all three body costs plus contact quality become stronger. Some factors contain Boolean gates, so the resulting temporal series is **not guaranteed mathematically smooth** merely because the final blend is linear. Conversely a static controlled tilted state can lose motion-window attenuation without a physical task-completion event. These are possible incentive tensions to quantify in completed logs, not proof that this policy selected a particular action because of them.

### Unknown-contact/load distinction

In current `functional_lift_edge_v2`, unknown normalized bearing load does **not** invalidate and clear the complete CoM/geometry role tracker. Load entries are `None`, load-drop evidence becomes unavailable, and the report explicitly says `UNAVAILABLE_NOT_ZERO_UNLOADED`. The preparation/directional route is retained. If target load is unavailable, the transfer-progress unload multiplier falls back to `initial` (for FR: I+AIR), so transfer progress may weaken when that route is absent; `max(preparation,progress)` can still preserve some attenuation. Current support loss can separately reduce continuity. Truly invalid CoM/body/wheel geometry clears role samples and yields motion fraction zero. Do not describe either case as proof that an unknown force equals zero or that every unknown contact fully restores the penalty.

Sources: `semantic_transfer_roles.py:99–150,161–225,261–280`; `semantic_supervisor.py:31,695–731,1320–1336`; selected task YAML lines 22–36,136–162.

## Progress interaction to check, not a learned-cause conclusion

The same tracker also supplies the unfinished leg's existing 0.1 unload share in global physical potential. Each eligible unplaced leg contributes `0.1*workspace + 0.1*unload + 0.25*(0.25*I + 0.75*current_lift_credit) + 0.35*carry + 0.2*capture`, before the common `0.85/4` factor. Before qualified crossing, decaying transfer progress can therefore both reduce potential and restore stronger body/contact costs. After qualified crossing, unload credit is retained as 1 so actual capture loading is not treated as unload regression. Workspace preparation for later legs can earn only its existing 0.1 share before predecessor placement, not later-leg lift/cross/place credit.

This establishes candidate conflicts between necessary body motion/ongoing transfer and stability/smoothness costs. It does not establish reward domination, an RR-specific reward bug, or causality for the loaded actor. Compare actual signed contributions, progress gain, event chronology and applied action effects before choosing a repair. Frozen-actor evaluation never updates the actor/critic; changing only reward weights in an otherwise identical eval would not change its outputs or trajectory. Such a reward-only comparison is bookkeeping, not a learned behavior ablation.

Source: `semantic_supervisor.py:1067–1113`.

## Existing logging and exact later integrals

For completed B0/C0, `source/video_policy_decisions.jsonl` returned rows contain:

- `step_info.reward` and identical `step_info.reward_breakdown`: signed `families`, `unweighted_families`, `total`, `potential_before/after`, `potential_shaping`, `terminal_event`, `elapsed_physics_s`, and integrated `cost_components`.
- `step_info.semantic_task.physical_transfer_fraction` and full role/current-leg diagnostics at the **decision endpoint**, not every internal tick.
- `start_tick`, `end_tick`, `physics_ticks`, `request_phase` and `step_info.end_phase_id` identify exact intervals and handoff-spanning decisions.

For a bounded interval of complete returned decisions, sum signed `families` directly. Component decomposition is exact from logged integrals: body attitude/rate/acceleration each `-(0.4/3)*cost_components[name]`; contact `-0.2*contact_quality`; applied smoothness `-0.05*(actual_drive_first_difference+actual_drive_second_difference)`. Do not multiply these integrated components by `f` again, include diagnostic nominal/residual differences, or sum both reward aliases. Keep time, potential and terminal event separate. If discounted return is reported, explicitly use one gamma per policy decision and label it separately from the undiscounted signed integrals.

The 120 Hz `source/physical_observations.jsonl` includes tick zero and every measured physical tick: raw orientation/velocity, joints/wheels, exact contact flags/reactions, CoM/support, commands and actual Full12. `source/native_tick_audit.jsonl` contains per-tick source phase, nominal Full12, projected residual and native dispatch audit. `source/stage_transition_evidence.jsonl` records phase/history changes. These are enough to investigate actual transfer-window mechanisms through an explicitly validated offline replay of the **existing** evaluator/role producer; they are not direct per-tick reward-term logs. The physical-quality CSVs contain stability metrics, not the signed reward decomposition.

Do not allocate an eight-tick decision total uniformly across its ticks or assign all of it to the end phase. Exact I/Q/contact-window or cross-phase cost attribution requires source ticks and original sample/history semantics, including initial drive history and consecutive derivatives. Decision endpoint `f` alone cannot recover intra-decision weighting. Any reconstruction must be labeled reconstructed and checked against the logged complete-decision integrals before using it for causal claims.

Crucial endpoint caveat: `EndpointObserver` records the final physical tick then raises `_PhysicalEndpoint` on common physical success/failure, **before** `SemanticEpisodeEnv.step` reaches reward evaluation. Such a final `environment_step_returned: false` row contains no reward breakdown. Preserve the actual physical result but report the uncovered final interval separately; do not fabricate a logged terminal -40/+40 or silently claim full-episode reward coverage. Controller-only incomplete termination can return normally and then its terminal reward is logged. Pure video/frame-budget aborts are not a task terminal reward.

Sources: `semantic_env.py:113–124,176–236`; `semantic_legacy_evaluation.py:39–49,70–113`; `semantic_video.py:446–461,627–665`; training-equivalent logging is `residual_and_projection_audit.jsonl` via `semantic_training.py:1122–1128`, with the same dictionary under `applied_audit.reward`.

Existing read-only test review: `tests/unit/test_semantic_transfer_role_reward.py` covers the exact body blend, independence from phase labels, per-tick integration, AIR/contact non-penalty, applied-only smoothing and terminal arithmetic. Tests were inspected, not rerun, for this media-first map. No production changes are proposed or made by this report.
