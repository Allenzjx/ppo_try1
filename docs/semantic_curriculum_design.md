# Natural-prefix semantic suffix/window curriculum

Status: design only, based on the d86bbadb semantic interfaces. No sampler is implemented or enabled by this document. The running N=1 optimization and the separately reviewed N=8 integration remain the priority; unavailable suffixes do not block either. This review used PowerShell reads only during the live barrier.

## Decision and present evidence limit

Use a **continuous live semantic prefix, then switch which decisions receive PPO credit**. Keep the same physical scene, SemanticEpisodeEnv, controller/evaluator, reader, mapper, bridge, observation builder and clocks across that boundary. Do not restore a historical phase label or construct a new evaluator at an advanced physical pose.

The smallest source is zero-residual B or an explicitly frozen C actor, both using the current B/C runtime. For genuinely reachable later-stage initialization before C can reach it, the concrete A-teacher dependency-injection plan below is the useful next extension, not merely a recommendation to keep falling back to P01. In either case the prefix source is a reset-distribution generator, not an on-policy training trajectory. It runs from ordinary P01 and consumes real 120 Hz observations. The first credited observation is the actual live observation reached by that run.

The reported B382 diagnostic ended with a real P09 timeout, not P10 entry. It does not establish any current-runtime P10/P11/P12/P13 reset asset. Its P01–P09 labels are useful reachability diagnostics, not a blanket eligibility certificate under the subsequently changed evaluator. Fresh prefix outcomes under the active runtime determine availability. Thirteen-stage coverage is a target and a measured deficit, not something this sampler can guarantee before a legitimate trajectory reaches those stages.

## Why existing snapshots are not semantic continuation states

Relevant current interfaces:

| Owner / source | State that must remain continuous | Existing limitation |
| --- | --- | --- |
| `semantic_supervisor.py:109` TaskEvaluator | Per-leg sample deque `(time, bottom_z, hip, knee, air)`, AIR/TOP counts, active-lift/crossed/placed state, event chronology, stable-since, failure latch, last tick/time | `snapshot` at line 137 is reporting data, not a complete serializer; no restore API exists. |
| `semantic_supervisor.py:305` TaskStageSupervisor | Current stage, unique completed stages, transition evidence, stage/episode start times, progress window, termination, last observation tick | A new `initial_stage_id="P10"` does not rebuild prior task completion or clock/potential state. That constructor seam is not a production curriculum certificate. |
| `semantic_supervisor.py:434` / `:505` nominal provider / adapter | Persistent nominal, source-motion cursor/tracking, handoff state, controller tick/time/lifecycle and event history | A fresh provider starts from P01 source state; replacing a controller mid-task is not presently a supported adoption operation. |
| `semantic_env.py:20`, `:89`, `:144` | Previous raw/residual/applied/nominal histories, bridge residual and applied history, semantic observation builder derivative history, current frame and reward endpoints | Calling `reset()` at the credit boundary destroys continuity; advancing only backend physics leaves these histories stale. |
| `sensor_reader.py:253` and physical adapter | Contact debounce/history, geometry/readback, current native buffers, mapper requested/applied/compensation/tracking counters and final-drive slew state | Root/joint pose alone does not restore these or PhysX contact solver state. |

The current lift qualification is particularly important. Before crossing, a ground contact revokes `active_lift` and clears its sample window. The append-only first `event_ticks.active_lift` is not proof that the leg is currently qualified. Under the corrected 57fa rule, above-top AIR is required when earning the current uninterrupted lift qualification; valid early TOP contact followed by center crossing does not require AIR on exactly the previous tick. Restoring a Boolean, a historical tick, or current clearance alone can still fabricate eligibility.

`reference/ppo_phase_snapshots/manifest.json` and its P01–P13 assets are the old `wlr50_clean.ppo_phase_snapshot_manifest.v3` bundle from Trial043. For example, P10's restore contract explicitly includes `reference_entry_compatible` and the signed rebound entry. Those records establish historical source provenance, not acceptance by the new evaluator. Do not call the legacy restore/latch path and then rename its result semantic. `semantic_backend.py:78–88` currently accepts natural P01 only.

For a future durable semantic checkpoint, even a complete public Python-state serializer would also need source/version binding, a supported timebase, current physical reconstruction/prime, and fresh validity checks. No such serializer is needed for the initial live-prefix sampler. Use the whole existing in-memory core instead.

## Minimal proposed API and ownership

Implement later as a small adapter around SemanticEpisodeEnv and its existing RSL adapter, not another backend or PPO algorithm:

```python
PrefixRequest(
    kind="full" | "suffix" | "window",
    target_phase="P01",                 # requested credit-start phase
    end_after_phase=None,               # window ends after this task completes
    max_credited_decisions=None,         # optional external sampling cutoff
    prefix_source="semantic_zero" | "frozen_semantic_actor",
    prefix_checkpoint_sha256=None,
    max_prefix_sim_time_s=...,           # bounded, never extends the task's 200 s
    fallback="fresh_P01",
)

sampler.reset_to_credit_boundary(seed, request) -> ReadyWindow | PrefixMiss
sampler.step(raw_current_policy_action) -> ordinary_RSL_step
```

`ReadyWindow` holds the existing live core internally; it is not a restorable pose file. Its record contains the requested and actual start phase, source/runtime/checkpoint hashes, current task/source ticks and times, remaining task time, current entry report, measured task history/event digest, prefix observation/audit artifact binding and counter offsets. `PrefixMiss` records the actual physical termination, unavailable target, or bounded prefix exhaustion. Infrastructure/sensing corruption still raises an implementation fault; a valid failed prefix is retained as a reset-attempt result.

The prefix algorithm is deliberately small:

1. Perform one ordinary semantic P01 reset. Choose the source and requested target before this physical attempt. Freeze the source actor and all reward/projector/supervisor semantics for the collection/update boundary.
2. Execute the real core's 15 Hz `step()` throughout roll-in, including its eight actual 120 Hz ticks, native audit, observation/history and semantic reward calculations. B supplies zero raw residual; C uses the frozen actor's inference path. Never use `runner.alg.act` or append to RSL storage during initialization, and never update a normalizer from these excluded samples. Prefix outputs remain explicitly logged.
3. After a completed decision, open credit only when the current live phase equals the requested target, the actual task is nonterminal and physically valid, its current semantic entry conditions hold, and task time remains. Check current task prerequisites, not completion of the target goal, old support groups, an exact pose/velocity, or a fixed waiting delay. Do not duplicate-observe the same tick or advance a controller without physics. If the phase was already naturally completed before a credit decision existed, record that fact; do not force it back.
4. Return `core.observation` with its warmed history. Keep all physical/task/derivative/filter state untouched. Record counter offsets instead of clearing the core's physical episode state or return. The next raw action, old log probability, value and subsequent transition are ordinary current-policy PPO data.

Each request gets at most one bounded prefix attempt before its explicit fallback. If a later target proves unreachable, a new attempt can choose a previously reachable earlier target or P01. A saved observation from an earlier tick cannot be used to rewind the already-failed live scene. The fallback must be a fresh physical reset and another real prefix, not a label edit or cached-frame replay. This prevents endless reset retry loops becoming a new pretraining gate.

Full-task rollout remains the default when the source has no useful reachable suffix. Earlier B/C prefixes do not create P10+ coverage by themselves; the learner must improve reachability, or a separately validated initialization source must provide it.

## Reward, time, accounting and sampling boundaries

- Keep `episode_started_s`, `stage_started_s` and the 200-second task horizon from physical P01. At a suffix start after 60 seconds, the actor gets the real 140 seconds remaining, not a new 200 seconds. Do not silently refresh a stalled stage's age.
- Keep the actual task potential/history across the credit boundary. The first credited shaping term starts at the reached state; no reset-to-zero potential, historical event bonus or reward for having selected an advanced stage is added.
- Distinguish `prefix_physics_ticks`, `prefix_behavior_decisions`, `credited_policy_decisions`, credited versus physical episode return, and credited versus prefix stage occupancy. `SemanticEpisodeEnv` currently counts every call, so its cumulative counters require explicit offsets/labels; they must not be reported as optimizer-consumed decisions. Prefix inference/simulation also consumes resources and must remain visible and bounded, not advertised as free training.
- Preserve the existing authorized training-budget ledger. A sampler does not authorize more PPO decisions, repeated unbounded priming or reset budget accounting. Keep requested/stored/actual counters separate when integrating N=8 or tails.
- Never retrospectively discard a failed trajectory after credit began. Prefix status is established before collection; it is not a way to relabel inconvenient current-policy failures as initialization. Every valid suffix success/failure remains on-policy data.
- A window `end_after_phase="P10"` closes only after the measured P10 completion/forward transition, not merely on a P10 label. A decision-count window can end earlier and must be marked external truncation. Such a boundary is not task failure: retain final physical potential and bootstrap from the actual final observation **before** resetting. Use the reviewed peer-reset method if sharing the current RSL integration: add the separately logged `gamma * V(final_obs)` to storage reward, set `done=True` to sever GAE, and keep upstream `time_outs=False` to avoid its pre-action-value correction. True task success/failure/200-second termination has zero bootstrap and takes priority over a simultaneous window cutoff.
- An ordinary 128-step RSL rollout boundary does not reset the task. Continue the same live episode and use its unreset observation for `compute_returns`.

## Thirteen-stage coverage without invented availability

Keep the requested mixture (initially 40% full / 40% suffix / 20% adjacent windows) distinct from its achieved mixture. Maintain a small runtime-bound availability ledger and phase coverage counters in existing training manifests; no separate acceptance hierarchy is needed.

Availability means a fresh source actually reached a nonterminal, eligible current state. It does not mean that the suffix is successful, robust, byte-identical, or available on all later attempts. Count phases from the source phase of credited policy decisions and their actual physics duration, not reset labels or prefix observations. Track completed windows and failed attempts separately. Unreached P10–P13 remain explicit zero/debt entries until real credited transitions exist.

Give available undersampled phases their minimum opportunity and higher weight to P02/P03/P08/P12/P13, but never manufacture a start just to satisfy a quota. Retain a substantial P01 path and increase its share later as intended by the user. Windows such as P07→P10 and P09→P11 are meaningful only when their actual starting states are reachable; P11→P13 is currently an unproven request, not a training asset.

The first suffix sampler should be N=1. The N=8 draft currently has synchronous whole-batch reset and assumes every row contributes a current-policy transition at every decision. Mixing teacher-prefix rows with credited rows would need a separately reviewed storage/masking protocol; do not feed teacher actions into those rows' PPO log probabilities. Keep initial N=8 training at P01. A later homogeneous batch prefix is possible only with explicit common credit boundaries and per-row evidence; it is not required to start N=8 optimization.

## Optional A-teacher priming: allowed concept, not an existing handoff

The user's reset-only initialization allowance is compatible with an explicitly labelled **A-teacher physical priming source**, provided it remains initialization outside B/C policy credit and observes the rules below. It does not permit B/C to invoke a rejected legacy controller and force its private lifecycle forward. The minimal implementation above does not need A and should not wait for it.

- Run an unmodified A controller from a real natural start in a separately selected initialization mode. Its commands actually advance the frozen physical scene; historical observation files are never fed to the learner as if they were current sensors. A's failure/wait is not overridden. Teacher run data must belong to the training initialization split, not locked-test or checkpoint-selection observations.
- From the first real observation, run the current semantic evaluator and supervisor as a read-only shadow at every contiguous 120 Hz tick. The shadow derives its own stage completion/history; it never reads A's label, old guard success or stored latch as proof. Current qualification revocation, RR_FIRST, physical failures and task clocks remain active. A historical/full legacy success may still be ineligible under this shadow and cannot clear a semantic failure latch or deadline.
- Reach a shadow-eligible state, then transfer ownership through a future explicit public reset-only adoption API. Keep that same measured semantic history and clock; do not construct a new P10 supervisor with an empty completion list, serialize only `snapshot`, or secretly assign private fields. A and B/C controllers must not both dispatch a tick.
- The physical reader and RobotAdapter must persist across adoption. Preserve mapper compensation and final-drive state. A's nominal/tracking and normal/feedback biases can differ from B/C's, so zero residual alone does **not** prove a continuous handoff. The new nominal provider needs an explicit initialization from the last actual source command/dispatch context plus a versioned bounded transition to its advisory output, with real float32 target audit. Do not seed a fresh P01 nominal or erase bias/filter history and call the jump a safe reset.
- Any B/C takeover prime uses real physical steps, warms the actual B/C observation/projector history, remains excluded from credit, and still spends the original task time. If qualification is lost or failure occurs during prime, the attempt fails; there is no recovery by restoring old success bits. The same adoption mechanics must be shared by B and C initialization, never active during final P01 evaluation.

Current `PhysicalEvaluationRecorder` already records complete evaluator inputs (`semantic_legacy_evaluation.py:39–45`) and runs an independent TaskEvaluator at 120 Hz. It is a useful evidence/readout seam, but it has only the evaluator, not a complete transferable semantic controller/supervisor/nominal state. It is not itself an adoption implementation. Compact historical observation streams and old P10 snapshot payloads must not be presumed sufficient to reconstruct the new state.

## Concrete A-teacher dependency-injection patch plan

This approach avoids copying a live scene from an old backend or restoring any private legacy FSM state. Start the existing SemanticIsaacBackend with an explicitly selected reset-only prefix controller, using its existing `controller_factory(fsm_path, motion_contract_path)` seam (`semantic_backend.py:74, :124`). Its task owner is the **new** semantic supervisor from the first observation. During initialization only, an unmodified SensorFsmController supplies the physical command suggestion. After a declared handoff, that teacher command source is permanently disabled for the credited episode.

Proposed additive/public interfaces, with concrete responsibilities:

| File / change | Proposed API | Responsibility |
| --- | --- | --- |
| New `semantic_prefix.py` | `PrefixSemanticIsaacBackend(..., prefix_request)` | Thin subclass of SemanticIsaacBackend. Override `_atomic_apply` only to call the existing implementation exactly once and record its verified dispatch receipt; never replace mapping or step physics there. Use the existing factory to construct ResetOnlyPrefixController. |
| New `semantic_prefix.py` | `ResetOnlyPrefixController(frozen_teacher, supervisor, contract, request)` and `record_verified_dispatch(receipt)` | Exposes the established controller/frame interface, semantic `task_snapshot`/progress, and fixed 120 Hz clock. At each real observation it advances the shadow semantic supervisor once, selects either the teacher's command/tracking/bias or the adopted semantic controller, and emits one Full12 frame. Teacher labels/results are separately logged, never used as semantic completion. |
| `semantic_supervisor.py` public constructor extension | `SemanticControllerAdapter.from_live_prefix(supervisor, nominal_seed, current_tick, current_time_s)` | Accept the very same shadow supervisor/evaluator object. Reject terminal history or inconsistent clock. Prepare a current handoff frame without observing the same tick again; next normal `step()` consumes tick `current_tick + 1`. No old FSM private state is changed. |
| `semantic_supervisor.py` nominal initialization extension | `NominalMotionProvider.from_handoff(contract, task_spec, seed)` | Seed persistent nominal and tracking from the actually dispatched source command. Start the advisory cursor for the already legitimate current semantic stage; do not use a P01 absolute anchor. Apply the existing nominal slew principles to later suggestions. |
| New `semantic_prefix.py` / small RSL adapter hook | `prepare_credit(core, request)` and a reset callback replacing unconditional `core.reset()` | Drive the actual SemanticEpisodeEnv during initialization, then return its current observation and a receipt/counter offset. No physical/core reset occurs at the marker. Existing RSL collection begins only afterward. |

`LiveNominalSeed` must include at least: verified command tick and source observation tick/time; logical nominal mapper input Full12; tracking names; **total controller post-mapper bias**; native-before-bias and final-drive Full12; and the actual ACK/mapper-state digest. During A-teacher priming raw residual is zero, so its dispatch bias must be demonstrably controller-owned rather than a hidden teacher PPO residual. The existing `_atomic_apply` arguments and returned ACK supply this without a second mapper advance. The thin subclass records the receipt before the parent goes on to step physics and call the controller.

This receipt hook matters: inside `controller.step(observation_n)`, the parent's `_last_atomic_ack` is not yet updated to the command just executed (`isaac_fsm_backend.py:2006–2018`). Reading that field through a closure can give a one-tick-stale seed. Also, the old teacher may already have computed an output for observation n that has **not** been dispatched. Neither is the seed. Use the captured receipt for the actual n-1→n physics transition.

The handoff timeline is:

1. Execute the selected teacher command from source observation n-1 through the existing projector/post-mapper pipeline, save its verified receipt, advance physics once, and read observation n.
2. Update the same semantic supervisor with observation n. On a configured decision boundary, if its current physical target/entry set is satisfied and no actual task/source failure ended initialization, select the new semantic controller using that supervisor and the verified previous-dispatch seed. Do not select by A's phase name.
3. Emit the first semantic frame for observation n with continuous nominal/tracking/controller-bias context. The next physical step is still n→n+1, using the same RobotAdapter, reader, contact bank, mapper and final-drive buffers. The legacy teacher is not called again after ownership changes.
4. Continue real zero-residual B takeover prime if needed to retire the teacher's controller bias through a versioned bounded transition and warm B's current history. Preserve task and stage ages. Open credit at the next eligible decision boundary once the resulting B/C state is the declared sampler state, then retain every C outcome.

Controller-bias retirement must not be omitted: old A has normal/feedback drive bias whereas the current semantic controller emits zero bias. Initial nominal/tracking equality alone does not eliminate that difference. Keep the measured initial bias and retire it during excluded-credit takeover, bounded by the existing actuator/nominal engineering rates in a versioned reset-only transition; audit actual target changes and finite values. Do not leave an unobserved perpetual teacher bias in training or add a fixed policy bias. If this small transition needs additional state, it remains entirely before credit; the credited actor starts in ordinary B/C execution semantics. Source bias exactly zero needs no artificial delay.

Because the SemanticEpisodeEnv has been processing every teacher/prime tick from P01, its observation derivatives, actual action histories and bridge already correspond to real execution. It is not necessary to copy private core fields into a newly constructed environment. A teacher zero-residual step is only an initialization command: its reward calculation may run to maintain diagnostics, but no teacher transition or old log probability enters RSL.

Practical later-phase requests can specify a measured predecessor window as well as a target stage, for example `switch_when = semantic P09 with current qualified RR lift and near crossing` followed by credited P09→P11, or direct `switch_when = semantic P10 entry` after measured RR placement. These are current physical predicates, not a known future success label or an old timed waypoint. Use this to begin before the desired difficult transition when takeover would otherwise consume the whole stage. If a stage naturally completes during excluded prime, record its zero credited coverage and choose an earlier switching predicate on a later physical attempt; never hold back the supervisor solely to manufacture samples.

Fresh A reaching P10–P13 under the **same shadow supervisor and evaluator**, then retaining validity through the handoff, would establish usable later curriculum starts. A previously verified TaskEvaluator-only whole-task result is encouraging but does not alone prove this: the shadow supervisor also has current per-stage clocks and entry/completion sequencing. The one useful prerequisite for an A-derived start is therefore its actual contiguous observed result, not another exact-state snapshot gate. Failed A-prefix/handoff attempts are initialization failures, not a veto on ongoing P01 optimization.

Initial rollout plan: implement the thin controller/receipt injection and one P09-or-P10 start first; execute one short teacher→B takeover with native audit, then one small C update. Extend the same request predicates to P11/P12/P13 as they are actually reached. Keep full P01 C episodes in the existing mixture and do not require every later source to be ready before using the first legitimate one.

## Focused implementation tests and next useful live check

These tests support the sampler change; they do not require all stages to succeed before more optimization:

1. Compare uninterrupted semantic execution with the same executed actions split at a credit marker. Assert identical subsequent observations, task histories/potential, nominal/residual/applied and float32 native targets; no extra physics step or controller observation occurs at the marker.
2. Exercise the real TaskEvaluator chronology: a measured lift then pre-cross ground contact must revoke qualification; a matching historical event tick must not restore it. Qualification requires genuine joint motion, upward gain, AIR and above-top clearance within the same uninterrupted process; a valid near-edge TOP contact chain does not require AIR on exactly the preceding tick. Verify skipped/noncontiguous observations fail and missing geometry/contact fails closed.
3. At a prefix stop, verify actual task/remaining-time and stage ages persist, derivative/mapper/bridge histories are warm, and the first credited action/log probability is from the current PPO policy. Prefix actor inference must not touch official rollout storage or normalizer statistics.
4. Test unreachable target, target naturally completed without credit, prefix failure and one fresh-P01 fallback; no private stage assignment, cached-state rewind, infinite retry or loss of a credited failure is allowed.
5. Test a real task termination and a nonterminal window cutoff with distinct final/reset observations and a known critic. Check no terminal bootstrap, correct external final-observation bootstrap, and no phase-potential reset.
6. Confirm prefix/stored/actual budgets and per-phase credited durations remain distinct. Verify that P10–P13 stay unclaimed when the source terminates at P09.
7. For the A-prefix extension, add a true source-command/mapper/bias handoff test and one short live A-prefix→B takeover with contiguous raw observations and native targets. Its outcome is an initialization test, not C learning success. No A adoption claim follows from a mock observer test.

Next useful curriculum extension after the existing training run: implement the concrete A-prefix injection, choose one physically qualified late P09 window or P10 start, test a bounded A→B takeover, and immediately collect a small ordinary C update if the resulting state remains eligible. An early B prefix can exercise the cheap credit-boundary plumbing while that extension is being built, but cannot substitute for the later-stage reachability objective. Report actual start/coverage and overhead; keep unavailable later phases explicit. Final qualification, paired A/B/C metrics and videos still use a fresh uninterrupted P01 episode with no teacher prefix or suffix splice.
