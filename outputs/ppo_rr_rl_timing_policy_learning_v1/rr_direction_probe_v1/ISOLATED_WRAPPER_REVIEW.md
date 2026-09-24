# Isolated pure-headroom wrapper review — NOTEXECUTED in Isaac

Root explicitly authorized evaluation of an **in-memory, separate-process diagnostic wrapper**, not a production source hook. This supersedes the earlier assumption that a new production callback was the only allowed route. No production files or currently running interpreter were changed.

Artifacts:
- `isolated_headroom_wrapper.py`: reviewable scoped wrapper, no simulation/model runner.
- `target_plan.py`: one-shot RR6/7 target planner.
- `test_isolated_headroom_wrapper.py` + `test_target_plan.py`: **22 passed**, base Python,0.019s (13 planner +9 wrapper). The wrapper tests call the actual pure production headroom projector and patch only a private SimpleNamespace. They assert Torch/Isaac are absent and the real module function remains unchanged.

## Feasibility and preserved execution

The adapter locally imports the headroom function and invokes it after its unique mapper advance. The native auditor also locally imports it and recomputes actual and zero-current-policy branches. A dedicated process can temporarily replace **only** that pure module function, while the wrapper always calls the captured original projector with the selected diagnostic request. Final hard clamp/slew and the existing one articulation write remain downstream and unchanged.

No native target is guessed from `frame.info.mapped_nominal_full12`. The wrapper receives the actual current mapped native/controller values as arguments from the real call. It composes only RRhip/knee; other10 requests and their original headroom projection remain exact.

One immutable pending context is armed by the existing tick observer from current physical evaluator/source owners and the adjacent real ACK. Source observation/episode tick and absolute adapter dispatch tick must be recorded separately. `read_committed_adapter_tick` distinguishes:
- dispatch t before write: adapter tick=t−1; planner may advance ONCE;
- audit of that dispatch after write: adapter tick=t; planner must not advance.

The first call seals the planned absolute RR targets. Replays recompute required residual and call the original projector again, not return a cached headroom/ACK result. Input native/controller must remain identical, and only actual-request/zero-policy replay branches are accepted. If a probe would first activate from an already-written zero-fast-path call, it fails closed; runtime preflight must confirm eligible P09 uses the normal pre-write composition path. The current declared FL-assist object makes that path normal even when FL assist is inactive, but this must be verified in the actual process, not assumed.

WAIT/ineligible prefix and RELEASED state return the original projector unchanged, including an already-written zero path. The old N prefix is never turned into RR intervention. Once active, loss of current permission releases permanently. The future run must still proceed with the policy after release, not terminate a successful capture early.

### Reviewed trigger correction — still NOTEXECUTED

The diagnostic now requires the actual current AIR-safe-drop geometry: RR within_top_xy, nonnegative front distance, positive measured clearance, lateral validity, current lift/continuation/body evidence and the existing minimum other measured bearing supports. The support force floor is read from the existing task spec and supplied as `force_noise_floor_n`; it is not a new tuning threshold. This follows the AIR branch of `SemanticSupervisor._rr_late_reconfiguration_ready`, but does **not** require RR contact and does **not** release late FL/RL actions. Existing current-stage/owner and evaluator expectations remain live, not cached from an earlier phase.

Before anchoring or starting its finite timer, WAIT checks the full absolute−20deg hip candidate and THEN-current previous-final knee hold against same-tick native+controller, residual cap, headroom and hard bounds, plus the initial anchor's residual-rate/final-slew continuity. An unexpressible endpoint (for example native hip+65deg,24deg residual cap) waits instead of starting a doomed one-shot. Endpoint admission is not an instantaneous−20deg step. A later feasible tick uses its updated real final knee/hip, not the rejected earlier candidate's anchor. After ACTIVE begins, dynamic loss of permission or actual compensation feasibility still permanently releases; no silent clipping or re-triggering.

These gates apply only to this finite exogenous diagnostic, not formal PPO, task success or a prescribed successful pose. Pure additions test pre-edge/invalid geometry→WAIT→legal AIR trigger, cap/headroom/initial-rate infeasibility→WAIT→later fresh anchor, and real support evidence/ACTIVE geometry-loss release. They do not establish physical safety or capture success.

## Crucial audit and attribution rule

The auditor's zero-current-policy branch must retain the **same exogenous RR target intervention**, with only the remaining policy request removed. This is how the scoped wrapper behaves. It avoids:
- attributing the entire probe effect to policy;
- ambiguity when the original policy request itself is all-zero;
- double-advancing the one-shot timer while auditing.

Actual/zero branches both go through original headroom and downstream actuator reconstruction. Original readback, joint IDs, dtype, one-write and mapper-clock assertions remain enabled. These tests have NOT yet executed that Torch/native readback layer; the22 pure tests only establish function-level routing/arithmetic.

The returned headroom receipt explicitly includes `explicit_RR_direction_diagnostic`: original policy raw/logp, original filtered policy request, incoming counterfactual request, injected request, selected channels, target anchor,0 PPO credit and counterfactual semantics. Its existing `requested_policy_residual_full12` now describes the injected request and is explicitly marked accordingly. Outer actuation-plan/ACK raw and requested-policy fields remain the original proposal.

Existing legacy native-audit top-level summaries such as `all12_policy_channels_unmodified_at_actuator` and `policy_request_execution_semantics` were written for formal production paths and may be false descriptions of this diagnostic. **Do not republish them as evidence of pure-policy execution.** The diagnostic sidecar/video manifest must state their restricted interpretation and use the explicit diagnostic receipt. The physical target-match result can remain meaningful; it does not confer policy-action provenance or learning credit.

## Required future run wiring; not supplied as a runner here

- Use official current422 frozen-checkpoint loader. Preserve model/optimizer/normalizer bytes; no optimizer or rollout storage.
- Use current successful_nominal N+0 P01→P07 prefix semantics, not frozen-FSM teacher PrefixSemanticIsaacBackend. Same physical episode, no reset atP07, prefix0 learning.
- Freeze observer context for each next120Hz dispatch. At each new15Hz actor call, use `bind_decision_proposal` before the first physics tick so the armed observer context does not retain the previous decision's raw/logp. Carry that current proposal through the following seven tick contexts.
- Context `previous_effective_residual_full12` must be the adjacent actual post-headroom request (injected during active probe), not stale raw request history. Separate the normal policy REQUEST HISTORY from diagnostic effective correction; actual-applied HISTORY remains real final dispatch.
- Derive stable RR source-owner IDs, not a timestamp-changing whole-layer dictionary. Release if either owner changes. Read previous effective final target and measured joint pose separately.
- Assert rear capture assist=None, rear geometry=None and wheel helper=off; do not silently change this profile or wheel masks.
- Keep single Isaac process lock and camera/physical observer chaining. Capture normal-speed≤200s complete prefix, intervention, release and failure/success tail. No early end merely for RR TOP.
- Suggested truthful label: `CPxxxx_DIAG_RRhipMinus20_kneeFinalHold_NprefixP07`, never formalC/PPO full success.
- Install only in that independent diagnostic process, after compatible weight loading, and restore in a finally/context-manager scope. Record wrapper/planner artifact hashes, sourceHEAD, original checkpoint/manifest hashes and `in_memory_runtime_override=true`. **Unchanged disk hashes/frozen weights do not imply frozen runtime behavior.**
- Budget and accounting:0 PPO decisions/updates/optimizer steps; actual diagnostic policy decisions and N-prefix decisions logged separately. No diagnostic teacher trajectory counted as PPO samples.

## Remaining checks before real use

This is a feasible proposal, not a physically executed or fully integrated runner. Root must review:
1. real native auditor/readback passes with correctly labelled extra headroom receipt and same-probe zero-policy counterfactual;
2. actual unique dispatch-tick/previousACK context and owner mapping, including reset and stage transition;
3. actual contact/safety release at120Hz, cap/rate failure reporting, exact final knee hold after dtype conversion;
4. physical/video observer chaining and full failure-tail export.

The earlier README production-hook design remains an alternative only; this approved-in-principle isolated wrapper does not require changing production file hashes. No wrapper was installed into Isaac, no checkpoint was loaded and no physical conclusion was produced.
