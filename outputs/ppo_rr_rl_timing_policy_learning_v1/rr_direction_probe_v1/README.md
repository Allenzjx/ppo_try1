# RR direction probe v1 — NOTEXECUTED, design + pure planner only

Latest update: root authorized an **outputs-only runner draft**. See `RUNNER_REVIEW.md` and `diagnostic_runner.py`; JSONL only, no camera. **25 pure tests pass**;4 separate CPU-Torch/native-audit tests are prepared but explicitly deferred until root confirms idle Isaac. The older "no runner created" and hook-only sections below describe prior design history, not the latest artifact inventory. Nothing has been physically invoked.

Update: root subsequently authorized a dedicated-process in-memory pure-headroom wrapper proposal. See `ISOLATED_WRAPPER_REVIEW.md`; its22 pure tests pass. No wrapper was installed in Isaac. The production-hook-only assumption below is superseded as an implementation constraint; the exact same-tick/audit/attribution requirements remain.

No Isaac/Torch/model forward, production edit, robot command, video or PPO update was performed by this subtask. Per root's follow-up, **no nonfunctional runner was created**. This is an explicitly finite direction diagnostic, not a permanent RR capture assist or a learned-policy result.

Files: `target_plan.py` (pure target planning), `test_target_plan.py` (13 standard-library unit cases). Together with9 isolated-wrapper cases, base-Python tests: **22 passed**,0.019s. Synthetic vectors test wiring arithmetic, not real reachability or stability.

## Concrete current interface blocker

`semantic_env.py:133` receives one rawFull12 request and repeats it for8 physical steps. `semantic_backend.py:465` explicitly labels frame.info.mapped_nominal_full12 as **previous_dispatch_native_before_post_mapper_bias**. Using that value in an inverse tanh is not same-tick compensation.

The unique real mapper advances at `semantic_residual_adapter.py:186`; current corrected native exists at192–215. Post-mapper headroom is evaluated at221; final hard clamp/slew starts278; the one articulation write is328. There is no generic opt-in diagnostic callback at that seam. `SemanticActuationDispatch` at366 only forwards known production contexts. The old `outputs/ppo_task_conditioned_hip_wheel_v1/direction_probe.py` anchors a filtered residual, not a final knee target, and cannot by itself implement an exact knee hold through changing mapped N.

Therefore no second mapper, cloned predictor, monkey-patched internal function, stale-N inverse, direct articulation write or old rear-assist reuse is proposed. Exact execution needs a reviewed versioned hook; root has not authorized its production implementation yet.

## Minimal reviewed hook surface (proposal, not implemented)

1. Add an explicit diagnostic-only per-dispatch context/callback to `SemanticIsaacBackend._atomic_apply` (312/362) and `SemanticActuationDispatch` (366). DefaultNone preserves existing behavior. Refuse any invocation during learner/PPO collection, frozen-FSM mode or a conflicting assist. Require RR assist=None, RR wheel mode=off, rear nominal geometry=None, all12 ordinary channels open, and a422-compatible frozen checkpoint. The existing declared FL capture assist may remain because it owns different channels; record it.

2. In `apply_semantic_residual`, after the **single** real mapper/correction and the original policy headroom calculation, before the final-servo loop (roughly226–278), pass immutable current context:
   - unique mapped/correctedN and controller bias, exact dispatch tick;
   - previous adjacent committed ACK final target, effective residual and write count;
   - original raw/log probability, filtered policy request and effective headroom request;
   - real current evaluator/geometry, source-owner IDs for RR hip/knee, physical limits/caps/rates;
   - actual RR hip/knee, gap/front distance/body position and physics clock.
   No policy call or mapper advance occurs inside the hook.

3. The pure planner produces an optional RR6/7-only required correction:
   `required_residual[i] = desired_final[i] - (same_tick_corrected_native[i] + bounded_controller_bias[i])`.
   The other10 original projected/effective requests remain exactly unchanged. For selected channels validate declared caps, same-tick headroom, residual rate and final physical slew. If exact hold is unavailable, release and report saturation; do not silently clip and still label the knee held. Apply through the existing final clamp/slew and single atomic write. Disabled hook must preserve original zero/nominal fast path; active hook cannot fall into that shortcut.

4. Extend `actuator_target_effect.py:49` audit input and its independent `physical_targets` calculation (311) to replay the declared diagnostic correction. Its final target readback check389 currently reconstructs only known production transforms and will correctly reject an unexplained extra transform. Do **not** disable this check or relabel the diagnostic as RR capture assist. Reproduce the pure plan from immutable pre-state/context for auditing, without advancing the real planner or mapper twice. Assert all10 unselected native actuator targets match the original-policy branch, expected joint IDs and one write.

5. ACK/decision receipt must preserve original raw sample, original Gaussian logp and original filtered REQUEST separately from diagnostic-injected residual and actual final actuator targets/readback. Do not invent an injected raw logp: this post-mapper diagnostic is not a new Gaussian policy sample. Existing raw/filtered-request HISTORY remains the actual policy request history; previous_applied HISTORY must remain the real final dispatch. This is an acknowledged exogenous diagnostic control, with0 PPO samples/updates, not an on-policy training path.

A source hook changes the strict runtime fingerprint even when weights are frozen. Bind a separately reviewed diagnostic runtime/profile/manifest and preserve original checkpoint bytes; do not bypass `load_semantic_checkpoint` contract checks or claim old runtime hashes still match. Do not apply such a hook while an episode is running.

## One-shot target plan

At the first permitted P09 carry window after a real N+0 P01→P07 prefix and frozen-policy continuation:

- Require evaluator valid/no abort and existing accepted physical-evidence status; **current** RR qualified lift, motion-continuation permission, body-control evidence, AIR/contact_modeAIR, no actual ground/obstacle contact, **within_top_xy, nonnegative current front_distance**, positive top gap and lateral validity. Other supports need current support/bearing flags, non-AIR real ground/top contact and measured force at least the existing task-spec noise floor. AIR feet do not count. Historical placed/lift bits alone never trigger it. This is the existing AIR-safe-drop geometric branch, not the late FL/RL group's separate RR-bearing permission; late FL/RL remain governed by unchanged production rules.
- Anchor RR hip and knee to the adjacent **actually executed final target**, not actual joint pose, old nominal or zero residual. Log actual joint pose separately.
- **Before consuming the one-shot**, require the full−20deg candidate and previous-final knee hold to fit current same-tick mappedN+controller, residual caps/headroom and physical joint bounds. Require the initial anchor compensation to satisfy actual residual rate and final slew too. Endpoint admission does not demand moving to−20 in one tick: the ramp remains continuous. Pre-edge or currently unexpressible candidates stay WAIT, leaving ordinary policy/nominal untouched; a later feasible tick anchors its then-current executed final targets. In particular, mapped hip+65deg with24deg residual cap cannot admit−20deg merely because its first ramp sample is attainable.
- When admitted, follow one quintic3s ramp toward−20deg, keep knee at the captured final value, total intervention at most4s. This is a candidate absolute target, never “subtract20 every tick.” Feasibility is checked at every dispatch against the real current mappedN and constraints; no promise that−20 remains reachable. These admission checks are **diagnostic-only**, not new fixed-pose, phase-completion or success conditions, and do not affect formal PPO.
- On actual RR contact/qualification loss, safety, RR source-owner change, cap/headroom/rate/slew inability or measured no-response, relinquish permanently for that episode. Do not restart at another phase. The original policy continues through the normal final slew; no extra probe release tail extends past4s, no whole-body freeze and no wheel modification.
- No-response diagnostic defaults: after1s of attempting at least1deg negative target movement, require at least0.25deg measured negative hip response. These are **reviewable diagnostic watchdog parameters**, not new task success thresholds, phase gates, force criteria or joint limits. Record gap/front/body response independently; hip response alone does not prove useful world movement.
- Contact/placement is not the end of recording. Continue the same episode to natural result/global200s, including failure tail. Probe end is not episode end or success.

The planner requires a context timestamp matching dispatch and the previous ACK exactly adjacent. It compensates both changing mapped N and controller bias; it rejects a supplied previous-dispatch mapped tick. The future hook must verify post-write ACK equality before claiming knee hold.

## Current422 loader and correct N prefix

Use official `semantic_video_cli.checkpoint_loader` (164; load at182; audited deterministic request at205) or equivalent current422 `construct_semantic_runner + load_semantic_checkpoint`, no reinitialization/fitting. Record actor/critic/optimizer/normalizer hashes before/after, deterministic mode, checkpoint/manifest and frozen runtime IDs. No optimizer call.

The correct existing prefix is `CheckpointPolicyPrefixRequest(target_phase='P07', source='successful_nominal')` in `semantic_checkpoint_prefix.py:35`, with zero callback and explicit successful-nominal provenance installed as in `semantic_cli.py:1342–1355`. It follows the same ordinary semantic episode/controller and current N. **Do not substitute** `PrefixSemanticIsaacBackend/PrefixRequest`: that is the frozen-FSM teacher/takeover pathway and not this requested currentN+0 prefix. Preserve source/contact/HISTORY; no teleport, cache reset, teacher success credit or hidden prefix failure retry presented as one episode.

For a full continuous video, the existing `capture_semantic_video` starts naturally atP01 and records every physical frame. A future diagnostic action wrapper can implement the same N+0 prefix semantics until the validated P07 boundary and then use the frozen current422 actor. Record the prefix switch explicitly; do not claim a natural-P01 full-policy evaluation. The capture manifest needs the diagnostic role/provenance, not a misleading formalC success label. Existing PhysicalEvaluationRecorder/EndpointObserver/ActiveViewportVideoRecorder can be reused after this diagnostic context is supported; no source media was created here.

## Required tests before any real invocation

Pure tests already cover previous-final knee anchoring, changing same-tickN+controller compensation, stale-N rejection, tick idempotence, current-vs-history qualification, absolute one-shot trajectory, remaining10 unchanged, contact/owner/safety release, saturation release, no-response release,4s cap, zero PPO accounting. Three added cases cover safe-drop rather than pre-edge admission; unexpressible endpoint/hold waiting then later anchoring; real-support evidence and permanent release when active safe geometry disappears.

Before physical use, additionally test real adapter/auditor integration:
- disabled hook bitwise equivalent to existing path and one mapper/one articulation write;
- active currentN/previousACK provenance, transformed indices exactly6/7, final knee equality after dtype/native-axis conversion, and original request/logp retained;
- diagnostic actual targets independently read back and counterfactual baseline kept separate;
- source stop/owner change and contact release at120Hz, not delayed8 ticks;
- strict422 loader and actual N-prefix receipts; learner collection rejects diagnostic mode; checkpoint/model hashes unchanged;
- fail-safe video export on physical failure/exception, complete prefix+tail,≤200s unaccelerated.

Current disposition: **NOTEXECUTED / hook required**. The proposed diagnostic may clarify direction but cannot itself establish a better learned policy or full RR/RL task success.
