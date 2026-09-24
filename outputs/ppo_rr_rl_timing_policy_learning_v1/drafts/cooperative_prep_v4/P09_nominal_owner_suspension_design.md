# P09 issued-owner suspension — design only, current runtime unchanged

**Conclusion:** a source-clock pause cannot solve the reported late6202 / support-loss6232 sequence. A small opt-in execution-boundary interface is feasible, but **not as a same422 stateless edit that preserves the present absolute residual composition**. This is a new, explicitly observable controller transform, not an RR capture assist; it cannot guarantee that RR physically stays loaded.

## Existing facts and the narrow interface

`semantic_residual_adapter.apply_semantic_residual` advances the real mapper once, then composes its `mapping.applied_drive_command_deg`, bounded tracking correction and projected policy request before final hard clamp/slew/write. `tracking_reference_evidence.mapper_pre_state` and the adjacent ACK bind that real mapper. In contrast, `_semantic_nominal_command_history` / `nominal_command_servo_deg` are explicitly **diagnostic, never dispatched**; they cannot be relabeled executed N. Final slew makes `FINAL - raw request` an invalid decomposition. Even freezing the real mapped N alone leaves an outstanding combined target for final slew to chase.

Add a current **winning-source-owner receipt** per affected servo, from the existing source layering: P09 late event648 and its generation, replaced immediately by a newer owner. Scope only indices **0,1,4,5 still owned by that event**, and only when the existing required current RR bearing is lost **before RL has a qualified current AIR continuation**. Do not key only on semantic P09: old ownership can persist into later stages. An active FL capture owner/new source wins; FR/RR indices2,3,6,7 and all wheel channels remain untouched. Both source clocks, including the authored wheel stop, keep running.

At the first suspended dispatch, take **H = previous committed FINAL** and **r₀ = previous ACK's projected/HISTORY policy request** on those channels. H is a total executed command anchor, not an estimate of nominal; r₀ is an input reference, not an inferred executed contribution. While suspended, use an explicitly declared relative request:

`relative_request = clamp_to_existing_residual_capacity(r_now - r₀)`

`candidate = H + existing_headroom(H, controller_bias=0, relative_request)`

Then use the unchanged final hard limits/slew, one actual target history and single atomic write. Continue the unique mapper advance for provenance, **but exclude its owned native/tracking pull from these candidates**; no second mapper or measurement-q home. Constant policy request now holds H, while independently changed policy requests retain bidirectional authority on all12 channels. This deliberately changes only the suspended channels' offset reference; it is **not** a claim of the old absolute-residual semantics, or of mathematically recovering an executed N. Do not feed nominal-trace values into H or silently allow twice the original residual capacity through differencing.

Restore the normal path on the existing live handoff predicate / legitimate new owner / RL current swing, with existing final slew; never replay event648 or its negative wheel pulse. A renewed loss creates a fresh committed-target anchor. Retiring stale holds must be per owner/channel, not a phase reset. The design suspends old nominal pull only: policy can still choose an unsafe motion, which remains a learning/safety issue.

## Observability and migration

Persisting H/r₀ requires **8 new visible scalars plus per-channel suspended bits**; expose winning-owner identity or prove its existing public reconstruction, and expose the exact current eligibility bit if not derivable from current features. Thus at least12 appended values are likely; final dimension must follow the actual owner/eligibility contract, not be guessed here. Existing changing previous-action slots cannot replace fixed entry anchors. Reset gives explicit inactive/zero state; normal phase handoff does not clear a still-owning hold. Use a new controlled-MDP revision, retain all learned weights/Adam/LR/RNG/Identity, zero-append only new actor/critic input columns/moments, fresh rollout and0 migration credit.

Minimum code boundary: source winner receipt, adapter composition branch + matching independent actuator audit, public state/codec, strict version migration. Frozen A/default path remains unchanged. No production implementation or fit is supplied in this note.

## Required counterexamples (no physical success claim)

1. Freeze source cursor / freeze real mapped N / reuse diagnostic N separately: all can keep moving under final slew; all must fail the claimed “executed hold” test.
2. Adjacent verified ACK, constant request, owned late channel: candidate remains prior FINAL despite far source and tracking correction; changed request still moves in either direction. Saturated/final-slewed prior samples must work without subtraction-based attribution.
3. New owner with numerically identical target clears stale ownership; FL capture and FR/RR policies are not masked. Exact0 residual permissions never erase nominal on unrelated channels.
4. Current verified RR TOP+bearing and RL qualified AIR are separate release cases; historical placed + AIR0N is not bearing. Grounded RL with stale lift history cannot release.
5. Wheel event7.2s executes once at its original source time during holds; no pulse replay or wheel freeze. Cross-stage persistence and genuine reset are distinct.
6. Nonadjacent ACK, wrong write count/order/owner generation, nonfinite anchors or unavailable current support fail explicitly. Logging original native target separately from selected held candidate must reproduce the **actual** single write.

The companion stdlib algebra tests illustrate only items1–2 and capacity/slew semantics; they do not validate this interface in the real mapper, contact system, or simulator.
