# P06 finite wheel tail → measured retirement: opt-in nominal candidate

Status: **design only; not selected, implemented, tested or simulated**. Based on runtime `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`. Adoption must wait for the current8,192-decision training block to finish and its latest saved natural-P01 evaluation. No production/configuration, current training process, budgets or task gates are changed by this report.

## Observed motivation, not causal proof

`p06_first_episode_134849_readonly.md` fixes the first natural-P01 episode of run134849:939 decisions,7,512 ticks,62.6s, P06 incomplete. Both rear wheels remained outside the workspace; the best recorded individual distances were RR−256.393mm and RL−350.562mm, not a simultaneous configuration. All600 P06 decision-end retirement peaks were0, so wheel_gain=1 throughout. Nevertheless, the finite source wheel stop caused the nominal to become zero by recorded tick5,792, leaving216 policy decisions /14.4s before the P06 deadline. Residual and actual commands remained nonzero; this was not policy masking. Current data do not prove that longer rolling would have completed the workspace, avoided retreat or improved stability.

The actual compact P06 source has an initial zero anchor, a time0 `wheel all 0.300` waypoint, and time25.533333333331882s `wheel stop`; end_full12 wheels are zero. `NominalMotionProvider._continuous_advisory` currently multiplies the current P06 source sample by the measured retirement gain. Multiplication cannot retain a nonzero suggestion after that source becomes zero. The existing regression `test_old_config_behavior_and_finite_p06_zero_tail_preserved` explicitly documents that behavior, including peak0/gain1 but zero tail.

## Proposed smallest semantic change

Introduce one absent-by-default opt-in under `nominal`, for example `p06_wheel_tail_semantics: measured_workspace_retirement_after_finite_source`. This name is a proposal, not a current supported configuration. It requires the already enabled `continuous_channel_inheritance` and `p06_rolling_retirement: measured_workspace_interior_peak`; unknown modes or incompatible configurations fail validation before nominal mutation.

At provider construction, bind the extension vector directly to the real P06 source's sole nonzero rolling tuple **[0.3,0.3,0.3,0.3] rad/s**. Verify its wheel-only rolling/zero-tail structure, rather than reading arbitrary prior commands or relying on an unchecked new amplitude. No source contract, frozen MotionExecutor, speed cap or residual scale changes. A changed source containing other rolling magnitudes/reversals or an unexpected tail must not silently inherit this interpretation.

The sole behavior seam is the P06 layer's owner contribution inside `_continuous_advisory` (`semantic_supervisor.py:817–832`):

1. Advance each existing layer's original MotionExecutor once, update `last/sample/touched`, and keep source `elapsed_s` / `endpoint_issued` exactly as today.
2. Update the existing P06 monotonic peak from the same validated current workspace measurement. Do not add a second resettable peak or use a reference snapshot.
3. For a live P06 layer **only after its verified finite zero endpoint has been issued**, use the source rolling tuple instead of zero as that layer's wheel base suggestion. Its contribution is still `source_rolling[i] * (1 - existing_peak)`.
4. Apply it only to wheel channels already owned/touched by that P06 layer, within the existing old-to-new layer composition loop. Servo targets, tracking and other layers are untouched. Before the source tail, behavior is unchanged.
5. Keep the existing outer persistent nominal slew/handoff and final hard clamping. The wheel slew is3rad/s² at120Hz, at most0.025rad/s per tick. This is not a direct setter action and never changes a PPO request.

The source endpoint flag identifies which advisory tail is being replaced; it does not become a task entry/finish requirement. The actual P06 goal and40s stage deadline remain unchanged, as do later deadlines, body/wheel-only failure detection, history and final physical success. A source endpoint can be true while wheels are still advised to roll and the physical task remains incomplete.

## Measured exit and ownership requirements

Reuse the current retirement fraction exactly: if both rear lateral flags are true, `fraction=clip((min(front_RL,front_RR)-workspace_min)/existing_XY_tolerance)`; otherwise fraction=0. Current bounds are−0.22m and width0.005m. Keep `peak=max(old_peak,fraction)` and gain=1−peak, so one lagging rear wheel prevents full retirement; both at−0.2175m give gain0.5 and both at−0.215m retire the contribution. These are existing soft nominal-retirement values, not new hard workspace entry requirements.

- **Do not stop the inherited P06 layer merely because the stage changes.** Early P06→P07→P08/P09 keeps the same layer/peak and lets current measurements continue its existing exit. A retreat after peak1 cannot restart P06 rolling.
- **Never replace the whole final wheel vector after layer composition.** Newer owners retain priority channel-by-channel, including their eventual zero commands. Actual P07 takes FR ownership at source0.333333s with−0.63rad/s and later stops it; P06 must not reassert FR+0.3 after that stop. P08 changes no wheels. P09 takes all four at source1.933333s with+0.3, later stops/reverses selected channels; those newer changes remain authoritative.
- Preserve the existing qualified-lift carry override and P13 endpoint stop; this candidate neither suppresses those rules nor grants itself priority over them. The PPO Full12 residual remains open and independently projected/composed, so it can oppose or augment the prior as before.
- **Terminal inputs cannot activate or rearm the tail.** Require the existing current live measurement status and absence of task/evaluator termination for the replacement. Retain `terminal_no_new_retirement`; do not create a new P06 layer, earn a new peak or inject a fresh rolling target after a terminal. Existing controller termination/stop handling remains authoritative. Invalid/missing/stale measurements keep the existing fail-closed validation before any source-clock/layer mutation.

## Teacher takeover and observability

`NominalMotionProvider.from_handoff` starts from the actually executed receipt and has no reconstructed predecessor layers. A takeover directly into P09/P10/P12 with **no live P06 layer** therefore gets no extension and no fabricated peak: retain the real receipt's wheels and existing next-owner behavior, and report `not_applicable_no_P06_layer`. Do not infer P06 ownership from phase history, nonzero inherited wheels or a teacher's historical success.

If the explicitly selected live takeover is P06 itself, the subsequent ordinary live P06 evaluation can create its current layer using the existing code, sample current geometry and slew from the actual receipt. It does not import teacher elapsed time/retirement peaks or replay a teacher's old P06. The teacher/TAKEOVER/exact receipt paths are not edited.

The static source tuple/mode, current goal measurements, existing peak/layer ownership and mapped nominal describe this candidate. No new hidden timer/counter, historical joint checkpoint or policy observation dimension is introduced. Optional diagnostic additions should only identify `finite_source_tail_replaced`, the P06 source tuple, existing layer origin/peak/gain and effective inherited wheel channels; labels must say **nominal suggestion before slew**, not actual physical actuation or guaranteed motion.

## Focused positive and negative tests if subsequently selected

These are proposed tests, not executed evidence:

1. Absent/None opt-in preserves every existing finite-tail result and v2 default, including zero after3090 ticks with peak0.
2. Enabled, both rear fronts−0.30m with live lateral geometry: the source endpoint remains true and clock progresses normally; P06 wheel suggestion stays0.3 after25.533333s while task completion remains false. No artificial source restart or second mapper advance.
3. Incomplete far-workspace trace from the first report: gain stays1 across the source stop; compare only old/new nominal proposals on the same recorded task states, not a fabricated improved physical rollout.
4. Partial measured retirement at−0.2175m yields0.15; both−0.215m yields0 under existing slew; subsequent retreat cannot restore the prior. One lagging leg and lateral-invalid cases retain current fraction/peak rules.
5. Early phase handoffs before and after source expiry preserve first handoff target/tracking, then continue the same P06 contribution until retirement or legitimate newer owner changes.
6. P07's FR reverse and subsequent zero override only FR; other P06-owned wheels may continue. Once P09 owns all wheels, its stops/reversals win permanently over the old P06 layer. No global `.3 ×4` post-loop overwrite.
7. Existing qualified rear carry override and P13 endpoint stop still work; all servo/tracking/source sample fields match the unmodified provider. Every resulting wheel step respects0.025rad/s.
8. Terminal at the first zero-tail tick, terminal after partial retirement, malformed geometry or stale observation: no newly activated tail/peak/layer and no extra dispatch/state mutation. Preserve the original terminal reason and safe exit.
9. P09 direct teacher handoff with no P06 layer retains the exact incoming receipt and no fabricated P06 diagnostic; P06 takeover creates only a current layer and uses actual receipt slew/current measurements.
10. Nonzero/negative PPO residual still reaches the unchanged physical composition and can cancel the0.3 prior within existing limits. Retired or extended nominal must not mask, relabel or count nominal motion as residual effect.
11. Loader rejects unexpected P06 source amplitudes, extra wheel reversal structure or a nonzero endpoint; legitimate absent mode remains compatible. Existing hard task events/reward configuration remain unchanged in paired same-observation tests.

## Risks, decision and attribution boundary

Longer rolling increases wheel travel exposure. If active lifting or current support is poor, it can push a rear wheel into the front wall, cause wheel-only climb, carry the chassis into the obstacle, destabilize supported front wheels or produce lateral drift. The existing lateral rule only prevents earning retirement when invalid; it is **not** a steering recovery or collision shield. Keeping that rule means this candidate does not solve lateral escape. The current body-collision episode confirms a real logged safety failure exists in this run but does not establish that a tail extension would help or cause it.

The monotonic peak can also retire the prior after a brief real entry into the interior, even if the robot subsequently retreats. This candidate intentionally preserves that design; adding reactivation, wider geometry bands, lift-order gates or larger speeds would be separate changes. The policy may continue opposing the rolling advice; longer nominal duration is not proof of usable body translation or a successful lift/capture.

This would change B nominal as well as C's residual-relative inputs and execution. It must be described as a separately versioned nominal/MDP factor, not PPO-only learned improvement. If later selected, use an explicit preserved-weight checkpoint boundary and report the exact change and actual outcomes; do not overwrite old artifacts or introduce A5/5/full-success requirements before optimization. No reward/entropy/std changes are bundled into this candidate.

**Recommendation now: retain as a narrow candidate only.** Wait for current8192 completion and its latest saved P01 evaluation before deciding whether finite-source exit is recurring enough to justify this single-factor experiment. No new run, gate or outcome is requested by this proposal. Only this outputs report was added; no Python, Isaac, runtime/config edits or commit were performed. Analysis complete; stop.
