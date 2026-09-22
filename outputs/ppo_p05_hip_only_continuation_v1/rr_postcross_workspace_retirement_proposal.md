# RR post-cross receiver-space credit retirement — proposal only

Decision: **supportable as a small, explicitly versioned candidate**, not a proven fix for hovering. No production/configuration changes, no new physics, and no live learner changes were made. Current block04 continues unchanged.

## Evidence and target

Current `semantic_supervisor.py:1228` still calls the live workspace helper for unplaced RR, including after qualified crossing. `:1306` combines `.5*edge + .5*receiver_workspace`; `semantic_transfer_roles.py:258` defines the receiver term as `.5*FL_margin + .5*FL_contraction_response_over_0.5s`. Unlike unload, that receiver preparation term is not retired after crossing. The physical reward consumes the potential at `semantic_reward.py:436` with `5*(.9985*Phi_after-Phi_before)`.

Existing matched evidence in [CP201728_RR_postcross_reward_readonly.md](CP201728_RR_postcross_reward_readonly.md), 7152→7160, remains directly relevant to the unchanged formula: RR was qualified/crossed/AIR with three real other supports; useful descent 62.283067→62.196051 mm increased the capture potential by 0.000006074, but decaying FL contraction/margin reduced workspace potential by 0.000266264. This demonstrates a local competing preparation term. Repeated learner post-cross hover motivates testing, **not** a causal conclusion that this term explains all hover, or that global PBRS task optimum is wrong. No new block04 numerical replay is claimed here.

## Smallest candidate

Keep the RR edge-distance half unchanged; retire **only the receiver-space half** of its existing workspace budget when all of these current physical facts hold:

- Evaluator valid and no physical termination.
- RR actual qualified-lift history and actual crossing history are true.
- RR **current** lift validity is true, no ground contact, current legal top XY/lateral region, and wheel center is at or beyond the front plane.
- Current capture-region geometry remains usable for preparation credit: AIR with signed collider gap at least the **existing** `geometry.top_gap_min_m` (currently -0.015 m), **or** actual verified TOP surface/contact. This is deliberately not a newly invented zero-height gate; small negative AIR gaps during final descent preserve preparation credit without claiming TOP/load/placement. An old crossing plus a new low AIR lift at -0.04 m cannot retire preparation.

Then `workspace_RR = .5*edge + .5*1`; otherwise use the old `.5*edge + .5*receiver_workspace`. This changes at most `.85/4*.1*.5 = .010625` of total potential. It awards no extra reward family or additive event bonus. Crossing can increase, never drop, this preparation share because the existing receiver score lies in [0,1]. Setting workspace to zero at crossing would be wrong; replacing the entire workspace term by 1 would unnecessarily remove current edge-distance shaping.

The proposed opt-in config key is:

```yaml
rr_postcross_workspace_semantics: current_qualified_RR_over_top_receiver_retirement_v1
```

Draft source patch: [rr_postcross_workspace_retirement_proposed.patch](rr_postcross_workspace_retirement_proposed.patch). It is stored under outputs and has **not** been applied. Configuration enablement and runtime contract/revision changes are deliberately left to the legal-boundary implementation owner.

## What remains real/current

- Unqualified early XY crossing does not retire anything. Stage ID alone does nothing.
- Historical crossing after RR ground recontact is insufficient: current lift validity must be reacquired, and a requalified wheel still behind the front plane or below the existing capture lower tolerance without real TOP contact remains on old shaping. If it requalifies and physically returns over the platform within that existing tolerance, retirement can resume. This does not certify a fresh hard crossing event or reconstruct an attempt ID from stale history; it is a soft earned-preparation rule with current physical corroboration.
- Current support loss that revokes lift validity, retreat before the plane, or leaving the top region returns to live preparation shaping. This conservative option is **not** a permanent once-per-episode reward latch. It therefore avoids retaining preparation on a failed attempt without adding hidden attempt memory.
- Valid TOP landing may retain current lift validity; the rule does not demand AIR, fresh upward motion, or ongoing FL contraction/contact. Two-other-support corroboration remains the existing evaluator's rule, not a new named support template.
- RR lift, carry, descent, actual TOP samples, placement/retention, all other legs, final success and safety are unchanged. No fake TOP/load is inferred. P09/P10 source control, masks, wheel speeds, N_ref, sigma/AUX and stage transitions are untouched.
- No mutable field or observation dimension is added; 389 remains. Eligibility uses existing evaluator measurements and existing qualification/crossing information. In particular, `within_top_xy` is already a physical geometry predicate used by capture and is **not claimed to be a new explicit actor bit**. The existing task-potential observation scalar changes meaning, so this is still a semantic migration, not a silent compatible-resume declaration.

## Directed CPU tests actually executed

[rr_postcross_workspace_retirement_draft_test.py](rr_postcross_workspace_retirement_draft_test.py) implements the candidate only in an outputs-side subclass, calling the unchanged production `physical_potential`. **19 tests passed**; helper process exited. These synthetic dictionaries are formula tests, not valid Isaac trajectories or task-success evidence.

Covered: old/missing/null mode exactness; bad/dependency-invalid mode rejection; exact RR-only half-share; no potential loss at crossing; no ongoing FL contraction demand after crossing; unchanged pre-cross exploration; unqualified crossing/ground/current-lift revocation/retreat/illegal XY negatives; support-loss handling without rewriting history; retained edge sensitivity; unchanged descent and real TOP increments; exact placed-retention branch; no mutated event/placed/contact data; phase-label independence; no repeat bonus; valid→invalid→valid chatter telescopes under the actual discounted PBRS rule; invalid/terminal retirement disabled.

Additional signed-gap boundary coverage: AIR +0.001→0→-0.001→existing -0.015 m keeps the receiver share constant while capture and total-potential increments equal the unchanged old formulas; one representable value below the existing minimum and -0.04 m are negative cases. A test changes the existing spec minimum to -0.007 m to prove the draft reads that field rather than hardcoding zero or -0.015. AIR stays AIR with zero TOP samples and no placement credit throughout. The existing capture formula uses absolute gap, so it still penalizes increasing negative magnitude after zero; this draft adds no second preparation drop there and does not redefine the capture objective.

Repeated identical eligible states still have the existing negative discount/time contribution, not repeated positive crossing bonus. For a complete low→high→low cycle, discounted shaping sums to `5*(gamma²-1)*Phi_low`, not a positive accumulated event payout. This does not eliminate PPO variance near an eligibility boundary.

## Tradeoffs and boundary

Pros: removes the evidenced demand for unnecessary continued FL contraction only after real useful RR geometry, allows alternative joint configurations, and retains all current physical landing evidence. There is no new latch or phase-dependent success shortcut.

Cons: threshold eligibility can still switch on current lift/XY loss or when AIR falls below the existing capture tolerance; this restores old preparation demand on retreat/support loss and may add variance. The -0.015 m limit is an existing geometric tolerance, **not** proof of contact, usable support, a newly completed crossing, or permission for penetration. Its relevance is only that preparation credit should not introduce a stricter final-descent plane than the current capture envelope. A future measured failure of that envelope would require separate evidence, not silently relaxing it here. Current edge shaping can still compete beyond its existing interval; this candidate intentionally does not widen scope. Replacing receiver credit by its completed value is a deliberate semantic judgment (task achievement substitutes for one preparation proxy), not proof the actual FL proxy reached 1. PBRS and local reward comparisons cannot demonstrate a globally optimal policy defect or guarantee feasible landing.

Recommendation: if adopted, do so **after the current block ends**, version task/reward/observation-scalar semantics, retain compatible actor/critic weights, explicitly document optimizer/Identity-normalizer/RNG handling, discard any unfinished old rollout, and collect fresh on-policy data. Frozen-model reevaluation alone cannot establish learned correction. A deterministic natural-P01 evaluation must still report real final completion or the first unfinished task; no new gate is proposed for the ongoing run.
