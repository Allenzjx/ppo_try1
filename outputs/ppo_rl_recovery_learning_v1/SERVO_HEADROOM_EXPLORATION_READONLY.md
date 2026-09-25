# Servo headroom and effective exploration — read-only, 72e runtime

Inspected HEAD `72e63592bdf412d375abfda29b03cf456fbf4f7e` while its P10 training runs. No production edits, model/Torch imports, extra simulation, or action/MDP change. Numerical examples below are standard-library arithmetic, not policy replay or physical improvement evidence.

## Finding

**Current code has asymmetric safety *clipping*, not asymmetric action *scaling*.** The policy first requests a symmetric `C*tanh(raw)` offset. The real same-tick mapper/controller then determines the feasible residual interval and clips many requests to its endpoint. In suspended FL/RL owner lanes, the anchor-relative request difference is clipped again. This explains a genuinely flat request-to-target region despite all12 permissions being open. It does not establish that clipping is the sole cause of failed task progress or actual joint overshoot.

The known first118/75-ground observations are from the preserved f6d run, not new72e results. These controller files are unchanged by the72e reward-only boundary.

## Exact current order and code facts

1. `semantic_rear_owner_actor.py:62-67`: shared389 REQUEST-history center, `mean=.1*head_mu+.9*history_center`; then the raw Gaussian. `semantic_history_actor.py:225-240` uses previous raw195:207 normally; only a declared phase entry with an enlarged cap recenters from `atanh(previous_filtered_REQUEST/current_cap)`. It does **not** recenter on post-headroom effective residual or actual target.
2. `action_projection.py:515-544`: `tanh(raw)` times the symmetric phase capacity, mask, then REQUEST slew. `execution_profile.yaml:47-57`: P06–P13 FL knee and RR knee both C=36deg; servo REQUEST rate60deg/s means0.5deg/120Hz tick. No Gaussian action clipping/rejection occurs here. `semantic_env.py:163-165` records the filtered REQUEST separately from actual drive; the held15Hz raw sample advances through8 physical ticks.
3. `phase_action_masks_v2.py:256-263, 371-389`: ordinary phase handoff retains the permitted REQUEST for its first tick, then resumes slew. `action_projection.py:586-596` deliberately defers **servo** headroom to the mapper; it does not clip against stale logical N. Wheel logical bounds remain separate.
4. `semantic_tracking_reference.py:221-232,254`: eligible frozen mapper tracking uses the **previous ACK filtered REQUEST**, not the current sample, effective residual or final target. Original correction is bounded to±10deg and the desired reserved interval; mature compensation has its own slew/history. Thus a currently clipped request can still affect later mapper-reference computation. It is wrong to infer policy effect solely as FINAL minus an independently recomputed nominal.
5. `semantic_residual_adapter.py:184-229`: exactly one real mapper advance → optional geometry correction (currentlyOFF) → headroom around `b=mapped_native+bounded_controller_bias`. `semantic_headroom.py:87-99`: knee hard limits[-60,210], reserve2 → L/U=[-58,208]; feasible request interval `[min(0,L-b),max(0,U-b)]`; `effective=clip(filtered_REQUEST,interval)`. No scale is reduced before tanh. Baseline outside reserve is not relocated by a zero residual.
6. Adapter `:232-279`: declared FL capture assist can override its FL lanes (currentlyON); rear capture assistOFF. Then `:281-310`: owner suspension may override selected FL/RL servos; FL assist ownership retires FL suspension. Owner receives the **original filtered REQUEST**, not the already-clipped effective residual.
7. `semantic_rear_owner_recovery.py:77-95`: for an active selected lane, `v=clip(r-r0,[-C,C])`, then headroom clips v around frozen entry FINAL H, and target becomes H+effective(v). Neither stale N nor its tracking correction enters this selected target. Only FL/RL indices0,1,4,5 can take this path; RR knee stays on ordinary composition. Constant r=r0 holds H, not actual q.
8. Adapter `:300-310` and `robot_adapter.py:709-734`: final hard clamp and150deg/s slew (1.25deg/tick), then one atomic actuator write. `adapter.py:364-367`: `drive_target_full12` is FINAL; `native_drive_target_full12` is mapped N. These must not be interchanged. Target reserve does not certify dynamic actual-joint safety.

## Existing real examples (not a new trial)

| Window | Relevant exact data | First flat layer |
| --- | --- | --- |
| First episode tick6103, FL knee | mapped N/b=-41.4deg; raw=-1.862968; filtered request=-34.306282; effective=-16.6; FINAL=-58 | Ordinary same-tick headroom. Source late N moved from-12.15 toward-41.4 while request stayed near-33/-34. No sampled input owner was active. |
| Third episode, first RR-ground tick6328, FL knee | public H≈-55.302874; r0≈-20.815821; request=-34.687776; FINAL=-58 | Owner relative difference≈-13.871955, but negative target room only2.697126deg. Old N=-41.4 is no longer the selected anchor. |
| First118 aggregate, RR knee | 24/118 lower-headroom clips/FINAL=-58; sampled clipped raw[-.637984,-.540931], sigma[.019484,.020897] | Ordinary headroom, not owner suspension. Full same-clock baseline/request is required to quantify each threshold. |

First118 FL knee clips6/118. The75 RR-ground/unqualified-RL endpoints all have FL FINAL=-58 although conditional means span-2.085 to-.946. These are pre-existing sealed/explicitly bounded reports, not a claim that the current72e policy remains identical.

At the owner example, exiting the flat *filtered-request* region requires r > r0+(L-H) ≈-23.512946deg. With fixed b and settled slew only, ordinary b=-41.4 gives a raw cutoff≈-.498722, and the owner example≈-.780751. These are algebraic frozen-context cutoffs, **not historical Gaussian escape probabilities**: the real request is rate-limited, the mapper and support state evolve, and one action spans8 ticks.

The physical failure was actual FL knee≈-60.011567deg while FINAL=-58. At the final rows FINAL was already above actual q; moving target mapping inward would not retroactively remove momentum/contact coupling. Do not label current saturation a mask bug or loosen the actual hard limit.

## Minimal next-version hypothesis — not an implementation recommendation yet

If the new block again gives flat effective exploration, a narrow **negative-side usable-range mapping** is more relevant than merely enlarging sigma. First consider only demonstrated FL-knee late/recapture lanes (and separately RR-knee if supported); preserve accepted early front behavior, all positive/negative permissions, hard limits, source stops, raw samples and final slew.

For an ordinary lane with C fixed and filtered r∈[-C,C], one candidate is:

`Dminus=min(C,max(0,b-L)); Dplus=min(C,max(0,U-b))`

`effective=r*Dminus/C` for r<0; `r*Dplus/C` for r≥0.

This is a changed control transform, not the current code. At b=-41.4,C36, raw-2/-1.5/-1 with settled requests currently all map to-58; the candidate would give approximately-57.403/-56.425/-54.042deg. Zero remains zero; when both directions have≥C room it equals the old mapping. It suppresses duplicated endpoint targets but also changes how the nominal contributes as b moves. It must not be called a reward-only revision or network learning.

**Owner lanes need a separate anchor-aware derivation.** Scaling the ordinary N headroom first is insufficient because the owner later overwrites it. Preserve H at r=r0 and preserve the old public H/r0 state. A candidate negative branch for r<r0 is:

`target=H + Dminus*(r-r0)/(C+r0)`, with `Dminus=min(C,max(0,H-L))`.

This normalizes the remaining negative REQUEST interval[-C,r0] into available negative target room; a positive branch may initially retain the existing bounded difference to avoid needlessly weakening recovery. At the logged H/r0, raw-2/-1.5/-1 would give approximately-57.770/-57.393/-56.475 instead of three identical-58 targets. This is frozen-context arithmetic only. Near r0=-C its denominator becomes tiny; an unconditional implementation can create excessive sensitivity and requires a reviewed limiting/degenerate rule. No such rule is selected here. Adding a persistent new anchor or mode would require explicit observed state; don't silently add a cache.

## Necessary counterexamples / implications before considering a change

- At b=L (or H=L), there is **no legal negative room**; any outward proposal must remain flat/zero. Reparameterization cannot manufacture mechanical range. With actual q already below its limit, a legal target alone cannot prove safe recovery.
- With ample headroom (e.g.b=0,C36), do not gratuitously change r→target. Preserve exact zero/zero-history N+0 and every legal explicit stop. In owner mode r=0 is not r=r0: policy withdrawal legitimately moves relative to the frozen request anchor; do not “fix” this by zeroing HISTORY.
- Active FL assist owns a different transform. Any experiment must exclude/disclose it and must not reinterpret its motion as policy exploration. The first hard-limit episode had no sampled active owner; an owner-only change cannot explain or guarantee fixing that failure.
- `semantic_observation.py:433-440` already exposes prior mapped N, actual/applied/REQUEST histories and full mapper state; owner17 exposes H/r0/active/winning-owner. Same-tick post-mapper b is nevertheless produced **after** sampling, with feedback/scheduling over8ticks. Audit the new actual scales at dispatch and ensure the actor/critic can identify any new task-local gate/state; do not assert439 is sufficient solely because it contains previous nominal. Do not use an extra mapper advance to estimate current b.
- Keep old/current log probability on the exact raw Gaussian action (`semantic_training.py:421,1773,1862,2178`). A deterministic state-dependent environment transform—even many-to-one—does not require a tanh/headroom Jacobian in this raw-action PPO convention. Never substitute applied action into old raw logp or call a clipped target a Gaussian sample. If the Gaussian mean/sigma or HISTORY coordinates also change, sampling/current likelihood/audit must share the same revised kernel.
- Preserving the current REQUEST-history meaning is the smallest candidate: scale **after** request filtering, keep r observations/previous-ACK semantics distinct from effective offsets, log both, and independently replay actual/zero-current counterfactuals through the same transform/owner state. Whether previous-REQUEST tracking compensation partly counteracts the new scale needs explicit sequence evidence; don't silently switch to realized residual feedback.
- Same raw weights under a different transform do not preserve physical actions or on-policy data compatibility. Any approved transform needs a version boundary, compatible weight/state preservation, fresh rollout and consistent train/eval mapping. No new migration is prepared here.
- Removing a flat region does not guarantee useful variance: tanh at strongly negative means still has a small derivative, available room may be tiny, final slew may make nearby requests indistinguishable for a tick, and PPO is not differentiating through this projection. Wider sigma can mainly add endpoint mass; rescaling can also reduce effective degrees-per-raw. Decide from actual request/effective/FINAL response diversity and task progress, not algebra alone.

Immediate recommendation remains: finish and inspect the current72e real update before choosing any action-semantic experiment. This memo supplies a falsifiable local hypothesis, not a proposed hot change or a success claim.
