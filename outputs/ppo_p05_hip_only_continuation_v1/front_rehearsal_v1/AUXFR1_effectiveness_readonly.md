# AUXFR1: why the saved-state action change was small

Read-only design review during real block06. No optimization, model forward, GPU/Isaac launch, runtime/config edit, or bound-helper edit was performed. Existing recorded inspection/JVP and execution arrays were recomputed with PowerShell only. This review is not a training/evaluation gate.

## Finding

**The actual stopping condition was the reviewed 32-attempt budget, not a trust rejection.** All 32 proposals changed parameters and were accepted at fixed LR500; `stop_reason=finite_budget_exhausted`. The small REQUEST changes agree with the already-recorded initial gradient response. They do not establish a broken gradient, mask, hard clipping, or a proven rank/capacity ceiling.

| Saved-state channel | Initial JVP ×500, one-step tangent | Frozen initial JVP ×16000, not a fit prediction | Actual final maximum absolute REQUEST change |
|---|---:|---:|---:|
| FL knee | 0.00861417° | 0.27565350° | 0.24035832° |
| FR knee | 0.00213074° | 0.06818366° | 0.05908418° |
| FL wheel | 0.000114573 rad/s | 0.003666350 rad/s | 0.003257450 rad/s |

Actual changes are about 87–89% of that simple frozen-gradient tangent. The true selected-gradient norm fell from `5.7746678e-5` to `4.2210773e-5` (about 27%); curvature and changing errors therefore matter. The recorded first gradient's P01/P02 column norms were `2.4525725e-6` / `5.7694571e-5`. This identifies a weak/differently distributed response in the selected basis, not a gradient rescaling error.

| Cumulative constraint, both datasets | Largest observed | Limit | Fraction of limit |
|---|---:|---:|---:|
| Joint REQUEST change | 0.240358° | 3° | 8.0% |
| Wheel REQUEST change | 0.00325745 rad/s | 0.15 rad/s | 2.17% |
| Forward full-Gaussian KL | 0.30428012 | 0.5 | 60.9% |
| Reverse full-Gaussian KL | 0.29407475 | 0.5 | 58.8% |
| Absolute log-sigma change | 0.17214823 | 0.25 | 68.9% |

These limits were checked but **not active as stopping constraints**. They nevertheless are not unlimited future headroom: KL grew `0.000407 → 0.024292 → 0.089618 → 0.185839 → 0.304280` at attempts1/8/16/24/32. Simply multiplying the budget cannot be assumed to produce proportional useful physical motion while satisfying the same Gaussian bounds.

At the training row with maximum final forward KL (row27), the mean-change component is `0.19926346` and sigma-change component `0.10501667`. Thus roughly one-third, not all, of that KL is sigma drift. Validation's maximum is `0.18863621 + 0.10693423 = 0.29557045`. This shared-trunk edit is **not mean-only** even though its objective only fits raw means. Narrow learned Gaussian sigmas can also make small raw-mean changes materially large in KL units.

## Parameterization and objective explain the local response

For a front-phase state, changing `W[:,0:2]` changes only that phase's 256 first-layer preactivation offsets. Every state's response still passes through the unchanged nonlinear trunk and both final heads. It cannot select twelve independent output corrections per physical state. This structural restriction is real; a numerical Jacobian-rank or optimality limitation was **not** measured here.

The exact objective is `L = mean((mu - saved_raw12)^2)/2`, with `mu = 0.1*f_mean(x) + 0.9*history_center(x)`. Its first-layer gradient averages across the selected states and all12 channels and retains the genuine 0.1 factor. The REQUEST derivative adds `cap*(1-tanh(mu)^2)`. These normalizations are intentional, not bugs to undo. FR knee's recorded initial means were within `[-0.405352, 0.080516]`; its tanh derivative is at least about0.85. Strong FR-knee tanh saturation therefore does not explain the tiny response.

Overall train/validation raw MSE fell **4.65% /5.16%**, but this was not equivalent FR-task recovery. FR-knee REQUEST mean absolute target error changed:

- Train: `0.84349287° → 0.84195775°` (almost unchanged).
- Validation: `0.93608165° → 0.93853384°` (slightly worse).
- FL-knee errors improved much more: train `0.99517155° → 0.89160490°`, validation `0.99121231° → 0.86115086°`.

The fit had no physical-progress objective or proof that each random raw sample was a uniquely desirable deterministic action. Its 95 training and93 interleaved validation rows come from one earlier locally successful FR segment; validation is not an independent current-policy trajectory. The source's later P12 failure remains a failure and was excluded from positive labels. The actual AUXFR1 natural-P01 deterministic rollout subsequently ended at15.583333s in P02, `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`, not full success.

## Closed-loop limitation

On ordinary raw-history ticks, a simplified local recurrence is `delta_mu_t ≈ 0.1*delta_f_t + 0.9*delta_mu_(t-1)`. A constant network correction can accumulate over time rather than remain at its one-step value. But actual state feedback, contact, mapper history, and cap-transition REQUEST-history entry rules change that recurrence. The recorded JVP holds all old 389-state inputs/history fixed. It does not predict the current deterministic predecessor distribution or the sign of a closed-loop clearance change. **Neither small one-step changes nor lower offline MSE justify automatically increasing AUX steps.**

## At most two future options — neither executed or required now

1. **Minimal: retain the exact512-column parameterization and existing kernel.** Finish block06 and its actual latest-checkpoint deterministic evaluation first. Only if its physical front-segment evidence supports another local rehearsal, inspect that latest compatible checkpoint against still-valid successful raw front data plus current-trajectory diagnostic holdout, then authorize at most one new explicitly bounded fixed-LR AUX event. Any changed budget must be justified against fresh response/KL/sigma measurements and the physical discrepancy, not this old MSE alone; no LR search or assumption that larger steps fix P02. Current failed actions may be negative/diagnostic holdout but must not become successful raw targets. This needs **no new actor/policy or389-observation schema**. An unchanged helper can append a fresh source/data/budget-bound AUX event; a changed objective/helper would require its own versioned AUX method/report and reviewed ledger binding, never overwriting event1. Same-input P03+ invariance and existing Adam remain exact for the AUX edit. No numerical new budget is recommended by this review.

2. **Reserve only if current-state evidence shows the same basis repeatedly ineffective: add an explicit phase-gated mean-only adapter.** For example, zero-initialized `u[P01/P02,12]` added to the network mean **before** the existing0.1 HISTORY combination; sigma head/kernel remains untouched. This gives directly controllable per-phase mean coordinates, but only phase offsets, not a rich state-dependent solution. It is newly added policy capacity, **not an edit of existing compatible weights** and not a reset/replacement of the mean head. It needs a new actor/policy version and topology migration, new metadata/report/ledger method version, exact preservation of every old weight and old Adam entry, explicit zero initialization of new parameters/Adam state, unchanged389 observation schema, empty rollout, zero migration PPO credit, and save/reload plus real evaluation. The adapter's gate must be zero for P03+; zero initialization must reproduce the old full Gaussian exactly, and trained same-input P03+ invariance must be tested. Later PPO can continue learning all original stages normally. Limited old raw data may overfit even24 added parameters; this is not recommended merely because one32-step fit failed. No evaluation-only rescue, alternate controller, algorithm switch, or success claim is implied.

Current priority remains block06 real PPO and deterministic evaluation. There is no evidence here to interrupt that work or relax physical acceptance.

## Evidence and integrity

- `CP209920_aux_execution_32x500.json`: all32 actual proposals, original/final Gaussian arrays, gradients, JVP, targets, source/data/helper bindings.
- `CP209920_readonly_final.json`, `budget_root_reviewed_32x500.json`, `actual_aux_audit.md`: inspection, reviewed budget, independent real saved-state integrity audit.
- `../DELIVERY.md`: sealed AUXFR1 physical/video failure and current real training handoff.
- Bound `front_rehearsal.py` still matches SHA256 `d72ff9dd9de8f2e302d2d41d9660a70ba1c880eeb2505b6cd8db26da37895321`.
- This review adds **0 policy decisions, 0 PPO updates, 0 AUX steps**, and no new physical evidence.
