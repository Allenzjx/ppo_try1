# Two completed episodes: reward and return audit

Scope: only ep0 and ep1 from real run `20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff`, plus their relevant sealed return tensors. Read-only CPU inspection; no runtime/config/checkpoint changes. Runtime revision a802b24d78df, current reward profile `task_conditioned_hip_wheel_quality_v1`.

## Terminal treatment is present and consistent

`semantic_reward.py:436–449` applies, once per issued policy decision:

`r_t = 5(γ Φ_after − Φ_before) + terminal_event − 0.02 Δt − weighted_quality_costs`.

- γ=0.9985, actual PPO GAE λ=0.99, rollout128. Reward and saved runner agree.
- Every nonempty non-success task termination, including `INCOMPLETE_CONTROLLER_BLOCKED`, physical failure and task timeout, gets **−40**; actual SUCCESS gets **+40**. Operational/infrastructure exceptions are not silently converted into successful training terminals.
- Every task terminal forces reward `Φ_after=0` and disables bootstrap. Actual measured terminal potential remains separately visible for diagnostics; it is not used as a terminal reward remainder.
- Both episodes end after a short5-tick interval (0.0416667s). Costs use actual physical time; γ is still applied once per issued action, not raised to5/8. Ordinary phase changes do not terminate the episode.

## Actual trajectory sums

Discounted values below use the actual float32 rewards sent to PPO, summed over each complete recorded path from its reset. `episode_return` is the different, undiscounted float64 reward-breakdown sum.

| | ep0: P12 incomplete, RR placed earlier | ep1: P02 incomplete |
| --- | ---: | ---: |
| Global decisions | 203777–205497 | 205498–205732 |
| Decisions / actual duration | 1721 /114.708333s | 235 /15.641667s |
| Initial Φ | 0.04514193655 | 0.04514193655 |
| Diagnostic physical terminal Φ | 0.62407572175 | 0.05399231237 |
| Reward terminal Φ_after | **0** | **0** |
| Terminal event | **−40** | **−40** |
| Terminal transition reward, float64 | −43.121126484 | −40.270699126 |
| Terminal discount γ^(T−1) | 0.075627377 | 0.703798269 |
| **Actual discounted path reward** | **−4.088725315** | **−28.647706374** |
| Logged undiscounted episode_return | −49.080845599 | −40.677814807 |

Thus the early-incomplete preference inferred from undiscounted scores is **not** the ordering of these actual discounted paths. This is not proof that the later trajectory is preferred specifically for its physical progress: much of its relative advantage comes from postponing the discounted negative terminal event.

Float64 discounted component reconstruction:

| Component | ep0 | ep1 |
| --- | ---: | ---: |
| Potential shaping | −0.225709683 | −0.225709683 |
| Terminal event | −3.025095063 | −28.151930762 |
| Actual-time cost | −0.821727577 | −0.263876926 |
| Enabled body/geometry quality family | −0.016192866 | −0.006189392 |
| Other cost families | 0 | 0 |

Potential is exactly continuous between adjacent nonterminal actions in both paths. With terminal potential zero,

`Σ(t=0…T−1) γ^t·5(γΦ_(t+1)−Φ_t) = −5Φ_0`.

Measured residual from this identity is below2e−15; float32 reward rounding only changes the full discounted sums by about1e−7. **No terminal-potential residue and no missing incomplete-event penalty were found.** The undiscounted sum of a γ-shaped reward does not telescope the same way, explaining why its ranking can differ.

## PPO return estimator is not the full-path sum

Official RSL PPO computes `δ=r+γ(1−done)V_next−V`, `A=δ+γλ(1−done)A_next`, and `return=A+V` over128-step batches. Nonterminal batch ends bootstrap the critic; task terminals do not. `normalize_advantage_per_mini_batch=false` means the stored advantages are normalized over the rollout.

Actual saved examples:

| Sample | Stored λ-return | Old V | Raw GAE | Stored normalized advantage |
| --- | ---: | ---: | ---: | ---: |
| ep0 first, rollout1558/index0 | −6.616279 | −7.010846 | +0.394567 | +2.642852 |
| ep0 terminal,1571/index56 | −43.121128 | −18.680981 | −24.440147 | −1.849533 |
| ep1 first,1571/index57 | −5.521269 | −5.980410 | +0.459141 | +0.891924 |
| ep1 terminal,1573/index35 | −40.270699 | −6.948671 | −33.322028 | −2.058831 |

Each terminal stored return equals its actual terminal reward, verifying no critic/reset bootstrap leakage. Early λ-returns depend on the current critic and batch horizon, so they need not equal the eventual full-episode Monte Carlo return or rank two reset states the same way. Policies are also updated between batches during these real episodes. Two realized paths do not establish causality, a stationary-policy value ordering or a global optimum.

## Separate algebraic issue: delaying an otherwise identical failure

There is a genuine but different discounted-objective incentive worth keeping explicit. From the same state, let F=40 be an unchanged eventual failure penalty, q_f the terminal transition cost, and q the cost of one additional no-progress action before the identical failure. Terminal-zero shaping cancels in either path:

`G_delayed − G_immediate = (1−γ)(F+q_f) − q`.

Pure synthetic positive/boundary/negative examples were checked, without physical execution. With q_f=0.02/15:

- q=0.02/15: delay increases return by **0.058668667**.
- q=0.060002: tie.
- q=0.07: delay decreases return by **0.009998**.

Even the upper bound from enabled time+quality costs per full action, (0.02+0.06)/15=0.0053333, is below the approximately0.06 discounted-failure saving. Conversely, delaying an identical success by one time-cost-only action reduces return by0.061331333; success versus failure at the same terminal time retains an80-point event margin before discount.

This confirms an abstract incentive to postpone a doomed no-progress failure, **not evidence that either recorded policy intentionally stalls**, and not the initially suspected incentive to fail early. The existing failure-avoidance-bound comment already notes that it does not order different-duration failures.

If a later revision specifically aims to remove this incentive, a necessary counterfactual acceptance condition is: adding a physically identical/no-progress idle segment before the same failure must not improve its discounted objective, while same-time real SUCCESS must still beat incomplete/failure and justified moving/settling must remain allowed. Algebraically, for a time-indexed positive terminal penalty D_t, require `D_t − γD_(t+1) ≤ q` in that no-progress comparison (including any shifted terminal transition cost consistently). This is a design constraint, **not a proposal to increase costs blindly, retain terminal potential, weaken safety, or change reward now**. No implementation or migration is included in this audit.
