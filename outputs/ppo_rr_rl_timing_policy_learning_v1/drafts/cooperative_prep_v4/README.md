# Cooperative rear preparation sigma — OUTPUTS-ONLY DRAFT

Status update: after the 1536-decision course naturally sealed and Isaac exited, root authorized promotion. The actor/profile/sigma and registration/likelihood/request/prefix hooks are now installed in the **uncommitted working tree**, based on `d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd`. The active course was not hot-modified. New CPU tests19 plus relevant old regressions137 passed (156 total), with `CUDA_VISIBLE_DEVICES=-1`; no Isaac/physical run was made by this work. Migration/loading and final freeze remain separate root/artifact-agent responsibilities. No real PPO credit is claimed for synthetic tests.

The three module files here preserve the original review draft; promoted files are under `src/wlr50_clean/ppo/`. The new production tests are `tests/unit/test_semantic_rear_cooperative_prep_policy.py`.

## Scope and exact observable gate

New policy version: `rear_cooperative_prep_history_v1`. Existing observation layout remains `role419_p02_progress_v1`, 422 features, unchanged numerical encoder. The actor subclasses the current P02-progress actor, adds no parameters or buffers, and delegates deterministic inference directly to its parent. Full12, Identity normalization, HISTORY rho=0.9, actual learned weights and Adam are retained. This preserves the initial same-input deterministic function; it does **not** promise identical future stochastic trajectories or unchanged means after learning.

The shared tensor kernel reads existing observable fields, without a new latch:

| Input | Existing index/source | Purpose |
|---|---|---|
| `rr_carry_capture` | 410 | Current carry/recapture task request |
| `rr_top_reachable` | 404 | Existing physical task candidate; not contact or bearing |
| `rl_prep_transfer` | 412 | Existing preparation request |
| `receiving_continuation_active` | receiving-parent evidence: phase P10–P12 plus RR placed-history index 157 | Suppress the additional FL-wheel factor during the parent's receiving window |

`prep_allowed = (rr_carry_capture && rr_top_reachable) || rl_prep_transfer`.

This is an exploration permission, **not** a rewrite of `support_transfer_permitted`, the P09 source clock, late group ownership, contact qualification, P12 RL cursor, or reward. It neither authorizes unloading a necessary support nor supplies a target angle. It permits policy sampling of positive or negative changes; no bias toward either sign is introduced.

## Total rear multiplier relative to the receiving parent

| Channel | Index | Old rear factor | Draft rear factor |
|---|---:|---|---|
| FL knee | 1 | 2 in RL prep, otherwise 1 | 2 when `prep_allowed`, otherwise 1 |
| FR knee | 3 | 2 in RL prep, otherwise 1 | 16 when `prep_allowed`, otherwise 1 |
| RL hip | 4 | 1 | 2 when `prep_allowed`, otherwise 1 |
| RL knee | 5 | 1 | 8 when `prep_allowed`, otherwise 1 |
| RR hip | 6 | 4 when carry and reachable, otherwise 1 | **Exactly unchanged** |
| FL wheel | 8 | 1 | 2 only when carry and reachable and **not** parent receiving, otherwise 1 |
| FL hip, FR hip, RR knee, FR/RL/RR wheels | 0,2,7,9,10,11 | Existing factors | **Exactly unchanged** |

FR-knee total 16 replaces the old rear factor: it is **not** 16 multiplied on top of the previous 2. FL knee likewise stays total 2 in postcontact RL preparation. The receiving parent actually expands **FR/RR wheel indices 9/11**, not FL wheel 8. Disabling the new FL factor there avoids concurrently widening another wheel; it is not cancellation of an existing FL factor.

The FL gate above is the selected observable formula, not an additional AIR test. In v3, `rr_carry_capture` can also describe lost-bearing recapture; a weak/transient TOP contact may therefore coexist with this gate before the parent receiving window. Calling this a “precontact window” must not be mistaken for measured AIR or zero-contact evidence. No new contact predicate is silently added.

## Risk of ineffective/saturated exploration

At the reported CP221696 AIR/reachable input (not a model forward performed here), the chosen factors imply:

| Channel | Reported effective raw sigma | Candidate effective raw sigma |
|---|---:|---:|
| FL knee | 0.11675 | 0.23350 |
| FR knee | 0.01902 | 0.30432 |
| RL hip | 0.08211 | 0.16422 |
| RL knee | 0.005988 | 0.047904 |
| RR hip | 1.13082 | **1.13082** |
| FL wheel | 0.20696 | 0.41392, only outside parent receiving |

These are dimensionless **conditional raw Gaussian** standard deviations, not joint degrees or final-actuator variation. The current P09 requested capacities remain FL knee 36°, FR knee 112°, RL hip 24°, RL knee 36°, RR hip 24°, FL wheel 1.2 rad/s. Existing tanh, residual slew, same-tick headroom and final physical limits remain unchanged.

FR-knee 16 is intentionally substantial: at raw mean zero, just the nominal transformation `112*tanh(0.30432)` is about 33° for a +1σ displacement, before history/rate/headroom effects. It can increase the fraction of different raw samples that collapse to the same physical limit. A nonzero current mean and limited measured headroom can make that worse; a raw-sigma table alone cannot estimate actual projection or prove useful joint response. RR hip already has the largest reported raw sigma and is deliberately not widened. FL wheel doubling is symmetric, so it can increase reverse as well as forward proposals; this is not a forward-wheel fix. Values have not been silently reduced.

The initial candidate should be evaluated using per-channel raw/request/applied variation, tanh/headroom/rate saturation, and measured joint/contact response. If added variance produces only more clipping or destroys capture support, that is evidence to revise this candidate at a later sealed boundary, not a reason to claim learning progress now. No optimality or success claim is made.

## Shared likelihood and integration

`semantic_rear_cooperative_prep_sigma.cooperative_prep_effective_log_std` is the single intended tensor kernel for actor sampling, current likelihood and audited request evidence. It is parameter-free/differentiable with respect to the existing log-sigma head, adds logarithms of positive observable factors, and introduces no sampling rejection or action transform. Original raw samples and their old Gaussian log probabilities must remain the PPO data. Applied physical projections do not replace those raw samples.

Promoted production explicitly registers the new version/class in policy contract, runner and exact saved/reloaded CLI/prefix validation. `audited_history_policy_request` now accepts the exact new class and selects **this same kernel**, retaining all existing HISTORY/capture/P02 observations. New evidence schema `wlr50_clean.actual_rear_cooperative_prep_request.v1` includes the observed gate booleans and total Full12 multipliers. The official optimizer audit enables the existing head-gradient instrumentation; every new-policy minibatch additionally recomputes sigma with this shared pure kernel from its actual head and saved observations, checks equality with the official Gaussian cache, and records gate evidence. This makes no extra actor forward or random draw. Migration/load/namespace integration is owned by the separate artifact agent; activation still requires its review and the final frozen version.

Migration is same422-to-same422. Bind to the actual latest compatible **sealed** checkpoint after the active course, not CP221696 by default. Strict-copy all actor/critic tensors, full Adam state and parameter groups/actual LR, Identity normalizers and RNG. No new columns, mean-head zeroing or optimizer reset. Old unfinished rollout data must not enter the changed-kernel update; collect fresh on-policy data after the version boundary. The source schedule, evaluator, rear-assist-OFF settings and original successful N remain unchanged by this factor.

## Test status

Executed with base Python, no Torch: `test_cooperative_prep_scalar.py` — **7 passed**. Covers the full 16-case boolean truth table, selected CP sigma arithmetic, total-versus-compounded postcontact factors, receiving-window FL exclusion, identity of unselected channels/front off-gate arithmetic, invalid flags and AST parsing without importing Torch.

`test_cooperative_prep_tensor.py` here remains the archived draft. Its tests were promoted and extended into `tests/unit/test_semantic_rear_cooperative_prep_policy.py`: **19 passed**, covering tensor/scalar agreement, old front sigma equality, strict state-dict/full Adam preservation, same-input deterministic equality, original raw sample/log-probability agreement, ratio=1 without update, exact request audit/prefix registration, head gradients, and a synthetic official optimizer update plus save/reload. The synthetic update128/1/20 is software-test evidence only, never real PPO credit.

Relevant existing regressions also passed: P02-progress policy16, rear timing actor/observation47, rear optimizer audit1, receiving-wheel sigma33, and checkpoint-prefix policy40 = **137**. Combined production tests: **156 passed**. The original scalar draft tests7 also passed separately. No full-suite gate was added.

Files in this directory are review artifacts; the promoted working-tree runtime is separately identifiable. No physical run, real checkpoint migration, real optimizer update or claimed task success resulted from this subtask. Nothing was committed by this agent.
