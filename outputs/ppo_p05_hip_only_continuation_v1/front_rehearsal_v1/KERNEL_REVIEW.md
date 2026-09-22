# Finite front-stage rehearsal kernel

`front_rehearsal.py` is an output-only candidate. Importing it does not load a checkpoint, optimize, publish, sample, or run Isaac. The data loader, source checkpoint verification, ledger, save/reload, and real execution authorization belong to the separate `reviewed_data.py` / `rehearsal_cli.py` layer.

## Scope

- Exact current `SemanticP05CaptureHistoryMLPModel`, 389 observations, raw 12-action Gaussian, Identity normalizer, `p05_capture_request_history`, unchanged rho=.9 and receiving-wheel sigma kernel.
- Only temporary leaf `actor.mlp.0.weight[:,0:2]`, 256 x 2 = 512 scalars. These are P01/P02 one-hot input columns. No bias, other actor column/layer, critic, PPO Adam, normalizer, or RNG update is permitted.
- Saved raw 12-action conditional-mean half-MSE is an explicitly auxiliary supervised objective, not PPO or nominal imitation. P01/P02 **means and sigmas can both change** through the shared trunk. Their entire Gaussian is evaluated, not merely its mean.
- Same-input P03+ Gaussian is checked bitwise on provided real holdout rows. All P03–P13 indicators also receive synthetic algebraic probes. The whitelist proves `DeltaW*x = 0` whenever `x[0:2] = 0`. Synthetic phase coverage is not real physical coverage; changed earlier motion may create different later inputs.
- `cap*tanh(raw mean)` denotes requested residual before mapper/filter/projection/nominal, **not final actuator drive** or physical behavior. Units: first 8 channels degrees, final 4 rad/s; order FL hip/knee, FR hip/knee, RL hip/knee, RR hip/knee, FL/FR/RL/RR wheel.

## Budget and acceptance

No automatic real-data default. Caller supplies constant learning rate, 1–32 maximum attempts, twelve cumulative training REQUEST bounds, twelve cumulative validation REQUEST bounds, maximum per-state full-Gaussian KL, and maximum absolute log-sigma change. Both KL directions are evaluated from actual mean/sigma in float64 with stable `expm1` arithmetic. All trust comparisons are cumulative against the original actor. Validation raw-target MSE is reported, not used as a mechanical acceptance-rate threshold.

Every candidate uses true autograd, including the .1 HISTORY mean chain; no rescaling. First rejection restores the complete 512-scalar temporary leaf and stops without LR search. Zero gradient stops before SGD. Float-rounding/no-change attempts receive no accepted credit. Only the final accepted temporary leaf is copied. Any later invariant failure restores the complete original actor and raises; no checkpoint publication occurs in this helper.

Inspection computes raw/sigma/request errors and a read-only JVP along the negative initial gradient. Its unit-LR tangent is not a finite SGD step, trained policy, or closed-loop success prediction. Inspection does not invoke an optimizer. The CLI performs actual inspection on CPU; authorized future fit supports the original CPU/CUDA device so a CUDA source need not be relocated or metadata falsified. No GPU was used for these kernel tests.

## Verification

30 synthetic current-389 CPU tests passed with `CUDA_VISIBLE_DEVICES=-1`, including:

- exact official deterministic mean and receiving-wheel sigma;
- .1 derivative plus read-only JVP numerical check;
- exact state whitelist, populated PPO Adam, Identity, RNG, and critic preservation;
- full-Gaussian sigma-only KL detection;
- P03+ same-input full-Gaussian bitwise invariance and synthetic P03–P13 coverage;
- invalid phase / leaked partition / 372-layout rejection;
- first and later rejected-step full512 rollback, final-exception whole-actor rollback;
- zero-gradient and numerical-noop credit refusal;
- explicit finite budgets and source-device propagation checks without GPU execution.

These tests give zero real policy decisions, zero PPO updates, zero real AUX updates, and no physical success evidence. No runtime source/config/scripts or accepted N_ref files were edited by the kernel implementation task.
