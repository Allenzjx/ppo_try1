# Actual mean-head AUX event3 — independent CPU audit

**PASS.** Plain CP213376 `8e5961e3e78260f12124447bff72a7459b7323d2da8f83cc467a59ddc9290e54` → unique AUX candidate `945b05e2763396c0f83c582eb85d34d778e6fb8c74b041177c3bdea9f7077cd3`. Actual source-device official save and independent fresh reload are recorded true; embedded/sidecar, tensor and complete optimizer hashes match. Latest pointer remains the plain source; this audit changed no pointer or checkpoint.

- Actual **32 accepted / 32 attempted**, LR4 finite independent SGD. Exactly **3084** existing mean-head scalars changed: 3072 weights and 12 biases. All trunk, log-sigma rows, other actor state and critic tensors are exactly unchanged.
- Same-input sigma/log-sigma are **bitwise equal** in independent CPU checks on all train/validation/19 protection/2 unlabeled P01 probes, and follow algebraically from the frozen trunk, sigma head and state-dependent sigma kernel. This is not a future-trajectory statement.
- Full PPO Adam/moments, LR1e-5, Identity, runner CUDA:0, full Python/NumPy/CPU/CUDA RNG and all three origins/migrations are identical. PPO remains **213376 / 1632 / 32640**, adds **0**. Old events1/2 remain exactly intact; front AUX now **96/96**, older ledger **7/8** unchanged.

| Fixed-state metric | Before | After |
| --- | ---: | ---: |
| P01_train | 0.000236830412177 | 3.46322485711e-05 |
| P02_train | 0.00022834369156 | 1.18359066619e-05 |
| P02_validation | 0.000229137542192 | 1.1861636267e-05 |
| total_explicit_objective | 0.000115232687676 | 1.74189772224e-05 |
| protection_half_MSE | 0 | 8.65148012963e-06 |

Across all 32 proposals relative to the original model, the 19 real P03–P12 protection states changed: max raw μ 0.0222686678, full Gaussian KL 0.381115714 (limit .5), largest joint REQUEST shift 0.391274452° (limit1°), largest wheel REQUEST shift 0.0258513689 rad/s (limit .03). FL knee 0.391274452°, FR knee 0.282173157°. σ shift is zero. Independent final CPU distributions agree with the actual CUDA receipt within declared numerical comparison tolerance.

This **does not preserve P03+ means globally**: the mean head affects all phases, and measured protection drift is nonzero. P13 has no real protection observation. Only one P01 positive exists and it has no independent validation; the 85 P02 validation rows are correlated with the 85 training rows from one historical trajectory. Positive observations are explicitly field-reconstructed historical389 inputs, not new on-policy data. These lower losses establish neither current physical success nor stability improvement.

Actual ordinary PPO carry after event3 is still unverified. CPU-only read audit; no optimizer step, GPU/Isaac, production edit, frozen-helper edit or checkpoint write. Helper exits after this report.
