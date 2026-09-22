# Actual RR AUX event4 — independent CPU audit

**PASS.** Plain CP214400 `8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9` → unique RR AUX `9987a4df3b4a2afaeddac794aa97b650e6133fe7ec89e8484cf59084cda69b6a`. Both saved payloads use old5fd; this audit does not consult or rebind a subsequent live runtime migration.

- Actual **7 accepted / 8 attempted**, fixed SGD LR0.25, no retry. Attempt8 exceeded protection KL: **0.567010767 > 0.5**, so the full3084 proposal was discarded. Saved final outputs and change metrics exactly equal accepted attempt7, not rejected attempt8; independent CPU evaluation reproduces the saved receipt.
- Exactly **3072 mean weights +12 mean biases** changed. Every trunk/sigma-head/other actor tensor, critic, full PPO Adam/moments, Identity, PPO LR1e-5, CUDA:0 runner and Python/NumPy/CPU/CUDA RNG is unchanged. Same-input σ/logσ are exactly equal on all342 train/171 validation/305 protection/2 P01 probes.
- PPO stays **214400 / 1640 / 32800**, +0. Old full events1–3 stay exact (**front96/96**); new **RR7/8** gives mixed ledger **103/104**, while older separate AUX **7/8** remains unchanged. All original origins/migrations persist. Official CUDA save and independent fresh reload are recorded successful. No pointer or checkpoint was written by this audit.

| Actual-raw target MSE (independent CPU) | Before | After |
| --- | ---: | ---: |
| train_P09 | 0.009428841993 | 0.009111006744 |
| train_P10 | 0.03232318908 | 0.03141816333 |
| train_P11 | 11.87680912 | 11.86894321 |
| validation_P09 | 0.008570441976 | 0.008210707456 |
| validation_P11 | 0.5993668437 | 0.6008877754 |

Saved protection max KL is **0.485127939**; largest joint REQUEST drift **0.350753784°**, wheel drift **0.0220548809 rad/s**. σ is fixed, but means may change across all phases. These are finite fixed-state guards, not global P03+ mean invariance or future-trajectory safety proof.

Targets remain all12 actually executed stochastic raw values, not source μ, final transformed targets or nominal labels. Only current X17 was explicitly derived on237 historical rows. P10 has one train/no validation row; validation is correlated within one episode. P13 has no real protection coverage. Lower supervised loss establishes neither current RR placement nor full-task success. Ordinary real PPO carry after event4 is not yet verified.

CPU-only audit; zero fit, GPU/Isaac, production/frozen-helper edits or checkpoint writes. Helper exits after this report.
