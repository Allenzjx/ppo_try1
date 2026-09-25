# 72e P12 first-episode credit / horizon snapshot

Read-only snapshot of `runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0835078527430Z_g72e63592bdf4_eb09a77b75b0420997a80291feb869d8`: first 452 policy decisions, completed updates 1736–1738. This report does not describe later training progress. No runtime/configuration/model changes, Torch/PXR/Isaac imports, model forwards, or optimizer calls were performed for this analysis.

## Measured episode and saved-update evidence

All 452 requests and end phases were P12. Ordinary phase changes do not set `done`: `semantic_training.py:286` uses `step.terminated`. This particular episode contained no phase transition to test empirically.

- RL current-lift validity: decisions 1–5 (inherited qualification tick 6212), 10–21 (fresh qualification tick 6294), and 114–124 (fresh qualification tick 7124): 28 decisions total. RL crossing and placement remained false throughout.
- RR remained outside top XY from decision 141 / tick 7344 onward. First measured ground contact: decision 152 / tick 7432. RR ground contact appeared in 295 of 452 decision-end records; not every intermediate physics tick is represented by that count.
- Decision 452 / global decision 227012 / tick 9832 / episode time 81.933333 s ended with `INCOMPLETE_CONTROLLER_BLOCKED`, not a hard-limit termination. Reward = −43.0867767334, old critic value = −12.0903148651; terminal event = −40, terminal potential shaping = −3.0854431368. `terminal_bootstrap_allowed=false`, `time_outs=false`.
- Updates 1736–1738 each completed 128 samples and 20 optimizer steps with actual learning rate 1e−5. All three have terminal_count=0 and a nonterminal bootstrap tail. The terminal at decision 452 lies in the then-unfinished fourth rollout; its official returns/GAE were not yet saved in this snapshot.

| Update | Local decisions | Mean reward | Mean old V | Mean raw GAE | Positive/negative standardized GAE |
|---|---:|---:|---:|---:|---:|
| 1736 | 1–128 | −0.0111820 | −12.878639 | +0.409024 | 52 / 76 |
| 1737 | 129–256 | −0.0063834 | −12.830874 | +0.444420 | 70 / 58 |
| 1738 | 257–384 | −0.0057456 | −12.389981 | +0.588876 | 58 / 70 |

## Actual per-row credit near requalification and retreat

Raw GAE below is the stored Float32 `returns − values`, not a new critic estimate. Standardized GAE is the official whole-rollout-standardized value, also present in the actual minibatch likelihood audit. Each completed rollout index appeared five times in the 20 recorded minibatches, with the same advantage value.

| Local decision / tick | End-of-action physical evidence | Reward | Old V | Raw GAE | Standardized GAE |
|---|---|---:|---:|---:|---:|
| 114 / 7128 | RL requalified at 7124; RR TOP contact | +0.479578 | −13.211000 | +0.292653 | −0.229188 |
| 120 / 7176 | RL still qualified; RR TOP contact | +0.012644 | −12.878820 | −0.726575 | −2.236520 |
| 125 / 7216 | RL validity and RR TOP contact lost | −0.512920 | −13.193138 | −0.281851 | −1.360653 |
| 141 / 7344 | RR begins persistent top-XY exit | −0.001912 | −13.050222 | +0.972229 | +1.478113 |
| 152 / 7432 | RR first measured ground contact | −0.031370 | −13.047647 | +0.985990 | +1.516649 |
| 452 / 9832 | Controller-blocked terminal | −43.086777 | −12.090315 | Not yet saved | Not yet saved |

Negative standardized GAE at an RL-qualification row **does not prove that this action's policy probability fell**. Positive GAE during RR retreat likewise does not prove that the policy learned to retreat. PPO clipping, gradients shared across samples, repeated minibatches, and changed distributions must be considered before such an attribution. Actor hashes, aggregate losses, and advantage signs alone do not establish physical learning.

## Measured versus inferred bootstrap

Measured: `advantage_audit.jsonl` records gamma 0.9985, lambda 0.99, three nonterminal tails, and `last_values_measured_separately=null`. Source calls official `compute_returns(obs)` at `semantic_training.py:2254`, then saves the rollout at :2259 onward.

Inferred only: for a nonterminal final storage row, `(stored_return_tail − stored_reward_tail) / gamma` implies bootstrap values approximately −13.0138155, −13.0126700, and −12.3064044 for updates 1736–1738. These contain Float32 rounding and are **not independently logged or re-evaluated critic outputs**. The next rollout's old value need not match because an optimizer update occurs between rollouts.

Saved-rollout access used only Python standard-library ZIP reading, a restricted unpickler that replaced the known tensor rebuild globals with inert storage descriptors, and `struct` Float32 decoding. Per-row stored reward/value exactly matched decision JSON; reconstructed raw-GAE means exactly matched the existing aggregate audit. No Torch or model was loaded. The official stored standardized advantages matched the corresponding likelihood audit.

## Horizon interpretation and bounded recommendation

128 decisions = 8.533333 s at 15 Hz. With `gamma × lambda = 0.988515`, the GAE exponential time constant is 86.569 decisions / 5.771 s and half-life is about 4.000 s. The TD-residual propagation factors `(gamma × lambda)^n` are 0.227960 at 128, 0.051966 at 256, and 0.002700 at 512 decisions. These are GAE propagation factors, not probabilities or success estimates.

The observed data supports a **future versioned 256-step comparison**, not an immediate change or a claim that rollout length is the sole obstacle. A 256-step window would cover both the decision-114 RL requalification and decision-152 RR ground contact within one collection block, reducing reliance on the intervening tail estimate. However, the immediate RL validity loss at decision 125 was already visible inside the first 128-step block.

A 512-step first window would encompass this particular episode's decision-452 terminal, but that outcome is 338 decisions / 22.533 s after decision 114; its GAE propagation factor back to that row is only 0.020153. Longer collection alone therefore does not guarantee adequate terminal credit or corrected control. The three existing rollouts were collected under successively updated policies and must not be retroactively concatenated into a purported on-policy long-rollout experiment.

Current rollout length is also part of runner configuration and migration/storage-shape checks (`semantic_training.py:50,169,1052`). Any future change requires a legal saved boundary, versioned compatible treatment, and fresh collection; it is not a hot-edit of a constant.

**No new training gate:** finish the authorized active P12 1536 and following natural-P01 1536 work. This optional horizon experiment is not a prerequisite for their training, checkpoint saving, deterministic evaluation, or video delivery.
