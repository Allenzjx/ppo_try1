# RR410 front retention candidate — preparation only

No real inspection, optimizer step, checkpoint save, simulator or production edit has been run for this candidate yet. The ongoing v9 rear curriculum is not gated by this work. A later invocation must name the actual newest chosen **sealed v9 same-branch checkpoint**, its SHA256 and its manifest SHA256 explicitly; it must not silently roll back to the initial221952 v9 publication.

Static review correction: reset row0 is excluded from supervised current410 admission because its complete measured pre-state is not available for current Phi/index17 recomputation. The immutable original95-row selection remains untouched; this candidate uses94 training rows, retaining the real P01 decision2. Every selected input now has an adjacent measured evaluator and an explicit old/current Phi equality check. All465 original source rows remain available for provenance and pre-P09 WAIT induction, not an unproved reset-Phi claim. Encoder index138 and v9 P09/P12-only wheel scope are reviewed static premises bound to frozen module hashes. Checks and tests remain unexecuted pending the CPU window.

## Scope

- `reviewed_data_410.py`: uses94 train /93 validation P01–P02 raw12 labels after excluding only original reset row0, plus13 P03–P06 observation-only invariance rows. It validates the original465-row contiguous source prefix and pinned rollout/audit bytes. No source label or old helper is changed.
- The original389 observation is directly saved. The new410 observation is **derived**, not originally saved: old389 is retained, with21 zeros admitted only after RR WAIT-before-P09 induction, actual RR history/contact checks, inactive P09/P12 wheel scope, and current-vs-old per-input Phi/index17 checks. No reset frame is invented or included as a supervised current410 input. Missing evidence fails closed.
- `front_retention.py`: narrowly adapted from frozen AUXFR1 kernel SHA`d72ff9dd9de8f2e302d2d41d9660a70ba1c880eeb2505b6cd8db26da37895321`. It uses the actual410 actor and original389 HISTORY prefix, unchanged receiving-wheel sigma kernel and caps. Only the512 scalars `actor.mlp.0.weight[:,0:2]` may change.
- `retention_cli.py`: defaults to CPU read-only current actor inspection. It reports initial raw/request error and first-gradient REQUEST direction for all four wheels. A JVP is not a finite-step prediction and does not prove physical recovery; an unfavorable FL-wheel direction does not trigger any automatic fitting.

P01/P02 means **and sigmas** may change. For legal P03–P13 one-hot inputs the two selected columns multiply zero; all other actor parameters and the kernel stay unchanged, so the complete same-input Gaussian must remain exact. The13 real holdout rows cover only P03–P06; P07–P13 algebraic probes are synthetic, not physical evidence. Different future trajectories are not claimed invariant.

The old front episode really placed FR and continued through assisted FL capture/P06, then failed in P12. These are **local front raw-action labels**, not whole-task success, not pure-policy FL capture, and not current on-policy data. Old conditional means are provenance only; old logp/GAE are not learning targets. The original AUXFR1 did not establish deterministic P02 recovery. The same correlated single episode supplies train and validation; only1 P01 row enters training and there is no independent P01 validation.

## Execution remains separate and bounded

Future explicit `--execute-aux` requires a same-source/hash/helper/data inspection receipt, a unique same-branch AUX filename and a reviewed finite budget. No learning rate or trust budget is selected here. Budget fields are:

`max_attempts` (1–32), `learning_rate`, `maximum_train_request_shift_full12`, `maximum_validation_request_shift_full12`, `maximum_per_state_full_gaussian_kl`, `maximum_abs_log_sigma_change`.

All limits are cumulative relative to the original candidate, not reset each step. Independent temporary SGD stops on the first rejected proposal, rolls back all512 proposed scalars, and does not retry/search LR. PPO Adam/LR, critic, normalizer and full Python/NumPy/CPU/CUDA RNG are preserved. Save/fresh-reload use the actual source device and full original CUDA visibility, not a CPU metadata rewrite. Rollout must be empty.

New AUX counts go exclusively into `rr_capture_transfer_branch.front_retention_auxiliary`, with compact source/helper/data/fit-report bindings. Old rr_postcross_workspace AUX events1–4 (front96/96, RR7/8, mixed103/104) and old historical7/8 records remain unchanged. PPO counters do not advance. Same-branch routing/v9 receipt and all origins remain unchanged. Neither main nor branch latest pointer is published by the AUX saver. Subsequent real PPO carry is not claimed tested before such a run actually occurs.

## Commands — only after root grants the CPU window

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m pytest -q outputs/ppo_rr_capture_then_rl_transfer_v1/front_retention_410_v1/test_front_retention.py outputs/ppo_rr_capture_then_rl_transfer_v1/front_retention_410_v1/test_retention_data_and_publication.py
```

Read-only real inspection (no `--execute-aux`):

```text
python outputs/ppo_rr_capture_then_rl_transfer_v1/front_retention_410_v1/retention_cli.py
  --checkpoint <actual latest sealed v9 branch immutable checkpoint>
  --expected-source-sha256 <actual file SHA256>
  --expected-manifest-sha256 <actual sidecar SHA256>
  --expected-head 3edda51732f4ff85717fcb3491bf5c8c5766474d
  --report <new absolute path in this directory>
```

This preparation does not authorize an actual AUX update. Do not run a real fit until the source-specific inspection, direction and finite budget are reviewed explicitly.
