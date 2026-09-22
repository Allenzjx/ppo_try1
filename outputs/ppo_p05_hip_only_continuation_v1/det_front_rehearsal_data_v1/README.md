# CP201728 deterministic P02 candidate data (read-only preparation)

Directory correction: this entire self-created candidate folder was safely moved under `outputs/ppo_p05_hip_only_continuation_v1/`. No files were removed. `candidate_manifest_pre_directory_correction.json` retains the original path receipt (SHA256 `08f5bf4d08dcf13c82e1c6866e937b9861aa935d2d176ff1270cb0c5675f7b67`); the active manifest updates only the three moved artifact paths and the builder hash after its relative ROOT changed from `OUT.parents[1]` to `OUT.parents[2]`. NPZ and row-check bytes are unchanged. Initial builder/API and CPU/CUDA numeric-check stop records remain preserved, with their resolutions stated. Use the active `candidate_manifest.json` and `read_candidate.load_candidate` for new reads.

254 rows: decisions 3–256, pre-action ticks 16–2040 (0.133333–17.0 s), actions end at tick 2048. Only P02; P01/reset is excluded.

`candidate.npz` contains reconstructed float32 X389, unchanged recorded raw12 deterministic actions, recorded conditional μ/σ/history, source IDs and suggested split indices. `row_checks.jsonl` records every source-row binding and replay error. `candidate_manifest.json` binds source checkpoint, source-prefix bytes, helper and artifacts.

The source did **not** save the 389 input tensor. This is field-supported reconstruction, not a directly recorded tensor and not a bitwise proof: original execution was CUDA, replay is CPU. Matching 12 outputs alone cannot establish all 389 inputs; explicit field/time provenance is also recorded. No guessed histories, zero-filled missing fields, or reconstructed video pixels were used.

All 254 rows passed the fixed predeclared absolute tolerance 1e-06. Maximum μ error 2.98023223877e-08; base μ 5.96046447754e-08; history 0; learned σ 5.96046447754e-08; effective σ 2.23517417908e-08; effective logσ 4.76837158203e-07. Previous raw, appended assist/continuation and history center match exactly. All 2032 action-interval physical ticks have all12 residual permission and zero assist ownership/correction.

Local evidence only: FR actual legal TOP/support and placed tick 2072 (decision 259 endpoint); FL placed tick 3257, legal TOP/support at decision 408 endpoint3264, followed by actual P06 decision409. FL continuation may include the declared assist; this is not a pure-policy FL or whole-task success claim. Later rear outcomes are not analyzed or used as positive labels.

Suggested fixed split, not fitted: train indices `range(0,254,3)` (85); validation `range(1,254,3)` (85); retain remaining `range(2,254,3)` (84) as source-only. Interleaving is temporally correlated and is not independent-episode validation or proof of closed-loop improvement.

The labels are the actual source deterministic conditional mean/raw actions, not independently recomputed means, tanh requests or final servo targets. Source checkpoint is CP201728, not a proposal to roll back the latest model. Source task-potential values are retained; current-policy compatibility/admission would require separate review.

Preparation added **0 AUX updates / 0 PPO decisions / 0 PPO updates**. No fitting, simulator, GPU, production edits, checkpoint writes, or edits to existing hash-bound v1 rehearsal helpers. CPU process exited after writing and independently re-reading these candidate arrays.
