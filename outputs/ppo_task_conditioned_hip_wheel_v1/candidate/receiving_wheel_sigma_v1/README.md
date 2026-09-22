# Receiving-wheel sigma experiment — sealed candidate, not applied

Seven runtime modules only; `adopt_candidate.patch` is an exact, read-only-verified apply_patch transformation from production HEAD `97ecd305afb5c43b095e1206ce8917e6e742b1bb` to these candidate bytes. `SEALED_FILES.json` records all seven final hashes. No configuration, existing history actor, N, evaluator, mapper, actuator, physics, wrapper, or reward file changes. Diff: +492 / -26 lines, including the two new modules.

Only P10–P12 and observed RR **historical** placed (index 157) use 3× the parent conditional Gaussian sigma for FR/RR wheels (indices 9/11). Current GROUND/AIR/AMBIGUOUS does not undo that history gate. This is continuation/recovery, not proof of current support. Mean, HISTORY/rho, caps, other ten sigmas, and P13 are unchanged. There is no sign bias or sampled-action selection. It also enlarges negative exploration; improved contact or task success is not established.

The new profile is `task_conditioned_receiving_wheel_sigma_x3_v1`. Sampling, actual-request evidence, PPO log-probability/entropy/KL, stochastic evaluation and checkpoint metadata select the same kernel. The immutable parent profile remains separately supported. New deterministic output is bitwise identical for the same weights/state; changed closed-loop stochastic trajectories are expected.

## Verification completed

`final_cpu_tests.xml`: **47 passed** in one coherent candidate package. This includes complete phase/history/channel gates, positive full12 sigma, one official Gaussian draw, unchanged deterministic cache/RNG, current likelihood ratio 1, exact metadata, real-source metadata/full code-scope checks and bounded tamper negatives. The official CPU chain populated Adam with a synthetic 128-sample PPO update, saved non-default 2.25e-5 LR/custom Adam settings, passed the actual plan builder/revalidator and CLI preflight, migrated with all actor/critic/Adam/Identity/RNG/branch/AUX/counters preserved, collected fresh 128 samples, performed the official update, saved and exact-reloaded the new profile, and constructed an independent exact-profile frozen prefix. Missing plan, partial storage, pending transition and non-P01 first migration were rejected.

The CPU source explicitly contains synthetic model state and reference metadata; it is not an adoptable checkpoint, physical trial, or new PPO/AUX credit. Source CP196608 was used only for bounded real metadata checks. No real plan, checkpoint, pointer or Isaac instance was changed.

## Safe adoption boundary

1. Let the running old-profile P12 block finish; select its **actual latest** valid sealed checkpoint. Do not replace it with the CPU fixture checkpoint number.
2. Root reviews/applies the seven-file patch at that safe boundary, verifies `candidate_file_sha256`, and commits. Bind the actual new HEAD/runtime to the actual source checkpoint through `build_migration_plan(..., receiving_wheel_sigma_review={reason, reviewed_code_sha256})`. All seven reviewed hashes are required; all six selected configurations remain byte-identical. No other factor may accompany this migration.
3. Use the existing `--resume-migration` route for train/v3/N1/full_episode/P01, zero offset and default prefix. The loader preserves source effective LR in every Adam group and `alg.learning_rate` (not a forced 1e-5), full tensors/moments, Identity, RNG, lifetime spending, original branch origin and its unchanged one-event AUX 7/8 ledger. It requires new empty 128×1×372/12 storage and adds zero learning credit at migration.
4. Continue fresh on-policy sampling; inspect the first actual 128-sample update/checkpoint receipt and sigma request/likelihood evidence. Subsequent exact resume, ordinary suffix-prefix selection and stochastic evaluation use the saved new profile. No re-run or reset of AUX is authorized here.

Persistent receipt key: `receiving_wheel_sigma_migration`; plan factor: `receiving_wheel_sigma_factor`, schema `wlr50_clean.receiving_wheel_sigma_same372.v1`. This is **same physical MDP, changed stochastic kernel**, not quantity-only. Existing media quantity/AUX bridges must not silently accept it; a separately reviewed adapter must disclose the profile and retain the method label **PPO + LIMITED AUX + receiving-wheel sigma experiment**. Media compatibility is not an optimizer-start gate.

CPU command: use the pinned env_isaaclab Python with CUDA_VISIBLE_DEVICES=-1, pytest `--confcutdir` this directory, and the three test files listed in `SEALED_FILES.json`. `conftest.py` selects the complete isolated candidate once; do not import the older parent candidate bootstrap.
