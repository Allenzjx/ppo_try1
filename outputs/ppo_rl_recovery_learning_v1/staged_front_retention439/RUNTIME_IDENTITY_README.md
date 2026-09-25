# Dormant front-retention439 identity and AUX ledger candidate

**Superseded status:** deployed at 65a9255 with independently reloaded identity
and AUX32 publications. See [DEPLOYED_BOUNDARY.md](DEPLOYED_BOUNDARY.md). The
unapplied/pending wording below is retained as the original preparation record.

Status: **UNAPPLIED**. No fit, migration, checkpoint publication, model import, CPU numerical test, Isaac run, CLI change, or production edit was performed by this staging task. Existing 72e training is unaffected. This is a fallback candidate, not authorization to execute AUX.

## Exact scope

`candidate.apply_patch` / `candidate.patch` contain only three runtime paths and one normal test file:

1. New `src/wlr50_clean/ppo/semantic_front_retention439.py`: same439 full-state identity, independent optional top-level ledger, evidence validation, immutable identity publication API.
2. `semantic_rear_policy_timing_migration.py`: optional outer namespace dispatch, before the existing RR retention dispatch.
3. `semantic_training.py`: identity-load dispatch; validate present identity/ledger on official load and save; carry both keys through every normal PPO save.
4. `tests/unit/test_semantic_front_retention439.py`: focused real-metadata lineage, receipt rejection, full439 CPU identity/reload, and synthetic AUX then actual ordinary PPO-update carry tests.

No CLI, runner, wrapper, controller, actor, reward, distribution, HISTORY, observation, sampling, return profile, or configuration change. Old checkpoints without either new top-level key follow the old route. The runtime contract must differ in exactly those three runtime paths; tests are not runtime paths.

Historical `rr_retention_reward_migration`, `rear_owner_recovery_migration`, all other `*_migration`/`*_branch` objects and the old nested AUX ledger remain byte-equivalent JSON values. Namespace inversion is:

`new identity -> reconstruct 72e -> original RR-retention validator -> reconstruct f6d -> original owner validator -> existing cooperative/earlier namespace`.

The source of this identity is the **then-current last complete learned 72e checkpoint** and its explicit checkpoint/sidecar SHA256, checked against the branch's current immutable pointer. It is never a hard-coded CP226432 or older fallback. The probe-v2 CP225280 hashes in this module identify only reviewed training data, not the weights to resume.

## Public API at a future authorized, simulation-free boundary

After applying, testing, reviewing, and committing the exact runtime delta, construct the ordinary current runtime contract with the existing project API. Select the actual latest complete 72e source, preserve its original seed/device, and call:

```python
record = build_front_retention439_identity(
    source_checkpoint, current_contract, reason=explicit_reason,
    expected_source_sha256=selected_checkpoint_sha256,
    expected_manifest_sha256=selected_sidecar_sha256)
# Save record as a reviewed JSON plan, then:
receipt = publish_front_retention439_checkpoint(
    source_checkpoint, current_contract, plan_path, unique_immutable_output)
```

The existing full-state publication pattern preserves all actor/critic tensors including learned columns 422:439, Adam state, actual learning rate, Identity normalizers, RNG, all PPO counters, and historical receipts. It requires empty rollout storage, discards unfinished rollout data, independently reloads the output, and does not promote any latest pointer. Numerical identity is verified at publication; ordinary descendants are not frozen to source weights.

If a separately authorized output-side kernel produces an accepted finite fit, append only its accounting using:

```python
new_infos = append_front_retention439_event(
    loaded_infos, source_checkpoint=identity_or_learned_immutable_checkpoint,
    expected_source_sha256=selected_checkpoint_sha256,
    expected_manifest_sha256=selected_sidecar_sha256,
    data_receipt={"path": selected_data_receipt_path, "sha256": data_receipt_sha256},
    fit_report={"path": accepted_fit_report_path, "sha256": fit_report_sha256})
# The existing save_semantic_checkpoint writes and round-trip verifies runner/new_infos.
# Then independently official-load and evaluate; collect only a fresh PPO rollout.
```

The fit kernel and data admission remain output-side; this runtime module does not fit or deploy a teacher. Its ledger accepts only sparse first-layer columns `[1,4,5,8]` (P02/P05/P06/P09, 1024 scalars), bound executed-student raw targets before intervention (indices <=997), balanced contiguous disjoint train/holdout windows, and the reviewed probe file hashes. Every fit has separate accepted/attempted AUX totals and zero added PPO decisions/updates/optimizer steps. The revised receipt distinguishes **real P10–P12 full-Gaussian same-input checks** from **synthetic P13 checks and the zero-selected-column algebra**. It requires actual P13 rows=0 and forbids claiming synthetic P13 as real coverage or placing it in fit targets. Future trajectory invariance and physical success remain explicitly unclaimed. The raw target contract separately labels finite transformed physical actions rather than incorrectly equating them to raw Gaussian samples.

Source/file/branch/dataset/counter changes are rejected. At unchanged PPO counters the saved actor and protected state must exactly match either identity source (no AUX) or the last fit report (AUX). Later genuine PPO updates may change learned tensors and preserve the independent ledger. Finite-budget/zero-gradient/targets-already-match stops do not fabricate a rejected proposal; only the rejected-proposal stop claims rollback.

After this explicit publication, ordinary existing CLI training/evaluation resumes the new-runtime checkpoint without a new ResumeMigration flag, new runner, or new course wrapper. Existing stage/prefix authorization remains separate.

## Verification completed now

- AST parse of all four generated candidate files: PASS.
- `git apply --check candidate.patch`: PASS.
- `git diff --stat -- src tests configs`: empty at verification; production unchanged.
- `test_runtime_stdlib.py`: 9/9 PASS; executes only AST-selected receipt helpers and the dormant kernel's pure-stdlib report validator, explicitly checks no Torch/PXR/Isaac import. `test_stdlib_contract.py`: 11/11 PASS including raw-versus-physical units and available-real/synthetic phase scope.
- `candidate_manifest.json` records exact source and staged file hashes.

## Required numerical checks NOT RUN

Only after current simulation exits and explicit approval to apply:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = 'src;tests/unit'
# Optional explicit source must still be the selected latest complete 72e checkpoint.
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m pytest -q tests/unit/test_semantic_front_retention439.py
```

The CPU fixtures isolate GPU-bound historical receipt semantics only for numerical full-state/carry tests; real source metadata tests separately exercise the unchanged ancestral validators. Do not weaken a production receipt to make a CPU fixture pass. Additional fit-kernel numerical checks are listed in the artifact-owned `NUMERIC_TESTS_PENDING.md`. Those checks and any actual AUX execution remain pending.

The patch is a candidate, not evidence of physical improvement, RR capture, RL crossing, or a new verified PPO checkpoint.
