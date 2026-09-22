# RR receiver retirement v2 — pending production patch

Base: 5fd88852bf20c94cd74405c791a13a9fd9e0a3d8.
Patch SHA256: 12d51ed035e71d1d51d80a4701a602d69b87cc17e8a86f09b82b08a2a2180599.
No production edits, tests, optimizer/fit, or Isaac execution by this preparation.

8 targets: 5 runtime/config files, a minimal old v1 regression-fixture adjustment, and 2 new unit tests. The old test now verifies adopted v2 config and explicitly runs its existing v1 fixture; no old v1 assertions were removed.

Static checks passed: exact complete-line in-memory application, Python AST parsing, unchanged TaskEvaluator/NominalMotionProvider and unrelated supervisor AST, training AST differing only in its carry-key tuple, task YAML differing only in its versioned mode. See rr_receiver_v2_patch_static.json. The helper is read-only and assumes this unmodified base; do not run it after adopting the patch as if it were a post-adoption test.

## Metadata

- schema: wlr50_clean.rr_postcross_workspace_same389.v2
- factor key: rr_postcross_workspace_factor
- mode: established_RR_over_top_receiver_retirement_v2
- new receipt: rr_receiver_retirement_v2_migration
- new branch: rr_receiver_retirement_v2_branch
- derived counts: rr_receiver_retirement_v2_branch_counts

The existing rr_postcross_workspace_branch/migration and every other branch remain whole and hash-preserved, including nested front_rehearsal_auxiliary future event4 and original auxiliary_mean_learning 7/8. No source checkpoint or future AUX counts have been invented or pinned here. Source must be the actual saved v1 checkpoint after the authorized AUX step, selected by root.

v1 factor validation remains strict. v2 accepts only v1→v2 mode, same389/N1/policy contract, explicit reviewed runtime hashes, other five configs byte-identical, full original actor/critic/Adam/effective LR/Identity/RNG/counters and fresh storage. Same-physical-state action equivalence is not claimed because Phi at observation index17 changes.

## Post-adoption tests for root

Run only after source-bound AUX is saved and root adopts the patch. Set CUDA_VISIBLE_DEVICES=-1 and PYTHONPATH=src; use the existing env_isaaclab Python:

python -m pytest tests/test_semantic_rr_workspace_retirement.py tests/unit/test_semantic_rr_workspace_migration.py tests/unit/test_semantic_rr_receiver_retirement_v2.py tests/unit/test_semantic_rr_receiver_v2_migration.py -q

The new integration test uses a temporary git/byte fixture and both actual migration validators, plus the actual official publisher/load/save and normal-training carry whitelist. It populates synthetic Adam moments without a forward/loss/fit; counter-only later-save propagation is explicitly not PPO learning.

Missing-evidence negatives concern this new gate's finite scalar fields and required flags. Missing top-level trusted evaluator/history/current_legs schema objects retain existing schema-error behavior. CurrentQ, placement, nominal/control, capture reward, caps and sigma are unchanged. This v2 has no effect on the reported 389 postcross hover inputs whose currentQ was already true.
