# Optional finite RR mean-row AUX candidate

Prepared output-side only. No AUX optimizer or physical run was executed by this preparation.

The real COMPLETE `train_tracking_fixed512_1e10d39` block was checked with stdlib:
11 continuous rows, tick 8224 through 8312, parent CP228352 / local3072 / PPO6 /
original Adam120 / AUX0. Package digest:
`984b4d2e9c7781b1939bba45a3af689d32b0484fbda3048b9dd899452b169fed`.
These cover terminal approach from 15.7 mm plus first TOP and a 0.5 s hold;
they do not demonstrate deterministic recovery from the 59 mm activation entrance,
RL completion, or natural P01 full success. First TOP-producing action belongs
to AIR-before (4 rows); subsequent requests belong to TOP-before (7 rows).

## API and safeguards

1. `prepare_success_package(run, expected_runtime=..., parent_checkpoint=...,
   checkpoint_sha256=..., manifest_sha256=...)` requires the sealed result CP,
   hashes the full source, and extracts actual issued conditional raw and complete
   448 observations. It rejects an unsealed run. FINAL targets remain evidence,
   never labels. Parent runtime is the source's exact fixed-tracking 448 runtime.
2. `recipe(steps=8, learning_rate=.001, max_shift_sigma=.25)` returns a fixed
   recipe. Budget must be 1–64 actual steps. Defaults are proposals only; root
   seals the actual recipe at the cold boundary. No automatic retuning.
3. `fit_rr_mean_rows(runner, package, sealed_recipe, isaac_stopped=True)` only
   after strict load/rebind of the exact parent, with empty rollout. Temporary
   two-row Parameters use independent SGD, zero momentum and weight decay;
   original parameter objects and original PPO Adam are not replaced.

Only last-local-Linear mean rows 6/7 and biases may change. Loss uses the current
single-HISTORY conditional mean versus actual raw targets, sigma-scaled SmoothL1;
it does not regress an amplified inverse-HISTORY label. Prior, trunk, std rows,
other 10 mean rows, critic, Adam including LR and moments, and full RNG must
remain exact. Fixed reference observations include all 41 successful episode
states to bound RR mean shift; this is a trust check, not imitation of all 41.

Return packet is `{event, receipt, aux_optimizer_state, counts_after}`. Receipt
contains every candidate, accepted/rejected/failed attempts, exact protections,
and AIR-before/TOP-before RR mean/target/RMSE before and after. All actual AUX
optimizer steps count, including rejected steps; PPO decisions/updates/original
Adam credit remains zero. Nonfinite failure evidence is explicitly labelled and
JSON serializable, not dropped. Failures require retained receipts; do not publish
a failed-invariant actor. Helper neither publishes nor updates a live pointer.

The shared stdlib validator is `validate_local_auxiliary_events(events, counts)`.
It returns a deep-copied complete ledger with all extra receipt fields retained,
rejects duplicate event IDs and mismatched counts, and accepts absent ledger only
when AUX count is zero. Schema is `wlr50_clean.finite_rr_mean_row_aux_event.v1`.

Root owns strict metadata-only rebind, unique new-HEAD `_auxNNNNNN` checkpoint,
sealed receipt/AUX state, strict reload, fresh on-policy data and real evaluation.
An improved fit is not a success claim. An untested checkpoint stays untested;
a full deterministic rollout is not a mandatory gate before every 512-step update.

## Tests

`test_finite_rr_mean_aux_stdlib.py`: 5 tests PASS using Python 3.13, with no Torch
or PXR import. Actual sealed package preparation above also passed with stdlib.

`test_finite_rr_mean_aux_cpu.py --isaac-stopped`: 3 prepared, not run by this
agent. Synthetic CPU fixture tests only: two-row updates/protected-state equality,
rejected-step accounting, and an injected post-step failure retaining its count
and evidence. Root runs these sequentially only after Isaac exit. They do not
certify the real actor, real checkpoint publication, or physical success; helper's
same checks also run on the actual loaded actor when root explicitly adopts AUX.
