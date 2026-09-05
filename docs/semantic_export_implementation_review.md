# Checkpoint-bound semantic training export proposal

Status: docs-only; not applied, imported, tested, or run. The live N8 barrier remained active throughout drafting. `semantic_export_implementation.patch` proposes one standard-library-only `tools/export_semantic_training.py` and two focused unittest cases. No Torch, Isaac, project runtime import, optimizer update, or source checkpoint rewrite is performed by the proposed exporter.

## Actual historical evidence inspected

The retained chain is initial → 128 → 1408 → 4352 → 4480. The 128 and 1408 sidecars have no `resume_ancestry`. For 1408, its real `run_manifest.started.json` resolves `arguments.checkpoint` to immutable step128. The original launcher passed `checkpoint_last.pt`; the exporter must **not resolve today's mutable last pointer** to reconstruct that historical launch. It checks the resolved immutable launch record, checkpoint/sidecar SHA, and first-update actor-before hash against the step128 actor hash. The 4352 and 4480 sidecars additionally contain explicit checkpoint/manifest SHA and budget ancestry.

The old 3e9 run contains global audit rows 129–1415 but its final logged update is PPO11/global1408/220 cumulative optimizer steps. Its explicit external-stop record binds checkpoint1408 and identifies seven uncredited decisions. The exporter selects only 129–1408 from this run, then selects 1409 onward from the d86 child run. It does not glob/concatenate all runs, count intermediate checkpoints twice, or retain old unoptimized overlapping rows.

At drafting, selected checkpoint4480 has 35 PPO updates, 700 optimizer steps, actor hash `e98d6930d1285950df6883a44d1dbd86157e0b3ba717cba7dd4c85394ef53a51`. The same explicit ancestry traversal supports subsequent N8 checkpoints; no latest-path inference or hardcoded 4480 limit is used.

## Output contract

Every export uses a new exclusive directory under `outputs/ppo_semantic_v2/metrics/training_exports/<version>`. Existing output directories fail without overwrite. Exceptions leave `export_failed.json`; any partial files in that directory are not delivery artifacts.

- `residual_and_projection_audit.csv`: exactly one row per checkpoint-proven optimized global policy decision, with source JSONL line/run/checkpoint/SHA, source HEAD and runtime/reward/task/supervisor hashes, raw PPO sample and old log probability/value, stored PPO reward/done, physical task terminal versus peer truncation, five reward families, actual per-environment physics ticks, raw/nominal/projected/applied/drive Full12, and last-tick native/counterfactual/delta Full12. Native servo values are radians; logical servo commands are degrees. Native wheel values are rad/s. Last-tick native values are not mislabeled as every-tick target tensors.
- `phase_coverage.csv`: P01–P13 including zero visitation, separately grouped by runtime/task/reward semantics. Time is summed per environment, not wall time. Training visitation and historical training successes are not deterministic current-head physical evaluation results.
- `training_manifest_summary.json`: exact optimized/requested counts and rounding difference, ancestry intervals, excluded audit-tail count, actual optimizer ledger diagnostics and actor hash chain, input checkpoint/sidecar/launcher/run/rollout hashes, and output CSV hashes. It does not independently deserialize model parameters or repeat the historical Torch save/load test; checkpoint byte integrity and upstream recorded optimizer/roundtrip evidence are identified separately.

Missing historical all-tick audit/write fields stay `unavailable`; they are never replaced by `true` or inferred from one valid last-tick audit. Only the documented historical single-env ABI permits interpreting PPO `done` as physical task terminal and peer truncation as false. Vector data requires explicit task/peer flags.

## After review and barrier clearance

Apply via `apply_patch`, then run the proposed focused standard-library tests and an actual 4480-prefix export before using a later selected checkpoint. This draft has only passed read-only `git apply --check`; it has not run Python. Root should inspect the real output totals and seven excluded rows. Do not run export against an actively growing source run; the exporter requires a finalized run/external stop and checks input stability.

Example (substitute exact paths; the version directory must not exist):

```powershell
& $python tools/export_semantic_training.py --project-root $project --checkpoint $immutableSelected --initial-checkpoint $immutableInitial --output-dir "$project/outputs/ppo_semantic_v2/metrics/training_exports/export_20260905_v1"
```

The helper path is outside the runtime inventory, but committing any new files still changes HEAD. Do not create a new training contract accidentally during a live run: apply/run only at root's approved boundary or materialize the reviewed helper as an external artifact without changing HEAD. Existing source/checkpoint/migration artifacts remain immutable.
