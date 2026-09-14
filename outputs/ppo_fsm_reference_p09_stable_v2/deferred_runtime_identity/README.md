# DEFERRED ONLY — actual live import identity / exact HISTORY372 resume

Status: **not applied to production; no tests run; no Torch/Isaac imported by this drafting work.**
Source baseline: commit 28609010db4e57c5b34304a4ae2563c69f9d00b9.
Do not place this directory on PYTHONPATH or execute these mirrored CLI files.

## Scope and immediate decision

The ongoing fixed-semantics 512 P01 + 1024 P06 training and natural P01 evaluation take priority.
This work is not an optimizer-start, evaluation, video, or task-success gate.
It does not retroactively claim the current live process's module paths.

Existing evidence establishes the launched Python executable, committed runtime source/hash,
the live nominal provider's source FsmSpec path, and reset controller hash.
The recorded offline probe's class files/sys.path are explicitly pre-revision offline evidence.
They do **not** prove every imported class __file__ or sys.path in the currently running Kit process.

The existing exact-resume path has three fixed-324 assumptions: migration plan dimension,
load-time storage check/config construction, and CLI preflight config construction.
Ordinary same-runtime372 resume already works. A logger-only runtime revision must use a
reviewed **exact resume**, not new-MDP warm start: no Adam reset, no network remapping.

## Copies and limits

Three source mirrors contain a proposed minimal diff:

- semantic_cli.py: record live already-reset backend/controller/provider/source motion,
  adapter/mature mapper, reader/contact/geometry, loaded FsmSpec path, active recovery
  (deliberate None for new semantic N), class module files/MRO, sys.path/executable/cwd.
  Logging performs no new imports, sensor reads, simulation steps, control changes or resets.
  Evaluation records immediately after reset. N1 training records after actual reset and
  checkpoint/prefix installation, immediately before credited training.
  Already-imported frozen reference classes are explicitly separate from active objects.
  Slot-based AuthoritativeFrame is read through its existing stored fields.
  A per-run live_runtime_identity.json is opened exclusively; evidence is never overwritten.
  Synthetic/incomplete backend objects are not represented as initialized live proof.
  This does not add instrumentation to the separate N8 or original A dispatch path.

- semantic_migration.py: additional same-observation factor only for verified v3 N1
  HISTORY policy, explicit role372 layout, canonical complete policy metadata, valid
  continuation topology, exact six selected configuration path/hash bindings, unchanged
  configuration/schema bytes, and validated target schema. It permits only the existing
  instrumentation allowlist, without mixed task/execution/video factors or file deletion.
  Existing324 plan shape stays unchanged. Scope equivalence is a reviewed assertion,
  not a theorem that arbitrary edits to whitelisted modules are non-behavioral.

- semantic_training.py: validate actual policy/layout/storage against that revalidated
  exact factor; pass the explicit layout into runner configuration comparison.
  Preserve the existing actor/critic/Adam/normalizer hashes, adaptive optimizer learning
  rate, RNG restore, counters and completed-block accounting. Reject partial rollout
  storage or a pending transition. No policy/value first-layer mapping is introduced.

No action, reward, phase supervision, nominal motion, physics, residual range, prefix code,
normalizer preprocessing, checkpoint policy, source FSM, or historical data was edited.

## Future boundary procedure

1. Let the currently authorized training block and natural P01 evaluation complete.
2. Resolve the actually latest immutable completed checkpoint and manifest; use their
   actual counters, not the older 10,112/44 or any prewritten checkpoint path.
3. Review source delta against the then-current production HEAD. Apply only reviewed
   hunks from the three mirrors (do not blindly replace files if production advanced).
   Reconcile independently deferred video code separately; this plan cannot mix it.
4. With Isaac stopped, run the deferred test draft after reviewed production application.
   It imports the applied package, not these mirrors. Isolate CPU and set
   CUDA_VISIBLE_DEVICES to the empty string for its nonempty-Adam test. Also run existing
   migration/policy/layout/return/CLI regression tests. There is no pass receipt yet.
5. Commit the instrumentation-only production revision. Build and persist a migration
   plan using build_migration_plan for the exact latest checkpoint/current runtime,
   exact changed-file list and reviewed reason. Do not use --new-mdp-warm-start.
6. Resume with --resume-migration, preserving seed and all learning contracts. The new
   process logs its own actual identity only after its own reset. Check the identity
   artifact alongside load-time state hash/RNG/counter verification.
7. Continue learning at the current curriculum choice. A fresh legal P01 physical reset
   and empty new rollout are required; prior incomplete rollout data is not reused.
   A reset-only P01-to-prefix roll-in is not policy credit or full-task success.

## Deferred tests

tests/unit/test_runtime_identity_exact372_deferred.py includes unrun tests for:
exact372 binding and round-trip plan validation; missing/mixed policy metadata, topology,
selected configuration and schema negatives; protected and mixed-factor negatives;
CLI preflight372; actual-object/MRO/slots identity with control-call sentinels and
exclusive artifact creation; a real CPU128-decision PPO update producing nonempty Adam,
exact reload with actor/critic/Adam/normalizer/adaptive-LR/RNG/counter equality, deterministic
history-conditioned action equality, and rejection of partial rollout reuse.

This is a bounded draft, not a production patch application, test pass, live identity
capture, training continuation result, or PPO task-success claim.

