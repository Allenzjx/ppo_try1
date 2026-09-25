# Dormant front-retention439 candidate

**Superseded status:** the reviewed candidate was deployed at 65a9255 and one
finite AUX32 was executed. See [DEPLOYED_BOUNDARY.md](DEPLOYED_BOUNDARY.md) for
actual test/fit scope. The following describes the earlier preparation state.

Status: **outputs-only preparation; not executed**. No model was loaded, no
dataset was written, no SGD step ran, and no AUX checkpoint or ledger event
exists. This candidate is not an optimizer or video gate.

## Exact scope

- Source evidence is the sealed student-entry probe-v2 run. Its zero-based
  `decision_index` values `0..997` are pre-intervention deterministic student
  requests with `original_student_raw_full12 == applied_raw_full12`. Every row
  at index `998` or later is strictly excluded.
- Targets are the student's actually executed conditional raw requests. They
  are not a teacher, task-success label, PPO sample, advantage, nominal command,
  mapper output, FINAL actuator target, or proof the whole failed trajectory was
  successful.
- Eligible fit phases are only P02/P05/P06/P09. The only mutable tensor slice is
  `actor.mlp.0.weight[:,[1,4,5,8]]` (1024 scalars). P01/P03/P04/P07/P08 are
  explicitly not fit because their source coverage is sparse.
- P10–P12 use separately bound available real same-input observations and
  bitwise equality of mean, log-sigma, sigma and requested residual. There is
  no reached P13 row: its actual row count is explicitly zero. P13 instead
  has clearly labeled synthetic same-input unit checks plus the strict
  zero-selected-phase-column argument. No successful P13 run is required to
  prove this local sparse-layer invariant; no physical P13 coverage is claimed.
- The numeric kernel uses the exact
  `SemanticRearOwnerRecoveryHistoryMLPModel` and
  `rear_owner_effective_log_std`; it does not substitute the older 419 kernel.

`data_contract.py` is pure stdlib. It binds the sealed run manifest, decisions
JSONL, source checkpoint and sidecar, validates all 1368 zero-based decision
indices, and deterministically selects equal-sized contiguous train/holdout
windows from one actual contiguous event per target phase. It keeps vectors in
memory and emits only a compact receipt; it never writes a teaching dataset.
The four raw copies (original student, applied raw, audit-selected raw and
step raw) must agree exactly. `step_info.applied_action_full12` is a separate
physical-unit transformed request: it must be finite, but is never compared
to raw values or used as a raw target. Caps/HISTORY/mapping can legitimately
make it numerically different without constituting a raw override.

`front_retention439.py` lazily imports numeric dependencies only when an
explicit caller invokes `inspect` or `fit`. `fit` has no CLI, save, publisher,
or ledger code. A future authorized `fit` must bind the compact selected-row
receipt by exact `{path, sha256}`. It permits one reviewed constant learning rate and at most 32
independent temporary-SGD attempts. Trust checks remain cumulative against the
original model: per-channel REQUEST movement, bidirectional full-Gaussian KL,
absolute log-sigma movement, nonselected state, and exact same-input invariance
on real P10–P12 and separately synthetic P13. The evidence records both matrix
hashes, actual per-phase counts including P13=0, three synthetic-only P13 rows,
and no future-trajectory claim. Per-attempt reports retain actual cumulative
train/holdout per-channel REQUEST maxima, both KL maxima and max log-sigma
movement; these are audit values, not new acceptance gates.
If a proposal is rejected, that first rejected proposal is restored in full
and stops the run; a finite run may also exhaust its reviewed budget without a
rejection. Only the last accepted sparse leaf can be copied to the actor.

The optional lineage candidate owned by the sibling task uses a separate
top-level `front_retention439_auxiliary` ledger. It must record actual accepted
versus attempted AUX steps separately from PPO and must not change the existing
rear-policy, owner, reward-retention, or historical AUX records.

The sparse algebra requires the exact Identity-normalized actor and identical
full 439 input/HISTORY. Only first-layer columns `[1,4,5,8]` may change; all
biases, other weights and distribution parameters stay fixed. For any finite
P10–P13 same-input vector those columns are zero, hence `delta_W @ x = 0`.
The unchanged trunk and exact effective-log-sigma kernel then preserve the
full Gaussian, and therefore log probability of the same raw action. A
non-Identity normalizer, changed bias/log-sigma state, or different HISTORY
invalidates that proof. Synthetic P13 vectors are not reconstructed physical
robot states and never enter train/holdout targets.
