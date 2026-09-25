# P13 evidence correction and raw/physical action admission

This revision removes an unnecessary physical-coverage requirement, not a
weight, likelihood, or trust-region protection. Existing actual P10/P11/P12
rows remain required and individually counted. Current actual P13 rows are
explicitly **zero**. Its coverage is synthetic same-input testing plus the
strict first-layer zero-column proof; neither a reached P13 state nor future
closed-loop success is claimed. Synthetic rows never enter fitting targets.

Data and kernel now accept actual invariance phase columns 9/10/11. The full
protected algebraic range remains P10–P13. Reports and the outer runtime
ledger use distinct real-P10–P12, synthetic-P13 and algebraic-invariance flags,
matrix hashes, and a per-phase actual count. The legacy ambiguous aggregate
flag is rejected; a claimed real P13 sample or synthetic fitting sample is
also rejected.

Raw admission keeps exact equality of original student raw, applied raw,
audit-selected raw and step raw. The separately transformed physical-unit
`step_info.applied_action_full12` must be finite but is never a raw label.
This fixes the observed failure admitting the sealed probe's legitimate
physical transform; it does not authorize raw substitution/intervention.

At root's request, each fit-attempt record now additionally includes observed
train/holdout cumulative maximum absolute per-channel REQUEST displacement,
each direction's maximum Gaussian KL, and maximum absolute log-sigma change.
No budget, optimizer, acceptance condition, data cutoff or sparse column
selection changed.

Verification by this subtask: 11 data/kernel stdlib tests + 9 runtime receipt
stdlib tests pass. Candidate AST and apply checks passed before the root
agent applied the three runtime paths. No model or fit was run by this
subtask. Root separately reports successful real-source read-only inspection
and is running official CPU tests after applying the reviewed candidate.

One additional synthetic CPU kernel test is isolated in
`additional_numeric_test.apply_patch`; it was emitted after root's initial
application, so it is a separate test-only follow-up. It perturbs only a
temporary selected first-layer leaf (no fit/optimizer step), requires exact
full Gaussian and same-raw log probability on synthetic P10–P13 inputs, and
tests nonzero-selected-phase and non-Identity-normalizer counterexamples.
Its standalone patch passed `git apply --check`; numerical execution is left
to root's single model-test process. No overlay or new runner was created.

The original complete candidate patch remains the recorded already-applied
artifact. Do not rerun its assembler against a tree containing those routing
changes; the source-anchor assertion intentionally fails safely. Future
changes, including this one test, use explicit small follow-up patches.
