# Numeric tests pending an explicit legal boundary

**Status update:** selected tests, the real zero-update inspection, and one
finite fit have since run; see [DEPLOYED_BOUNDARY.md](DEPLOYED_BOUNDARY.md).
This original checklist is not a claim that every listed test was executed.

These tests were specified but **not run**. They require an explicitly selected
immutable current student checkpoint, empty PPO rollout, source-device runner,
and separately bound available real P10/P11/P12 observations. No real P13
exists in the current evidence; its count must be zero, not manufactured.

1. Official-load the current 439 checkpoint, confirm the exact owner actor,
   `Identity` normalizers, 256/256 mean+log-sigma MLP, LR, Adam, RNG, counters,
   and empty rollout. No ancestor may be substituted for the current student.
2. Hash the sealed probe-v2 decisions file once, stream all 1368 rows through
   `data_contract.plan_probe_rows`, and assert selected row IDs are all `<=997`,
   balanced P02/P05/P06/P09 contiguous windows, with no intervention row.
3. Evaluate `distribution` against the official deterministic actor on every
   selected row. Mean and `rear_owner_effective_log_std` must match exactly;
   there must be no extra random draw or actor cache mutation.
4. Run `inspect` only. Actor, critic, PPO Adam/options/steps, LR, Identity
   normalizers and full RNG hashes must remain unchanged; record the sparse
   256x4 gradient and per-channel request response.
5. If and only if a fixed budget is separately approved, run `fit` once. Do not
   search learning rates. Check every accepted proposal against original-model
   REQUEST, bidirectional-KL and log-sigma limits. Inject a deliberately too-
   strict bound and verify the first rejected 256x4 leaf restores bitwise and
   stops without incrementing accepted credit.
6. Compare all actor state tensors before/after: only columns 1/4/5/8 of
   `mlp.0.weight` may differ. Critic, PPO Adam, LR, Identity normalizers and RNG
   must be bitwise/hash equal.
7. On separately bound real P10–P12 inputs, require bitwise equality of full
   conditional mean, effective log-sigma, sigma and requested residual. Test
   P13 separately with explicitly synthetic fixtures (replace only the phase
   one-hot in one row of each actual phase); never label these reached states.
   The normal CPU test `test_sparse_P10_P13_full_gaussian_and_raw_logp_synthetic_unit_only`
   perturbs a temporary selected leaf without fitting and checks both full
   Gaussian and same-raw likelihood. It also rejects a non-Identity normalizer
   and demonstrates that a selected nonzero phase is not invariant. None of
   these checks claims an unchanged future trajectory or actual P13 success.
8. A zero-accepted report produces no ledger event or checkpoint. A nonzero
   result remains finite AUX—not PPO, teacher deployment, or physical success—
   and still requires the independent lineage/publisher task to authorize save.
