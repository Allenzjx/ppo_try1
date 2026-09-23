# v9 migration — installed and CPU-tested at the legal boundary

Update: root authorized installation after Isaac naturally ended. This agent installed only the new migration module/test and the four paths in `integration.patch`; the wheel was owned/applied separately by root. The three tests below ran with CUDA hidden and OMP/MKL1: **118 passed, 1 skipped**, exit0; 119 collected. The skip is unavailable Windows fixture symlink creation. `git diff --check` passed. CPU processes exited. No commit, real checkpoint publication, simulator or real PPO/AUX update was performed.

Installed module SHA256: `c695472d8b9fc42a37454d9a8994139a4af95355e94eba3e1f8dc757ad180cc6` (only staging docstring wording removed).
Installed test SHA256: `1e2a5b01f311408099982c49a849cc051464b25848c1c79feaf2878fac0652f6`.

The earlier preparation notes below describe scope and publication commands; their original "not run here" status is superseded by this dated boundary update.

Source is the **learned existing branch** checkpoint, not its initial ancestor:

- `branches/ancestor220544_signed_wheel_v8/checkpoints/history/checkpoint_step_000221952.pt`
- Checkpoint SHA256: `746c3abb9aa6bfced2c194a45b7c1c9db2cf543a25ab1c033108233bce1c685a`
- Manifest SHA256: `8e4caab3cc7a1bf66c0b1e41617413c693405367efa68f627a461d99f51a629e`
- Runtime: `d1871df37d6ea909657511d0e43e7435198f6ccd`; actual counts **221952 / 1699 / 33980**.
- Original branch origin stays **220544 / 1688 / 33760**; earned branch counts stay **1408 / 11 / 220**. Main-branch 640 decisions are not borrowed.

## Prepared files / installation scope

1. Add `semantic_rr_postcapture_wheel_migration.py` to `src/wlr50_clean/ppo/` from this staged copy.
2. Apply `integration.patch` from repository root: only CLI, training, generic migration routing and execution-profile revision.
3. Add `test_semantic_rr_postcapture_wheel_migration.py` to `tests/unit/` from this staged copy.
4. Primary agent's separately reviewed wheel candidate is the sixth runtime path; its tests are separate.

The exact six runtime paths are wheel, new migration module, semantic CLI/training/migration and this namespace's execution profile. No old migration validator, RR assist, context, backend, task spec, observation codec, actor, reward, sigma or physical limits may change in this factor.

Schema: `wlr50_clean.rr_postcapture_wheel_same410.v9`.
Factor: `rr_postcapture_wheel_v9_factor`; persistent receipt: `rr_postcapture_wheel_v9_migration`.
Exported builder/validator/recorder are `build_rr_postcapture_wheel_migration`, `validate_rr_postcapture_wheel_migration`, `record_loaded_rr_postcapture_wheel`.

The new factor records origin221952; the entire old v8 receipt still records its true zero-update origin220544. The branch's original `checkpoint_output_routing` and v5 source selection are unchanged. New v9 routing is checked before old receipts and fails closed rather than falling back. Both natural P01 and explicit checkpoint-policy suffix resumes use the same existing branch.

## Semantics and preservation

Same410 layout/codec and same-input policy distribution are preserved. X409's armed timing extends to explicit P12 post-stop support retention; this is declared changed observation/control semantics, not a same-MDP claim. No new mutable state or RL source cursor is introduced. P12 must prove its own adjacent committed authored wheel stop; its complete RL source endpoint is not required, and valid +0.3 nominal feedback is not mislabeled as a fresh source owner. Only currently bearing wheels are selected; actual RR TOP/bearing remains sensor-dependent.

All compatible actor/critic parameters, Adam moments/steps/groups, actual LR, Identity, full training RNG, existing branch origins and AUX ledgers are preserved by the established official loader/save/fresh-reload path. Old rollout is discarded. Migration adds zero policy decisions, PPO updates, optimizer steps and AUX updates.

The wheel's original numeric primitives/constants and explicit P12 proof strings are checked statically. Full wheel bytes are reviewed and hash-bound. Four P12-parameterized function bodies are **not** falsely declared byte-identical; existing P09 behavioral tests are the appropriate equivalence check.

## Focused legal-boundary checks (not run here)

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m pytest -q tests/unit/test_semantic_rr_postcapture_wheel_migration.py tests/unit/test_semantic_rr_signed_wheel_migration.py tests/unit/test_semantic_checkpoint_output_branch.py
```

New tests cover exact learned-source pins and preserved origins/receipts, extra-path/config rejection, P12 proof/defaults, malformed-v9-no-fallback routing, natural/suffix same-branch routes, official CPU full-state save/fresh-reload and ordinary-save receipt carry. Synthetic counters are not real training credit. The parent separately runs bounded old P09/new P12 wheel tests.

## Publication after reviewed commit only

`publish_rr_postcapture_wheel_v9.py --expected-head <actual full committed HEAD>` validates only. Add `--publish` only when authorized and no Isaac process is running. For actual publication, retain the source CUDA device and full GPU RNG visibility; do not inherit `CUDA_VISIBLE_DEVICES=-1` from CPU tests.

Unique output stays inside the original branch:
`checkpoints/history/checkpoint_rr_postcapture_wheel_v9_step_000221952_g<HEAD12>.pt`.

Source history, main and branch `checkpoint_last.pt`/`resume_state.json` remain unchanged. Future normal PPO saves advance only this branch. The target HEAD is required explicitly; no future commit/hash or physical result is pre-recorded.

Status: staged code/tests only; no production modifications by this agent, no tests, no checkpoint writes and no model updates performed for this task.
