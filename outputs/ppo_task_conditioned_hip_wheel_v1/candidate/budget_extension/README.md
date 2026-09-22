# Quantity-only continuation candidate (not deployed)

This directory is an isolated candidate, not a running policy or a physical test.
No production file, active configuration, checkpoint, controller or simulator is
changed by the candidate CPU tests. Synthetic samples carry **zero real PPO credit**.

Scope: the task-conditioned experiment's lifetime `full_episode` requested-decision
ceiling changes explicitly from 100000 to 131072. `smoke=10000` and
`phase_suffix=100000` remain unchanged. This is a quantity/scheduling declaration,
not a change to the task, policy, observations, reward, physics or a claim that the
user's task is complete at a particular sample count.

`STAGE_BUDGETS` remains byte-for-byte unchanged. The existing entropy assignment
continues to divide global decisions by `sum(STAGE_BUDGETS.values()) = 210000`.
It is not recomputed from the expanded quantity ceiling.

Four prospective deployment files only:

- `src/wlr50_clean/ppo/semantic_training.py`: one explicit experiment ceiling
  helper; contract-bound budget check; same-layout budget receipt load/persistence.
- `src/wlr50_clean/ppo/semantic_cli.py`: all remaining-budget checks and live/vector
  dispatch use the same ceiling; runtime contract binds the declaration in the
  execution profile; the first budget migration is natural-P01 N1 training only.
- `src/wlr50_clean/ppo/semantic_migration.py`: narrow quantity-only factor and
  source/target/AST/configuration validation (separately reviewed in this candidate).
- `configs/ppo_task_conditioned_hip_wheel_v1/execution_profile.yaml`: only
  `training_budgets.full_episode: 131072`.

No new-MDP warm start, network/Adam reset, optimizer or normalizer replacement,
RNG reset, branch reset, stage relabeling, or altered mean/sigma is authorized.
The old runtime hash cannot be silently accepted. A reviewed explicit migration
must bind the immutable source checkpoint and the actually committed target.
The plan adds no decisions or updates and loads into fresh rollout storage.
Immediate cross-runtime checkpoint-policy suffix entry is deliberately not added;
the first new natural-P01 checkpoint can subsequently use normal exact-resume,
evaluation and prefix provenance.

Run the new tests with `--confcutdir` to prevent the parent candidate's historical
`conftest.py` from substituting a different candidate implementation:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' -m pytest `
  --confcutdir=outputs/ppo_task_conditioned_hip_wheel_v1/candidate/budget_extension `
  outputs/ppo_task_conditioned_hip_wheel_v1/candidate/budget_extension/test_budget_quantity.py -q
```

Final isolated-candidate result: 53 tests passed (17 quantity/real-source read-only,
27 migration validators, 1 official CPU loader integration, 8 portable tests).
The quantity tests include unchanged entropy AST and exact
values for every integer global decision 0 through 260000, old-experiment cap
rejection, explicit task-cap acceptance without counter reset, malformed contract
rejection before sampling, and an official CPU synthetic update crossing 100000.
The production training/CLI/v3-continuation/experiment-path suites also passed
(53 tests) when process-locally importing the candidate.
Final same-process combined rerun: **106 passed in 92.23 seconds**, exit code 0.
All five prospective deployment file hashes match `SEALED_FILES.json`; production
`src`, `scripts`, `configs` and `tests` remained clean at final verification.

The migration suite binds actual saved CP190976 to a temporary future runtime,
then uses the real builder/validator and rejects unrelated control, reward,
configuration, entropy, counter and migration changes even when their review
byte hashes are recomputed. This synthetic future-HEAD plan is **not adoptable**.
After deployment, generate a new plan from the actual latest immutable checkpoint
and actual committed runtime contract; do not reuse a test plan or reset counters.

CP190464 was independently loaded only as CPU tensor data: checkpoint/actor/critic,
complete Adam, normalizer and embedded-info hashes matched its genuine manifest.
The original runner device remains `cuda:0`. The test did not pretend to exact-load
that real CUDA checkpoint into a CPU runner, create a device-relocation copy, or
consume training RNG. Its actual per-parameter Adam step is 5980; lifetime optimizer
steps are 29060. These distinct existing ledgers are preserved rather than forced
equal.

The official CPU integration uses a clearly synthetic task-policy source with
populated Adam state, LR 2.3e-5, betas (.87, .996), eps 2e-8. It runs the real plan
validator, preserves actor/critic/Adam/all LR/Identity normalizer/RNG/lifetime/stage/
branch state, proves empty rollout storage, performs one further 128-decision
official PPO update, saves, and exact-resumes. Partial/pending storage and missing
plan are rejected. First migration through evaluation, suffix/offset, N8 or prefix
is rejected. All synthetic optimizer activity remains zero physical training credit.

The only test proposed for production is
`tests/unit/test_semantic_training_quantity.py` (8 tests). It has no historical
checkpoint, outputs directory, bootstrap or old-HEAD dependency. Keep the other
tests here as isolated historical/candidate evidence.

To run all 53 isolated tests from the project root:

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$budgetCandidate = 'outputs/ppo_task_conditioned_hip_wheel_v1/candidate/budget_extension'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' -m pytest `
  --confcutdir=$budgetCandidate $budgetCandidate -q -o addopts=
```

No production adoption, commit, new Isaac run or genuine policy decision was
performed by this candidate work. The parent alone decides deployment at a safe
boundary and validates the genuine latest CUDA checkpoint through the existing
exact loader.
