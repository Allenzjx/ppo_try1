# N8 CLI integration review draft

Status: **docs-only proposal, not applied, imported, tested or simulated.** Written while the single-environment live barrier was active. It layers on `semantic_vector_implementation.patch`; it does not replace that draft's eleven CPU tests or its one-scene batched backend.

## Scope

- CLI and PowerShell gain `--num-envs` / `-NumEnvs` (only 1 or 8) and an explicit `--vector-smoke-evidence` / `-VectorSmokeEvidence`.
- N8 initially supports GPU, seed1001, preflight, functional smoke and training. A/B/C evaluation stays single-environment. N8 training always loads a real existing checkpoint; it cannot initialize a fresh actor instead.
- No observation/action/reward/nominal/task/physics configuration changes. Observation is **324**, action is 12. Frozen A remains unchanged.
- The budget label `phase_suffix` remains available, but checkpoint and result metadata explicitly report **P01-only synchronous reset; phase-suffix curriculum not implemented**. Fast full-reset clones are not credited as curriculum.

## Explicit reviewed execution factor and migration

The companion kernel patch adds exactly three runtime modules: semantic_vector_backend.py, semantic_vector_env.py and semantic_vector_training.py. This integration patch only changes the known CLI, training, migration and PowerShell plumbing, plus a dedicated test file. Native smoke and GPU helpers stay inside semantic_cli.py; pure execution/topology helpers stay inside semantic_migration.py.

No new hardcoded target source hashes or review placeholders are introduced. The reviewer approves the exact declared file delta, target Git commit and full runtime contract; the immutable migration plan records every before/after file hash and its own SHA is captured at use. Only the complete three-file vector addition is allowed. A topology switch cannot silently revise already existing vector modules. Observation/action/reward/supervisor/backend/physics remain protected. Historical d86 qualification plans remain unchanged.

Generate a new immutable migration plan after the runtime commit and a real N8 interface smoke:

```python
plan = build_migration_plan(
    source_checkpoint, current_contract,
    allowed_changed_files=explicit_reviewed_runtime_delta,
    reason="Reviewed additive semantic N8 execution; unchanged learned state and MDP",
    execution_evidence={"target_num_envs": 8, "vector_smoke": vector_smoke_manifest},
)
```

If the selected source checkpoint additionally requires the already reviewed geometry/prior/qualification factors, supply their existing explicit evidence as well. Prefer the completed current N1 checkpoint so the N8 boundary does not unnecessarily repeat an MDP change. Old migration plans and checkpoints are never overwritten.

The plan binds source checkpoint+sidecar, source/target runtime, exact declared source hashes, current-head live smoke and its native/physical/GPU artifacts. CLI validates before AppLauncher. The loader then verifies source topology, unchanged PPO configuration, actual actor/critic/Adam/normalizer hashes and restored RNG, and **fresh storage `(128, 8, 12)` with step=0 and no active transition**. No old rollout, physical state, success latch or old value/log-prob buffer is imported. All target rows start from legal P01 reset. Global actual decisions, optimizer updates and requested stage budgets continue from the source.

Every new checkpoint declares its topology. Only genuinely pre-vector semantic checkpoints may infer legacy N1 from the absence of all additive semantic vector files; a vector-capable checkpoint missing topology fails closed. Switching back to N1 requires another explicit `execution_evidence={"target_num_envs": 1, "vector_smoke": ...}` plan, even under the same HEAD. The retained actor and optimizer are loaded, not recreated as a new training lineage. Deterministic single-env evaluation of an N8 checkpoint uses this explicit topology boundary too and performs zero optimizer updates.

## Real functional smoke, not a task-success gate

The proposed smoke executes two explicit same-backend resets, each followed by at most 64 decisions per row (up to 128 per row / 1,024 total). It uses the actual N8 semantic kernel, actual mapper and native target auditing; no actor or optimizer is constructed.

1. First reset: all eight rows receive exact zero, establishing physical control traces.
2. Second reset: eight zero decisions, then row0-only wheel excitation through decision31, then distinct bounded twelve-channel patterns for every row through decision63. Raw magnitudes correspond to 5–12% of the unchanged engineering caps, not the old tiny residual probe.

The kernel must verify eight row-bound native audit records, every physics tick, all four zero state-write counters, and exactly one scene step/write/contact capture per tick. Unexcited rows must have zero residual native effect. Every row must show actual resolvable target effect in the distinct-pattern window.

Physical isolation is not inferred only from Python identities: measured local base position and actual joint/wheel states of the seven unexcited rows are compared to the first reset's zero-control traces while row0 is excited. Row0 must show actual physical response. Distinct origins at least 6m apart and actual scene collision filtering/replicated physics are checked too. Engineering interface-repeatability bounds are explicitly 0.01deg servo, 0.002rad/s wheel and 20µm base for unexcited rows. These are reviewable probe bounds, not task/reward gates; they must not be silently relaxed if real reset nondeterminism makes the evidence inconclusive.

A real task failure is recorded and ends that probe segment; it is never relabeled success. If the segment ends before enough common physical ticks or before all rows have resolvable actuation, the interface proof is **inconclusive**, not a perfect-full-episode requirement. Do not use an inconclusive proof to authorize vector training.

Example after review and runtime commit:

```powershell
./scripts/run_semantic_ppo.ps1 -Command smoke -ExpectedHead $reviewedHead -NumEnvs 8 -MaxDecisions 128 -Seed 1001
```

The proof is `vector_smoke_manifest.json` in that immutable run, accompanied by `vector_native_audit.jsonl`, `vector_physical_probe.jsonl`, `gpu_memory.jsonl` and the finalized `run_manifest.json`.

## GPU measurement

The draft samples before/after vector scene construction, after both explicit resets, after training reset, after actual checkpoint reload, after each complete rollout and official optimizer update, and after final verified checkpoint. No sampler advances physics or acts on a controller.

It records torch allocated/reserved and torch peak statistics, CUDA total/free/device-used memory (which includes non-torch consumers), and raw `nvidia-smi` device/process reports. WDDM `N/A` process usage remains explicitly unavailable, never zero. Peak-of-sampled device usage is labeled as sampled, not a proven continuous whole-process peak. CUDA OOM remains a failure; a 12GB device label does not prove capacity. Preserve failed-run GPU evidence before considering fewer environments.

## Exactly accounted 100k stage

| Segment | Envs | Requested | Actual | PPO updates |
|---|---:|---:|---:|---:|
| Main | 8 | 99,328 | 99,328 | 97 |
| Explicit migrated tail | 1 | 672 | 768 | 6 |
| Total | | 100,000 | 100,096 | 103 |

The 96 extra actual decisions are the declared N1 whole-rollout rounding overrun. They are not charged twice, concealed, or mislabeled as requested budget. N8 itself never silently rounds a tail. A first one-update N8 trial (1,024 requested) is part of the 99,328 main segment, not extra budget; the remaining main request is 98,304.

```powershell
# All checkpoint and plan paths below must be exact existing immutable artifacts.
./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead $reviewedHead -NumEnvs 8 -Stage phase_suffix -Decisions 99328 -Seed 1001 -Checkpoint $sourceN1 -ResumeMigration $toEightPlan -VectorSmokeEvidence $n8Smoke
# After a completed or safely stopped N8 main, generate an explicit 8-to-1 plan.
./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead $reviewedHead -NumEnvs 1 -Stage phase_suffix -Decisions 672 -Seed 1001 -Checkpoint $completedN8 -ResumeMigration $toOnePlan
```

If the main segment stops early, recompute remaining requested budget from its verified checkpoint, not these example totals. The existing graceful sentinel still stops after a complete update and a verified checkpoint: N8's boundary is 1,024 total decisions, N1's is 128. Peer truncations remain separately reported and do not count as physical failures.

## Required merge-time tests and limitations

- Apply both patches only after the current live barrier is cleared; parser/import and focused CPU tests first. No Python execution was performed while authoring this integration patch.
- Retain the existing real official RSL 1024-decision test, then extend it to actual N1 checkpoint→N8 update→N1 tail reload, verifying actor/critic/Adam/normalizer/RNG/global counters and no old storage reuse at each boundary.
- Test undeclared or changed source hashes, incomplete vector additions, changed protected files, stale/missing/forged smoke, wrong actual storage shapes, missing topology, and N8 non-multiple requests before AppLauncher.
- Test exact 99,328+672 budgeting and safe stop/resume without duplicating the first update; distinguish environment reward from peer bootstrap correction in saved rollouts/audit.
- Test shared native tensors/wrong row requests/altered inactive rows/missing counters/repeated-reset disagreement are rejected. The proposed physical isolation thresholds require live evidence; no CPU fixture claims to prove them.
- Run actual N8 functional smoke, then one real update and checkpoint roundtrip with measured GPU use, before scheduling the remaining main segment. No legacy perfect GateA/GateB/matrix acceptance is introduced.

The included new tests are a starter set, not a claim that the migration roundtrip and live isolation have already passed. Any merge-time corrections must be reviewed and committed; generate a new exact immutable migration plan before collecting fresh on-policy data. This proposal is based on d86; rebase migration hunks over the subsequent reviewed TaskEvaluator-only boundary rather than overwriting it.
