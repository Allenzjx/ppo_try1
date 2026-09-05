# Fixed P09/P10 prefix entry and separate video migration proposal

Docs only: not applied, imported, tested, or physically run. The existing N8 training run is untouched. This integration layers on the separately drafted `semantic_prefix_implementation.patch`; it does not certify that draft's real backend/core/RSL credit seam or P09/P10 physical availability. The video implementation remains a separate additive proposal.

## Minimal entry

Add only `--prefix-target {P09,P10}` / `-PrefixTarget`. Absence preserves P01 training. First suffix training is N1, `stage=phase_suffix`, and requires an explicit saved checkpoint; it cannot initialize replacement weights. No mixture probabilities, nonterminal window cutoff, new reward, or additional decision budget is introduced.

The fixed request uses `PrefixRequest(target, maximum_prefix_decisions=1800, maximum_takeover_decisions=30)`. The 1800 limit bounds the **whole initialization loop**, including takeover; the 30-decision takeover cap is within that limit, not an extra 30. Earlier task/stage termination still wins. One valid prefix miss produces one fresh P01 fallback, not an infinite retry or a later-stage availability claim.

`dispatch_prefix` creates the explicit prefix backend and one SemanticEpisodeEnv, opens `reset_prefix_audit.jsonl`, then constructs PrefixRslAdapter. The adapter's constructor/reset performs teacher initialization without constructing a policy or calling `alg.act`. It retains the same physical scene, evaluator, 200-second task clock, mapper, observation history and potential at the credit boundary. Only after this does the training entry construct fresh official storage and reload actor/critic/Adam/normalizer/RNG from the selected immutable checkpoint. The existing global/requested budget accounting counts only the following current-policy steps.

The prefix smoke entry uses 8–128 post-prefix manual probe decisions and no optimizer. It records native effect/audits, initial actual credit state, attempts and fallback. It does not require whole-task success. Full A/B/C evaluation and video reject `--prefix-target`: they always begin from natural P01 without teacher assistance.

```powershell
# After review, implementation tests, approved commit, and explicit migration plan:
./scripts/run_semantic_ppo.ps1 -Command smoke -ExpectedHead $prefixHead -NumEnvs 1 -PrefixTarget P09 -MaxDecisions 8 -Seed 1001
./scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead $prefixHead -NumEnvs 1 -PrefixTarget P09 -Stage phase_suffix -Decisions 128 -Checkpoint $completedN1Smoke -ResumeMigration $prefixPlan -Seed 1001
```

The first physical probe may report `actual_suffix_available=false` and a P01 fallback; do not relabel that as P09 coverage. Prefix time/storage overhead must be reported. The inherited adapter automatically primes a new episode when a credited terminal is returned, including at a rollout's last decision; that extra bounded initialization is excluded credit but can delay the optimizer/stop boundary. Measure it before scheduling large suffix batches.

## Explicit reset-distribution boundary

```python
build_migration_plan(checkpoint, current_contract,
    allowed_changed_files=exact_reviewed_delta,
    reason="Reviewed fixed P09 reset-only teacher initialization; unchanged learned state",
    reset_review={"reason": "Reviewed continuous physical prefix and credit seam", "target_phase": "P09"})
```

The factor is separate from execution-topology, evaluator/geometry/prior and video factors. Finish the N8→N1 tail first; prefix reset-distribution changes do not silently change env count. The saved reset-distribution record contains the exact request, fallback, no-credit rule and original deadline. Later P09→P10 or suffix→P01 training changes require their own explicit distribution plan even if HEAD is unchanged. Deterministic evaluation of the weights does not resume the training reset sampler.

The prefix companion is **not merely an additive file**: it adds two public DI methods in semantic_supervisor.py and optional default-None constructor injection. The proposed narrow guard requires evaluator/task-supervisor prefix bytes unchanged; removes only `NominalMotionProvider.from_handoff` and `SemanticControllerAdapter.from_live_prefix`, normalizes exactly those two default-construction assignments/options, then requires the entire module AST equal to its predecessor. Existing nominal scheduling, ordinary controller behavior and evaluator logic cannot change through this factor. The actual added methods/module remain bound by the reviewed exact file delta, target HEAD/full contract, source/target hashes and immutable plan SHA, with no new hardcoded class hash, test certificate, or success gate.

Current topology v1 literally declares P01-only. The proposal preserves v1 reads and introduces a precise N1 fixed-prefix v2 record rather than saving contradictory P01 metadata. `phase_suffix_curriculum_implemented` is accompanied by `one_fixed_target_not_mixture`; actual achieved starts/coverage remain in prefix telemetry. It does not claim the user's full mixture has been implemented.

## Separate video instrumentation publication

```python
build_migration_plan(checkpoint, video_contract,
    allowed_changed_files=exact_reviewed_video_delta,
    reason="Add separately reviewed A/B/C capture only",
    video_review={"reason": "Reviewed additive capture CLI, recorder and wrapper; no training behavior change"})
```

This permits only the first complete addition of semantic_video.py, semantic_video_cli.py and scripts/run_semantic_video.ps1, plus already known instrumentation plumbing. It rejects any supervisor, nominal, reward/action/observation/backend/config change. It records actual file hashes and the existing complete contract/plan binding; it does not certify video success or improvement and introduces no new fixed source hash. Do not combine prefix and video publication into one migration boundary. Publish video against a later verified checkpoint already bound to the preceding prefix runtime, or keep video first as its own boundary; no extra physical-success gate is needed.

## Minimum checks before use

1. Companion prefix DI/body review plus actual backend/core/RSL credit-seam test: prefix calls never enter storage, no duplicate tick/adoption write, first PPO sample uses reloaded current actor, and ordinary constructor/nominal/evaluator remain unchanged.
2. Real saved N1 checkpoint → fixed prefix update → saved reload: actor/Adam/normalizer/RNG/budget ancestry and fresh `(128,1,12)` storage; prefix clock/return/coverage excluded from credit while physical deadline remains 200 seconds.
3. Pre-AppLauncher rejection of undeclared/reset-mismatched plans, unsupported N8 prefix, wrong stage, mixed migration factors, altered ordinary nominal/evaluator code, and incomplete video additions. Preserve all historical evaluator/execution plans byte-for-byte.
4. One fresh real P09/P10 initialization probe and one real update. Report actual reachability/fallback and overhead. Neither these docs nor prior synthetic tests establish availability, robustness, full-stage coverage or successful video capture.

The existing export helper already passed the actual 4480-prefix export; this proposal neither reimplements nor reruns it. Later export extension may read the explicitly separate prefix telemetry, never merge `reset_prefix_audit.jsonl` into optimized decisions.
