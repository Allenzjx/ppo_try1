# Joint task-state sigma / physical-quality migration candidate

Output-only; no production writes or Isaac execution by this subtask.

Implementation: `semantic_migration.py`, `semantic_cli.py`, and
`semantic_checkpoint_prefix_policy.py` in this candidate's `src/wlr50_clean/ppo/`.

- Explicit `task_conditioned_hip_wheel_factor`, schema
  `wlr50_clean.task_conditioned_hip_wheel_same372.v1`: FR physical-innovation
  policy / FL-quality experiment to task-conditioned policy and experiment.
  This is a joint reward/potential + sigma boundary, not old sigma-only logic.
- Source checkpoint and manifest hashes, actual target HEAD, exact changed-code
  review hashes, complete old/new six-configuration bindings. Four configs must
  be byte-identical; only the agreed exact reward and soft-retention fields may
  change. Protected evaluator, nominal provider, controller ASTs stay identical.
- All parameters and buffers, complete Adam (both groups and algorithm LR),
  Identity normalizers, RNG and lifetime counters are retained. No old partial
  transition/rollout is accepted. No physical trajectory equivalence is claimed.
- Explicit `archive_only_exact_bytes_factor` permits supported same-policy
  archive HEAD changes only when all runtime bytes and other contract fields
  are identical. It is not a generic no-op exemption.
- Preflight and frozen checkpoint prefix carry checked source/effective policy,
  runtime and plan provenance; prefix actions receive no optimizer credit.

CPU tests use the real installed RSL runner with synthetic observations/rewards.
No synthetic decisions are real training progress. The actual CP185856 manifest
is independently tested as 185856 decisions / 1417 updates / 28340 Adam steps.
The full synthetic old-policy -> true validator -> official new-policy load ->
frozen prefix -> update -> save -> exact reload chain preserves nondefault Adam
LR 2.3e-5, betas and eps, Identity/RNG and lifetime/branch counts.

The shared `candidate_bootstrap.py` removes import-order contamination: all 12
candidate runtime modules have candidate `__spec__.origin`/code filenames.
Five modules' `__file__` is a process-only production-config root anchor; the
pytest provenance output explicitly reports both. Genuine old reward,
supervisor, observation and fixture module objects are captured before the
overlay, so old/new physical comparisons are not self-comparisons.

`test_task_sigma_actor_integration.py` also verifies the new read-only official
minibatch head audit: identical 128-row storage, existing Adam and RNG produce
bit-identical actor/critic/Adam, algorithm LR, statistics and final RNG with
audit enabled or disabled. Exactly 20 actual head forwards in both branches;
head outputs, their true derivatives and postclip parameter norms match
independent read-only hooks and each minibatch's saved-observation Gaussian.

Evidence: `combined_candidate_tests.xml`, `task_sigma_with_head_hook_tests.xml`.
The synthetic target HEAD in migration tests is deliberately not an adoptable
real migration plan. After production integration, the parent must build and
validate a new plan against the actual committed HEAD and runtime bytes.
