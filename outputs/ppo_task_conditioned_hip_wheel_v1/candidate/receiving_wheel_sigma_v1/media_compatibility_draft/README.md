# Receiving-wheel media compatibility draft (not deployed)

Status: **DRAFT / NOT VALIDATED / NOT DEPLOYED**. This directory is post-hoc
artifact plumbing only. It is not imported by training and is not an optimizer,
migration, render, or release prerequisite.

The compatibility boundary is deliberately narrow:

- source policy: `task_conditioned_hip_wheel_sigma_v1`;
- target policy: `task_conditioned_receiving_wheel_sigma_x3_v1`;
- exact factor schema: `wlr50_clean.receiving_wheel_sigma_same372.v1`;
- only P10-P12 with observation history index 157 equal to one;
- only raw action channels 9/11 (FR/RR wheels), innovation sigma multiplied by
  3.0; deterministic mean, HISTORY kernel, caps, rewards, task acceptance,
  physics, full12 actions, observation372, N1, Adam and branch origin unchanged;
- preserved real LIMITED AUX lineage remains 7 accepted / 8 attempted and is
  still labelled **PPO + LIMITED AUX**. The sigma profile adds zero AUX updates.

The candidate migration binds two new modules (`semantic_receiving_wheel_profile`
and `semantic_receiving_wheel_sigma`) plus exactly five integration modules:
`semantic_policy_distribution`, `semantic_training`, `semantic_migration`,
`semantic_cli`, and `semantic_checkpoint_prefix_policy`. It does not change the
nominal geometry, residual mapper, evaluator/supervisor, reward/task-quality,
backend/physics, or any of the six selected configuration files.

`receiving_wheel_provenance.py` refuses a version-only allowlist. Formal use
must re-run the deployed production `validate_migration_plan`, match its result
exactly to the persisted `receiving_wheel_sigma_migration` receipt, verify the
historical source checkpoint through the existing LIMITED AUX validator, and
verify the checkpoint's embedded `infos` against its sidecar. With the current
pre-deployment production runtime this fails closed by design.

The earlier quantity boundary is revalidated separately by
`historical_posthoc.py`. Formal use must explicitly supply
`--historical-runtime-root` naming a clean, detached, real git worktree at the
old plan's `target_git_commit`; the helper never creates or changes a worktree.
It hashes every file in the historical target runtime contract, launches the
checkpoint-pinned Python with `CUDA_VISIBLE_DEVICES=-1`, imports the old
production validator from that worktree, reconstructs the saved quantity
receipt exactly, and runs the sealed LIMITED AUX validator in the same isolated
process. It never claims that the current production HEAD has the old value.

`receiving_wheel_media.py` keeps the existing reviewed media functions and
hash pins. The old quantity-only comparison is re-run at the historical sigma
source (the quantity target), then the sigma boundary is reported separately.
It reports the nominal/mapper/evaluator/reward/physics control chain and N as
the same. It does not claim full runtime-hash equality, and it does not relabel
old B as the target sigma policy profile: the policy distribution's stochastic
kernel is the sole behavior-level difference. Deterministic and stochastic C
captures must use the exact same checkpoint; stochastic is always reported
separately.

The training-receipt wrapper changes the frozen post-hoc policy constant to the
single exact target version inside an AST-copied, hash-pinned summarizer. Thus
all actual learner `policy_request.policy_version` rows must be the target
profile; the old policy check is not removed. It then attaches the same formal
sigma/AUX provenance. This wrapper only reads an already sealed run.

No success or quality-improvement claim follows from this adapter. A formal
failure video remains a deliverable, including its complete tail. No actual
video source exists at the time of this draft, so no render has been attempted.
