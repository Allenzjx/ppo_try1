# Staged actor448 v2 (not production yet)

Apply `actor448_v2.apply_patch.txt` only after Isaac exits at the complete
update boundary. This keeps one production actor implementation: explicit
`legacy447_migration_only=True` plus old layout is the constructor-only old
shape route. It can load the immutable old weights and perform deterministic
function comparisons; stochastic forward and production audit reject it.
The old model objects still have their original trainable-parameter structure
so actual Adam IDs/order can be restored before the448 structural migration.
The flag must never appear in a formal448 train/eval runner configuration.

Default actor mode is448/v2. Frozen prior remains439; appended observation
index447 is the boolean `current_attempt_capture_eligible`. The existing
indices439..446 and HISTORY are untouched. The new field does not mask
actions: eligibility0 afterGROUND leaves the latched active local module
free to recover. Current sensor TOP/bearing may be real while lift-attempt
eligibility is false; task logic, not actor geometry, decides valid success.

No mean scaling, sigma/LR changes, extra state machine, action projection or
extra HISTORY pass was added. Model loading and prior hash checks are shared
between legacy and v2. State-dict parameter names remain the same; only the
local first-input weight dimension changes.

`existing_actor_tests448.apply_patch.txt` adapts the existing actor tests'
synthetic observation fixture only. Add/execute the separate9 new cases in
`test_actor448_v2_cold.py` after integration. They cover explicit447 opt-in,
no legacy sampling/audit, eligibility observation validation, exact old
columns and learned heads, one HISTORY, probability, freeze, new-column
learning and actual save/reload.

Cold-boundary commands from repository root:

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' `
  outputs/ppo_rr_capture_first_cp225280_v1/staged_actor448_v2/test_actor448_v2_cold.py `
  --isaac-stopped
```

While Isaac runs, only `validate_staged_patch.py` may run with Python313;
it checks patch context and AST in memory, without executing/importing Torch
or writing production. Tensor tests are prepared, not yet passed.

Main route integration still owns: real448 schema/task production, old/new
runner construction, actual filtered Adam load, helper migration, preserved
effective LR/full RNG restore, versioned manifests, fresh rollout semantics,
and real reload/evaluation. The previous completed-block learning-signal
script loads historical447 checkpoints: after this patch it must explicitly
add `legacy447_migration_only=True` to that read-only constructor. It performs
deterministic forward only, so it remains compatible with the restricted path.
