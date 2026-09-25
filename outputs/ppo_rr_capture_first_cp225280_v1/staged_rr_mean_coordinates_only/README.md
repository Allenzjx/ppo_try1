# RR mean coordinates — historical preparation, deployed at cold boundary

Status update2026-09-25: below is the retained preparation record. Root deployed
the reviewed five-file production change as0d0f8948999222f9b8710b0eb913a9419537be83
after0ff CP228864 DET sealed incomplete. Actual cold CUDA migration/save/fresh
load succeeded with zero new learning credit. See parent RECOVERY.md and run
migrate_rr_mean10_CP228864_0d0f894/coordinate_receipt.json. Fresh512 PPO is running;
no physical success or new update is established by the migration.

This is an isolated candidate. Production, current rollout, reward, sigma, AUX
recipe and saved weights are untouched. No Torch/PXR or physical run performed.

Actor patch: fixed constructor option `local_mean_coordinate_gain_full12`, default
twelve1, optional RR6/7=10 only. Only active local **mean** is scaled, before its
existing one HISTORY; std is not scaled. Audit `local_raw_mean_delta_full12` stays
the effective raw contribution, and a separate gain field identifies coordinates.
Legacy447 accepts identity only. No extra observation/action or state_dict buffer.

Cold API, after a strict complete-checkpoint load into the patched identity actor:

```python
receipt = migrate_rr_mean_coordinates(
    runner, reference_observations_full448,
    source_runtime_gain=IDENTITY, target_runtime_gain=RR10,
    isaac_stopped=True,
)
```

The helper keeps the existing model/Adam objects; final local mean W/b rows6/7
become /10; their true Adam `exp_avg` becomes *10, `exp_avg_sq` and any AMSGrad
`max_exp_avg_sq` become *100. Step, group options/LR, all other parameters/moments,
critic, frozen prior and RNG remain. Actual named Parameters join actual live
optimizer group order to saved IDs; no assumed IDs or new optimizer. It requires
empty rollout, no pending action/gradients and existing complete moments.

Reference input batch must include both active and inactive valid448 inputs.
An inactive gate counterfactual can check function equivalence but is not training
or physical evidence. The helper checks effective local contribution, raw mean,
same-action Gaussian log probability, exact std/other10/inactive outputs, protected
state, Adam expected transform and RNG. It neither samples nor runs PPO/AUX.
FAIL or exception means do not publish the unpublished runner; preserve the old
checkpoint and the failure evidence. PASS means numerical migration only.

Root integration remains required if this option is adopted:

- Freeze a new source/runtime version at a full update. Start from the latest
  compatible complete checkpoint, including AUX64 provenance, not an older CP.
- Pass the tuple through `make_runner` and serialize the same explicit twelve
  values in **both** saved `runner.local_configuration['actor']` and runtime
  `local_contract`. The helper sets the former and returns the required latter
  binding; it does not build a runtime or loosen strict loaders. Runtime and saved
  actor option must agree; missing is identity only on the declared old source.
- The production request/profile coordinate contract and source hash must identify
  the change. Existing 448 observation and12 raw-action units need no alteration.
- Publish a unique checkpoint, strict reload with gain10 configured, validate
  same-input raw mean/std/logp, preserve exact prior and historical AUX ledger64,
  migration credit0. Collect fresh compatible PPO; no reuse of unfinished rollout.

This is deliberately **not optimizer-dynamics equivalent**: mean-row functional
steps are roughly10x under Adam, subject to epsilon, historical moments, actor
global gradient clipping, shared trunk and adaptive scalar LR. Those can change
other actor updates and std in later PPO. RR gain does not fix AIR/TOP sign conflict,
contact retention or wheel-induced retreat. Existing AUX used SGD; do not reuse
its old step/LR recipe unchanged (function-space SGD step could scale100x).

Tests: stdlib suite validates scalar math/config/guards and applies actor patch
in memory for AST validation. CPU synthetic tests are prepared but unexecuted;
run `test_rr_mean_coordinates_cpu.py --isaac-stopped` only after explicit exit.
Neither suite proves physical success. Prefer next saved/reloaded DET evidence
before deciding whether this optional change is necessary.
