# Proposed input-only migration, not applied

`draft_migrate_local447_to448.py` is an output-side draft, not production. No
live model, checkpoint, optimizer or configuration was modified. Its tensor
tests are prepared but **not executed while Isaac is running**.

New schema: unchanged439 frozen-prior input + unchanged8 local features +
one explicitly observed current-attempt lift-established feature =448.
The observation producer must use the real current-attempt eligibility field;
TOP can invalidate `current_lift_valid` while `lift_established` remains true.
Historical placed/cross is not a substitute for a fresh valid attempt after
GROUND. This draft does not implement or validate those physical semantics.

Cold-boundary integration:

1. Save/reload the complete447 update and retain its sidecar, runtime, RNG and
   empty-rollout evidence. Construct old447 objects with their real old actor,
   critic and optimizer order; load that original checkpoint normally.
2. Construct unpublished new448 actor/critic and an empty new Adam using the
   same qualified trainable-parameter order. The frozen prior remains439 and
   excluded. Do not deploy those constructor-initialized objects.
3. Call `migrate_local447_to448(..., isaac_stopped=True)`. It expands only each
   local actor / independent critic's first input Linear weight with a zero
   column. All other tensors, including learned mean/std heads, remain exact.
4. The helper joins real `Parameter` objects via qualified `named_parameters`
   with each optimizer's actual serialized group IDs. It refuses unknown,
   shared, reordered or frozen members. Adam `exp_avg`, `exp_avg_sq` (and
   optional AMSGrad `max_exp_avg_sq`) get zero columns only for those two
   weights. Steps, all other moments, group LR and options are retained.
5. Restore the saved full Python/NumPy/Torch CPU/CUDA RNG **after** constructing
   and migrating new objects. The helper itself consumes no RNG; constructor
   RNG consumption must not be called preserved continuation. Preserve the
   existing RNG record and effective algorithm LR as well as optimizer LR.
6. Compare original447 output against new448 for both new-feature values and
   identical old features/history. Zero new columns should initially preserve
   actor and critic functions within the declared CPU/GPU arithmetic tolerance.
   Verify frozen prior and Adam columns exactly. Save and actually reload the
   new package with a versioned448 manifest, then collect fresh448 rollouts.

No optimizer IDs should be invented from a 0..N assumption. No learned head or
normalizer is reset. Do not mix old447 rollout observations or task semantics
into new448 PPO storage. This migration is not permission to retain incorrect
GROUND eligibility or to declare TOP bearing without sensors.

After explicit confirmation that Isaac has exited:

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' `
  outputs/ppo_rr_capture_first_cp225280_v1/test_draft_migrate_local447_to448.py `
  --isaac-stopped
```

These synthetic tensor tests cover old-column identity, new-column zeroing,
learned-head/prior retention, Adam step/moments/LR, no RNG consumption within
the helper, real optimizer-order rejection, and a subsequent trainable new
column. They do not replace a real checkpoint migration/save/reload or physics.
