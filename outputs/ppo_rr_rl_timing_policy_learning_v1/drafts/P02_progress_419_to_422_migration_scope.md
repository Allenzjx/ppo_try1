# P02 progress state: bounded 419 → 422 migration plan

Planning only: no production edits, model load, optimization, publication or
tests were performed. The paused front-retention AUX candidate remains unused.

## Source and unchanged scope

Use the **actual latest sealed learned checkpoint** in
`branches/ancestor220544_recapture_v2`, after the current course reaches a full
128-decision update boundary. A stop at the next complete update is requested;
neither `221696` nor an unconsumed course target is a registered source or earned
count. Bind the actual checkpoint SHA, sidecar SHA, source runtime,
all three global counters and current branch counters only after sealing.
Keep branch origin `220544 / 1688 / 33760`, the branch route, main pointer, all
old checkpoints, append/recapture/live-swing receipts and AUX ledgers intact.
No return to the initial ancestor and no borrowing of main-branch credit.

The new P02 recovery flag is independent of rear mode
`rr_live_swing_evidence_v3`. Rear timing, current-RL qualification, assist-OFF
rules, raw12 action order/caps, HISTORY, sigma mapping, rewards, physical
acceptance and limits remain unchanged. The reviewed handoff candidate proposes
the ordered fields `p02_best_remaining_m` (metres, scale 1) and
`p02_progress_credit_fraction` (credit/capacity, range 0–1), both zero outside
P02, with opt-in `p02_measured_progress_credit_v1`. Reset/update rules and final
names still require root approval; do not silently default missing state.
Root approved a third ordered `current_eligible` boolean (scale 1) because the
exact AIR/current-lift/verified-support permission must not be hidden. Thus the
candidate dimension is 422. Final field names and eligibility semantics remain
subject to the reviewed handoff design; no implementation is authorized yet.

## Reuse the existing append mechanism, not its old source pins

`semantic_rear_policy_timing_migration.py` already provides the required pattern:
`zero_append_rear_policy_training_state`, source-device official load, mapped
state installation, RNG restoration, official save and independent fresh load.
Its implementation is deliberately exact `410 → 419` and pins the historical
ancestor; leave that migration strict. Add one small dedicated `419 → 422`
factor/loader using the same mechanics, with exact current source validation.

- Actor **and critic** first weights: `[256,419] → [256,422]`, old columns exact,
  new three columns zero. All other parameter/buffer values are copied exactly.
- Verify the official actor-then-critic parameter ordering before mapping Adam.
  Extend only the first-weight `exp_avg` / `exp_avg_sq` columns for parameter
  IDs 0 and 6; preserve every old moment, step, option and all other states.
- Preserve effective LR (currently `1e-5`, verify the sealed value), Identity
  normalizers, runner iteration, seed and full Python/NumPy/Torch/CUDA RNG.
  Storage starts empty; no inherited unfinished rollout, new sample or update.
- A **new** actor/profile/layout contract accepts 422 inputs. Keep the existing
  419 codec prefix numerically identical. Its full MLP may learn the new three
  columns; HISTORY uses the existing prefix and sigma calls the unchanged
  419 kernel with `observation[..., :419]`. No hidden timers or sigma change.
  With zero new weights, the Gaussian and critic function are mathematically
  unchanged at matching old inputs; test floating-point tolerance explicitly
  rather than promise bitwise output equality across a changed GEMM width.

## Minimal integration and metadata

Register the new profile/codec and a narrow append factor in the existing
generic migration, training loader and CLI paths. Update policy-request audit
and checkpoint-policy-prefix dispatch to use that same 422 profile while
retaining raw Gaussian sample/logp semantics. Add the new receipt to ordinary
save/carry and namespace validation **before** the historical 419 validators;
validate the preserved old chain against its reconstructed original contract,
not by pretending a 422 policy is an old 419 policy. Keep the same branch route.

The factor records actual source/target contracts, reviewed runtime/config
delta, exact three-state schema and mappings, source counters, preserved lineage
digests, `same_mdp_claimed=false`, `old_rollout_inherited=false`, and zero added
PPO/Adam/AUX credit. Do not replace `rear_policy_timing_branch` or its origin.
Publish one unique append-named checkpoint in branch history (never overwrite
the generic step checkpoint or promote either pointer during migration).
Following real PPO saves resume normal same-branch pointer behavior.

## Bounded checks after implementation is authorized

1. Populated-Adam synthetic fixture: both first weights and moments are exact
   zero extensions; all other state, counters, receipts, LR and full RNG survive.
2. Old419/new422 same-prefix Gaussian and critic comparison, nonzero appended
   state, shared sampling/audit sigma, and strict missing/invalid-feature tests.
3. Official source-device save + independent reload, empty rollout, unchanged
   main/branch pointers, and one normal save carrying the new receipt.
4. Reject wrong source/sidecar/runtime, wrong dimension/parameter ordering,
   stale/foreign branch and existing output. Existing P02 tests must prove the
   progress rule itself; migration tests do not claim physical task success.

No new scheduler or publication framework is needed. Final field names,
implementation path list, source SHA/counters and target commit remain unbound
until the real checkpoint is sealed and the narrow control design is approved.
