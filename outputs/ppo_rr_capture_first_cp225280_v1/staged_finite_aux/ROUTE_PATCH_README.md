# Finite AUX route support — UNAPPLIED, no AUX authorization/execution

Files owned by this subtask:

- `route_finite_aux.patch`: minimal production-route patch text, **not applied**.
- `semantic_rr_capture_local.py`: full candidate for review; not imported as production.
- `test_route_finite_aux_stdlib.py`: 15 passing stdlib/AST/mock-state tests. They use the sibling helper's actual pure ledger validator, not a second implementation. Model/serializer operations are JSON mocks, not CPU tensor tests.

`git apply --check` passes against the current production route; no src/config/live file was changed, no commit made, no Torch/PXR/Isaac imported or launched. Cold tensor save/reload remains for the parent after an actual safe boundary and a decision whether to adopt this optional candidate.

## Exact scope

Ordinary `load` still requires identical runtime. Optional `local_auxiliary_events` is validated and deep-copied through load/save; all nested/extra event fields survive. Saved embedded metadata is checked on round trip. An old checkpoint without this optional field is supported only with local `auxiliary_updates=0`; inherited prior AUX lineage remains separate.

The shared helper API is `validate_local_auxiliary_events(events, counts)`. Schema is `wlr50_clean.finite_rr_mean_row_aux_event.v1`; actual independent optimizer steps are summed against `counts.auxiliary_updates`. These are **not necessarily Adam steps**, and include executed steps subsequently rejected by helper constraints. `accepted_steps` is separate. The route does not run AUX, manufacture events or increment any count. The parent-owned driver must append the helper-returned event and use its `counts_after` before ordinary save; 0-step failures do not create fake events.

Positive AUX counts add `_auxNNNNNN` before `_lineage448_v2_gHEAD` in the checkpoint name. AUX zero preserves the ordinary filename. Thus same PPO count at different actual AUX steps or publication HEAD does not collide. Existing target files still fail closed rather than overwrite. `iter`, `local_policy_decisions`, `local_ppo_updates` and `local_optimizer_steps` remain PPO-only.

Video `control_contributions` includes the complete local event ledger, `local_auxiliary_optimizer_steps`, and `AUX_updates_during_evaluation=0`. This is training provenance, not a deployed rear controller or an evaluation intervention. Consumers should retain these fields when making a new export; no old video is relabelled.

## Additional metadata-only cold rebind

Allowed origin is exactly `1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81`; destination must be a different valid committed HEAD. The only existing file allowed to differ is `src/wlr50_clean/ppo/semantic_rr_capture_local.py`; the only added runtime file is `src/wlr50_clean/ppo/semantic_rr_capture_local_aux.py`. No removals. Every cfg/task/actor/physics/library value and hash is otherwise identical, including `local_contract` and `selected_configuration`. Tests/docs remain outputs-only for this exact inventory restriction.

The helper candidate must be installed under that one permitted production filename **only if the parent adopts the candidate at the cold boundary**. The route patch alone intentionally depends on its shared validator and is not a complete deployable change.

The original 5f→1e three-file tracking repair validator remains intact and still passes its actual archived-contract test. The new branch dispatches only for the exact1e origin. The common rebind procedure continues to call ordinary `load(runner, path, old_runtime)` and checks complete-update/empty-rollout state, explicit checkpoint and sidecar SHA, actual actor/critic/Adam hashes, unchanged optimizer identity/parameter order/effective LR, Identity normalizers, full RNG and historical lineage. Rebind appends its own clearly named metadata-only receipt and produces zero PPO/AUX steps. The next standard save uses the new gHEAD suffix.

No CLI `aux` mode, training algorithm, control/reward/observation change, relaxed loader or automatic next run is introduced. `train` and `main` ASTs are unchanged.

## Safe preparation test

```powershell
& 'C:\Program Files\Python313\python.exe' outputs/ppo_rr_capture_first_cp225280_v1/staged_finite_aux/test_route_finite_aux_stdlib.py
git apply --check outputs/ppo_rr_capture_first_cp225280_v1/staged_finite_aux/route_finite_aux.patch
```

Candidate SHA-256: `776d54508224b132a7bf2a5bc764593c61752c65a71a8bd9350e6358df43142c`.
Patch SHA-256: `2d021f015fbbd979bfacc0941f281f3e7bf6877416b9cd2596369bc451eac1cb`.
