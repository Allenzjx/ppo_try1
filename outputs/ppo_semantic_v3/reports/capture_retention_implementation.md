# Current platform-region capture retention — implementation and actual initialization

Status: committed and CPU-tested; real natural-P01 training launched and its immutable new-MDP initial checkpoint verified. **This is an initialization snapshot, not a completed training block or a task-success result.** No subsequent optimizer samples or planned endpoint are counted here.

## Committed change and attribution boundary

Commit `68631e932c7deb08a7a3f2a2787b79fa7eb569ef` changes exactly three files: `src/wlr50_clean/ppo/semantic_supervisor.py`, `configs/ppo_semantic_v3/stage_task_spec.yaml`, and the new `tests/unit/test_semantic_capture_retention.py`. The two production changes opt v3 into `capture_retention_semantics: current_platform_region_after_placement`; the new test file contributes55 parameterized cases. The final combined JUnit `C:/robotics_sim/wlr_robot/semantic_capture_retention_all_cpu.xml` records **896 tests,0 failures,0 errors,0 skipped,85.719s**. These CPU checks are not physical training or evaluation episodes.

Previously, a leg with historical placement contributed1 to its leg-progress component indefinitely. The opt-in reuses only the existing capture share: a historically placed leg now contributes `0.8 + 0.2 * retention`. Let `outside` be Euclidean XY distance to the same tolerance-expanded measured platform rectangle, `gap = max(0, existing_top_gap_min - current_clearance)`, and `scale` the existing positive minimum lift gain. Then `retention = min(clip(1 - outside / 0.25), scale / (scale + gap))`. The0.25m decay length is the existing workspace/carry scale; overall0.85 leg-progress /0.15 finish weighting is unchanged.

AIR above the platform remains fully eligible for this credit: no current contact, load, fixed pose, upper height ceiling or joint endpoint is imposed. Moving outside or below the existing region smoothly reduces only current credit, and returning restores it. Historical qualification/crossing/placement and predecessor ordering are not erased. Unplaced-leg preparation/lift/carry/capture formulas, hard physical success/failure, support/ROI/stop criteria, action caps, nominal motion including the preceding geometry feature, mapper, environment and frozen A are unchanged by this commit. This is reward/potential semantics, not a new completion gate or a nominal-motion fix. The added current-geometry diagnostic does not itself grant placement.

The fixed observation schema remains324-dimensional and preprocessing stays identical, but task-progress potential is an observation feature. Identical weights therefore do **not** establish old/new-MDP action or trajectory equivalence on separately constructed observations. The prior C54,656 P05 incomplete remains an actual historical outcome, not retroactively successful and not proof of improvement under this revision.

## Actual run and initial artifact

Started manifest: `runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189/run_manifest.started.json`.

It records command=train, lifecycle=RUNNING, source54,656, `--new-mdp-warm-start`, semantic v3, retained `heteroscedastic_log_v1`, N1/seed1001, natural P01, full_episode, teacher offset0, **8,192 requested decisions** and checkpoint cadence4 updates. The initial sidecar has `curriculum_epoch.prefix_request=null` and `P01_full_task_only_initial_version` sampling. Source physical state is not inherited; this is a legal new episode, not a physical suffix continuation or historical snapshot. Actual later samples/outcomes have not been audited or credited in this report snapshot.

Initial checkpoint:

`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000054656_sf24b35714c5b_g68631e932c7d_4514185096bcd8d85cee235533c911e3d7d2f1609d3f0a21efa5ccebf58a771a.pt`

Read-only PowerShell hashes of the actual files:

- Initial checkpoint SHA-256: `4726e5a3ec9ce2028c2bb3971de5f3fefb469f35a1e01ef02b1ce5f06ee05c04`, matching its sidecar; `save_load_round_trip=true`.
- Initial `_manifest.json` SHA-256: `87feb0065f2fc55f3b2de7417941b959befd3c54960163e0fef96caae8526a67`.
- Bound source checkpoint: `checkpoint_step_000054656.pt`, SHA-256 `f24b35714c5b4881f78713739640f2130f8678216e8394d1fd41c91734c9aec7`; source sidecar binding `609e2686d8288ff3ec01ee7197fb23a31f59fefb58113c26cb0b700c4e9a6566`. The source is preserved, not overwritten by the named initial artifact.

## Preserved learned state and deliberate optimizer reset

The source and actual initial sidecars compare equal for the following digests:

| State | Source54,656 and new initial SHA-256 |
|---|---|
| Whole actor, including learned mean and state-dependent log-std parameters | `cca90f60744f0e00068068c9160cd06648a129a881df9cb010380a17dc716f5d` |
| Critic | `1ed87c38bd86953a2e5d1b91fc96ef0581650d1f5bcf08d8a9f00e34455dce6d` |
| Identity RSL normalizer | `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` |

The retained policy contract is official `HeteroscedasticGaussianDistribution`, state_dependent_std=true, log std, ELU, actor hidden256/256,324 observations and12 unbounded Gaussian latent actions before the existing tanh projection. There is no scalar-Gaussian architecture fallback. The entire source/initial `training_rng_state` JSON compared equal, including stored Python/NumPy/Torch state fields. `_load_v3_warm_start` restores the verified source RNG after loading/replacing Adam; the actual initial sidecar corroborates that preservation rather than merely asserting a requested seed.

Source Adam digest `04bbd35fe486bb9d449143d202e6c27e13c571a4ac6ce55b2ca29fc576dee0c7` changes to fresh initial `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`. The source effective LR is1e-5 and the actual new initial LR is **3e-5**. This is the initialization value, not a claim that later adaptive-update LR stays constant. `new_mdp_warm_start.json` records `exact_mdp_resume=false`, `reset_all_moments`, `old_rollout_buffer_inherited=false` and `physical_state_inherited=false`. The production loader verifies fresh N1 storage `(128,1,12)`,324 observation width, `storage.step==0` and no pending transition before loading; it verifies source network/normalizer/Adam hashes, then creates fresh Adam. No old optimizer moment or rollout transition is intentionally used for a new update.

Initial counters remain **54,656 decisions /392 PPO updates /7,840 optimizer steps**, spent full_episode16,896 /phase_suffix27,648 /smoke0 and original origin10,112. These values match the source; the new-MDP boundary does not clear budgets, restart lifetime counters or count initialization as optimization.

## Same-state action diagnostic and evidence limits

`new_mdp_initial_action_comparison.json` is phaseP01/tick0/sim0, observation SHA-256 `2dfc53792d5671abf3805561149676bca0dbd27dc31251296a00cbabd77973a0`,0 optimizer updates and `actual_environment_or_bridge_history_modified=false`. Both physical execution profiles have unchanged SHA-256 `8020972778b63b64a14e916b059bb43d4f756e1e3510c72d22897ad23cf4ed60`; every12-channel old/new difference for bounded, scaled, masked, rate-projected, safe-projected and applied action is0. This compares the same raw mean on one same **new** observation and the old/new physical profiles. It is not native dispatch, old-MDP observation reconstruction, a physical trajectory comparison or a paired stability improvement.

The migration binds exactly the two changed production files. The selected action/execution/observation/quality/reward files have identical source/target hashes; the stage task spec changes `094bdadf3f1954c40bdff70fbe34ac754003081ff5872f4d300bb9155fa04044` → `1994953dfd447b0630b060368218d1a509e9a591d9e9cc5d62c76831160d705b`. No task-success requirement was introduced as an optimizer prerequisite. Latest completed full evaluation is still C54,656's P05 incomplete, and fourteen finalized training blocks remain the completed ledger. No new successful/same-condition paired video or future evaluation is credited.

This report was prepared only with read-only PowerShell and an outputs-only patch. No Python, extra Isaac, production changes, historical-run edits or commit were performed by this report task. Subsequent actual updates and final execution/outcomes await separate evidence.
