# Same389 RR workspace reward-semantic migration — isolated draft

Status: not adopted, no production/config/checkpoint edits or real learning. Proposed supervisor behavior is separately reviewed by the supervisor agent. Current Isaac/PPO continues unchanged.23 CPU tests passed:14 new isolated tests, including synthetic full-state publication through the generic strict loader, plus9 existing feedback-v2 regressions. Test modules were patched only in that CPU process's memory; synthetic checkpoints lived in pytest temporary storage. Both test and driver-help processes exited0.

## Minimal scope and exact contract

Use a dedicated schema `wlr50_clean.rr_postcross_workspace_same389.v1`, factor `rr_postcross_workspace_factor`, not the one-time capture-feedback factor and not NewMdpWarmStart. Generic strict resume/save/reload handles actual weights, optimizer, Identity and RNG; only its factor-selection/record/carry/count hooks are added. The existing source-device zero-update publisher mechanics are shared through one small private function, rather than duplicated as another publishing framework. Old strict factors/whitelists remain unchanged.

The sole parsed task-spec difference is:

```yaml
rr_postcross_workspace_semantics: current_qualified_RR_over_top_receiver_retirement_v1
```

This includes the supervisor's final existing `geometry.top_gap_min_m` AIR tolerance, not a new zero-plane cutoff. No task geometry number may change. All other five selected config files must remain byte-identical. Full source/target selected-configuration bindings, actual committed HEAD, every runtime file hash, and the exact reviewed delta are checked. Allowed delta is limited to supervisor, task spec, dedicated migration module and the three small shared migration/training/publication integrations. No actor, action schema, nominal, capture-assist controller, reward coefficients, physical assets, sigma/caps, observation schema or budgets may change.

Shape stays389 and distribution contract stays the same, but **observation index17 (`task_progress_potential`) and reward numerical semantics change**. The receipt explicitly sets `reward_changed=true`, `same_mdp_claimed=false`, `physical_dynamics_changed=false`, and `same_numeric_input_policy_mapping_preserved=true`; it does not claim identical actions from the same physical state. All actor/critic parameters and Adam moments/steps/groups/LR are retained exactly. Critic estimates must recalibrate from newly collected reward data; source values are not new-objective truth. No first-layer zero extension, normalizer reset, variance change or policy-head reset.

## Lineage and zero-update publication

New `rr_postcross_workspace_branch` origin uses the selected *actual future legal-boundary checkpoint's* decisions/PPO/optimizer counters. No207232/207872 or other source number is hardcoded. Migration adds0 decisions,0 PPO,0 optimizer,0 AUX. Old rollout is discarded; no physical snapshot is inherited.

Preservation covers the complete task branch including its `auxiliary_mean_learning` ledger: all historical events, reports, source/data/helper bindings, accepted/attempted totals and method label—not just the7/8 scalar totals. P05 and feedback-v2 branch objects, original origins, counts, immutable migration records and older branch/migration metadata are hash-bound and compared in full after real save/reload. Existing generic `resume_migration` may point to the newly applied boundary; it does not replace the immutable prior migration records. Normal subsequent PPO saves carry the new migration/branch and recompute all three independent origins.

Publish/reload the identity-migrated checkpoint under the new runtime **before** using it as a frozen P06-prefix source. No old-checkpoint/new-runtime prefix waiver is added. Publication uses the source runner device and original CUDA RNG visibility, only after Isaac exits. It does not change checkpoint pointers or overwrite an existing artifact.

## Draft artifacts and later adoption

- `rr_workspace_migration_proposed.patch`: complete proposed production/test additions, including shared integration. Does not include the separately reviewed supervisor/config behavior patch.
- `rr_workspace_migration_draft.py`: readable future dedicated module, same bytes as the combined patch.
- `rr_workspace_migration_integration.patch`: readable shared integration subset.
- `test_rr_workspace_migration_draft.py`: isolated in-memory test harness; combined patch supplies installed-runtime tests without this harness.
- `prepare_rr_workspace_checkpoint.py`: output-only plan/publication driver; not run beyond `--help`.

After root approves/adopts, commits and freezes the runtime at an actual safe boundary, the driver requires `--checkpoint <actual sealed path>` and `--expected-source-sha256 <actual SHA>`. Default validates and binds a plan only; `--publish` performs source-device identity publication. No hardcoded source checkpoint or future HEAD. Root alone selects the stop boundary and the remaining genuine PPO budget.

The23 tests establish wiring/state-preservation semantics, not physical success or policy improvement. New reward version still needs fresh real on-policy updates and a saved/reloaded natural-P01 deterministic evaluation.
