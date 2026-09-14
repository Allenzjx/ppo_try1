# DEFERRED: complete-rollout terminal reset scheduling

Status: design/patch/test mirrors only. Author has not applied this patch, imported production, executed tests, or launched Python/Torch/Isaac. Do not apply while the existing training process is running. Reviewed production base: `69aeaca777dcc8653e60f19da56ae1cd002e5271`.

## Files and scope

- `semantic_training.apply_patch`: relative project-root patch. The only production target is `src/wlr50_clean/ppo/semantic_training.py`; existing checkpoint-prefix tests receive the now-unnecessary final reset count correction and a stale pre-fsync-era evidence assertion correction.
- `test_semantic_rollout_tail_reset.py`: proposed new `tests/unit/test_semantic_rollout_tail_reset.py`, to install using apply_patch after review. No production module is shadowed by this output file.

No change to semantic_prefix.py, controls, rewards, task/event predicates, ordinary phase transitions, action distributions, 372 observations, normalization, RSL algorithms, rollout length, physics, or success definitions. Parent-class hooks automatically serve ordinary and prefix N=1 adapters. Vector and unsupported stateful/normalized/RND paths keep original behavior via a capability check.

## Exact sequence

1. Collector enables a scoped defer flag only for the final tick of a complete rollout. Direct adapter callers and non-final ticks retain eager autoreset.
2. If that step is a real terminal, retain the actual finite terminal observation, original reward, sampled latent/logprob/mean/std/value, done=True, and time_outs=False. Persist the same terminal audit/episode evidence once before returning. Mark pending reset rather than executing it.
3. Official process_env_step stores the already-sampled transition; official compute_returns applies the existing done mask. No new policy sample is drawn from terminal observation. Complete the usual update and verified checkpoint publication. A pending terminal forces checkpoint publication even off the ordinary cadence, before another reset can fail.
4. If this was the last requested update or an accepted stop-after-update boundary, end without an unused prefix. get_observations remains read-only and can return the terminal observation during checkpoint/report checks.
5. Only when starting another rollout, consume the pending legal reset/prefix, validate finite reset observation and unchanged curriculum/return contract, then call alg.act with the NEW reset observation. If reset fails or mutates provenance, no new action is sampled; the preceding update/checkpoint remains saved.

## Official RSL evidence and limitations

Installed RSL 5.0.1 ppo.py:139–205 stores the pre-step observation in act; process_env_step receives the post-step observation for normalizer/RND consumers. For current N=1 identity-normalized, nonrecurrent models without RND those consumers are inactive. compute_returns evaluates a finite last value but multiplies its bootstrap contribution by 1-done. Mathematically the terminal return is reward; float32 follows `(reward - value) + value`, so tests preserve the official operation-order rounding bound instead of modifying GAE to force a bitwise reward copy.

The optimization is NOT universally equivalent for stateful normalizers, RND, or RNNs; the helper deliberately leaves those on the old path. Current HISTORY372 actor is stateless: previous raw comes from observation195:207, not a hidden recurrence. Tests compare eager/deferred official CPU rollouts, optimizer/model/normalizer state, and RNG under an explicitly RNG-neutral deterministic synthetic reset. This does not prove all opaque Isaac reset internals are globally RNG-neutral or promise bitwise identical future physical trajectories after reordering reset versus update. Preserve actual checkpoint RNG on migration; do not add RNG reseeding/rewinding to this patch.

The runtime content hash changes. Parent-reviewed exact same372 migration must state **terminal-only autoreset scheduling correction**, not pretend the code change is only logging. No NewMdpWarmStart/Adam reset or allowlist expansion is proposed. Stop and apply only after the current old process saves and exits normally; its already-executing unnecessary prefix cannot be cancelled safely by this draft.

## Proposed required verification (not run here)

- New tests: true finite terminal obs/time_outs=false/read-only getter; last terminal128 with no extra reset; next rollout reset before action; failed/provenance-changing next reset after forced off-cadence checkpoint; middle terminal still eager; nonterminal phase handoff uncut; stop boundary no prefix; terminal fsync failure restores writer scope; unsupported normalizer/RND/RNN/vector behavior; HISTORY372 eager/deferred official rollouts/Adam/RNG equivalence for the supported deterministic reset fixture.
- Existing regressions: `test_terminal_evidence_before_prefix.py`, `test_semantic_training.py`, `test_semantic_prefix.py`, `test_semantic_checkpoint_prefix_training.py`, history-prefix/runtime372 resume/migration tests, and relevant vector training tests. Existing prefix-credit test now expects 128 teacher actions rather than129 for128 terminal policy decisions: initial prefix plus127 resets needed for actual future policy actions; all128 credited actions remain.
- Verify fresh save/reload of the actual latest checkpoint and empty rollout via the reviewed exact migration before live continuation. The efficiency fix is not a probe/task-success gate.

The existing `test_terminal_reset_provenance_mutation_aborts_before_storage_optimizer_or_save` terminates on its FIRST credited step, so its eager reset behavior does not change here. Its old expectation that terminal audit/episode files stay empty was already obsolete after the earlier fsync-before-reset change. The draft now checks the one true terminal row and episode both inside the changed reset entry and after the exception; optimizer log remains empty, rollout storage remains empty, no checkpoint is published, and all original actor/critic/Adam invariants remain. This does not credit or resume the rejected pending transition.

Static review only so far. A read-only exact-text check matched all 16 old hunk contexts in source order against current production/test files, including the two stale-test correction hunks. No hunk was applied; src/scripts/configs/tests remained clean. The new test mirror was manually reviewed, not parsed/imported/executed. Main agent must apply/test/fix after the running process reaches a safe boundary.
