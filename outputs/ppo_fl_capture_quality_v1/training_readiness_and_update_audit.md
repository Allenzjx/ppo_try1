# FL quality curriculum readiness / per-update audit

Read-only review of production `08a46f6e7c8ffef9e2f3024329a082f2a4a711e9`.
No Isaac, real policy decision, optimizer update, checkpoint or pointer was created by this review.

**Resolved at the safe boundary:** after the probe naturally ended, the root applied the prepared two-file patch and committed `f2e552406ea74a5aae2803347d1c8bebe261910d`. Independent read-only diff review found only the exact class acceptance/Identity and migration-scope changes described below. `prefix_migration_after_fix.xml` records21 tests,0 failures/errors. Use the newly bound `checkpoint178432_fl_capture_quality_prefix_migration.json`; the old08a46f6 plan remains historical and must not be passed to the new runtime. No additional code blocker was found for the planned first512/P04 collection. The pre-fix diagnosis below is retained as evidence, not the current state.

## Immediate finding

Natural P01 collection and nonzero P01/P02 reward are wired through the selected six `ppo_fl_capture_quality_v1` configs. The first512 CUDA command was already validated against the real CP178432 migration. No additional blocker was found in that path.

The planned P04 `checkpoint_policy` prefix **is blocked** at the frozen-prefix actor constructor: `semantic_checkpoint_prefix_policy.py` selected `SemanticHistoryMLPModel` only for the untempered HISTORY version and required exact `MLPModel` for every other version. The actual quarter actor is therefore rejected before P04 roll-in or policy collection. CPU reproduction:3 expected positive tests fail,4 tests pass; see `prefix_compatibility_before_fix.xml`.

Prepared, not applied during the active physical probe: `checkpoint_prefix_quarter_compatibility.pending.patch`. It maps declared plain/half/quarter HISTORY versions to their exact classes, retains forward/get_latent/distribution/Identity checks and independent-copy semantics, and binds this narrow constructor acceptance change in the FL migration. It does not change actor forward, sampling, HISTORY, nominal, optimizer, or credit accounting. Focused tests: `tests/unit/test_semantic_tempered_checkpoint_prefix_policy.py`.

## Curriculum facts and limits

- First512: full_episode / natural P01. Actual stochastic outcomes determine whether this reaches P05;512 requested decisions do not guarantee FL samples or completion.
- Next2048: phase_suffix / P04 / checkpoint_policy / offset0, using the **latest newly saved compatible checkpoint**. Its frozen independent deterministic actor constructs a physical P01→P04 entrance. This prefix excludes P01/P02/P03 from PPO credit; do not count those as new quality or P03 samples. It can form a policy-created entrance, not a direct state restore.
- Prefix miss: exactly one fresh-P01 fallback, not a nominal-prefix substitution. Count actual request phases in saved PPO rows and disclose fallback count/causes from `prefix_evidence.jsonl`.
- Final256: phase_suffix / P06 / successful_nominal. Its N+0 prefix has no optimizer credit; suffix success is not full PPO success.
- Source actor is frozen for each entire prefix-training block; it is not silently refreshed after every update. A later block can explicitly select the latest checkpoint.
- All2816 planned decisions are exact multiples of128. If all execute, the arithmetic is22 updates/440 optimizer steps; these are **planned**, not achieved counts.

## Per-completed-update acceptance and diagnostic checks

1. **Completion and identity.** Match `optimizer_updates.jsonl`, immutable checkpoint manifest and `rollouts/rollout_<update>.pt`; `advantage_audit.jsonl` explicitly precedes optimization and alone is not proof of a completed update. Expect128 actual N1 rows,20 optimizer steps, correct runtime/372/full12/quarter policy, and new `fl_capture_quality_branch_counts = lifetime -178432/1359/27180`. No initialization/prefix credit.
2. **Coverage.** Read `advantage_audit.by_request_phase.*.sample_count` and confirm against `applied_audit.phase_id` in the matching128 audit rows. Report P01 through P13, no missing phase filled with success/quality zero. At least one **completed** optimizer batch must contain actual P05 request transitions before saying P05 was trained. Report its exact count, gap/contact/placed progression and actual minibatch indices. Assess P03–P06 share from samples, never from the requested course name. P04-boundary decisions may contain P05 physical ticks but retain their true P04 request label.
3. **Front cost is applied, not only configured.** In `applied_audit.reward_breakdown`, require objective `fl_capture_front_body_quality_v1`, epsilon.03 and `front_quality_sample_audit` rows for actual P01/P02 physical ticks. `effective_beta_per_s` must be in[.015,.03], including preparation/carry; these labels cannot turn it off. Each sample has real time/dt, roll/pitch/rates, raw costs and weighted cost. Sum weighted costs and compare to `-families.body_stability` and to `front_weighted_tilt_cost + front_weighted_rate_cost`. Nonzero measured tilt/rate must produce positive cost; a genuinely zero-motion/zero-angle sample may correctly cost0. Other phases have body cost0 in this candidate.
4. **Avoid a false quality-training count.** Count only new on-policy P01/P02 rows that entered a completed update. P01/P02 traversed by frozen checkpoint or successful-N prefixes do not qualify. Report active beta by physical substate and task progress/clearance, not just a mean diluted by other phases or idle time.
5. **P05 reward and physical truth.** Check new stage/potential configuration binding; log signed potential shaping, terminal event, time cost and reward. FL fine-gap progress requires qualified crossing/current valid top XY, uses positive gap with.025/.003m scales, and retains a separate real-contact share. AIR with small gap is not placed. After physical placement, potential uses retention instead of repeated descent. Include actual FL gap/front/contact/bearing/support and P05→P06 evidence; task improvement is not proven by a new config or changed weights.
6. **Behavior likelihood.** Actual request audit provides base/conditional means, previous raw observation history, learned/effective sigma, selected raw/tanh, and sampled log-prob with zero extra actor forwards/random draws. Selected raw must equal saved Gaussian action, not projected actuator targets. Saved old mean/std/log-prob must match actual collection; only the first minibatch before the first update should have ratio≈1—later minibatches legitimately see changed weights.
7. **Learning direction.** Join actual P05 rows to `update_<update>_likelihood.json` by saved rollout indices. Report raw GAE, stored normalized advantage, actual minibatch advantage, `advantage * delta_log_probability`, and strictly active PPO clip branch. Keep helpful gap/contact transitions distinct from terminal penalties; positive advantage is not guaranteed for every locally useful action. Parameter change alone is not physical improvement.
8. **Continuity and termination.** Ordinary phase changes remain nonterminal. Check `ordinary_phase_change_samples`, terminal reason, finite terminal observation, no bootstrap on true task terminal and official last-value bootstrap on a nonterminal rollout boundary. Source endpoint is not episode end or loss of residual authority.

## Evidence locations in the existing production path

- `semantic_cli.py` selects new config_root; installs the independent frozen prefix only after verified checkpoint load.
- `semantic_checkpoint_prefix.py` records every physical prefix step separately, hands off the current observation/history without teleport or actor-data credit, and records accepted/fallback entrance.
- `semantic_env.py` persists the complete reward dictionary in both `reward` and `reward_breakdown` inside the actual decision info.
- `semantic_training.py` preserves that info in `residual_and_projection_audit.jsonl`, stores actual rollout tensors and phase/advantage summaries, and runs actual20-minibatch likelihood instrumentation for this experiment.
- Existing checkpoint manifests and official reload validate learned state; formal deterministic and explicitly seeded training-style stochastic videos remain separate later evaluations.

These are execution/learning-audit criteria, not a success gate requiring multiple full episodes. Failure samples remain legitimate on-policy training data; physical task improvement must be demonstrated by reloaded evaluation/video.
