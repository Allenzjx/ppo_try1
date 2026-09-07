# Versioned return-horizon continuation

## Scope and evidence at the revision boundary

Source is the immutable learned heteroscedastic checkpoint `history/checkpoint_step_000109824.pt`, at 109824 policy decisions / 823 PPO updates / 16460 optimizer steps. Its SHA256 is `065bf5de9604da5f6732d652dea1a3d9154750000d688e103c492b3b5b36067a`; the pointer's checkpoint and manifest hashes were independently matched before this revision. Source runtime is f1a9bbf650b1b8f80d5169e09173fcfc68797d99. Historical data, frozen A and earlier checkpoints are retained.

C109824 is a real natural-P01, fixed-mean evaluation: 668 decisions / 5344 physics ticks / 44.533333333 seconds, P05 INCOMPLETE_CONTROLLER_BLOCKED. FL qualified and crossed historically but never captured the top; it is currently AIR with zero load. This is not a sensor/execution failure or success. The latest P10 course supplied 4 P10 / 104 P11 / 916 P12 learner samples, but no RL hard qualification/crossing/placement and no P13. Its teacher RR events are not learner achievements. Detailed evidence remains in `eval_109824_diagnosis.md` and `p10_block_109824.md`.

The selected adjustment is an empirical return-horizon experiment, not a proven explanation or improvement. Previous diagnostics found no mismatched discount, double PBRS application, phase-label GAE cutoff or outside-XY zero-capture gate. At 15 decisions/second, gamma=.995 discounts a result 30 seconds later to .104806 and 60 seconds later to .0109843. Gamma=.9985 gives .508898 and .258978 respectively. A finite 128-step rollout still uses the successor critic; it is not a hard 8.53-second objective horizon.

## Actual production configuration selected

- Explicit profile `v3_gamma_09985_lambda_099_v1`: gamma .9985 and GAE lambda .99, replacing .995/.95 only in v3.
- N1; 128-step rollouts; 5 epochs x 4 minibatches; clip .2; adaptive KL .01; initial new-MDP Adam learning rate 3e-5. Official installed RSL PPO/GAE is unchanged.
- Existing 324-observation, 12-action, 256x256 ELU actor/critic and learned heteroscedastic log standard deviations are unchanged. No std floor, new network, action masking or entropy restart is introduced.
- Existing servo/wheel ranges, continuous mapper/request histories, whole-body lift predicates, legal prefix courses, hard task/safety rules, 200-second task limit and phase deadlines are unchanged.
- Five family weights remain 1/.4/.2/.1/0; potential weight 5, terminal events +/-40 and time cost .02/second are unchanged. Reward PBRS and PPO read the same versioned discount. The original failure-cost bound remains enforced: the selected bound is approximately 37 < 40. A .9995 candidate would violate that unchanged guard and was not selected.
- Frozen v2/A parameters remain .995/.95, including v2's actual historical failure bound 14.6133333333. No A 5/5 or successful manual-probe gate is added.

## Code and migration behavior

`semantic_return_profile.py` identifies the exact supported legacy/new parameter pairs. `semantic_training.py` takes runtime gamma from the selected reward configuration and validates cached runner metadata, live RSL configuration, actual algorithm gamma/lambda and the attached actual reward calculator before accepting transitions. `semantic_reward.py` rejects unmarked new gamma and unknown/mixed profiles. `semantic_policy_distribution.py` validates the complete historical runner configuration against its explicit profile, preserving all non-return metadata checks. `semantic_migration.py` binds the source reward bytes to their historical Git SHA and records the exact source/target return transition.

Ordinary resume cannot silently change return profiles. An explicit new-MDP warm start preserves compatible actor (including learned std), critic, identity normalizer, verified training RNG, lifetime counters, original 10112-decision origin and spent budgets. Source Adam is restored and verified, then deliberately replaced by fresh Adam at 3e-5. Old unfinished rollout is discarded; physical state is not inherited and is legally reset. Same-version course switches after that boundary use ordinary resume and retain Adam.

## Verification and execution status

The first 149-entry CPU regression exposed one incorrect v2-bound expectation and one genuine missing-marker loader hole; both were corrected. The following 569-entry regression passed the 68 new return/reward/official-GAE/migration cases, all 19 v3 continuation cases and the learned heteroscedastic same-profile migration case. Three existing stop-progress assertions required isolation from the newly versioned gamma and the separately enforced workspace dependency. Final corrected-suite results and actual Isaac launch/update evidence will be appended below; no unrun decisions are credited here.

No successful full-P01 or suffix checkpoint, successful PPO video or paired stability superiority is claimed by this revision document.

### Final CPU verification

The corrected 17-file regression exited 0: **571 JUnit entries, 0 failures/errors/skips**, 51.367 seconds. Receipt: `C:/robotics_sim/wlr_robot/return_horizon_verified_regression_20260907.xml`. The 68 new return/reward/official-GAE/migration entries are included, not added again. The two stop-progress tests now independently retain the old v3 .995 calculations and validate the new .9985 calculations using matched loaded reward configs. The global-progress dependency case isolates the continuous-stop guard from the separate workspace guard. These changes do not weaken a production predicate. The source-fixture reward history is now real temporary Git history rather than a fictitious commit. No CPU fixture decisions count as Isaac training. The original failed receipts remain available.

### Actual Isaac launch and first update

Production and tests were committed as `f4bfe2560bfd228541f7829d830fa441054eab1d`; generated historical outputs remain untracked and preserved. Actual command is `scripts/run_semantic_ppo.ps1 -Command train -ExpectedHead f4bfe2560bfd228541f7829d830fa441054eab1d -SemanticVersion v3 -Stage full_episode -FromPhase P01 -NumEnvs 1 -Seed 1001 -Decisions 4096 -Checkpoint outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000109824.pt -CheckpointIntervalUpdates 8 -NewMdpWarmStart`.

Run `train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8` published its revision-bound immutable initial and began true Isaac sampling. The first 128 decisions produced **109952 / 824 / 16480**, a real actor-changing official PPO update with finite nonzero gradients and a saved checkpoint with roundtrip=true. P01/P02 counts are 1/127, all 1024 native ticks verify, the four in-episode state-write counts are zero, and the ordinary P01-to-P02 boundary preserves nonterminal bootstrap and action carry. All 128 PBRS terms exactly match the once-per-decision .9985 expression; no terminal occurs in that bounded window. Details are in `return_horizon_initial_109824.md` and `return_horizon_first128_109952.md`.

The first subsequent completed training episode is 651 decisions / 43.4 seconds, P05 INCOMPLETE_CONTROLLER_BLOCKED. Its physical evaluation is valid; FL qualified at tick1677 but did not cross, then touched ground before crossing at5165, correctly revoking that qualification. It does not count as success. The 4096 allocation remains in progress at this note, so it is not added to the finalized block ledger.
