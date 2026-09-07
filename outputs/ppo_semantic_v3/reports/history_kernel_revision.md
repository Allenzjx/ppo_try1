# History-conditioned exploration revision

Implementation boundary after checkpoint118016 and its completed natural-P01 evaluation. This is a conditional-policy change, not an environment/reward change, and not evidence of improved physical control. No source checkpoint, frozen A file or old run is overwritten.

## Measured motivation and limits

The unchanged f4bfe2560bfd runtime completed P06 preparation, P07 and P10 courses with genuine on-policy updates. In the most recent P10 first128, P12 RL hip raw was negative126/126; knee raw positive4/126 still produced negative filtered requests126/126. All actuator requests reached the audited native path. Front-wheel positive samples sometimes reduced negative nominal motion but did not cancel the strongest sampled segment. This identifies an observed difference between instantaneous Gaussian sampling and sustained filtered control; it does not prove that exploration persistence is the unique cause of incomplete lifting.

The P07 block116992 had genuine RR qualification and closest decision-end front distances of -25.133/-12.074mm, but no crossing or placement. The P10 block118016 had no hard RL qualification/cross/placement. Reloaded C118016 completed669 decisions/5352ticks/44.6s, P05 incomplete: FL Q1843/C2869/noP, current AIR/load0, front+6.726686mm/gap+31.968420mm. Its first missing task is real FL capture. There is no full/suffix success, successful PPO video or paired improvement claim.

## One selected policy change

Version: `history_conditioned_heteroscedastic_log_v1`.

For each stored observation, h is its existing `previous_raw_full12` feature, indices195:207, scale1, schema clip +/-20. The shared324-input, 256x256 ELU, 24-output mean/logstd actor retains every learned parameter and parameter name. The actor is stateless: it has no hidden action accumulator and does not take history from neighboring minibatch rows.

```
rho = 0.9
conditional_mean = (1-rho) * learned_base_mean + rho * h
conditional_std  = exp(learned_log_std)
raw_action ~ Normal(conditional_mean, conditional_std)
```

The learned sigma is the conditional innovation standard deviation, not a promised stationary marginal standard deviation. It is not rescaled, floored, clamped or reset. Finite positive representability is checked. Existing tanh, actuator headroom and slew projection remain downstream of the sampled latent; PPO likelihood uses the raw latent and actual conditional Gaussian.

RSL sampling, stored old mean/std/logprob, entropy, adaptive KL and shuffled-minibatch recomputation all use the actual conditional mean/std. Fixed-mean inference also changes to the conditional mean. This is not training-only noise. With identical sigma, the base-mean head's local derivative is scaled by0.1; this can alter adaptive-LR behavior and slow response to a necessary change. Persistent wrong-direction actions and greater raw wandering/saturation remain risks. No closed-loop improvement or ideal-AR stationary-variance guarantee is asserted.

Real episode reset supplies zero raw history; ordinary phase changes retain live environment history. The existing observation schema and encoder are unchanged. Unsupported JIT/ONNX export is rejected because the inherited export would otherwise omit the conditional-mean calculation. Normal deterministic evaluation and video call the actor's actual forward path.

## Migration and provenance

Use the explicit `--new-mdp-warm-start --target-policy-version history_conditioned_heteroscedastic_log_v1` boundary. The flag reuses the existing reviewed warm-start mechanism; the record explicitly states that physical MDP and reward are unchanged. Ordinary/exact resume does not convert kernels.

The source must be a verified v3/N1 learned heteroscedastic checkpoint. All six configuration SHA records and physical/runtime metadata must be unchanged. Runtime source changes are limited to the actor, policy contract, migration, training/CLI/wrapper and checkpoint-prefix integration files. Source and target code/config identities remain bound separately.

The loader verifies source embedded metadata, actor/learnedstd, critic, original Adam and identity normalizer before replacing Adam with fresh state at3e-5. It restores source RNG and preserves lifetime decisions/updates/optimizer counters, spent budgets and original10112 task origin. Storage is empty128x1x12; physical state is recreated through a legal reset. No old unfinished rollout is used.

The immutable initial filename also binds the target policy contract. A separate same-observation comparison records source base mean, actual history-conditioned target mean, shared sigma/value and conditional KL without sampling or changing the distribution cache/RNG. The older physical-profile projection comparison uses target raw in both profiles and is explicitly labeled as such, not old/new-policy equivalence.

Checkpoint-policy prefixes are forbidden only during the cross-kernel migration itself, preventing source-checkpoint identity from being mislabeled as a new kernel. Ordinary resumes of a saved new-kernel checkpoint can create a separately frozen new-kernel prefix. Both prefix boundaries validate the exact supported contract and actual actor/kernel identity; equal tensor hashes cannot disguise different inference laws.

## Unchanged production behavior

No change to nominal, mapper, phase/event conditions, reward or six physical configurations. Gamma.9985/lambda.99/rollout128, official PPO5epochs x4minibatches, all12 learnable channels, existing action caps/slew/hard limits and finite200s task deadline remain. Body collision/wheel-only climb and independent safety aborts remain unchanged. Teacher initialization is excluded from policy credit; ordinary phase changes do not end an episode.

## Verification status

Focused actor81/81, prefix120/120, CLI16/16 and corrected prefix-training4/4 CPU cases passed. The first expanded633-case regression had631 passes and two failures from a pre-existing fake reward-calculator object in the prefix-training fixture. That algorithm-only fixture now explicitly has no physical reward calculator; the production consistency validator was not weakened, and both tests also cover the new kernel. The failed receipt is retained.

Final expanded regression passed 643/643 cases, zero failures/errors/skips in45.220s. Receipt: `C:/robotics_sim/wlr_robot/history_kernel_verified_regression_20260907.xml`. This includes eight integration cases: actual official CPU source128/1/20, verified migration, immutable initial save/reload, target128/1/20, stored raw/history/logprob consistency and rejection cases. The temporary Git source blobs are provenance fixtures; this test does not execute an old historical runtime or simulate physics.

These CPU tests are algorithm/interface evidence, not Isaac task success. The eight production files and six test files were committed as `e8462f06690873c8ac4c4fec453f451f5d10d0e8`; generated outputs remain separate and preserved.

The selected physical course has now completed: run `20260907T0736098630371Z_ge8462f066908_a07ce6b30c64439e8e8735a0926c4cc6`, frozen-FSM initialization to P06 with160 teacher-offset decisions, then1024 fresh on-policy decisions, N1/seed1001. The actual increment is1024 decisions/8 PPO updates/160 optimizer steps, yielding119040/895/17900. Final `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000119040.pt` records roundtriptrue and the new policy contract. Full53504/suffix55424/smoke0/origin10112 reconcile to119040. Teacher transitions are excluded; all12 channels and ordinary cross-phase credit remain active.

The initial-boundary evidence is in `history_kernel_initial_118016.md`; the verified first128 control audit is in `history_p06_first128_118144.md`. The full-batch report is being finalized separately. This completed training block is not a successful task: the first578-decision episode endsP09 incomplete, and the remaining446 decisions form a nonterminal optimized tail. Reloaded natural-P01 evaluation of119040 has been launched and is not yet credited with any outcome here. No successful PPO video or stability-superiority claim is made.
