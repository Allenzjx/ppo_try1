# Block09 first actual PPO carry — RR AUX4 + receiver-v2

**PASS.** Migrated CP214400 `a039f071def736bcb8d2f9b6fb0c7e4691d9aeb74715918521b2268e9a521d9a` → actual CP214528 `e7ef82ceedb613866a72e31245d0fda4754225acd4af100bb759b1c1325d5e75`. New **128 decisions /1 PPO /20 Adam**, cumulative **214528 /1641 /32820**. No new AUX.

All four full AUX events carry exactly: prior front96/96 + RR7/8 = mixed103/104; older separate7/8 unchanged. Original three origins/migrations persist, and the new receiver-v2 origin214400/1640/32800 records exactly128/1/20. The migration record and plan binding remain intact.

Frozen actual migrated-source prefix: **264 actions /2112 ticks**, reaching P04 before learning; every prefix action has zero PPO credit. The128 learner inputs are **{'P04': 1, 'P05': 127}**, with 1024 verified physics ticks, 1 ordinary phase handoffs, 0 terminal samples and 0 assist-owned endpoints. Handoffs remain continuous and do not set done. This is a P04-initialized suffix, not full P01 learned success.

Direct389/raw12/μ/σ/logp/value/reward/done match the sealed rollout and synchronous native audit. Each original sample is used5 times in20 minibatches; CPU Gaussian logp max error **3.81469727e-06**. All12 residual permissions and separately observed FL-only assist execution validate. RR qualified/crossed/placed input counts are0: this first front-state batch does not yet prove active RR reward learning.

Actor and Adam truly change; all12 Adam states advance20, effective LR1e-5 and Identity persist. Full saved RNG schema/CUDA count remain and state advances. Embedded/sidecar/tensor/optimizer hashes verify and actual official save/reload is recorded true. PPO may now change sigma/trunk normally; mean-head AUX invariance is not falsely carried over to learning.

Read-only CPU audit of first sealed rollout1641 and actual source/target; no physics/GPU/fit/production edits/checkpoint writes. CPU helper exits after writing this compact report.
