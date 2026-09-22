# Block06: first genuine PPO update carries the actual AUX ledger

**PASS.** CPU-only read of the first sealed batch, not a synthetic test or a new optimizer run. The live Isaac instance was not touched.

Source: `checkpoint_aux_frontrehearsal_step_000209920_v1.pt`, SHA256 `3c45e7325210487431ebe8b5d1cbd92b480734e3d53829b04429869c4773e3e8`.

First normal PPO save: `checkpoints/history/checkpoint_step_000210048.pt`, SHA256 `40adc1a766d512daff53074e2bd365d63ca8b86613b832b5d1ab9c1134e6460e`.

- Actual counters: **210048 decisions /1606 PPO updates /32120 Adam steps**; this batch adds **128 /1 /20**, AUX adds0.
- `rr_postcross_workspace_branch.front_rehearsal_auxiliary` is exactly equal to source, including its full event, source/data/helper/report hashes,32 accepted/32 attempted counts. Previous limited AUX7/8 is exactly preserved.
- Twelve complete branch/migration dictionaries, including the new RR branch, are unchanged. The seven derived branch-count fields are correctly recalculated—not incorrectly claimed byte-identical after real PPO. All three origins remain199680/1525/30500,203776/1557/31140,207872/1589/31780.
- Corresponding current branch totals are10368/81/1620,6272/49/980,2176/17/340. All old historical branch and migration lineage is retained.
- Actor and Adam hashes actually changed; all12 Adam parameter states advanced exactly20 steps. Recorded finite nonzero gradient range1.00917–1.41421. LR1e-5, full runner config, Identity normalizer, runtime and policy contract remain unchanged. Actual saved actor/critic/Adam hashes and embedded/sidecar equality were independently checked; normal save/load roundtrip is recorded true.
- Real fresh storage is128×1×389 with12 raw actions. Original raw samples, μ,σ,logp,reward,done exactly match their synchronous audit. Recomputed Normal logp maximum difference3.815e−6; likelihood ledger shows each original sample used5 times across20 minibatches.
- Natural P01, no prefix: requested inputs **P01=2, P02=126**, P03–P13=0;1024 physical ticks,0 terminal samples,0 assist-owned endpoints. All12 residual masks were1 and native dispatch/state-write checks passed. This first batch proves ledger propagation, not front-stage or full-task success.

Evidence run: `runs/ppo_p05_hip_only_continuation_v1/train/20260922T1041563379791Z_g5fd88852bf20_087b67f789494ca695e6d94de78c5623`.
Bounded files: `rollouts/rollout_001606.pt`, `rollouts/update_001606_likelihood.json`, first128 `residual_and_projection_audit.jsonl` rows, first `optimizer_updates.jsonl` row, source/target checkpoints and sidecars.

The previously pending **first real ordinary PPO carry** is now verified. Later batches, episode completion and physical improvement are outside this audit. No bound helper or production file was modified; CPU process exited0.
