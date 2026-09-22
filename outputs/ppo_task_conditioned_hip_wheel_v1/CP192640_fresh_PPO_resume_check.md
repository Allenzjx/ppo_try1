# First real PPO update after finite auxiliary mean learning

Scope: read-only CPU inspection of exactly block10's first sealed rollout/update
1470 and CP192640. No model forward, optimizer step, Isaac call or production
edit performed by this audit. It is not an additional training gate.

## Source, provenance and counts

- Run: `20260921T1043344171428Z_g97ecd305afb5_0168bb1943e74402accb92ea31b10de3`.
- Actual source: `checkpoint_aux_flmean_CP192512_budget16_01.pt`, SHA256
  `89d7e1e0371f2a661753c3a0f4d594933301e253b7ce6b8899b42088ce2f861c`.
  Sidecar SHA `99f11cba839b736980f908fdd6a7bc767c6d2d59b4da36ca418525b6b39f9561`.
- CP192640 SHA `46e5176aeb7deeb2dd800eb59836a20a37a0efad7e79c4dd982187a2733ba507`;
  declared save/load roundtrip true. All three involved checkpoint file hashes
  and payload optimizer hashes were recomputed on CPU and matched manifests.
- Resume source SHA and update's pre-actor hash match the actual aux checkpoint.
  The complete nested auxiliary ledger equals the source ledger: exactly one
  event, 7 accepted / 8 attempted auxiliary steps. Its provenance was not dropped.
- PPO alone advanced **128 decisions / 1 update / 20 Adam steps**:
  192512/1469/29380 → 192640/1470/29400. Original task-branch origin retained;
  branch counts now 6784/53/1060. Auxiliary updates are not added to these counts.

## Adam continuity and fresh live data

Original CP192512 and aux CP have identical complete PPO Adam hash
`891b4fdb53cdb505edefe57eb9bf37456ea4a291c80daf325c4b52beb97cba8b`.
After the real PPO update, all 12 parameter-state step counters increased by
exactly 20, parameter groups are unchanged, and LR remains 1e-5. Adam was not
reset for auxiliary supervision or this continuation. Its moments appropriately
change during the new PPO update; this is not a claim that moments stay frozen.

`rollout_001470.pt` contains original `[128,1,372]` inputs and live policy actions
for decisions 192513–192640, ending at physical ticks 8–1024. Counts are P01=2,
P02=126; no P05, P06 or rear-stage coverage is inferred for this one update.
All128 stored raw actions, old means, sigmas, log probabilities and observable
raw HISTORY match the corresponding actual decision logs exactly. All128 use
`training_style_conditional_gaussian`, one genuine policy draw per decision.
The 75 original diagnostic approach-observation hashes have **zero** overlap
with these rollout inputs. No diagnostic action selector or old approach data
was observed in this fresh natural-P01 rollout.

Rollout SHA: `5a33eeabdccdf0d1560fd49af8c427c4d1757c952ab4ef7ea1b6cdc508f156b8`.

## Full12 learning remains active

Every one of the128 actual dispatch audits has all12 phase permission entries=1.
Each channel's stored Gaussian sigma is strictly positive and finite; the
minimum across all samples/channels is .00845407. All128 physical-step records
retain the no-in-episode-state-writes proof.

All20 actual official PPO minibatches record finite likelihood, advantage,
conditional distributions and direct loss derivatives. Both direct mean and
log-sigma derivative groups have nonzero aggregate magnitude for every channel.
All24 corresponding post-clipping head-row gradient norms are finite and
strictly positive in every minibatch (minimum mean-row norm .00713388;
minimum log-sigma-row norm .00301021). Thus the continuation is not a row0-only
optimizer or wheel-masked update. Logged extra model forwards/random draws=0.

KL mean=.0223755882751. Likelihood SHA:
`7daf70254be8eb5edacef38ab1c75f4b2fc44a124632fbe94e2b67c785377c69`.

These checks establish actual fresh full-channel PPO continuation and persistent
explicit auxiliary lineage. They do **not** establish improved FL capture,
retention, rear clearance, stability or complete obstacle success; those require
the subsequent real trajectory and teacher-free full evaluation.
