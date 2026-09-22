# Feedback v2: first real on-policy update

Read-only, CPU-only audit of update1558 from run `20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff`. No synthetic fixture paths, simulator calls or runtime edits.

- Real checkpoint: `outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000203904.pt`.
- Verified file SHA256: `68df381aca0de94320a204b726e09522fd42f042700d3550de69a18fc4a38b52`.
- Actual counters: **203904 decisions / 1558 PPO updates / 31160 optimizer steps**.
- Actual source: `checkpoint_capture_feedback_v2_step_000203776_ga802b24d78df.pt`, verified SHA256 `d26eb196a213f45e75a7f3b6c8c4b382fe9605cce97f8dfc076f0b4b459691c3`.

| Persistent lineage | Original origin: decisions / PPO / optimizer | Actual added counts |
| --- | --- | --- |
| Original P05 branch | 199680 / 1525 / 30500 | **4224 / 33 / 660** |
| Feedback v2 branch | 203776 / 1557 / 31140 | **128 / 1 / 20** |

Both lineage objects, the immutable migration record and the task/AUX branch dictionary exactly equal the published-source values. Thus the ordinary real-training save preserves the old P05 origin and separate feedback-v2 origin, not just the zero-update publisher/test fixture. Revision is `hold_to_air_progress_window_v2`; original migration credit remains zero. Migration plan SHA `a739c6d8762149649aeba7a55bf6de5feedeca6e8690a5f98cdcf586aede35d8` matches its actual file and saved record. Target runtime HEAD `a802b24d78df5f1b8f0caf914ce7a8a68a98db8e`, runtime SHA `fc4e9aa6da24b56186441d05917f8e8bad49533d4a886913dc272943c39b0f92` matches the checkpoint and sealed rollout.

Sealed `rollout_001558.pt` contains 128 new samples, global203777–203904, policy/critic dimension389 and raw-action dimension12. Natural P01 curriculum has no prefix: P01=2, P02=126, total1024 physical ticks. Every raw sample, conditional mean/sigma and old log probability matches collection records; raw samples also match selected-policy and native-dispatch audit evidence. Independent CPU Gaussian logp maximum error is `3.814697265625e-6`. One sample/action; all128 samples appear exactly5 times in the20 official optimizer minibatches.

Every endpoint snapshot declares feedback-v2; all12 assist observation features match the preceding actual state. Assist is WAIT for all128 inputs, so this batch verifies the new contract and provenance, **not an active HOLD-to-AIR recovery or task success**.

The real checkpoint sidecar records successful actual save/load roundtrip; CPU loading verifies embedded metadata against that sidecar and recomputes exact actor, critic and Adam hashes. Actor and Adam changed from the real published source; all12 Adam parameter step counters advanced20. Identity-normalizer hash and full runner configuration are unchanged; actual LR=`1e-5`. Completed update has finite nonzero gradients. Audit process exited0; no subsequent batches inspected.

## Later bounded follow-up: first RR-captured learner batch

Sealed `rollout_001566.pt`: 128 actual inputs/actions, global204801–204928, 389/12 dimensions, 1024 physics ticks, no done samples. Input states are verified against adjacent, same-episode, nonterminal endpoints204800–204927; requested phases P09=68/P10=1/P11=59 (endpoint phases67/1/60).

| Actual input state | Samples |
| --- | ---: |
| RR crossed history | 128 |
| RR current valid lift | 70 |
| RR placed history | **61** |
| Placed history and current classified TOP | **31** |
| Placed history but not currently TOP | **30** |

Placement history first appears in decision204867's endpoint: the recorded physical event is tick8726 /72.716667 s. That decision ends at tick8728 /72.733333 s, when RR is already classified AIR again. Thus the event is real placement-history evidence, **not a claim of continuous current TOP contact**. Placed-history policy inputs begin204868 and extend through204928; the last endpoint at76.8 s is AIR with4.30565 mm top clearance. No later P12 or final-block state was included in this audit.

All61 placed-history inputs and all31 current-TOP inputs are genuine stored Gaussian PPO samples. Each appears exactly5 times in20 official optimization minibatches. Raw action, conditional mean/sigma and old logp exactly match collection/storage/native audit; independent CPU logp maximum error remains3.8147e-6. Original single-draw semantics, all12 residual permissions and native verification pass. Update1566 records20 actual optimizer steps, changed actor parameters, finite gradients and LR1e-5.

The entire batch has assist mode RELEASED, owner indices `[]`, and RR pre/post-assist candidates identical. No RR or other assist target replacement occurs in this batch. Curriculum remains natural-P01 current-policy training with `prefix_request=null`, not frozen-prefix/teacher credit. This does not relabel the earlier FL-assisted preparation as pure-policy execution. Placement-history samples, current contact retention and full traversal success remain separate claims. Read-only CPU process exited0; no checkpoint or runtime writes.
