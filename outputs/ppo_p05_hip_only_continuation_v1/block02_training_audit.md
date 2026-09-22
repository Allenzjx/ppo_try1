# Block 02: sealed full training audit

Read-only CPU audit of run `20260922T0523430327244Z_g0001c3138b0b_f01402fe07fa4d85892560249297522b`, original runtime `0001c3138b0b38a7278edc288b8ddc7444815770`. Only original recorded native audit flags were consumed; the subsequently modified capture controller was **not imported or used to replay old receipts**. No GPU, Isaac, source/config edits or checkpoint writes.

## Completed updates and checkpoint

- New work: **2,048 policy decisions / 16 PPO updates / 320 Adam steps**; requested block completed, no rounding overrun.
- Latest sealed checkpoint: `outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000203776.pt`.
- Lifetime counters: **203,776 decisions / 1,557 PPO updates / 31,140 optimizer steps**.
- Independently read file SHA256: `8e4a72db49c01faee9749db6752daab3db2428a89ce67995c34e515f2526bff8`, matches manifest. Recorded save/load round trip passed.
- All 16 update records have finite metrics, nonzero gradients, changed actor hashes and a continuous actor hash chain. Each has 20 optimizer steps; all 12 Adam parameter states advance by exactly 320 relative to CP201728. Six actor and six critic state tensors changed. Final LR is `1e-5`; manifest retains the identity-normalized fixed 389-feature contract.

## Exact stage counts

Counts distinguish policy-input/request stage from post-action endpoint. The frozen CP201728 prefix is physically executed but not included in any PPO rollout.

| Stage | Credited requests | Credited endpoints | Uncredited prefix requests |
|---|---:|---:|---:|
| P01 | 0 | 0 | 6 |
| P02 | 0 | 0 | 762 |
| P03 | 0 | 0 | 9 |
| P04 | 0 | 0 | 3 |
| P05 | 0 | 0 | 444 |
| P06 | 863 | 860 | 0 |
| P07 | 3 | 3 | 0 |
| P08 | 3 | 3 | 0 |
| P09 | 1,179 | 1,182 | 0 |
| P10 | 0 | 0 | 0 |
| P11 | 0 | 0 | 0 |
| P12 | 0 | 0 | 0 |
| P13 | 0 | 0 | 0 |
| Total | 2,048 | 2,048 | 1,224 |

This block intentionally samples P06-initialized preparation/rear suffixes. Its zero credited P01–P05 counts must not be described as full-P01 learning coverage.

## Actual RR and capture coverage

| RR condition | Policy input | Action endpoint |
|---|---:|---:|
| Currently qualified | 515 | 516 |
| Crossed history | 421 | 422 |
| Crossed but not placed | 421 | 422 |
| Placed history | 0 | 0 |
| Actual TOP / top-surface contact | See note | 0 / 0 |

The 421 crossed-input samples were actually used by PPO: every saved rollout sample occurs exactly five times in its 20 official minibatches, including all 421 crossed observations (2,105 sample uses). They all come from the second physical episode; this is new post-cross learning coverage, unlike block 01. No real RR landing sample exists in this block.

Exact current TOP is not an explicit raw-389 classifier feature. Among the 2,045 inputs matchable to a preceding credited physical endpoint, TOP-positive count is zero. The three prefix-to-learner input points have no exact current-RR-TOP classifier in the selected prefix journal; they are **not silently filled as zero**. RR qualification/crossing/placement input counts above are direct saved observation bits.

Capture assist owns only FL hip/knee at **896 endpoints**: P06 860, P07 3, P08 3, P09 30. The other 10 candidate channels are unchanged by the assist. Remaining 1,152 endpoints report all 12 policy channels unmodified at actuator. All 2,048 endpoints have all-one residual permission masks; this is compatible with explicit assist ownership and is not evidence of pure-policy FL capture.

Observed assist input modes: HOLD 90, DESCEND 156, BLOCKED 620, RELEASE 33, RELEASED 1,149. Endpoint modes: HOLD 87, DESCEND 156, BLOCKED 620, RELEASE 33, RELEASED 1,152. Mode-count differences are normal within-decision state changes.

Pending FL, permission-to-continue-pending, and scheduler-advanced-pending counts are all **zero**, in both saved policy inputs and endpoints. Every frozen prefix reaches P06 after actual FL placement; this block does not exercise pre-placement pending overlap.

## Real episodes, prefix exclusion and physical ticks

All three accepted frozen-prefix runs contain 408 decisions / 3,264 ticks / 27.2 s, then open learner credit at P06 without resetting physics. All 1,224 prefix decisions explicitly have `policy_credit=false`; every credited row rejects both teacher and checkpoint-prefix data in PPO storage.

| Episode | Credited decisions | Credited ticks / seconds | Total physical time including prefix | Result |
|---|---:|---|---:|---|
| 0 | 756 | 6,045 / 50.375 | 77.575 s | P09 `INCOMPLETE_CONTROLLER_BLOCKED`; RR never crossed |
| 1 | 803 | 6,420 / 53.500 | 80.700 s | P09 `INCOMPLETE_CONTROLLER_BLOCKED`; RR crossed, never TOP/placed |
| 2 | 489 | 3,912 / 32.600 | 59.800 s | Partial P09 episode at completed-update block boundary; not terminal or successful |

Credited simulation totals **16,377 ticks**, not 16,384: the two terminal decisions legitimately execute 5 and 4 ticks. Prefix simulation contributes **9,792 ticks**; total task simulation is **26,169 ticks**. Reset/settling steps are outside these counts. Recorded native all-tick verification covers every credited and prefix task tick; no in-episode root/velocity/force/gravity writes are reported.

## Stored rollout / likelihood checks

All 16 sealed rollouts have policy/critic observations `(128,1,389)` and raw actions `(128,1,12)`, finite tensors and the original runtime binding. Stored raw actions, conditional means/sigmas, log probabilities, rewards and terminal flags match collector records; raw actions also match the policy-selected sample and native-audit request. All actions record one sample and zero extra random draws.

Independent CPU Gaussian log-probability recomputation has maximum error `5.7220458984375e-6`. All 16 official likelihood logs contain 20 minibatches and use each saved sample exactly five times. The 12 assist and five continuation input features exactly match the recorded policy-request evidence. This verifies actual optimization use, not task success.

## Outcome

The training block completed and learned from real RR-crossed states, but produced **no RR TOP placement and no full-task success**. CP203776 still needs its separately scoped reload/full-P01 evaluation; the partial suffix episode and successful optimizer lifecycle must not be promoted to traversal success.

Detailed machine-readable counts: `block02_training_audit.json`. Reproducible outputs-only CPU reader: `audit_block02_cpu.py` (prints JSON, writes nothing, imports no project runtime/controller modules).
