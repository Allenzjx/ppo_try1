# First completed-update evidence for current-clearance lift credit

Read-only PowerShell analysis; no Python, simulation, production edit, checkpoint load or parameter change was performed. This is an early fixed-prefix observation, not the final 6,144-decision result or a training gate.

## Evidence boundary

- Run: `runs/ppo_semantic_v3/train/20260906T0636173277212Z_g64abc5357d00_6614a15089014b2ebef51695b9b8cc22`.
- Runtime: `64abc5357d00763419fe51c8126db8454fcf7697`; source checkpoint global 21,888. Requested run budget: 6,144 decisions.
- Snapshot limited to `optimizer_updates.jsonl` updates **137–141**, global **21,889–22,528**: **640 completed optimized decisions / 100 optimizer steps**. Every update records changed actor parameters. All later rows were excluded, even if already present during analysis.
- Source rows: `residual_and_projection_audit.jsonl`, specifically `applied_audit.reward_breakdown`, `semantic_task.physical_evaluator`, and native tick summary. Phase counts: **P06 303, P07 1, P08 1, P09 335**. All 5,120 credited physics ticks have verified native audit. No terminal or success occurs inside this snapshot. Teacher reset motion is not among these credited global rows.

## The new credit reaches reward exactly once

All 640 rows retain measured FR/FL placement while RR/RL remain unplaced. Independently reconstructing the physical potential, including workspace, available support/unload, initial clearance and `0.99 * 0.008 / (0.008 + max(0, -clearance))`, agrees with logged `potential_after` to **1.11e-16** maximum error.

For all 640 rows, each of the following recorded equations has maximum error **0**:

`potential_shaping = 5 * (0.995 * potential_after - potential_before)`

`families.task_progress = potential_shaping + terminal_event - 0.02 * elapsed_physics_s`

`reward_breakdown.total = sum(families)`

The top-level training reward differs from the double-precision breakdown by at most **6.84e-9**, consistent with its float32 adapter conversion. This verifies the logged reward path, not a separate reload of the saved rollout tensor.

P09 has **25 decision-end RR soft-earned samples**, all initial=true, AIR, ground=false, obstacle=false, qualified=false and cross=false. Their clearance range is **−48.913 to −37.715 mm** relative to obstacle top. This is real sub-top progress evidence, **not qualified lift, crossing, placement or suffix/full-task success**.

## Height direction versus discounted task reward

Among adjacent P09 samples in the same earned AIR segment, six rise and six descend. The examples below separate the RR lift contribution `phi_lift = (0.85/4)*0.25*0.75*lift_credit` from the entire task reward.

| Completed global pair | Clearance, mm | Change in phi_lift | Discounted lift contribution to task reward | Actual destination task reward |
| --- | --- | ---: | ---: | ---: |
| 22370 → 22371 | −47.434 → −40.478 | +0.000816847 | +0.003921498 | −0.009431366 |
| 22277 → 22278 | −37.715 → −41.637 | −0.000545399 | −0.002885933 | −0.016238797 |

For both pairs, the change in total phi equals the change in phi_lift to rounding precision: there is no height plateau in these observed pairs. Nevertheless, the rising pair's total task reward remains negative because the existing gamma discount on retained total progress and elapsed-time cost outweigh that increment. This is not evidence of a reversed height signal or duplicated penalty. Conversely, other physical terms can change: at 21936 → 21937 clearance rises while total phi falls; height alone does not determine total reward sign.

First P09 soft acquisition is decision **22199**, time **50.600 s**, earned physics tick **6070**, clearance **−45.504 mm**. Its total task contribution is **+0.0823955**. This acquisition also includes existing initial-history progress; it must not be attributed wholly to the new height term.

## Actual weighted family scale

These are already weighted, signed contributions, not costs before weighting. P09 covers 22.3333 seconds of credited physics.

| Family | All 640 sum | P09 335 sum | P09 mean/decision |
| --- | ---: | ---: | ---: |
| Task progress | −8.240318 | −4.446002 | −0.013271647 |
| Body stability | −0.442094 | −0.336142 | −0.001003408 |
| Contact motion quality | −0.002213 | −0.001826 | −0.000005452 |
| Control smoothness | −1.765494 | −0.942003 | −0.002811950 |
| Control regularization | 0 | 0 | 0 |

All-family sum is **−10.450119**. On this prefix, smoothness is the largest non-task cost; contact cost is small. This is an observed scale comparison, not a weight recommendation or an inference from advantage signs.

The actual smoothness family equals `−0.1 * (actual_drive_first_difference + actual_drive_second_difference) / 2` from its time-integrated diagnostics to **1.31e-18** maximum error. Nominal and residual first differences are logged diagnostics, but are not added as second copies of this cost. All **25** P09 soft-earned samples have **zero confirmed_post_touchdown_rebound**; active AIR does not automatically incur rebound cost in these observations. Body/contact quantities are decision integrals over real 120 Hz samples; the compact log does not expose every raw physics sample for a separate causal reconstruction.

Updates 137–141 report PPO clip fractions **0.1234375, 0.3796875, 0.196875, 0.403125, 0.275**. These are optimizer ratio-clipping statistics, not physical actuator saturation rates.

## Limits

The fixed prefix demonstrates the new credit's live reward connection and signed height sensitivity. It does not establish policy improvement, preservation of above-top height after qualification, a paired causal comparison, or task success. Soft demand/response remains a temporal physical heuristic rather than proof of actuator work. Training continued unchanged; no new gate or tuning was introduced.
