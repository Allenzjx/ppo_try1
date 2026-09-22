# Block 03: sealed natural-P01 training audit

Run `20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff`, runtime `a802b24d78df`. Independent CPU reads/counts and Gaussian arithmetic only: no Isaac, policy inference, runtime replay, production edits or checkpoint writes.

## Completed work

- **+2048 policy decisions / +16 PPO updates / +320 optimizer steps**, exact requested budget; all sampled from natural P01, no frozen/teacher prefix.
- P05-branch cumulative **+6144 / +48 / +960**. Feedback-v2 branch now has real learning **+2048 / +16 / +320**; its preceding migration itself added zero.
- CP205824 lifetime **205824 decisions / 1573 updates / 31460 optimizer steps**.
- [checkpoint_step_000205824.pt](checkpoints/history/checkpoint_step_000205824.pt), independently hashed SHA256 `c1615e5d6987306d41dcdbbd4c3767bd0d28d30fda559d149e7c5b2ade886735`; matches manifest, recorded save/load round trip true.
- All 16 updates **1558–1573** have finite metrics, nonzero gradients, changed actor hashes and a continuous actor hash chain; 20 optimizer steps each. Identity normalization/389 inputs retained, LR `1e-5`. No new AUX.

## Exact P01–P13 sample counts

| Stage | Credited policy requests | Action endpoints |
|---|---:|---:|
| P01 | 6 | 3 |
| P02 | 601 | 603 |
| P03 | 4 | 4 |
| P04 | 1 | 1 |
| P05 | 176 | 176 |
| P06 | 197 | 197 |
| P07 | 1 | 1 |
| P08 | 1 | 1 |
| P09 | 432 | 432 |
| P10 | 1 | 1 |
| P11 | 167 | 167 |
| P12 | 461 | 462 |
| P13 | 0 | 0 |
| Total | 2048 | 2048 |

This block supplies this branch's first credited P10–P12 samples; P13 remains unobserved. Request phase is the action's original phase, not retrospectively replaced with the endpoint label.

## Actual physical-event learning coverage

| Leg | Input lift bit | Input crossed | Input placed | Endpoint lift equivalent | Endpoint crossed | Endpoint placed | Endpoint TOP |
|---|---:|---:|---:|---:|---:|---:|---:|
| FL | 1435 | 1297 | 1260 | 1436 | 1298 | 1261 | 549 |
| FR | 1831 | 1439 | 1437 | 1833 | 1440 | 1438 | 1429 |
| RL | 9 | 0 | 0 | 9 | 0 | 0 | 0 |
| RR | 554 | 974 | 630 | 554 | 975 | 631 | 84 |

Input counts come from all 16 saved rollout tensors. RR's lift bit means **current qualified lift**, whereas other legs use the schema's active-lift history. Placed is historical achievement, not present bearing. Of 2045 inputs matched to the immediately preceding credited endpoint, exact TOP-positive counts are FL549/FR1428/RL0/**RR84**. Exact TOP is not an explicit raw389 classifier; the three initial natural-P01 reset observations lack that exact selected-journal classifier and are not silently filled as zero.

All saved samples were actually used five times in their 20 official minibatches: therefore RR placed-input samples have **3150** optimization uses, RR crossed-input samples **4870**, known RR TOP-input samples **420**, and RL lift-bit samples **45**. These are learning coverage counts, not independent episodes or repeated physical events.

Capture assist owns FL hip/knee at 238 endpoints; all 12 residual permission masks remain enabled. Pending FL appears at 37 endpoints but scheduler-advanced-pending is zero: this training block does not exercise deadline-based pending handoff (the preceding CP203776 deterministic video did).

## Real episodes and task outcomes

| Episode | Decisions | Actual ticks / seconds | Result |
|---|---:|---|---|
| 0 | 1721 | 13765 / 114.708333 | P12 `INCOMPLETE_CONTROLLER_BLOCKED`; FR/FL/RR placed, RL not crossed/placed |
| 1 | 235 | 1877 / 15.641667 | P02 `INCOMPLETE_CONTROLLER_BLOCKED` |
| 2 | 92 | 736 / 6.133333 | Partial P02 at completed-update boundary; no terminal or success |

Total **16378** actual credited task ticks, all recorded native-tick verification passed; no prefix or in-episode state writes. The two terminal decisions each have five ticks, explaining the six-tick difference from 2048×8; settling/reset simulation is excluded.

Episode0 authoritative event ticks: FR lift/cross/place **26/2252/2268**; FL **2288/3390/3687**; RR **5496/5976/8726**; RL lift **10090**, no crossing/placement. Thus this is genuine new RR placement and RL-lift coverage, not full traversal. The later loss of usable RR support is detailed in [bounded handoff report](block03_RR_capture_handoff_readonly_11304.md).

## Storage checks and limitation

All 16 rollouts have `(128,1,389)` policy/critic observations and `(128,1,12)` original raw actions; finite, equal actor/critic observations, exact stored action/mean/std/logp/reward/terminal agreement with collector. One sampling draw and zero extra draws per decision. Independent CPU Gaussian logp maximum error **5.7220458984375e−6**. This verifies real optimizer use without replaying physics or claiming a better policy solely from successful training.

The trained checkpoint still needs its separately running saved/reloaded full-P01 deterministic evaluation. **No full-task success or stability superiority is established by this block.** Detail: [JSON audit](block03_training_audit.json). Reproducible reader: `audit_block03_cpu.py` (stdout only, no project runtime imports or file writes).
