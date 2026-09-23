# CP220544 ancestor contact-onset v6 media QA

This directory is an outputs-only rendering of the sealed deterministic natural-P01 evaluation source
`20260923T0829436381354Z_g97c4367ee293_5795b2a4e4384ec6be3589fa43477a5c`.
It is not a success artifact and records zero new PPO updates for the v6 migration/evaluation.

## Actual outcome

- Physical duration: 121.666667 s; final physics tick: 14600; terminal phase: P09.
- Semantic terminal reason: `INCOMPLETE_CONTROLLER_BLOCKED`.
- Terminal source: `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`.
- The outer source-acceptance text is `SemanticVideoError: episode did not meet common physical task`; this is distinct from the semantic terminal reason above.
- RR qualified and crossed, but never obtained TOP contact or placement; P10 and RL were not reached.
- Terminal RR clearance: -0.003537004 mm; contact state: AIR; obstacle-normal force: 0 N.
- Assist remained `DESCEND_PROGRESS`; travel was 52.041667 degrees, below the 53-degree finite-search limit. The run therefore ended at the finite task deadline, not because the search-travel budget was exhausted.

## Rendered media

| Artifact | Frames | Duration | SHA-256 |
|---|---:|---:|---|
| `CP220544_DET_v6_full_attempt_INCOMPLETE.mp4` | 1825 | 121.666667 s | `25ed70822115aab094fdc2c21fd78d38e212b829042cfa4c5e7331b2fe50e699` |
| `CP220544_DET_v6_RR_window_RL_not_reached_detail.mp4` | 789 | 52.600000 s | `eb8eb698a1a9c107d18cc4ef4ff2c2cf99979b0b73549c828b4c68719c87e8b7` |
| `historical_N_vs_CP220544_DET_v6.mp4` | 1825 | 121.666667 s | `bc0de967539164b06e0c1a47773590ce888da7514ccf1d006cb142dd6050b8a6` |

All three videos fully decoded at 15 fps with monotonic, continuous PTS and zero detected black frames. The detail is the contiguous same-episode interval `[1036, 1825)`, corresponding to ticks 8296 through 14600, and retains the complete failure tail. The comparison explicitly identifies N as historical, not a fresh B or same-controller evaluation, and freezes the 73.808-second N reference for its remaining duration.

## Terminal sensor evidence

- First RR ownership: tick 9516 / 79.300000 s, clearance 27.865899 mm, AIR, `BLOCKED`, travel 0 degrees.
- First RR crossing: tick 9555 / 79.625000 s, clearance 27.510446 mm, AIR.
- First contact-onset increment: tick 14596 / 121.633333 s, clearance +0.020992 mm, AIR, `DESCEND_PROGRESS`, travel 52.008333 degrees, onset increment 0.008333 degrees.
- Terminal: tick 14600 / 121.666667 s, clearance -0.003537004 mm, front distance 20.146016 mm, AIR/no bearing/no placement, travel 52.041667 degrees, onset increment 0.041667 degrees.

Machine-readable provenance, decoder checks, and selected four-wheel evidence remain in `export_receipt.json` and `CP220544_v6_selected_fourwheel_evidence.json`; neither sealed input nor rendered receipt was rewritten after export.
