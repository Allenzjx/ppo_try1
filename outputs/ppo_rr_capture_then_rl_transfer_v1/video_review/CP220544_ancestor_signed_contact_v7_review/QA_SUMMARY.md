# CP220544 ancestor signed-contact v7 media QA

The sealed deterministic episode remained a physical task failure: P09 ended at
tick 14715 / 122.625 seconds with `INCOMPLETE_CONTROLLER_BLOCKED` from
`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`. The outer run lifecycle is
`DIAGNOSTIC_FAILURE`, and the outer source-acceptance text is
`SemanticVideoError: episode did not meet common physical task`. These are
separate fields. The existing export receipt is preserved; its empty
`semantic_terminal_reason` resulted from an older direct `step_info` lookup.
`terminal_reason_QA.json` records the sealed nested
`step_info.semantic_task` values.

At the terminal sample RR clearance was −0.010906366 mm, current contact was
AIR with 0 N obstacle-normal force, no TOP contact or RR placement had occurred,
and neither P10 nor RL was reached. The signed-band permission allowed the
bounded search to continue to 53 degrees; it did not infer contact, bearing, or
success and did not introduce a weak-contact controller.

## Rendered media

| Artifact | Frames | Duration | SHA-256 |
|---|---:|---:|---|
| `CP220544_DET_v7_full_attempt_INCOMPLETE.mp4` | 1840 | 122.666667 s | `5dcebcdc0a0a58292bc071e4d452106af6899088e484a32e63737e9e632fe6d4` |
| `CP220544_DET_v7_RR_window_RL_not_reached_detail.mp4` | 804 | 53.600000 s | `e8ac907f270dd7b564b3c08b7e2ad2ae5d9f5f93b3ce03923a26c0c720a7f2a4` |
| `historical_N_vs_CP220544_DET_v7.mp4` | 1840 | 122.666667 s | `157a2180f09336f9937e8c22f3641822338cf555417978dc45633dd723ae2899` |

All three outputs fully decoded at 15 fps with monotonic continuous PTS and
zero detected black frames. The detail is the contiguous same-episode interval
`[1036,1840)`, ticks 8296–14715, and retains the full terminal tail. The N
comparison explicitly marks its reference as historical/not a fresh B and
freezes it after its own endpoint; no same-controller/runtime claim is made.

Visual review of the decoded first/terminal full frames, detail onset, and pair
terminal frame found the whole robot and HUD legible. The terminal HUD shows
P09, `BLOCKED / finite_search_travel_or_margin`, travel `40+12+1` degrees,
AIR, no TOP/placement, and no P10/RL, with all four wheel source/target/measured
channels visible.
