# Actual CP221184 / v4 media export

Export completed successfully after the source simulator exited. This file supersedes the pre-execution status in the separate DRAFT notes; the executed script and immutable export receipt are unchanged.

Physical outcome: **P05 incomplete at tick 7288 / 60.733333 s; RR and RL windows were not reached.** The source's acceptance error is `episode did not meet common physical task`, not an encoding failure. This is not PPO task success or evidence that wheel-v4 solved RR capture. RR assist remained WAIT and no wheel-envelope milestone occurred.

| Output | Frames / duration | SHA256 |
|---|---|---|
| CP221184_DET_v4_full_attempt.mp4 | 911 / 60.733334 s | eaa0069c58ed63759a4a52be96ebdebff50abfc4df4ea75c22a7eafe4880e419 |
| CP221184_DET_v4_predecessor_failure_tail_detail.mp4 | 900 / 60.000000 s | b0bff26457c8dfdd22b72a1c11b81278dbe04d688e3947dfed1bb00e66b81a5a |
| historical_N_vs_CP221184_DET_v4.mp4 | 1108 / 73.866667 s | 5db3cdad7c57cca8eedeeea9b1dfa11e944481f842d7f2463e28ee8b2c88d6ca |

All three passed complete decode, 15 fps, monotonic continuous PTS and zero black-like frames. Full/detail dimensions are 1280×720; comparison is 1920×610. The full episode contains every source frame without an intro or speed modification. The detail is contiguous source frames [11,911), episode ticks 96–7288, including the actual terminal tail; it is explicitly labelled as a predecessor tail, not RR detail.

The comparison uses the accepted **historical** N_ref, visibly labelled `NOT FRESH B`. After candidate frame 910 ends, its 197 added comparison frames are visibly marked `RUN ENDED - FROZEN, NOT NEW PHYSICS`. No fresh same-controller paired-baseline claim is made.

Visual review inspected the full first frame, full terminal frame, full frame 750 (episode tick 6008 / 50.067 s), detail terminal frame, and comparison terminal frame. Scene and robot remain visible beneath the fixed top HUD; FL/FR/RL/RR canonical target and measured-qdot rows are legible, labels retain failure status, and the comparison freeze warning is visible. No unobserved movement, contact or native joint ID was added.

The migrated checkpoint remains SHA `2fed192f9c5141a40f769893d0266ec3b95c91d55a4b3091acc411cd99672f90`. Real prior training is 640 decisions / 5 PPO / 100 Adam steps; this evaluation and the control migration add no optimization credit. Historical limited AUX and FL/RR/support-wheel controllers are disclosed separately from policy learning.

`export_receipt.json` contains source/checkpoint/migration bindings and full validation records. `CP221184_v4_selected_fourwheel_evidence.json` contains the real terminal four-wheel evidence; missing RR milestones and native actuator IDs are not invented. The export Python process exited 0, and the final single-frame QA FFmpeg process exited 0. No simulator or production edit was performed during export.
