# Sealed N+0 refinement — task incomplete, video complete

The full labelled video is `zero_Nplus0_full_review.mp4`; the unmodified-frame remux is `zero_Nplus0_full_native.mp4`. Both passed full decode and native cadence checks: 1108 frames at 15 fps, encoded 73.8667 s for 73.8083 s of actual physics. The final partial frame interval is disclosed, not extra physical settling. No source video or historical successful zero was changed.

## Actual outcome

The source retained `DIAGNOSTIC_FAILURE`, `INCOMPLETE_CONTROLLER_BLOCKED`, `POST_COMPLETION_LOSS`: not currently controlled at the fixed post-completion endpoint. All four placements were recorded and final region/support remained valid; that does not turn incomplete controlled stopping into success.

At tick8857, measured RR wheel angular velocity was -0.370923 rad/s against the unchanged absolute limit 0.25. All four wheel commands were zero. Body speed 0.012815 m/s and angular speed 0.061457 rad/s were inside 0.05 / 0.30 limits. Region-loss latch was false. No smoothing or endpoint replacement was used.

Source home was requested at observation8737, first physically dispatched at8738. Its eight-servo target was [0.5, -0.7, 3.7, 0.4, -2.6, -3.9, -0.5, -6.0] degrees, not eight zeros. Terminal maximum error against this actual source target was 5.437734 degrees (FL knee). The evaluator's separate legacy zero-home error was 8.115364 degrees; these are different references. Neither was substituted for physical stopping.

| Wheel | Final1s >0.25 rad/s | Final0.5s >0.25 rad/s |
| --- | ---: | ---: |
| FL | 0/120 | 0/60 |
| FR | 0/120 | 0/60 |
| RL | 3/120 | 0/60 |
| RR | 18/120 | 17/60 |

This is repeated RR threshold crossing after home, not a sole anomalous terminal sample. First all-three-speed-threshold control after home dispatch was tick8779; the longest uninterrupted segment was8779–8798 (20 samples, 0.1667 s sample exposure, 0.1583 s endpoint span). Home-entry body speed rose from0.02788 to a peak0.31342 m/s, with angular-speed peak0.46232 rad/s, before settling. These are time-aligned observations, not an isolated causal attribution to a motor or load path.

Full per-tick exceedance intervals and 0.1s source-home/actual traces are in `zero_terminal_motion_review.json`; raw compact final2s data are in `zero_review_aggregate.json`.

## Camera review

All fifteen decoded keyframe images `camera_key_01.png` through `camera_key_15.png` were visually inspected, covering start, FR/FL lift/cross/place, RR lift/place, RL lift/cross/place, P13 entry, source-home entry and terminal. The `.media.json` binds each to its actual ledger frame/tick, including 0–7tick event-to-frame offsets.

The new framing fixes the former lower-edge clipping of near-side FR at the end. The robot's outer silhouette stays inside the image in all inspected keyframes, and the obstacle front edge, near-side RR motion and visible home recovery are readable. Labels are clear and do not cover the original rendered image.

Limitation: the far-side RL wheel/contact point is still partly self-occluded by the robot body at several keyframes. This is improved framing, **not** proof that all four contact points are visible every frame. Keyframe review is also not a claim of manually inspecting all1108 frames. The full video and same-tick physical contact records remain available; no hidden geometry or synthetic view was invented.
