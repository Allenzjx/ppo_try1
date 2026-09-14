# Video3: incomplete-task diagnostic copy

Verified 2026-09-10 12:00:39 UTC. This is the natural-P01 video evaluation of checkpoint 145920, not a successful PPO crossing and not a success publication.

[Play the diagnostic video](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video3_145920_p01_incomplete.mp4)

The original viewport capture decoded correctly but reported an invalid MP4 container duration of 286,331,153 seconds. A separate `ffmpeg -c:v copy -movflags +faststart` output now reports 50.53 seconds (FFmpeg display precision), below 200 seconds. No re-encoding, trimming, concatenation, interpolation, speed change, or simulation was performed.

Both files fully decode to the same 758 unique H.264/yuv420p frames at 1280×720 and 15 fps, with zero black-like frames. The complete decoded frame-checksum, PTS, PTS-delta, and key-frame sequences match exactly. Decoded media duration remains 50.533333 seconds; the physical episode was 6060 ticks / 50.5 seconds. The 0.033333-second difference is the existing last-frame quantization, not added physics or a timing edit.

The original source remains in its run directory and its before/after SHA256 is unchanged: `2c770e40365925c1a829a893c5518b152bd614bf26efac8a7438c4582d5afc79` (71,009,845 bytes). The diagnostic copy is `a569c7353afea2c949a96e436788099832046831baadd3cea639bc7b1385c69d` (70,922,669 bytes); changing container metadata necessarily changes the whole-file hash, while frame content and timing remain identical.

The source task verdict is unchanged: `physical_task_success=false`, `diagnostic_only=true`, with `SemanticVideoError: episode did not meet common physical task`. No success-publishing function was called. This video provides failure/incompletion evidence only; it does not establish full-task stability superiority over FSM.

[Verification receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video3_145920_p01_incomplete.remux_receipt.json) records the exact tool arguments and hashes. The output-only reproduction script refuses to overwrite an existing diagnostic file. Production, four main reports, checkpoint files, and active training processes were not modified.
