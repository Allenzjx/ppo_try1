# Prepared export workflow (no new run or video claimed)

`sealed_rr_review.py` processes **only explicitly named finalized sources**. It does not poll, simulate, load tensors, call a policy, optimize, or edit original source videos. Its tests use wiring fixtures, not synthetic physical-success evidence.

After root confirms that a formal `residual_rr_fix_v1` evaluation has sealed:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs/video_review_v1/sealed_rr_review.py --source '<sealed C run>\source' --name '<unique CP and run name>' --zero-native-receipt '<new sealed v2 zero export>\zero_Nplus0_full_native.media.json'
```

The optional zero receipt is the **native** B0 receipt emitted by the existing `sealed_zero_review.py`, not its labelled review receipt. No output directory or media is overwritten.
It may be a sealed TASK_INCOMPLETE zero; no zero-success prerequisite exists. Omit this optional argument to export the formal C attempt independently of zero or comparison readiness.

Outputs in the new directory:

- `PPO_full_native.mp4`: unchanged decoded frame/PTS/keyframe stream-copy of the entire actual single episode.
- `PPO_full_actual_wheels_RR.mp4`: original 1280×720 viewport plus an external 160px bottom panel. Native-axis measured wheel angular velocities are individually labelled FL/FR/RL/RR, explicitly distinguished from wheel-center displacement. RR gap, unsupported continuous-air gain, current-lift flag and contact mode use the exact task-observation tick. The nominal source readiness check displays its own tick separately. Missing values stay N/A.
- `PPO_latest_failure_context.mp4` on recorded non-success: same episode, normal speed, final relevant phase context (12–30 seconds when available), preserving the actual endpoint. The full failed attempt remains available.
- Optional `zero_full_actual_wheels_RR.mp4` and `zero_vs_PPO_same_camera.mp4`. Each short side freezes only **after** its actual last frame and shows `END - frozen last frame, not new simulation` outside the robot view.
- Source-bound `.frame_data.json`/`.media.json`, decoded camera keyframes at observed leg events plus start/P13/end, and `review_export_receipt.json`.

Every output is 15fps, ≤200s, fully decoded and checked for exact N/15 PTS. Source ticks are mapped from the actual frame ledger; no speed change, phase synchronization, cross-run splice, generated scene, or missing-data interpolation is used. Original video frames are not cropped. Comparison scales each complete viewport/panel proportionally.

## Limits and disclosure

Formal learned labels require the actual immutable checkpoint load binding and positive `rr_task_branch_counts` for decisions, PPO updates, and optimizer steps—not merely a large lifetime counter or an initialization checkpoint. All native-tick residual permission masks are checked before labelling FULL12. Baseline raw/projected residuals are checked as zero.

The comparison presently requires equal committed code inventory/build and camera/seed, and identical five non-stage configuration hashes. It permits **only** the recorded zero-to-RR stage-spec changes: revision, free-air RR qualification, and current-lift source readiness. Both must use terminal home v2. Consequently it is a descriptive same-camera/physics comparison, **not identical nominal scheduling or identical task acceptance** and not a claim of measured initial-state equality. Any additional code/config difference stops the comparison for explicit review instead of being silently waived.

Full decode is not a camera-occlusion or causal-mechanism certificate. Inspect all emitted event keyframes and the RR/failure interval before claiming visibility or valid traversal. A recorded success label is preserved as recorded; the helper does not independently re-adjudicate wheel-only-climb causality. Zero's unsupported-air-gain field may legitimately be N/A because its older RR semantics do not record the new residual-only diagnostic.

Angular velocity provenance: labels read `height_diagnostics.jsonl.joint_velocity_native_rad_s.value` by startup native joint-name indices (`robot.data.joint_vel`). The separately saved `physical_observations.jsonl.wheels.*.velocity_rad_s` values are canonical and are **not** labelled native. Every selected frame cross-checks native = canonical × `[-1,+1,-1,+1]` for FL/FR/RL/RR. Both original value sets and native actuator targets remain in the per-frame JSON, with no unit conversion from rad/s.

The old sealed zero exporter remains reusable unchanged for the forthcoming v2 zero. No current run is credited or exported by this preparation.
