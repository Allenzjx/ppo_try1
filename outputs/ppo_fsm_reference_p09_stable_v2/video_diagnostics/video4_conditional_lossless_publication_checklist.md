# Conditional video4 publication: container-only repair

Read-only review against production `7db0d17f398d393ce026b6990bd2566d53366407`. This note does not assert that video4 exists or has completed the task. No video validation, FFmpeg operation, test, simulation, production edit, or successful-file publication was run for this review.

## Existing path is sufficient for a container-only fault

The recorder already permits an otherwise valid raw capture with a corrupt duration atom: [video_capture.py:590](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/infrastructure/video_capture.py:590) calls full-decode validation with `require_sane_container_duration=False`. It preserves the raw MP4 and records the invalid container metadata honestly. Therefore this issue alone should not turn a physically successful, otherwise complete capture into a diagnostic-only source.

[validate_semantic_video_source:789](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_video.py:789) still requires genuine success candidate/non-diagnostic status, natural P01 proof, physical task success, zero optimizer updates, matching captured runtime/configuration, intact source artifact hashes, a successful independent replay of every physical observation, complete native-action and decision evidence, and a full-source frame/PTS ledger. Only raw container-duration strictness is waived at line916; decode, cadence, frame count, content and ≤3000-frame/200-second task-window checks remain.

[publish_success_source:994](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_video.py:994) calls that validator first, then packet-copies the original into a distinct non-existing output with `-copyts -start_at_zero -c:v copy -movflags +faststart`. It strictly validates the new container and verifies unchanged decoded frame checksums and exact PTS-delta sequence. It then writes a separate publication manifest binding the original source-manifest hash and output hash. Original source files/manifests are never rewritten. C is still `candidate_not_improved`, with `improved_claim=false`.

## Minimal operation after a real successful capture

1. Wait for capture finalization and Isaac exit. Keep the complete run/source directory untouched. Use the same clean production HEAD/config/runtime as the captured source: the validator reconstructs the contract and refuses a different current HEAD ([semantic_cli.py:88](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_cli.py:88)). Do not rerun `run_semantic_video.ps1` merely to post-process; that launcher starts another simulation.
2. Confirm the actual source manifest has both physical-success fields true, `success_candidate=true`, `diagnostic_only=false`, and no acceptance error. These are prechecks, not a replacement for independent validation. If any fails, do not edit it into eligibility.
3. In a small post-processing Python call, use the existing API below with the actual source directory and a fresh output subdirectory. The allowed C basename is `semantic_C_candidate.mp4` (also `ppo_success_clean.mp4`); use the candidate name to avoid implying a stability advantage. The full validator runs inside this call, so a separate duplicate replay is unnecessary.

```python
from pathlib import Path
from wlr50_clean.ppo.semantic_video import publish_success_source

# Resolve these only after the genuine run exists and qualifies.
record = publish_success_source(actual_success_source_directory,
                                fresh_output_directory / "semantic_C_candidate.mp4")
```

4. Add a small output-only identity receipt: original SHA/size before and after; full source/output `decode_frame_timeline` equality (frame index, **absolute** PTS, checksum, key flag); matching PTS-delta hash; and explicit `0 < container_duration_s <= 200`. Reuse the validation ideas, not the hard-coded run or failure-only assertions, in [video3 remux helper](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/remux_video3_diagnostic.py). Keep the native final partial-frame display quantization explicit; no trim, speed adjustment, extra hold or synthesized frame is needed.

## Limits / actual possible blockers

- Existing publication verification guarantees exact PTS **deltas**, not absolute timestamps or key-flag equality ([video_timeline.py:514](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/evaluation/video_timeline.py:514)); `-start_at_zero` permits rebasing. For the current zero-start capture, require absolute equality in the separate receipt before describing PTS as fully unchanged. Do not silently waive a mismatch.
- Generic MP4 validation allows a one-frame duration tolerance around the cap; source count is separately limited to3000. The extra explicit container≤200 check avoids overstating that tolerance as a strict numeric cap.
- True frame loss, failed encoder finalization, missing evidence, incorrect reset/load proof, runtime mismatch or failed physical replay is not a container-only repair. Physical success alone does not permit publishing a source already marked diagnostic-only; the existing encoder-failure regression explicitly preserves physical success while blocking publication. Leave it diagnostic and report the distinct artifact problem; no manifest edits to manufacture pass.
- A failed post-copy check may leave an output file without a valid publication manifest; the function refuses overwrite on retry. Preserve that failed attempt, use a fresh output directory, and do not deliver it as validated success. No code changes or new framework are needed for the expected container-only case.

Until full-task success actually occurs, there is no eligible C success source to publish. File-integrity work remains post-processing, never a training-start gate or a claim that PPO is more stable than FSM.
