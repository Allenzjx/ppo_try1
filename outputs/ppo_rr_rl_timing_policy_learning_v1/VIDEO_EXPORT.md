# Rear-policy video export

`export_policy_rear_no_assist_video.py` is an outputs-only postprocessor. It
does not import the live control/migration implementation, start Isaac, load a
model, or alter the sealed source. It accepts only a closed deterministic
natural-P01 `rr_rl_timing_policy_learning_v1` source whose manifest says:

- front FL capture assist is `p05_hip_only_continuation_v1` and is not PPO;
- rear task assist, RR capture assist, rear wheel projection, and nominal
  geometry advisory are off;
- nominal timing is `rr_capture_before_rl_transfer_v1`;
- the saved 419-column policy checkpoint and lifetime/branch counters match
  the explicit command-line hashes and counts;
- the new branch added no AUX updates, while the two inherited AUX ledgers
  remain migration-hash-bound and separately reported.

After root reports the final checkpoint and sealed video hashes, run once from
the repository root with the locked CPU environment (placeholders must be
replaced by actual values):

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$source = '<absolute-sealed-source>'
$checkpoint = '<absolute-checkpoint-loaded-by-this-video.pt>'
$sourceManifestSha256 = '<source-manifest-sha256>'
$runManifestSha256 = '<run-manifest-sha256>'
$checkpointSha256 = '<checkpoint-sha256>'
$checkpointManifestSha256 = '<checkpoint-manifest-sha256>'
$step = 0          # replace with the actual positive checkpoint counter
$ppoUpdates = 0    # replace with the actual lifetime counter
$optimizerSteps = 0 # replace with the actual lifetime counter
$historicalN = Join-Path $repo 'runs\ppo_task_conditioned_hip_wheel_v1\video_eval\prior_B\20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1\source'
Set-Location -LiteralPath $repo
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = Join-Path $repo 'src'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' `
  (Join-Path $repo 'outputs\ppo_rr_rl_timing_policy_learning_v1\export_policy_rear_no_assist_video.py') `
  --source $source `
  --destination (Join-Path $repo "outputs\ppo_rr_rl_timing_policy_learning_v1\video_review\CP${step}_deterministic_rear_no_assist_review") `
  --source-manifest-sha256 $sourceManifestSha256 `
  --source-run-manifest-sha256 $runManifestSha256 `
  --expected-head '<capture-evaluation-HEAD>' `
  --checkpoint-runtime-head 'fa4b98ed506eb2230e61e13bd93ceecdcb6cfdad' `
  --checkpoint $checkpoint `
  --checkpoint-sha256 $checkpointSha256 `
  --checkpoint-manifest-sha256 $checkpointManifestSha256 `
  --expected-global-policy-decisions $step `
  --expected-ppo-updates $ppoUpdates `
  --expected-optimizer-steps $optimizerSteps `
  --expected-new-auxiliary-updates 0 `
  --historical-n-source $historicalN
```

`--checkpoint` is path-bound as well as hash-bound: use the exact path in
`semantic_video_source_manifest.json -> checkpoint_load_provenance.source.checkpoint`,
not an equivalent `latest` pointer or copied file. Its sidecar is resolved as
`<checkpoint stem>_manifest.json`. Source/run hashes must be taken only after
both manifests are closed. The destination must not already exist.

If—and only if—the sealed source reports the exact
`viewport callback_count=0, expected 1` artifact abort, append
`--diagnostic-capture-abort`. That path requires `DIAGNOSTIC_FAILURE`, a
finalized fully decodable continuous ledger prefix, and exactly one missing
final render interval. It emits a visibly labelled `CAPTURE_ABORT` prefix and
available-tail detail. It never calls that prefix a task success/failure, never
uses the unrendered endpoint as video evidence, and never fills the missing
frame. Without the explicit flag, artifact failures remain rejected. A launch
failure before source creation has nothing to export and is not accepted.

Outputs are immutable and use actual outcome-dependent names:

- `CP<step>_DET_full_policy_rear_no_assist[_INCOMPLETE].mp4`;
- `CP<step>_DET_RR_capture_to_RL_detail.mp4` only for an actual successful
  episode; an incomplete episode that reached the RR window instead gets
  `CP<step>_DET_RR_attempt_to_actual_tail_INCOMPLETE.mp4`, while an episode
  that never reached RR gets an explicitly named
  `RR_NOT_REACHED_<phase>_failure_detail` from this same episode;
- `N_vs_CP<step>_DET_same_camera.mp4`, with historical N and either ended side
  visibly frozen as non-physical comparison padding.

The export receipt retains full decode/PTS/frame/checksum/black-frame QA,
source and checkpoint hashes, actual terminal result, full failure-tail proof,
assist split, PPO/AUX credit split, and detail availability reason.
`placed_hist` in the HUD is explicitly historical evaluator state; a transient
RR TOP/placement followed by AIR is not labelled as capture success.

For the reviewed media-only capture repair, `--expected-head` is the committed
evaluation/capture runtime recorded by the sealed source, while
`--checkpoint-runtime-head` remains the checkpoint's original `fa4b98ed...`
runtime. The exporter accepts that split only when the live source contains the
exact `semantic_video_media_only_checkpoint_compatibility.v1` receipt: both
runtime identities, the two reviewed media-file hashes, all other runtime
files, all seven selected configurations, and the official checkpoint/sidecar
binding must agree. This exception is video-eval-only and is not a weight,
control, MDP, or policy-distribution migration.

For later exact-runtime recapture-v2 checkpoints, omit
`--checkpoint-runtime-head`; it defaults strictly to `--expected-head` and no
media compatibility receipt is expected. Such checkpoints are accepted only
when `rear_recapture_migration` proves the registered initial419 ancestor,
zero-update identity migration, current target runtime and
`rr_recapture_current_support_v2`, and when `checkpoint_output_routing` binds
the checkpoint to the isolated `ancestor220544_recapture_v2` branch with no
main-pointer promotion. The original rear-policy migration, inherited AUX
ledgers, rear-assist OFF/front-FL-assist ON contract, policy shape and actual
checkpoint hashes remain mandatory.

The exact-runtime recapture-v2 invocation therefore uses the same frozen
commit for the sealed source and checkpoint and adds no media-compatibility
flag or receipt:

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' `
  (Join-Path $repo 'outputs\ppo_rr_rl_timing_policy_learning_v1\export_policy_rear_no_assist_video.py') `
  --source $source `
  --destination (Join-Path $repo "outputs\ppo_rr_rl_timing_policy_learning_v1\video_review\CP${step}_deterministic_recapture_v2_review") `
  --source-manifest-sha256 $sourceManifestSha256 `
  --source-run-manifest-sha256 $runManifestSha256 `
  --expected-head '44219b4fdc4d36d33be489b833c03b897766045b' `
  --checkpoint $checkpoint `
  --checkpoint-sha256 $checkpointSha256 `
  --checkpoint-manifest-sha256 $checkpointManifestSha256 `
  --expected-global-policy-decisions $step `
  --expected-ppo-updates $ppoUpdates `
  --expected-optimizer-steps $optimizerSteps `
  --expected-new-auxiliary-updates 0 `
  --historical-n-source $historicalN
```

The checkpoint path must be beneath
`branches\ancestor220544_recapture_v2\checkpoints\history`; the sealed
checkpoint's `checkpoint_output_routing` must point to that exact branch and
must set `main_latest_pointer_promotion` to false.

The later live-swing v3 continuation remains in that same isolated branch.
For it, use the sealed evaluation/checkpoint runtime as `--expected-head` and
continue to omit `--checkpoint-runtime-head`. The exporter first reverses the
finite `rear_live_swing_migration` delta to reconstruct the exact 44219 parent
runtime and validate the inherited v2 recapture receipt; it then validates the
v3 receipt, learned-source CP221568 binding, zero migration learning, preserved
route, and actual descendant counters. The source manifest must report
`rr_live_swing_evidence_v3` while rear task assist remains OFF and FL capture
assist remains ON. This is distinct from the optional a982 media-only load
compatibility receipt, which remains forbidden when source and checkpoint use
the same runtime.
