# CP225280_local000000 direction-diagnostic export

Prepared only; execute after both `source/source_manifest.json` and the parent
`run_manifest.json` exist naturally and the source recorder has closed.

- Runtime HEAD: `ecf205e80094693938057589fc91f7f31082ef89`
- Composite checkpoint: `checkpoint_CP225280_local000000.pt`
- Checkpoint SHA-256: `9c0188cf1b8c31cbad393d65684f43323ad14a1cc2268b934131dfcd12ee6d82`
- Sidecar SHA-256: `8d0fd893427b443aaeb89a45fbbac6f8dc2fa417874eae5c2e6ceff89c13998b`
- Local learning counts: `0 decisions / 0 PPO / 0 Adam`
- Mode: independent direction diagnostic, PPO credit zero
- Comparison: accepted original49eb CP225280, not N

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$source = "$repo\runs\ppo_rr_capture_first_cp225280_v1\diagnostic_hipminus25_kneeplus20_ecf205e\source"
$destination = "$repo\outputs\ppo_rr_capture_first_cp225280_v1\video_review\CP225280_local000000_direction_diagnostic_review"
$checkpoint = "$repo\outputs\ppo_rr_capture_first_cp225280_v1\checkpoints\history\checkpoint_CP225280_local000000.pt"
$sourceSha = (Get-FileHash -Algorithm SHA256 -LiteralPath "$source\source_manifest.json").Hash.ToLowerInvariant()
$runSha = (Get-FileHash -Algorithm SHA256 -LiteralPath "$source\..\run_manifest.json").Hash.ToLowerInvariant()
$env:CUDA_VISIBLE_DEVICES = '-1'
& 'C:\Users\kskzz\miniconda3\python.exe' `
  "$repo\outputs\ppo_rr_capture_first_cp225280_v1\export_rr_capture_first_video.py" `
  --source $source `
  --destination $destination `
  --source-manifest-sha256 $sourceSha `
  --source-run-manifest-sha256 $runSha `
  --expected-head ecf205e80094693938057589fc91f7f31082ef89 `
  --checkpoint $checkpoint `
  --checkpoint-sha256 9c0188cf1b8c31cbad393d65684f43323ad14a1cc2268b934131dfcd12ee6d82 `
  --checkpoint-manifest-sha256 8d0fd893427b443aaeb89a45fbbac6f8dc2fa417874eae5c2e6ceff89c13998b `
  --expected-local-policy-decisions 0 `
  --expected-local-ppo-updates 0 `
  --expected-local-optimizer-steps 0 `
  --allow-diagnostic
```

Expected derived labels contain `DIAGNOSTIC`, `PPO CREDIT 0`, frozen
`CP225280` prior and `local0`. Any observed local physical success remains a
diagnostic result, not learned-PPO success and not full-task success.
