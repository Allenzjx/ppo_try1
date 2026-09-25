# CP229120 reward-only72e deterministic export — dormant until seal

Do **not** run this command while Isaac PID `41560` / root exec `81629` or the
source writer is active. The source path below is known, but its run/source
manifest hashes must only be read after the process exits and both manifests
are sealed. No outcome is asserted here.

The immutable checkpoint sidecar is structurally accepted by the existing
`export_rl_recovery_video.py`; no exporter edit is needed:

- checkpoint: `checkpoint_step_000229120.pt`;
- checkpoint SHA-256:
  `a02bfd50e17cad3bebe083493b0da9611cfe607fa18dad6a4ed83abdecb91f1c`;
- sidecar SHA-256:
  `7c2b8958f695387f9c186978ec6a4b80df496220cfbd4d0eabfe62dc6e683036`;
- lifetime counters: `229120 / 1755 / 35100`, round-trip true;
- turn credit from CP225280: `+3840 / +30 / +600`, new AUX `0`;
- exact runtime: `72e63592bdf412d375abfda29b03cf456fbf4f7e`;
- policy/layout remain `rear_owner_recovery_history_v1` /
  `role422_rear_owner_recovery_v1`, 439 observations, raw12;
- route remains `ancestor220544_recapture_v2`;
- inherited f6d owner439 and zero-credit 72e reward-only publications remain
  mandatory and separately visible. The reward-only boundary does not claim a
  controller, distribution, HISTORY, physics, or AUX change.

After explicit seal/process-exit authorization, run exactly once:

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$python = 'C:\Users\kskzz\miniconda3\python.exe'
$run = Join-Path $repo 'runs\ppo_rr_rl_timing_policy_learning_v1\video_eval\validation\20260924T1008401445075Z_g72e63592bdf4_3f1b5877a524484aab1eecca97694515'
$source = Join-Path $run 'source'
$sourceManifest = Join-Path $source 'semantic_video_source_manifest.json'
$runManifest = Join-Path $run 'run_manifest.json'
$checkpoint = Join-Path $repo 'outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000229120.pt'
$destination = Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\video_review\CP229120_deterministic_rr_retention_72e_review'

# Evaluate only after the sealed notification and PID41560 exit.
$sourceManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceManifest).Hash.ToLowerInvariant()
$runManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $runManifest).Hash.ToLowerInvariant()
$env:CUDA_VISIBLE_DEVICES = '-1'

& $python -B (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\export_rl_recovery_video.py') `
  --source $source `
  --destination $destination `
  --source-manifest-sha256 $sourceManifestSha `
  --source-run-manifest-sha256 $runManifestSha `
  --expected-head 72e63592bdf412d375abfda29b03cf456fbf4f7e `
  --checkpoint $checkpoint `
  --checkpoint-sha256 a02bfd50e17cad3bebe083493b0da9611cfe607fa18dad6a4ed83abdecb91f1c `
  --checkpoint-manifest-sha256 7c2b8958f695387f9c186978ec6a4b80df496220cfbd4d0eabfe62dc6e683036 `
  --expected-global-policy-decisions 229120 `
  --expected-ppo-updates 1755 `
  --expected-optimizer-steps 35100 `
  --rear-owner-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\rear_owner_CP225280_gf6d1d2df8d87_publication.json') `
  --rear-owner-publication-sha256 161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267 `
  --rear-owner-plan-sha256 f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af `
  --rr-retention-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\rr_retention_CP226048_g72e63592bdf4_publication.json') `
  --rr-retention-publication-sha256 8c2fd9193ab6d3088cffcd620f9577c660fc2904d11ca2ffe23d5dc76b3a4402 `
  --rr-retention-plan-sha256 e1050240996e1669731a041f339924ef5ff7eba68983e628b806368e0b5dff9d
```

The exporter derives the truthful terminal/detail plan from this one sealed
episode. It must retain the continuous failure tail when incomplete, emit an
explicit predecessor-failure detail if RR/RL is not reached, and keep the N
comparison labelled as historical/frozen rather than a fresh same-version B.

