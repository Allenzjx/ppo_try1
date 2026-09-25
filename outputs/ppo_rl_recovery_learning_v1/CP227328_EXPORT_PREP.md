# CP227328 front-preserved deterministic video export — checkpoint verified, source active

This command is dormant. Do not read or export the source while its writer or
Isaac process is active. The checkpoint side is verified; the source/run
manifest hashes and episode result must come only from the natural sealed run.

Verified checkpoint identity:

- evaluation/runtime HEAD `892385cba8a7089b52567bb7f558c018fac82a77`
- checkpoint SHA256 `5b871af2df8d6b9c0180f7a51863b18b2153148d8dc2a94876f21f5bd73791b4`
- sidecar SHA256 `0200daa9a22652e1f1c64e5c5ff82a0c977f0418c508158455b3a5ce85bac308`
- lifetime `227328 / 1729 PPO / 34580 Adam`, with `save_load_round_trip=true`
- front-preservation branch delta from CP225280: `+2048 / +4 PPO / +80 Adam`
- front replay: `2560` row exposures, `80` minibatches, `0` on-policy
  samples, and `0` separate AUX optimizer steps
- route `cp225280_front_preserved_v1`; no new AUX
- front-preservation publication SHA256
  `29bd2ba487088f98d76944566959057a8ffb280ab72ae51eae47f65eff0e0379`
- immutable replay dataset SHA256
  `d78686d75a8ab69e492d6504340030992891b64ab279a19de374dc91233d2e38`

After root confirms natural seal and process exit, run exactly:

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$py = 'C:\Users\kskzz\miniconda3\python.exe'
$source = Join-Path $repo 'runs\ppo_rr_rl_timing_policy_learning_v1\video_eval\validation\20260924T1842131087811Z_g892385cba8a7_c074d1ebec6947cc830fb3e0ec86dcec\source'
$runManifest = Join-Path (Split-Path $source -Parent) 'run_manifest.json'
$sourceManifest = Join-Path $source 'semantic_video_source_manifest.json'
$sourceManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceManifest).Hash.ToLower()
$runManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $runManifest).Hash.ToLower()

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $py (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\export_front_preserved_video.py') `
  --source $source `
  --destination (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\video_review\CP227328_deterministic_front_preserved_v1_review') `
  --source-manifest-sha256 $sourceManifestSha `
  --source-run-manifest-sha256 $runManifestSha `
  --expected-head '892385cba8a7089b52567bb7f558c018fac82a77' `
  --checkpoint (Join-Path $repo 'outputs\ppo_rr_rl_timing_policy_learning_v1\branches\cp225280_front_preserved_v1\checkpoints\history\checkpoint_step_000227328.pt') `
  --checkpoint-sha256 '5b871af2df8d6b9c0180f7a51863b18b2153148d8dc2a94876f21f5bd73791b4' `
  --checkpoint-manifest-sha256 '0200daa9a22652e1f1c64e5c5ff82a0c977f0418c508158455b3a5ce85bac308' `
  --expected-global-policy-decisions 227328 `
  --expected-ppo-updates 1729 `
  --expected-optimizer-steps 34580 `
  --rear-owner-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\rear_owner_CP225280_gf6d1d2df8d87_publication.json') `
  --rear-owner-publication-sha256 '161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267' `
  --rear-owner-plan-sha256 'f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af' `
  --front-preservation-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\staged_cp225280_front_branch\front_preserved_CP225280_g892385cba8a7_publication.json') `
  --front-preservation-publication-sha256 '29bd2ba487088f98d76944566959057a8ffb280ab72ae51eae47f65eff0e0379' `
  --front-replay-dataset (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\staged_cp225280_front_branch\CP225280_front_replay_dataset.json') `
  --front-replay-dataset-sha256 'd78686d75a8ab69e492d6504340030992891b64ab279a19de374dc91233d2e38'
```

The exporter must derive the outcome and detail filename from the sealed
same-episode evidence. It retains the continuous normal-15fps episode and full
failure tail, and labels historical N as a frozen different-version reference.
No result is asserted here while the source remains active.
