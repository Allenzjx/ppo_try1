# CP226304 front-preserved deterministic video export — dormant

Do not run while the source writer is active. The checkpoint-side preflight is
complete; only the two sealed source hashes are intentionally pending.

Verified checkpoint identity:

- HEAD `892385cba8a7089b52567bb7f558c018fac82a77`
- checkpoint SHA256 `c0f4727bcc4e1963b5e9d079158dde1f5f662390397b61049ff893c40e1910f6`
- sidecar SHA256 `4d1e103507e7083f9d1ecb6d5ff230bd3d106054da2af0ef85e0ce81b9c8569b`
- lifetime `226304 / 1727 / 34540`
- branch delta from CP225280 `+1024 / +2 PPO / +40 Adam`
- front replay `40 minibatches / 1280 row exposures / 0 on-policy
  samples / 0 separate AUX optimizer steps`
- last complete update contains its actual `20` minibatches and `640` row
  exposures
- no new AUX; original f6d source AUX lineage remains independently labelled

After natural seal and process exit:

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$py = 'C:\Users\kskzz\miniconda3\python.exe'
$source = Join-Path $repo 'runs\ppo_rr_rl_timing_policy_learning_v1\video_eval\validation\20260924T1641350708339Z_g892385cba8a7_de3f11b3a669411abf3e6658002008cc\source'
$runManifest = Join-Path (Split-Path $source -Parent) 'run_manifest.json'
$sourceManifest = Join-Path $source 'semantic_video_source_manifest.json'
$sourceManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceManifest).Hash.ToLower()
$runManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $runManifest).Hash.ToLower()

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $py (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\export_front_preserved_video.py') `
  --source $source `
  --destination (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\video_review\CP226304_deterministic_front_preserved_v1_review') `
  --source-manifest-sha256 $sourceManifestSha `
  --source-run-manifest-sha256 $runManifestSha `
  --expected-head '892385cba8a7089b52567bb7f558c018fac82a77' `
  --checkpoint (Join-Path $repo 'outputs\ppo_rr_rl_timing_policy_learning_v1\branches\cp225280_front_preserved_v1\checkpoints\history\checkpoint_step_000226304.pt') `
  --checkpoint-sha256 'c0f4727bcc4e1963b5e9d079158dde1f5f662390397b61049ff893c40e1910f6' `
  --checkpoint-manifest-sha256 '4d1e103507e7083f9d1ecb6d5ff230bd3d106054da2af0ef85e0ce81b9c8569b' `
  --expected-global-policy-decisions 226304 `
  --expected-ppo-updates 1727 `
  --expected-optimizer-steps 34540 `
  --rear-owner-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\rear_owner_CP225280_gf6d1d2df8d87_publication.json') `
  --rear-owner-publication-sha256 '161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267' `
  --rear-owner-plan-sha256 'f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af' `
  --front-preservation-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\staged_cp225280_front_branch\front_preserved_CP225280_g892385cba8a7_publication.json') `
  --front-preservation-publication-sha256 '29bd2ba487088f98d76944566959057a8ffb280ab72ae51eae47f65eff0e0379' `
  --front-replay-dataset (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\staged_cp225280_front_branch\CP225280_front_replay_dataset.json') `
  --front-replay-dataset-sha256 'd78686d75a8ab69e492d6504340030992891b64ab279a19de374dc91233d2e38'
```

The exporter derives SUCCESS/INCOMPLETE, RR placement and RL reachability from
the sealed same-episode evidence. It retains the complete failure tail, uses
normal15fps, and labels historical N as a frozen non-fresh comparison.
