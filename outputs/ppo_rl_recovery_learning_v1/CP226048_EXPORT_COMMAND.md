# CP226048 owner439 deterministic export (dormant until seal)

Do **not** run this command until the run is explicitly reported sealed and
the Isaac process is gone.  The active source is not an input to this
preparation.

The immutable checkpoint sidecar is structurally accepted by the current
`export_rl_recovery_video.py` descendant path:

- checkpoint: `checkpoint_step_000226048.pt`;
- checkpoint SHA-256:
  `fbd27dea193153c7150881b1258dbf5ce249cfd5d1529e22b066be0abf0da973`;
- sidecar SHA-256:
  `93f41386de6d46fc204f2a22cb2c712ea83eb9b1d6b29c32a7276e2d9da6df53`;
- lifetime counters: `226048 / 1731 / 34620`;
- owner439-origin delta: `+768 / +6 / +120` from
  `225280 / 1725 / 34500`;
- exact runtime/head: `f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b`,
  runtime digest `d1561db383c0b359526889f3a2844dbcf97b2ef60a9f30741ecd9c8767a43dba`;
- exact policy/layout/mode remain `rear_owner_recovery_history_v1`,
  `role422_rear_owner_recovery_v1`, and
  `issued_rear_transfer_owner_suspension_v1`;
- route remains `ancestor220544_recapture_v2` and the checkpoint is in that
  route's `checkpoints/history` directory.

Thus the adapter does not pin ordinary learning at the migration origin.  The
remaining checks necessarily wait for the sealed source: its manifest hashes,
checkpoint-load provenance, deterministic seed/mode, and closed media ledger.

After seal, run exactly once into the new destination below:

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$python = 'C:\Users\kskzz\miniconda3\python.exe'
$run = Join-Path $repo 'runs\ppo_rr_rl_timing_policy_learning_v1\video_eval\validation\20260924T0651033031701Z_gf6d1d2df8d87_b494c761b75a4755bd13d45572d8a2c1'
$source = Join-Path $run 'source'
$sourceManifest = Join-Path $source 'semantic_video_source_manifest.json'
$runManifest = Join-Path $run 'run_manifest.json'
$checkpoint = Join-Path $repo 'outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000226048.pt'
$destination = Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\video_review\CP226048_deterministic_owner439_recovery_review'

# Only evaluate these after the writer is sealed.
$sourceManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourceManifest).Hash.ToLowerInvariant()
$runManifestSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $runManifest).Hash.ToLowerInvariant()
$env:CUDA_VISIBLE_DEVICES = '-1'

& $python -B (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\export_rl_recovery_video.py') `
  --source $source `
  --destination $destination `
  --source-manifest-sha256 $sourceManifestSha `
  --source-run-manifest-sha256 $runManifestSha `
  --expected-head f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b `
  --checkpoint $checkpoint `
  --checkpoint-sha256 fbd27dea193153c7150881b1258dbf5ce249cfd5d1529e22b066be0abf0da973 `
  --checkpoint-manifest-sha256 93f41386de6d46fc204f2a22cb2c712ea83eb9b1d6b29c32a7276e2d9da6df53 `
  --expected-global-policy-decisions 226048 `
  --expected-ppo-updates 1731 `
  --expected-optimizer-steps 34620 `
  --rear-owner-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\rear_owner_CP225280_gf6d1d2df8d87_publication.json') `
  --rear-owner-publication-sha256 161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267 `
  --rear-owner-plan-sha256 f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af
```

The exporter will derive the truthful outcome/detail plan from that same
sealed episode and produce the continuous full video, same-run detail (or an
explicit predecessor-failure tail if the rear window is not reached), and the
historical-N same-camera comparison.  Visible labels remain `FL ASSIST ON`,
`REAR OWNER SUSPENSION/PROJECTION ON`, and
`REAR CAPTURE/GEOMETRY COMPLETION OFF`; no override or new AUX is inferred.
