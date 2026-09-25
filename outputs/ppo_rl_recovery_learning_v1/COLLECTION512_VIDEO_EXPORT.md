# collection512 video export (dormant command)

Status: exporter support is ready, but no collection512 descendant video source is sealed yet. Do not run this command against the zero-update publication as though it were a learned evaluation.

The boundary source is CP229632 at `229632 / 1759 / 35180`. A valid learned descendant must have, for integer `k >= 1`, counters `229632 + 512*k / 1759 + k / 35180 + 20*k`. Historical 128-step counters stop at the bound 65a source and are not reinterpreted.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& 'C:\Users\kskzz\miniconda3\python.exe' `
  'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rl_recovery_learning_v1\export_rl_recovery_video.py' `
  --source '<SEALED_SOURCE_DIR>' `
  --destination '<NEW_OUTPUTS_VIDEO_REVIEW_DIR>' `
  --source-manifest-sha256 '<SEALED_SOURCE_MANIFEST_SHA256>' `
  --source-run-manifest-sha256 '<SEALED_RUN_MANIFEST_SHA256>' `
  --expected-head '59e868f3e223c589e7645a0f5d63f91fa6119fb6' `
  --checkpoint '<ACTUAL_COLLECTION512_DESCENDANT_CHECKPOINT>' `
  --checkpoint-sha256 '<ACTUAL_CHECKPOINT_SHA256>' `
  --checkpoint-manifest-sha256 '<ACTUAL_SIDECAR_SHA256>' `
  --expected-global-policy-decisions '<229632+512*k>' `
  --expected-ppo-updates '<1759+k>' `
  --expected-optimizer-steps '<35180+20*k>' `
  --rear-owner-publication 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rl_recovery_learning_v1\rear_owner_CP225280_gf6d1d2df8d87_publication.json' `
  --rear-owner-publication-sha256 '161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267' `
  --rear-owner-plan-sha256 'f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af' `
  --rr-retention-publication 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rl_recovery_learning_v1\rr_retention_CP226048_g72e63592bdf4_publication.json' `
  --rr-retention-publication-sha256 '8c2fd9193ab6d3088cffcd620f9577c660fc2904d11ca2ffe23d5dc76b3a4402' `
  --rr-retention-plan-sha256 'e1050240996e1669731a041f339924ef5ff7eba68983e628b806368e0b5dff9d' `
  --front-retention439-publication 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rl_recovery_learning_v1\front_retention439_CP229120_g65a9255be6d9_identity_publication.json' `
  --front-retention439-publication-sha256 '5178c92547a73fe8cdd05c0087a5883b0f05819895de2cbe06b0f658a79b0917' `
  --front-retention439-plan-sha256 'faeb48d9d59c0d9ba5225f54bd124857ee348a221decc1c17e9cc3204d1a92a7' `
  --collection512-publication 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rl_recovery_learning_v1\collection512_CP229632_g59e868f3e223_publication.json' `
  --collection512-publication-sha256 '8c79be67d82843fd517269fe28da4078c79348c499bb20be484056085eb26f6c'
```

The visible identity remains FL assist ON, RR/RL task-completion assist OFF, and issued-owner suspension/projection ON. The export receipt keeps front-retention AUX `32/32` separate from PPO and reports the collection512 boundary as zero-credit plus only the actual post-boundary `+512/+1/+20` multiples.
