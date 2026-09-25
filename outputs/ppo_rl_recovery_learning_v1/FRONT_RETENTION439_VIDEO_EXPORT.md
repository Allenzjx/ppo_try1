# 65a front-retention439 video export boundary

The outputs-only exporter now accepts the exact 65a accounting runtime, its
ordinary same-branch PPO descendants, and the separately carried
`front_retention439_auxiliary` ledger. It first reconstructs exact 72e, then
reuses the existing reward-only72e and owner439 ancestry checks. It does not
load Torch or Isaac.

For a future **sealed** deterministic source, invoke:

```powershell
$py = 'C:\Users\kskzz\miniconda3\python.exe'
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$out = "$repo\outputs\ppo_rl_recovery_learning_v1"

& $py "$out\export_rl_recovery_video.py" `
  --source '<SEALED_SOURCE_DIRECTORY>' `
  --destination "$out\video_review\<UNIQUE_REVIEW_DIRECTORY>" `
  --source-manifest-sha256 '<SEALED_SOURCE_MANIFEST_SHA256>' `
  --source-run-manifest-sha256 '<SEALED_RUN_MANIFEST_SHA256>' `
  --expected-head 65a9255be6d9fd4e19590a48a3a650ae606b04f7 `
  --checkpoint '<ACTUAL_65A_CHECKPOINT_OR_DESCENDANT>' `
  --checkpoint-sha256 '<ACTUAL_CHECKPOINT_SHA256>' `
  --checkpoint-manifest-sha256 '<ACTUAL_SIDECAR_SHA256>' `
  --expected-global-policy-decisions '<ACTUAL_GLOBAL>' `
  --expected-ppo-updates '<ACTUAL_PPO>' `
  --expected-optimizer-steps '<ACTUAL_ADAM>' `
  --rear-owner-publication "$out\rear_owner_CP225280_gf6d1d2df8d87_publication.json" `
  --rear-owner-publication-sha256 161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267 `
  --rear-owner-plan-sha256 f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af `
  --rr-retention-publication "$out\rr_retention_CP226048_g72e63592bdf4_publication.json" `
  --rr-retention-publication-sha256 8c2fd9193ab6d3088cffcd620f9577c660fc2904d11ca2ffe23d5dc76b3a4402 `
  --rr-retention-plan-sha256 e1050240996e1669731a041f339924ef5ff7eba68983e628b806368e0b5dff9d `
  --front-retention439-publication "$out\front_retention439_CP229120_g65a9255be6d9_identity_publication.json" `
  --front-retention439-publication-sha256 5178c92547a73fe8cdd05c0087a5883b0f05819895de2cbe06b0f658a79b0917 `
  --front-retention439-plan-sha256 faeb48d9d59c0d9ba5225f54bd124857ee348a221decc1c17e9cc3204d1a92a7
```

The exporter reads AUX counts from the checkpoint ledger. It does not hardcode
`32/32`; the current official AUX checkpoint happens to carry 32 accepted and
32 attempted steps, while PPO counters remain `229120/1755/35100`. HUD and
receipt label those counts as separate finite AUX, not PPO, inherited AUX,
teacher deployment, or physical success. No command should be run against an
active writer-owned source.
