# CP225280 front-preservation video export (dormant)

Status: exporter adapter and stdlib tests are ready. The zero-update boundary
was actually published/reloaded under frozen HEAD
`892385cba8a7089b52567bb7f558c018fac82a77`; no learned descendant video is
claimed here. Run only after the selected descendant checkpoint has been
officially saved/reloaded and its natural-P01 deterministic source is sealed.

The adapter accepts only `front_preservation439_branch_identity` on branch
`cp225280_front_preserved_v1`. It independently binds the original f6d439
CP225280 source, the new zero-credit publication, actual descendant counters,
the immutable front replay dataset, and the sealed video manifests. It rejects
the later 72e/65a/AUX32/collection lineage rather than relabelling it.

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$py = 'C:\Users\kskzz\miniconda3\python.exe'
$source = '<ABSOLUTE_SEALED_SOURCE_DIRECTORY>'
$checkpoint = '<ABSOLUTE_cp225280_front_preserved_v1_CHECKPOINT>'
$publication = Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\staged_cp225280_front_branch\front_preserved_CP225280_g892385cba8a7_publication.json'
$destination = Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\video_review\CP<STEP>_deterministic_front_preserved_v1_review'

$env:CUDA_VISIBLE_DEVICES = '-1'
& $py (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\export_front_preserved_video.py') `
  --source $source `
  --destination $destination `
  --source-manifest-sha256 '<SEALED_semantic_video_source_manifest_SHA256>' `
  --source-run-manifest-sha256 '<SEALED_run_manifest_SHA256>' `
  --expected-head '892385cba8a7089b52567bb7f558c018fac82a77' `
  --checkpoint $checkpoint `
  --checkpoint-sha256 '<ACTUAL_CHECKPOINT_SHA256>' `
  --checkpoint-manifest-sha256 '<ACTUAL_SIDECAR_SHA256>' `
  --expected-global-policy-decisions '<ACTUAL_LIFETIME_DECISIONS>' `
  --expected-ppo-updates '<ACTUAL_LIFETIME_PPO_UPDATES>' `
  --expected-optimizer-steps '<ACTUAL_LIFETIME_ADAM_STEPS>' `
  --rear-owner-publication (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\rear_owner_CP225280_gf6d1d2df8d87_publication.json') `
  --rear-owner-publication-sha256 '161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267' `
  --rear-owner-plan-sha256 'f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af' `
  --front-preservation-publication $publication `
  --front-preservation-publication-sha256 '29bd2ba487088f98d76944566959057a8ffb280ab72ae51eae47f65eff0e0379' `
  --front-replay-dataset (Join-Path $repo 'outputs\ppo_rl_recovery_learning_v1\staged_cp225280_front_branch\CP225280_front_replay_dataset.json') `
  --front-replay-dataset-sha256 'd78686d75a8ab69e492d6504340030992891b64ab279a19de374dc91233d2e38'
```

Expected outputs remain the existing three-video set: a continuous normal-speed
full episode with its real failure tail when incomplete, a same-episode detail
(or explicit predecessor/RR-not-reached tail), and the fixed historical-N
same-camera comparison. The N reference is historical, not fresh same-controller
evidence, and its frozen tail is not physical evidence.

Visible accounting is deliberately separate:

- restored source: exact f6d439 CP225280 learned state;
- new branch learning: actual `lifetime - 225280/1725/34500`, exactly
  `+512 decisions / +1 PPO / +20 Adam` per completed update;
- replay: `640` immutable front-row exposures per update in the same official
  PPO Adam minibatches, adding zero on-policy samples and zero separate AUX
  optimizer steps;
- AUX: legacy source lineage only; the later 65a front-retention AUX32 is not
  inherited and cannot be supplied through this CLI.

The replay KL is not a closed-loop physical-retention or task-success proof.

Actual zero-credit boundary, for lineage reference only (do not substitute it
for a later learned video checkpoint):

- checkpoint: `checkpoint_front_preserved_CP225280_g892385cba8a7.pt`
- checkpoint SHA256: `4240785e0fb18f1e91c2b0b8ce2592304bf84cca20a4c6853fe707371b9b5d9c`
- sidecar SHA256: `8b37a904ce5b0c32a48f125f1c76ba0326dee69756c8f56189821595b49de8e3`
- counters: `225280 / 1725 / 34500`
- front branch counts and replay counts: all zero at publication
- publication SHA256: `29bd2ba487088f98d76944566959057a8ffb280ab72ae51eae47f65eff0e0379`
