# Post-train receipt and media commands

Do not run these while the current Isaac training process is alive. Replace
`<FINAL_CP>` and `<FINAL_DECISIONS>` only after the run is sealed. The same
literal checkpoint path must be used for deterministic and stochastic C.

```powershell
$project = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$python = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
$head = '649ccd906421d06c8b5c699f28101730910e885c'
$run = "$project\runs\ppo_task_conditioned_hip_wheel_v1\train\20260921T1355291518150Z_g649ccd906421_aac78369d59846409cb2d4039083e87a"
$checkpoint = "$project\outputs\ppo_task_conditioned_hip_wheel_v1\checkpoints\history\<FINAL_CP>.pt"
$aux = "$project\outputs\ppo_task_conditioned_hip_wheel_v1\finite_aux_CP192512_execution_01.json"
$historical = 'C:\robotics_sim\wlr_robot\ppo_historical_validation_97ecd305'
$bridge = "$project\outputs\ppo_task_conditioned_hip_wheel_v1\candidate\receiving_wheel_sigma_v1\media_compatibility_draft\receiving_wheel_media.py"

# CPU-only post-hoc receipt. The output must not already exist.
$env:CUDA_VISIBLE_DEVICES = '-1'
& $python $bridge --aux-receipt $aux --historical-runtime-root $historical `
  training --run $run `
  --output "$project\outputs\ppo_task_conditioned_hip_wheel_v1\receiving_wheel_train_<FINAL_DECISIONS>.receipt.json"
if ($LASTEXITCODE -ne 0) { throw 'Receiving-wheel training receipt failed' }
Remove-Item Env:CUDA_VISIBLE_DEVICES

# Run sequentially, only after training/any other Isaac process has exited.
$detRun = & "$project\scripts\run_semantic_video.ps1" `
  -SemanticVersion v3 -ExperimentId task_conditioned_hip_wheel_v1 `
  -ExpectedHead $head -Stage full_episode -MaxDecisions 3000 -Seed 4001 `
  -Checkpoint $checkpoint -Mode semantic_residual_eval -Device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Deterministic video capture failed' }

$stochRun = & "$project\scripts\run_semantic_video.ps1" `
  -SemanticVersion v3 -ExperimentId task_conditioned_hip_wheel_v1 `
  -ExpectedHead $head -Stage full_episode -MaxDecisions 3000 -Seed 4001 `
  -Checkpoint $checkpoint -Mode semantic_residual_eval -Device cuda:0 `
  -StochasticPolicy -PolicySeed 4101
if ($LASTEXITCODE -ne 0) { throw 'Stochastic video capture failed' }

# CPU post-processing. Both source manifests must be sealed first.
$env:CUDA_VISIBLE_DEVICES = '-1'
$detReview = "$project\outputs\ppo_task_conditioned_hip_wheel_v1\receiving_wheel_<FINAL_DECISIONS>_deterministic_review"
$stochReview = "$project\outputs\ppo_task_conditioned_hip_wheel_v1\receiving_wheel_<FINAL_DECISIONS>_stochastic_seed4101_review"
& $python $bridge --aux-receipt $aux --historical-runtime-root $historical `
  export --source (Join-Path $detRun 'source\semantic_video_source_manifest.json') `
  --destination $detReview
if ($LASTEXITCODE -ne 0) { throw 'Deterministic review export failed' }
& $python $bridge --aux-receipt $aux --historical-runtime-root $historical `
  export --source (Join-Path $stochRun 'source\semantic_video_source_manifest.json') `
  --destination $stochReview
if ($LASTEXITCODE -ne 0) { throw 'Stochastic review export failed' }

$detReceipt = (Get-ChildItem -LiteralPath $detReview -Filter 'PPO_PLUS_LIMITED_AUX_*.media.json' -File -ErrorAction Stop).FullName
$stochReceipt = (Get-ChildItem -LiteralPath $stochReview -Filter 'PPO_PLUS_LIMITED_AUX_*.media.json' -File -ErrorAction Stop).FullName
& $python $bridge --aux-receipt $aux --historical-runtime-root $historical `
  modes --deterministic-receipt $detReceipt --stochastic-receipt $stochReceipt `
  --output "$project\outputs\ppo_task_conditioned_hip_wheel_v1\receiving_wheel_<FINAL_DECISIONS>_same_checkpoint_modes.json"
if ($LASTEXITCODE -ne 0) { throw 'Same-checkpoint mode proof failed' }

$bReceipt = "$project\outputs\ppo_task_conditioned_hip_wheel_v1\B_current_review\B_current_Nplus0_P01_full.media.json"
& $python $bridge --aux-receipt $aux --historical-runtime-root $historical `
  pair --b-receipt $bReceipt --c-receipt $detReceipt `
  --destination "$project\outputs\ppo_task_conditioned_hip_wheel_v1\receiving_wheel_<FINAL_DECISIONS>_B_vs_C_pair"
if ($LASTEXITCODE -ne 0) { throw 'Archived-quantity plus sigma pair proof failed' }
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

The render commands intentionally keep scene/reset seed 4001 fixed; stochastic
policy sampling is separately labelled with policy seed 4101. A failed physical
attempt remains a formal deliverable and must not be renamed as success.
