# v10 video export — dormant sealed-source command

Status: adapter prepared as outputs-only text. It has **not** been imported,
compiled, tested, or executed while the natural-P01 viewport writer is live.
Do not run this command until the run is sealed and both manifest SHA values
have been read from the closed source.

The adapter is intentionally specific to learned CP222720. It binds the
99dff5f runtime, the exact v10 zero-learning publication, its AUX CP222592
source, the direct-parent receipt, ordinary PPO credit `+128/+1/+20`, and the
independent front-retention AUX ledger `32/32`. The overlay keeps three credits
separate: ordinary PPO, non-PPO v10 controller migration, and non-PPO retention
AUX. Stage/detail names come from this episode's measured phases and contacts;
an episode that does not reach RR receives a predecessor-failure tail, never an
RR-success filename.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' `
  'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\export_rr_capture_video_v10.py' `
  --source 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_capture_then_rl_transfer_v1\video_eval\validation\20260923T1241379571238Z_g99dff5fd366e_d4c5e1bc1e8a4bfeb2b946c7f12630dd\source' `
  --destination '<NEW_OUTPUTS_ONLY_DESTINATION_AFTER_SEAL>' `
  --source-manifest-sha256 '<SEALED_SEMANTIC_VIDEO_SOURCE_MANIFEST_SHA256>' `
  --source-run-manifest-sha256 '<SEALED_PARENT_RUN_MANIFEST_SHA256>' `
  --expected-head '99dff5fd366e9cad80899176bf8b008e02112daf' `
  --checkpoint 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_step_000222720.pt' `
  --checkpoint-sha256 '069a71f547427b69491ff749dccc14fcf403565d81b3c37db3c6ac9b1e739e55' `
  --checkpoint-manifest-sha256 '314f4e25189d895f24c51c590dbe3aa284c20cf9de7a049fad526bd13b2054e5' `
  --published-v10-checkpoint 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_rr_capture_reserve_v10_step_000222592_g99dff5fd366e.pt' `
  --published-checkpoint-sha256 '8f1634e58d5303a53a1abef39c3ef8cc459223b1a2ca582c85d6d978cdf0b223' `
  --published-checkpoint-manifest-sha256 '35a6c9453f0eda960b58ad6b61898c8b5a625e637373d320b63fba4c0a86a671' `
  --migration-source-checkpoint 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_aux_frontretention410_CP222592_01.pt' `
  --migration-source-checkpoint-sha256 '8aaf255c118fa9199467b8e4016ad6d8ac235d28a31f25335423c45e93e5d86e' `
  --migration-source-manifest-sha256 '32aa8dcd6eef618b6262342911f6971a85e1217102bb149ab0be93f08ad6cf67' `
  --migration-plan 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\staging_v10\CP222592_RR410_capture_reserve_v10_g99dff5fd366e_migration.json' `
  --migration-plan-sha256 '4fa505698d5fe094c4ea6c98b40a6e131e61a2b14443bef1e0e2fe0fe1fa883a' `
  --publication 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\staging_v10\CP222592_RR410_capture_reserve_v10_g99dff5fd366e_publication.json' `
  --publication-sha256 'ce49e5e3c1cb5492ba908ec13e212b0810b878698aeeb75f4104648fb04e7c8' `
  --publisher-script 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\staging_v10\publish_rr_capture_reserve_v10.py' `
  --publisher-script-sha256 '3a3c1f46748ad68283e79896f026681e12c84cba567463d19c6f9e5d7b05aa3e' `
  --front-retention-ledger-sha256 '52aad294be62450ec44fb3c28b4f20a0b3cf9d54b3cb653c23fddad5047ac815' `
  --expected-front-retention-accepted 32 `
  --expected-front-retention-attempted 32 `
  --expected-global-policy-decisions 222720 `
  --expected-ppo-updates 1705 `
  --expected-optimizer-steps 34100
```

No destination is pre-created. The command requires a new directory and uses
the existing media path's normal 15 fps timeline, full decode/PTS/black-frame
QA, full task tail, actual same-episode detail, and explicit frozen historical
N comparison.
