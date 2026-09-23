# Dormant strict v8 learned-branch video export

`export_rr_capture_video_v8.py` is outputs-only and accepts only a real learned
checkpoint from branch `ancestor220544_signed_wheel_v8`. It deliberately rejects
the zero-update CP220544 migration ancestor as the evaluated video checkpoint.
No source video, final checkpoint, direct parent, final counters, result, or
destination is filled before the actual training checkpoint is sealed.

## Frozen contract

- target HEAD: `d1871df37d6ea909657511d0e43e7435198f6ccd`
- schema: `wlr50_clean.rr_signed_wheel_same410.v8`
- factor/receipt: `rr_signed_wheel_v8_factor` / `rr_signed_wheel_v8_migration`
- control revision: `rr_capture_signed_wheel_continuity_v8`
- unchanged assist feedback: `signed_band_contact_formation_incremental_v6`
- reviewed publisher: `publish_rr_signed_wheel_v8.py`, SHA-256
  `4cb0f244b00d533603cbcab02a6ab9321b10de0843a19f1f1549a6f23b4627aa`
- published zero-update ancestor:
  `checkpoints/history/checkpoint_rr_signed_wheel_v8_ancestor_step_000220544_gd1871df37d6e.pt`,
  SHA-256 `339c2c779e0a9916b30af1d5ba0e76dfd76708f8d3d23b3284aca56fd5dcf072`
- immutable v7 source ancestor SHA-256
  `47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430`,
  manifest SHA-256
  `a0906988337c97f5d67108c289b4ec46d6497716cac56df39bedf8d9597362a1`
- source counters: `220544 / 1688 PPO / 33760 Adam`

The migration changes only the existing wheel transform for the already
observed signed task band: negative gap is clipped to zero for the nonnegative
support floor rather than selecting `release_slew`. RR assist feedback/state and
53-degree/45-second budget are unchanged. Sensor TOP, task acceptance, policy,
reward, physics, and action caps are unchanged. Signed AIR is not contact or
bearing, and no weak-contact controller is added.

## Command shape after the actual learned checkpoint and video seal

The actual saved/reloaded evaluation candidate is now CP221952 at
`221952 / 1699 PPO / 33980 Adam`, SHA-256
`746c3abb9aa6bfced2c194a45b7c1c9db2cf543a25ab1c033108233bce1c685a`,
manifest SHA-256
`8e4caab3cc7a1bf66c0b1e41617413c693405367efa68f627a461d99f51a629e`.
Its branch credit relative to CP220544 is the actual `1408 / 11 / 220`; new AUX
credit is zero. Its explicitly loaded parent is CP221056, SHA-256
`0c614de0d3743824a5775b411eb581fd3458314dffe2e3fe60ed425f1b1f048c`.
This identity does not predict the deterministic episode result.

All immutable checkpoint, parent, manifest, plan, publication, and publisher
hashes in the command are now filled from the actual files. The named live
source must not be opened or exported until writer closure. Planned
2048/16/320 values are not substituted for the actual CP221952 metadata.

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$python = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'

& $python `
  "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\export_rr_capture_video_v8.py" `
  --source "$repo\runs\ppo_rr_capture_then_rl_transfer_v1\video_eval\validation\20260923T1125359120678Z_gd1871df37d6e_ab5e4253c9fe431c9e6a77d3b4d19d3c\source" `
  --destination "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\video_review\CP221952_deterministic_signed_wheel_v8_review" `
  --expected-head d1871df37d6ea909657511d0e43e7435198f6ccd `
  --checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_step_000221952.pt" `
  --checkpoint-sha256 746c3abb9aa6bfced2c194a45b7c1c9db2cf543a25ab1c033108233bce1c685a `
  --checkpoint-manifest-sha256 8e4caab3cc7a1bf66c0b1e41617413c693405367efa68f627a461d99f51a629e `
  --published-v8-checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_signed_wheel_v8_ancestor_step_000220544_gd1871df37d6e.pt" `
  --published-checkpoint-sha256 339c2c779e0a9916b30af1d5ba0e76dfd76708f8d3d23b3284aca56fd5dcf072 `
  --published-checkpoint-manifest-sha256 72fad3591114e2bf5dba48a652bdebe2a37fc04eb4aef2aacbf0aea2005930b3 `
  --migration-plan "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\CP220544_RR410_signed_wheel_v8_ancestor_gd1871df37d6e_migration.json" `
  --migration-plan-sha256 126469d19bd7439d24f013470bb64babeba224e849176a65a4de68f6726b82e8 `
  --publication "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\CP220544_RR410_signed_wheel_v8_ancestor_gd1871df37d6e_publication.json" `
  --publication-sha256 addc54d70a07627631881eabeffbfd9d15687c47e483ff2df71966b79cc33b0b `
  --publication-state-proof-key exact_same410_source_state_preserved `
  --publisher-script "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\publish_rr_signed_wheel_v8.py" `
  --publisher-script-sha256 4cb0f244b00d533603cbcab02a6ab9321b10de0843a19f1f1549a6f23b4627aa `
  --migration-source-checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_signed_contact_v7_ancestor_step_000220544_g60abc00957c0.pt" `
  --migration-source-checkpoint-sha256 47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430 `
  --migration-source-manifest-sha256 a0906988337c97f5d67108c289b4ec46d6497716cac56df39bedf8d9597362a1 `
  --migration-source-head 60abc00957c0c988da6689e2fb3cfd5c8da22a47 `
  --migration-source-role front_validated_ancestor_control_eval `
  --checkpoint-output-branch ancestor220544_signed_wheel_v8 `
  --expected-parent-checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_step_000221056.pt" `
  --expected-parent-checkpoint-sha256 0c614de0d3743824a5775b411eb581fd3458314dffe2e3fe60ed425f1b1f048c `
  --expected-parent-checkpoint-manifest-sha256 77e3ecbe60f9f4c369aa2b5d3a2345c9a1d946ebb94ffe2599b8b7437f2553bf `
  --expected-global-policy-decisions 221952 `
  --expected-ppo-updates 1699 `
  --expected-optimizer-steps 33980
```

The exporter validates the inherited v8 receipt, source v7 receipt, unchanged
policy/optimizer/RNG/Identity and prior origins, actual branch route and counts,
and direct parent. Media labels use the actual branch counter deltas. Full video
keeps the normal-speed episode and complete failure tail; detail, RR/TOP/P10/RL,
and success labels are derived only from the sealed run. Historical N remains a
frozen, non-fresh, non-same-controller reference.
