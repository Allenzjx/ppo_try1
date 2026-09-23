# Dormant strict v9 learned-checkpoint video export

`export_rr_capture_video_v9.py` is an outputs-only adapter for an **actual
post-migration learned checkpoint** in the existing
`ancestor220544_signed_wheel_v8` branch. It is not an authorization to inspect
the active P07 training run, publish a checkpoint, run Isaac, or encode media.
The command remains incomplete until the future checkpoint, its immediate
parent, exact counters, and one sealed natural-P01 video source are known.

## Frozen v9 boundary

- target HEAD: `3edda51732f4ff85717fcb3491bf5c8c5766474d`
- schema/factor/receipt:
  `wlr50_clean.rr_postcapture_wheel_same410.v9` /
  `rr_postcapture_wheel_v9_factor` /
  `rr_postcapture_wheel_v9_migration`
- exact migration source: learned CP221952 at
  `221952 / 1699 PPO / 33980 Adam`, SHA-256
  `746c3abb9aa6bfced2c194a45b7c1c9db2cf543a25ab1c033108233bce1c685a`,
  manifest SHA-256
  `8e4caab3cc7a1bf66c0b1e41617413c693405367efa68f627a461d99f51a629e`
- officially published zero-update v9 checkpoint:
  `checkpoint_rr_postcapture_wheel_v9_step_000221952_g3edda51732f4.pt`,
  SHA-256 `05af3d48bf03cc3a82fb8f5ef5469bf885a30667d17a8ba715fdc2fbc2a087bf`,
  manifest SHA-256
  `770af1dcedbceff4c3ce3cf2f365b190d8e044f2b6bae9837cbc5dfaac640992`
- migration plan SHA-256
  `96e2df145f6e0b0724c70f40f790dab966fca42159f946aaf4a4bfd99d7cde83`
- publication SHA-256
  `8f9f58d5c04a343fbd6d7f30e4fa0ac35b1580ca975e4b8ca3821c18990244b7`
- publisher SHA-256
  `b75c02dd9b34de10221b204f5463dfd0f62816e15fa8d941770a406206052007`

The migration itself adds zero decisions/PPO/Adam/AUX. It changes only the
P12 post-authored-stop, current-bearing wheel-retention semantics while keeping
the v8 P09 path and RR assist feedback/budget unchanged. A later video must not
claim that this fixes FR/FL support, reaches RR/RL, or represents pure policy.
All physical labels come from the sealed episode.

## Command template after checkpoint and source seal

Every `<ACTUAL_...>` token is mandatory. Planned training counters must never
be substituted for saved/reloaded counters.

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$python = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

& $python `
  "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\export_rr_capture_video_v9.py" `
  --source '<ACTUAL_SEALED_NATURAL_P01_SOURCE>' `
  --destination '<NEW_UNIQUE_OUTPUTS_VIDEO_REVIEW_DIRECTORY>' `
  --expected-head 3edda51732f4ff85717fcb3491bf5c8c5766474d `
  --checkpoint '<ACTUAL_LEARNED_V9_CHECKPOINT>' `
  --checkpoint-sha256 <ACTUAL_CHECKPOINT_SHA256> `
  --checkpoint-manifest-sha256 <ACTUAL_MANIFEST_SHA256> `
  --published-v9-checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_rr_postcapture_wheel_v9_step_000221952_g3edda51732f4.pt" `
  --published-checkpoint-sha256 05af3d48bf03cc3a82fb8f5ef5469bf885a30667d17a8ba715fdc2fbc2a087bf `
  --published-checkpoint-manifest-sha256 770af1dcedbceff4c3ce3cf2f365b190d8e044f2b6bae9837cbc5dfaac640992 `
  --migration-plan "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\staging_v9\CP221952_RR410_postcapture_wheel_v9_g3edda51732f4_migration.json" `
  --migration-plan-sha256 96e2df145f6e0b0724c70f40f790dab966fca42159f946aaf4a4bfd99d7cde83 `
  --publication "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\staging_v9\CP221952_RR410_postcapture_wheel_v9_g3edda51732f4_publication.json" `
  --publication-sha256 8f9f58d5c04a343fbd6d7f30e4fa0ac35b1580ca975e4b8ca3821c18990244b7 `
  --publication-state-proof-key exact_same410_source_state_preserved `
  --publisher-script "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\staging_v9\publish_rr_postcapture_wheel_v9.py" `
  --publisher-script-sha256 b75c02dd9b34de10221b204f5463dfd0f62816e15fa8d941770a406206052007 `
  --migration-source-checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\branches\ancestor220544_signed_wheel_v8\checkpoints\history\checkpoint_step_000221952.pt" `
  --migration-source-checkpoint-sha256 746c3abb9aa6bfced2c194a45b7c1c9db2cf543a25ab1c033108233bce1c685a `
  --migration-source-manifest-sha256 8e4caab3cc7a1bf66c0b1e41617413c693405367efa68f627a461d99f51a629e `
  --checkpoint-output-branch ancestor220544_signed_wheel_v8 `
  --expected-parent-checkpoint '<ACTUAL_DIRECT_PARENT_CHECKPOINT>' `
  --expected-parent-checkpoint-sha256 <ACTUAL_PARENT_SHA256> `
  --expected-parent-checkpoint-manifest-sha256 <ACTUAL_PARENT_MANIFEST_SHA256> `
  --expected-global-policy-decisions <ACTUAL_DECISIONS> `
  --expected-ppo-updates <ACTUAL_PPO_UPDATES> `
  --expected-optimizer-steps <ACTUAL_ADAM_STEPS>
```

If and only if the actual checkpoint contains
`rr_capture_transfer_branch.front_retention_auxiliary`, append all four flags:

```powershell
  --expected-front-retention-ledger '<SEALED_CANONICAL_LEDGER_JSON>' `
  --expected-front-retention-ledger-sha256 <LEDGER_SHA256> `
  --expected-front-retention-accepted <ACTUAL_ACCEPTED> `
  --expected-front-retention-attempted <ACTUAL_ATTEMPTED>
```

The adapter otherwise rejects an unbound ledger. It reports that new ledger
separately from PPO and verifies that the historical finite-front AUX ledger is
unchanged. It reads the authoritative terminal reason from the last sealed
`video_policy_decisions.step_info.semantic_task` row, so predecessor failures
do not receive an empty reason. Missing RR/P10/RL events produce a truthful
predecessor detail rather than a fabricated RR clip.
