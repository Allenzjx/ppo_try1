# Owner439 RL recovery video export

`export_rl_recovery_video.py` is an outputs-only adapter over the established
rear-policy media pipeline. It does not start Isaac or load Torch. It accepts
only the frozen `f6d1d2df8d87...` owner439 publication (or an ordinary
same-runtime/same-branch descendant carrying that exact receipt), a sealed
deterministic natural-P01 run, and the fixed historical N reference.

Visible control labels are deliberately split: `FL ASSIST ON`, `REAR OWNER
SUSPENSION/PROJECTION ON`, and `REAR CAPTURE/GEOMETRY COMPLETION OFF`. A bare
`REAR OFF` label is not used because owner suspension/projection is an active
control repair even though no rear capture/geometry target assist is enabled.

The export names are:

- `CPxxxx_DET_full_RL_completion_attempt[_INCOMPLETE].mp4`
- `CPxxxx_DET_FL_to_FR_RL_recovery_detail[_INCOMPLETE].mp4`
- `N_vs_CPxxxx_DET_same_camera.mp4`

If the episode never reaches the RR window, the second file is still a real
continuous tail from that same episode and is labeled `RR NOT REACHED |
PREDECESSOR FAILURE TAIL`; it is never substituted from another run.

After the source is sealed, fill only the source/checkpoint-specific values:

```powershell
$repo = "C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1"
$python = "C:\Users\kskzz\miniconda3\python.exe"

& $python "$repo\outputs\ppo_rl_recovery_learning_v1\export_rl_recovery_video.py" `
  --source "<SEALED_SOURCE_DIRECTORY>" `
  --destination "$repo\outputs\ppo_rl_recovery_learning_v1\video_review\<UNIQUE_REVIEW_DIRECTORY>" `
  --source-manifest-sha256 <SEALED_SOURCE_MANIFEST_SHA256> `
  --source-run-manifest-sha256 <SEALED_RUN_MANIFEST_SHA256> `
  --expected-head f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b `
  --checkpoint "<ACTUAL_VIDEO_CHECKPOINT_OR_SAME_BRANCH_DESCENDANT>" `
  --checkpoint-sha256 <ACTUAL_CHECKPOINT_SHA256> `
  --checkpoint-manifest-sha256 <ACTUAL_CHECKPOINT_MANIFEST_SHA256> `
  --expected-global-policy-decisions <ACTUAL_LIFETIME_DECISIONS> `
  --expected-ppo-updates <ACTUAL_LIFETIME_PPO_UPDATES> `
  --expected-optimizer-steps <ACTUAL_LIFETIME_ADAM_STEPS> `
  --rear-owner-publication "$repo\outputs\ppo_rl_recovery_learning_v1\rear_owner_CP225280_gf6d1d2df8d87_publication.json" `
  --rear-owner-publication-sha256 161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267 `
  --rear-owner-plan-sha256 f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af
```

For the zero-credit published checkpoint itself, the exact checkpoint pins are
`fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f`,
manifest `f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d`,
and lifetime counters `225280 / 1725 / 34500`. A learned descendant must use
its own actual hashes and counters; the adapter does not hold those counters at
the migration origin.
