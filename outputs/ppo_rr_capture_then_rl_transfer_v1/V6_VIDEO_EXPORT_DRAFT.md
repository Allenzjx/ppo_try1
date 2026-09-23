# v6 video export draft — do not run before source seal

Exporter: `export_rr_capture_video_v6.py`

The current natural-P01 video is writer-owned. No preflight, Python import,
decode, or FFmpeg process is authorized until root confirms the source is
sealed. The exporter has no implicit latest resolution and requires every
identity below on the command line.

## Published ancestor evaluation

After seal, replace only `<SEALED_SOURCE>` and `<NEW_DESTINATION>`:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe -P outputs\ppo_rr_capture_then_rl_transfer_v1\export_rr_capture_video_v6.py `
  --source <SEALED_SOURCE> `
  --destination <NEW_DESTINATION> `
  --expected-head 97c4367ee293cb4dd191944ef663bbda63a5b235 `
  --checkpoint outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_contact_onset_v6_ancestor_step_000220544_g97c4367ee293.pt `
  --checkpoint-sha256 0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01 `
  --published-v6-checkpoint outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_contact_onset_v6_ancestor_step_000220544_g97c4367ee293.pt `
  --published-checkpoint-sha256 0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01 `
  --published-checkpoint-manifest-sha256 5672a9446485d09da35aae818dc010fc128d2334165194c2cad773d74ac3fc4c `
  --migration-plan outputs\ppo_rr_capture_then_rl_transfer_v1\CP220544_RR410_contact_onset_v6_ancestor_g97c4367ee293_migration.json `
  --migration-plan-sha256 e69a0ab2334c3699b7a8efe107cbc089370dbf18f9266524562c1c988afa80dc `
  --publication outputs\ppo_rr_capture_then_rl_transfer_v1\CP220544_RR410_contact_onset_v6_ancestor_g97c4367ee293_publication.json `
  --publication-sha256 3d5da5b508a65feb209a71e0e59061d49c19e3589484eb8522067e83aacbe0a9 `
  --publication-state-proof-key exact_same410_source_state_preserved `
  --migration-source-checkpoint outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_progress_handoff_v5_ancestor_step_000220544_gc53119ab332f.pt `
  --migration-source-checkpoint-sha256 308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895 `
  --migration-source-manifest-sha256 4fbf49e50fe7ab7903462478e45778b79d9cc0f757a99c0127aacb84a5fbb2e8 `
  --checkpoint-role published-ancestor-zero-learning `
  --expected-global-policy-decisions 220544 `
  --expected-ppo-updates 1688 `
  --expected-optimizer-steps 33760
```

The current exact live source is intentionally not copied into this runnable
block while it remains unsealed.

## Future ancestor-branch descendant

Use the same published-v6 and migration bindings. Replace `--checkpoint` and
its SHA with the exact immutable branch checkpoint, select
`--checkpoint-role ancestor-branch-descendant`, provide
`--checkpoint-output-branch ancestor220544_contact_v6`, provide the immediate
parent checkpoint SHA with `--expected-parent-checkpoint-sha256`, and provide
the three actual saved counters. The exporter then requires:

- the checkpoint path to be under that branch's immutable `checkpoints/history`;
- exact `checkpoint_output_routing`, no main pointer promotion;
- the persisted v6 receipt to equal the published zero-update receipt;
- immediate-parent ancestry to match the declared SHA; and
- RR branch counts to equal current counters minus 220544/1688/33760.

This branch support does not accept a different v6 migration source or a
different controller schema. A latest-learned publication requires its own
reviewed adapter/binding rather than a role switch here.

## Outcome handling

Full video is continuous normal 15fps and at most 200 seconds. RR contact,
TOP, placement, P10, RL, terminal result, and success are read from the sealed
episode. If RR is not reached, the detail becomes a truthful predecessor
failure tail and records why an RR detail is unavailable. Historical N is
explicitly non-fresh and freezes after its own endpoint.
