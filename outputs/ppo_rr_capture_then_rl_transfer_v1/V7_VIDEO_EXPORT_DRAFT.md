# Dormant v7 signed-AIR video export

This is an outputs-only command draft. It must not run until root names an
actually sealed v7 semantic-video source. No source path, outcome, contact,
placement, P10/RL event, or learned branch counter is prefilled here.

The reviewed adapter is `export_rr_capture_video_v7.py`. It requires exact
HEAD/checkpoint/manifest/plan/publication SHA arguments and refuses an existing
destination. It treats the configured signed AIR band only as permission for
the existing bounded action. AIR remains neither contact nor bearing, and v7
adds no weak-contact controller. Migration credit is always zero.

## Frozen ancestor bindings

- target HEAD: `60abc00957c0c988da6689e2fb3cfd5c8da22a47`
- v7 ancestor checkpoint:
  `outputs/ppo_rr_capture_then_rl_transfer_v1/checkpoints/history/checkpoint_rr_signed_contact_v7_ancestor_step_000220544_g60abc00957c0.pt`
- checkpoint SHA-256: `47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430`
- checkpoint manifest SHA-256: `a0906988337c97f5d67108c289b4ec46d6497716cac56df39bedf8d9597362a1`
- migration-plan SHA-256: `8a930bffceb0edda662b953c8ccc9249766ae5dfb47a82e044b3723ed0773895`
- publication SHA-256: `952ce797123ad8a175c84d6ffcce2677d5e06b8a1a28d3d262d63193f32ecbec`
- publication proof key: `exact_same410_source_state_preserved`
- source v6 HEAD: `97c4367ee293cb4dd191944ef663bbda63a5b235`
- source v6 checkpoint SHA-256:
  `0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01`
- source v6 manifest SHA-256:
  `5672a9446485d09da35aae818dc010fc128d2334165194c2cad773d74ac3fc4c`
- ancestor counters: `220544 / 1688 PPO / 33760 Adam`

The plan and publication *paths* below remain explicit placeholders because no
filename was guessed. Replace them only with the immutable files matching the
SHA values above. Replace the source placeholder only after writer closure.

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$python = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'

& $python `
  "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\export_rr_capture_video_v7.py" `
  --source '<EXACT_SEALED_V7_SOURCE_DIRECTORY>' `
  --destination "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\video_review\CP220544_ancestor_signed_contact_v7_review" `
  --expected-head 60abc00957c0c988da6689e2fb3cfd5c8da22a47 `
  --checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_signed_contact_v7_ancestor_step_000220544_g60abc00957c0.pt" `
  --checkpoint-sha256 47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430 `
  --published-v7-checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_signed_contact_v7_ancestor_step_000220544_g60abc00957c0.pt" `
  --published-checkpoint-sha256 47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430 `
  --published-checkpoint-manifest-sha256 a0906988337c97f5d67108c289b4ec46d6497716cac56df39bedf8d9597362a1 `
  --migration-plan '<EXACT_V7_ANCESTOR_PLAN_PATH>' `
  --migration-plan-sha256 8a930bffceb0edda662b953c8ccc9249766ae5dfb47a82e044b3723ed0773895 `
  --publication '<EXACT_V7_ANCESTOR_PUBLICATION_PATH>' `
  --publication-sha256 952ce797123ad8a175c84d6ffcce2677d5e06b8a1a28d3d262d63193f32ecbec `
  --publication-state-proof-key exact_same410_source_state_preserved `
  --migration-source-checkpoint "$repo\outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_contact_onset_v6_ancestor_step_000220544_g97c4367ee293.pt" `
  --migration-source-checkpoint-sha256 0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01 `
  --migration-source-manifest-sha256 5672a9446485d09da35aae818dc010fc128d2334165194c2cad773d74ac3fc4c `
  --migration-source-head 97c4367ee293cb4dd191944ef663bbda63a5b235 `
  --migration-source-role front_validated_ancestor_control_eval `
  --checkpoint-role published-ancestor-zero-learning `
  --expected-global-policy-decisions 220544 `
  --expected-ppo-updates 1688 `
  --expected-optimizer-steps 33760
```

## Ordinary branch descendant

For a later ordinary-PPO checkpoint, keep all migration/publication/source pins
above and change only the evaluated checkpoint binding plus:

```text
--checkpoint-role ancestor-branch-descendant
--checkpoint-output-branch <EXACT_SAFE_BRANCH_NAME>
--expected-parent-checkpoint-sha256 <EXACT_IMMEDIATE_PARENT_SHA256>
--expected-global-policy-decisions <ACTUAL_CHECKPOINT_VALUE>
--expected-ppo-updates <ACTUAL_CHECKPOINT_VALUE>
--expected-optimizer-steps <ACTUAL_CHECKPOINT_VALUE>
```

The adapter verifies that the checkpoint's actual branch counts equal those
counter differences and that its router preserves the original c531 ancestor
selection. It does not transfer learning credit from the v7 migration or from a
different latest-weight publication.
