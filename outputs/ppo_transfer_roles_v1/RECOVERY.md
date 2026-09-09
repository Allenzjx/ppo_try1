# Checkpoint 138496 recovery — goal unfinished

Snapshot: 2026-09-08 UTC. All 19 owned training runs and 8 natural-P01 C evaluations have exited. No Isaac process was left running. The latest formal C result is P02 incomplete, not a success; no first-full-success/improved checkpoint or success video exists.

## Saved state

- Project: `C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`; branch `ppo/task-semantic-exploration-v2`.
- Pinned runtime HEAD: `d7479d9fc41cacd98740a10fb47c2fee64fd74b7`.
- Latest verified history checkpoint: `outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000138496.pt`.
- SHA256: `e568ccd4573035fb3de734a09ac6afe1a7e212fb0c4c055d2d80e78f8e84279a`.
- Companion manifest: `outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000138496_manifest.json`.
- Latest pointer and resume ledger: `outputs/ppo_transfer_roles_v1/checkpoints/checkpoint_last_pointer.json` and `resume_state.json`; `checkpoint_last.pt` is the published convenience copy.
- Counts: 138496 policy decisions, 1047 PPO updates, 20940 optimizer steps. This turn restored 123136/927/18540, not the old10112 origin. New actual credit:15360/120/2400.
- Current d747 authority revision:4096 actual decisions/32 updates; this is a check budget, not convergence or a resource/permission limit.
- Stage budget spent: full_episode64000, phase_suffix64384, smoke0. Existing100000-per-stage limits remain unchanged.

Actor/critic, learned372-input columns, state-dependent std and HISTORY kernel are preserved. Same-runtime ordinary resume preserves Adam (last adaptive lr1e-5), identity normalizer, Python/NumPy/Torch CPU/CUDA RNG and cumulative budgets. Configuration hashes and runtime inventories are in the companion manifest. Do not repeat the324→372 append, earlier scale compensation, or new-MDP Adam reset.

No unfinished rollout is inherited. PhysX contact state is not serialized: use a legal reset and collect new real trajectories. Do not write root/contact values during PPO sampling or claim tick-exact physical continuation. Teacher/checkpoint-policy prefixes remain reset-only and excluded from reward, PPO decisions and full-success credit.

## Example next ordinary continuation

First check the actual latest pointer, manifest, protected worktree and processes. If another valid newer checkpoint exists, use it instead of reverting to this snapshot. If an Isaac run is active, coordinate its existing update boundary; do not start a second process. The launcher independently enforces exact HEAD, clean protected runtime and the single-process lock.

The following is a saved recovery command, **not an already executed additional1024-decision run**:

```powershell
Set-Location 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
.\scripts\run_semantic_ppo.ps1 `
  -Command train `
  -ExpectedHead d7479d9fc41cacd98740a10fb47c2fee64fd74b7 `
  -Stage full_episode -FromPhase P01 -Decisions 1024 `
  -SemanticVersion v3 -ExperimentId transfer_roles_v1 `
  -Seed 1001 -NumEnvs 1 -Device cuda:0 `
  -Checkpoint 'outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000138496.pt' `
  -CheckpointIntervalUpdates 1
```

No `NewMdpWarmStart`, `ResumeMigration` or policy-distribution migration is appropriate for an unchanged runtime. A real later change to task/reward/nominal/action/observation semantics needs a separately documented compatible migration and fresh rollout, not bypassing contract checks.

The next work still concerns natural-P01 FR/FL completion and policy-created RR clearance/crossing/placement. Do not rerun A/Recording to authorize the optimizer. Do not lower the front-edge or active-lift conditions because C8 missed them, label suffix/teacher placement as full success, or describe newest as best. Original A and historical artifacts remain untouched; its initialization/evaluator differ from C and it was incomplete.
