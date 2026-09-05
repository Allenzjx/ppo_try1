# Semantic exploration v2: protected starting inventory

The 2026-09-05 continuation request supersedes historical imitation constraints
for B/C and authorizes implementation and real PPO updates within the existing
10000 + 100000 + 100000 policy-decision budget.

Read-only inspection before any edits:

- Actual worktree: `C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`.
- Existing origin fetch/push: `https://github.com/Allenzjx/ppo_try1.git` (not changed).
- Starting branch: `ppo-try1-clean-main`.
- Starting HEAD: `9f64d4fae7697cffbfa72e0f300bf9a5050cfd88`.
- `git status --short`: empty.
- Unstaged diff: empty. Staged diff: empty.
- Untracked non-ignored files: empty.
- No AGENTS.md in current/parent directory chains, or tracked source subdirectories.
- Existing source/control worktree remains at `fsm_50mm_recording_shaped_clean_v1`,
  branch `main`, commit `7d6bfda0da593e2cace2accd8bc81d300bdd9288`.
- New branch: `ppo/task-semantic-exploration-v2`, created in place from the actual HEAD.

No old run, snapshot, baseline export, failed probe, manifest, or test record is
removed, overwritten, or relabeled. Old `PPO_PRETRAINING_GATE_B_BLOCKER.md` and
the exact old Gate B implementation are retained as historical evidence, not
edited to claim success. New experiments use `runs/ppo_semantic_v2`; new exports
use `outputs/ppo_semantic_v2`. Source changes are additive and independently
versioned; the A controller and mapper are not edited.

Plan: audit all restrictions; implement supervisor/prior separation and P10
paired tests; verify representative actual actuator/reset interfaces; then
perform the first real single-environment PPO update before vector capacity
selection. Save/reload smoke checkpoint, evaluate from P01, then continue the
remaining curriculum/full-task budget with real A/B/C attribution.
