# Applied video compatibility record — 64c0324

Recorded: 2026-09-10 08:29 UTC. Status: APPLIED / CPU REGRESSION PASSED / NO CURRENT SUCCESS VIDEO.

Commit: `64c03243ac05e43e7a5843eaee252eef1136c925`, applied after the previous
128-decision block completed and Isaac exited normally. Source checkpoint:
`checkpoints/history/checkpoint_step_000141696.pt`; its actual manifest records
141,696 global policy decisions, 1,072 PPO updates and 21,440 optimizer steps.

## Scope and review

Seven committed files: five production files (`semantic_video.py`,
`semantic_video_cli.py`, `semantic_migration.py`, `semantic_training.py`,
`scripts/run_semantic_video.ps1`), the new current-video test and the corrected
old v3 launcher test. No task/physics/nominal/observation/reward/config change.

The main agent reviewed the full patch and narrow selector correction; independent
reviews found no blocking issue. The selector only bypasses the existing HISTORY372
instrumentation validator when an actual VIDEO_FILES delta has explicit video
review. Pure instrumentation plus either empty or valid video review remains
rejected. Instrumentation/video factors remain separate; neither allowlist widened.

Only the current experiment uses true interval-end frames from natural P01 to
the actual terminal tick, with no additional PRE/POST physics, no duplicated tick0
frame, and explicit final partial-interval CFR display quantization.
At most24,000 physical ticks produce at most3,000 native15fps frames (<=200s).
P13's existing real completion observation remains. Failed/incomplete tails are
retained as diagnostic sources, not successful publications. Legacy windows remain.

## Verification receipts

- First failed receipt retained unchanged:
  [video_current_experiment_regression.xml](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_current_experiment_regression.xml).
  Its two failures were a stale PowerShell literal-path assertion and the
  pure-instrumentation/empty-video-review rejection-reason regression.
- Final receipt:
  [video_current_experiment_regression_v2.xml](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_current_experiment_regression_v2.xml):
  **231 passed, 0 failed, 0 errors, 0 skipped;17.411s**.
- PowerShell parse and `git diff --check`: PASS, reported by the main agent.
- The actual CPU128-decision synthetic HISTORY372 test passed: nonempty Adam,
  real save/reload through the video loader, deterministic actor actions, and
  optimizer/normalizer/RNG/counter preservation. It is NOT real Isaac/video/full-task
  success evidence. Done-only partial timeout and terminal encoder-failure tests
  also belong to the passed suite.

## Actual exact migration and next run

The actual plan is
[video_from_000141696_to_64c0324.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_from_000141696_to_64c0324.json).
It binds source06716a88/checkpoint141696 to target64c0324, exactly the five changed
production files, HISTORY372/N1 and all six unchanged selected config bindings.
It preserves learned/optimizer/normalizer/RNG/budget state and discards old rollout
storage; this is not a new-MDP warm start.

The next real P10 training run started:
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0828133987091Z_g64c03243ac05_8f4efe6ab2af4675b6e3f42469a8acfa`.
Its started manifest selects P10,128 decisions and this exact checkpoint/plan.
No completion or training result is inferred from that start receipt.

No current success video, complete P01 success or FSM-vs-PPO improvement is claimed
here. Original deferred mirrors/READMEs and first failure evidence are preserved.
This narrow applied record does not update the global training report.
