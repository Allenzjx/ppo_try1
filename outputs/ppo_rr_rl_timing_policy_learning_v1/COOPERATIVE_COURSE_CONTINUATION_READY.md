# Prepared continuation, not executed

`resume_cooperative2048.py` is an outputs-only, base-Python launcher of the unchanged normal PowerShell7 training wrapper. No Torch, optimizer, simulation, checkpoint save, or retry logic runs inside it.

Read-only actual-source inspection passed: checkpoint **223616 / 1712 PPO / 34240 Adam**, SHA `327f62688c41cb0d6137fb35de64592c12030389aa352c7199fd1f780079d632`, sidecar SHA `29146111e18ca20105fc6177088e06a14b4279aec3d63697920e6470d33e0abf`, runtime `49eb23163a6e20bc56301dbafb59b137ecebce66`, branch `ancestor220544_recapture_v2`. Eight pure scheduler tests passed; no child was launched.

The original `cooperative2048_g49eb23163a6e/curriculum.json` and its stop file remain immutable. Its real P07 **384 / 3 / 60** is verified against the sealed run and retained as prior evidence, never relaunched or credited as new work. Remaining order is exactly **P10 384 (successful_nominal prefix), then natural P01 1280**. The second block uses the first block's actual saved checkpoint, not a predicted filename or old ancestor. Actual phase counts come only from advantage audits whose optimizer update completed; teacher prefixes receive zero credit.

After the current Isaac video has naturally exited, root may run from the repository:

```powershell
& 'C:/Program Files/Python313/python.exe' outputs/ppo_rr_rl_timing_policy_learning_v1/resume_cooperative2048.py --execute --source outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000223616.pt --source-sha256 327f62688c41cb0d6137fb35de64592c12030389aa352c7199fd1f780079d632 --expected-head 49eb23163a6e20bc56301dbafb59b137ecebce66
```

Omit `--execute` for read-only inspection. The new independent ledger is `runs/ppo_rr_rl_timing_policy_learning_v1/curriculum/cooperative2048_continued_after_video_g49eb23163a6e/curriculum.json`; an existing directory is refused. `actual_counts` and `this_invocation_new_counts` contain only newly sealed work, while `course_counts_including_previous` explicitly includes the prior 384. Full completion would add 1664/13/260 to this invocation, but none of that is credited in advance. Both new and combined phase histograms are recorded. Main pointer and original files are hashed and checked unchanged.

For a between-block pause, put `stop_before_next_block.request.json` in the **new** continuation directory. For an active child, use the existing `stop_after_update.request.json` schema pinned to that actual child run and HEAD; the wrapper must return `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`, then this runner stops without launching another block. No process interrupt/partial-rollout credit. Child stdout/stderr are retained in `launch_00.json` / `launch_01.json`; no automatic rerun after failure. The normal wrapper retains its single-Isaac busy guard and OS lock. A completed budget is never labeled physical success.
