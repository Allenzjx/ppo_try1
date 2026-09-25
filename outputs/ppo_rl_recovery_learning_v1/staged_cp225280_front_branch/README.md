# CP225280 front replay dataset

This output-only builder extracts the saved **conditional Gaussian** on exact
439-dimensional CP225280 student observations.  It does not use nominal joint
angles as targets, fit a model, create PPO/AUX credit, or enter replay into the
on-policy rollout buffer.

Selection is deterministic and covers the complete time range of each front
phase: at most 64 uniformly spaced rows each from P02/P05/P06, plus every
scarce P01/P03/P04 endpoint. Alternating selected rows form training and
heldout partitions; the single P04 row necessarily has no independent
phase-specific holdout. The sealed trajectory reached P06 after real FR/FL
placement, but individual rows and offline replay are not success labels or a
closed-loop proof.

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& 'C:\Users\kskzz\miniconda3\python.exe' `
  'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rl_recovery_learning_v1\staged_cp225280_front_branch\build_front_replay_dataset.py'
```
