# 422 candidate ready for review — not applied or published

Use **production.patch**, not the superseded early integration.patch.
The patch passed git apply --check. It changes 8 integration files and adds
the profile, actor and migration modules. Supervisor/backend/configuration
remain root/handoff-owned and are deliberately absent from this patch.

Source is the real sealed branch checkpoint221696:
SHA da63ac820ab7b260e161a116b5c925370090c087b4cce222859626b8faec24e1,
manifest956737aab9e91c3fc11d77a237f4ec03613877c53081f16c4b6b81e82c5501b6,
global221696 / PPO1697 / Adam33940; original branch1152 / 9 / 180.
All this learning remains. Migration adds zero PPO, Adam and AUX credit.

Profile is rear_policy_p02_progress_history_v1 / role419_p02_progress_v1.
The actor and critic append 3 zero columns; Adam first-weight moments append
3 zero columns. All old tensors, Adam steps/options/LR, Identity, full RNG,
receipts, branches and AUX records are preserved. New features come directly
from semantic_task.p02_progress_credit, short keys best_remaining_m,
credit_fraction, current_eligible. The public labels carry p02_ prefixes.
Old419 sigma/HISTORY logic is reused; no new exploration scaling or mean reset.

Required root config changes: both task and execution flags
p02_progress_credit_mode=p02_measured_progress_credit_v1, execution revision,
observation marker p02_progress_features_version=role419_p02_progress_v1 plus
final group {name:p02_progress_credit_full3,size:3,scale:1}. Curriculum entries
stay P01/P07/P10 in that order but weights become 2/3,1/6,1/6, matched by
execution sampling_target. The plan records1024/256/256 as unearned future
allocation. No other parsed config changes are admitted; unrelated configs
must remain byte-identical.

## CPU candidate checks

16 focused tests passed with CUDA hidden and OMP/MKL threads1. Production was
not modified: candidate modules/integration were loaded only in that process.
Tests cover same-prefix Gaussian/critic and sigma, explicit feature failures,
codec prefix, frozen prefix, one-forward raw/logp audit, populated Adam mapping,
official save/load and independent fresh reload, unchanged old lineage/RNG,
ordinary serializer carry and foreign route rejection. One synthetic ABI
128-decision update exercised20 minibatches and five exposures of every saved
raw row; this is **zero real robot-training credit**.

Command: env_isaaclab/python.exe -m pytest
outputs/ppo_rr_rl_timing_policy_learning_v1/drafts/staging_p02_progress_422/test_p02_progress_candidate.py
-q --tb=short (CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=MKL_NUM_THREADS=1).
The test also supports normal imports after root applies the patch. Its
full-state fixture reads only the real source sidecar and creates synthetic
checkpoints in pytest temporary storage; it does not load real model weights.
The small test-loader switch itself was added after the16-case pass; the
tested candidate production code is unchanged.

Publication API: build_p02_progress_migration(checkpoint,current_contract,
reason=...), validate_p02_progress_migration(...), and
publish_p02_progress_checkpoint(checkpoint,contract,plan_path,unique_output).
Root must freeze the reviewed complete runtime before actual source-device
CUDA publication. No pointer promotion, new simulation, real AUX or optimizer
step was performed in this task. No further candidate expansion is planned.
