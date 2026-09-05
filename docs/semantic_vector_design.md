# Semantic vector N=8 review draft

Status: **DRAFT ONLY. Not applied or simulated.** The patch contains complete additive Python modules, not changes to the running runtime. After the root cleared the live barrier, the proposed source was compiled/imported only into in-memory modules and its CPU tests executed through a custom pytest collector. Existing single-environment execution remains untouched; no proposed runtime file was created.

## Scope and physical ownership

- First revision supports exactly 8 clones, natural P01 initialization, and synchronous full-batch reset. No independent subset reset, snapshots, 16/32 or curriculum claims.
- Reuses the existing one InteractiveScene/SimulationContext, 13 exact-pair sensor bank, coordinate-local row readers, global hard-reset transaction and `_BatchedCommandAdapter` physical mapping/write. Each row has a new SemanticControllerAdapter/TaskEvaluator and independent mapper, observation builder, transition bridge and reward history.
- Reset initialization always constructs a fresh P01 semantic supervisor. If a physically satisfied initial predicate immediately hands off at tick zero, it is not rejected merely because its display phase changed. Settle and initial history are excluded from policy credit; no replay-derived success latches are installed.
- Fixed gravity reference comes from the execution profile. Legacy per-episode mean-quaternion calibration is not used. Robot/obstacle/material/actuator configuration is unchanged.
- Row facades expose tensor slices and mapper metadata for read-only auditing. They expose no actuator write method. Native audit occurs after the one batched dispatch and before the one global physics step; no counterfactual mapper advance occurs.

## PPO peer-reset boundary

At any true task terminal, stop the action-repeat loop and capture all eight actual final observations. Terminal rows use the unchanged task reward with zero terminal potential and no bootstrap. Nonterminal peers keep their physical potential and receive `gamma * V(final_observation)` as a separately reported PPO storage-reward correction before the batch resets. All rows have `done=True` to stop GAE crossing reset. Upstream `time_outs=False` prevents RSL 5.0.1 from additionally substituting its pre-action value. Raw environment reward, correction, task terminal and external peer truncation are retained separately. Peer resets are not counted as physical failures or task successes.

The factory binds the existing official critic; no PPO optimizer/ratio/GAE implementation is replaced. Pure 128-step rollout cutoffs still call the existing `compute_returns` on the current unreset observation. The additive training wrapper requires exact multiples of 128*8=1024 total decisions so vector-tail rounding cannot silently expand a budget. The existing 10k single-environment run remains independent.

Budget limitation: 10,000 and 100,000 do not divide by 1,024. This draft alone cannot exhaust either exact stage budget. A later driver must use an explicitly ancestry-bound single-environment tail (or a separately reviewed exact partial-rollout implementation); it must neither silently stop short nor round up and hide the extra decisions.

## Proposed later entry point, after review and live-barrier clearance

1. Apply the draft only after reviewing its base dependencies against the next frozen runtime.
2. Construct `SemanticVectorIsaacBackend(app, num_envs=8)` and `SemanticVectorRslEnv(backend, seeds=tuple(range(1001,1009)))`.
3. Call `construct_semantic_vector_runner`, then explicitly verify/migrate checkpoint ancestry using the existing migration mechanism before collecting new data.
4. Use `train_semantic_vector(..., decisions=1024, ...)` for the first one-update trial only within the unspent authorized stage budget; preserve migrated `stage_requested_decisions`, never reset exhausted smoke accounting. The current CLI is deliberately not changed or auto-wired by this draft.

## Required evidence before claiming implemented

- Apply/parse/import and focused CPU regressions; real raw semantic observation/reward through official RSL, including one true failure plus seven peers with distinct final values. No peer-final bootstrap may use a reset observation or action-before value.
- Real N=8 short zero and identifiable residual action checks, full native float32 target audit in each row, varied row requests, mapper/evaluator/reader identity isolation, one global step/write/capture counter per physical tick.
- Repeat full-batch reset and verify native physics/contact view reconstruction; inspect independent physical observations after deliberately differing row actions. Do not substitute Python object identities for physical isolation evidence.
- One real PPO update and save/reload. Measure total GPU process/device memory as well as PyTorch peak stats through construction, repeated reset, rollout and optimizer. Current 12 GB GPU capacity does not prove 8, 16 or 32 are feasible.
- Short-clone success does not establish complete task success. The synchronous peer-reset scheme can severely reduce later-stage coverage when one row fails; expose truncation counts and effective phase occupancy before expanding throughput or claiming curriculum coverage.

## Merge boundaries

This additive draft uses internal reused helpers; their signatures/ack fields must be checked after concurrent semantic changes. The runtime source/config inventory and checkpoint migration must include the new modules after merge. Do not silently reuse an old runtime hash, old vector matrix acceptance, old 125-dimensional adapter or legacy phase-snapshot restoration. No runtime CLI integration or claim of live readiness is included here.

Static patch-container check: `git apply --check -- docs/semantic_vector_implementation.patch` passed against the current worktree. An in-memory CPU run passed **11 tests**, including unchanged official RSL 128x8=1,024 decisions / 20 optimizer steps, scalar semantic-kernel equivalence, peer-final-observation bootstrap, real batched mapper / float32 target audit, and actual semantic reset/frame-builder/step interfaces through two resets. Scene construction, physical integration and reader acquisition in that reset test remain explicit fixtures, not evidence of real physical isolation or reset correctness.

External JUnit: `C:/robotics_sim/wlr_robot/semantic_vector_draft_cpu_20260905T070831Z/junit.xml`. The custom collector loads only proposed additive modules from this patch into memory; all optimizer/checkpoint artifacts are outside the repository. CPU evidence does not replace the required N=8 live checks above.
