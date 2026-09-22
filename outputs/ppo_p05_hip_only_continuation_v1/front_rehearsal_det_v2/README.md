# Deterministic P02 reconstruction: inspection-only interface

This directory adds **no fit, execute, budget, ledger, checkpoint save, teacher deployment, or production change**. It does not choose an older checkpoint or automatically run after import. The latest real source must be explicitly provided by root after the ongoing run/evaluation; preparing this interface does not fix P02.

## Inputs and scope

`inspect_det_rehearsal.py` binds the unchanged v1 gradient/JVP kernel by SHA256. Its `--data` must explicitly name the reviewed sibling `det_front_rehearsal_data_v1/candidate_manifest.json`, which binds254 CP201728 deterministic P02 rows. It reuses that candidate's frozen `read_candidate.py`, retains its full field-provenance/error receipt, verifies all254 row hashes/temporal identities/numeric checks and the four bounded source prefixes, and compares actual recorded deterministic raw12 labels rather than nominal or actuator targets.

These389 inputs were **reconstructed from synchronized fields**, not directly saved. Original CUDA/source-actor versus CPU reconstruction replay agreed within fixed absolute tolerance1e-6; agreement in12 outputs alone does not prove389-vector identity. The receipt retains reconstruction details, per-field evidence, exact history/assist checks, and the actual maximum errors. The current actor is inspected on the same numeric reconstructed inputs only. The script **does not admit these data to the current MDP or to learning**.

Fixed suggested partition: P02 indices0,3,… (85 training-inspection rows),1,4,… (85 validation-inspection rows),2,5,… (84 source-only rows). No rows are optimized. Validation is interleaved from the same old trajectory, not an independent current-policy episode. No later failed segment is turned into a positive label.

With root's explicit approval, protection-only rows are independently read through the frozen v1 loader: two directly saved block03 P01 states and thirteen directly saved P03–P06 states. They are **not** new training/validation targets and not conflated with CP201728 reconstruction. P07–P13 appear only in clearly labeled synthetic algebraic phase probes, not claimed real coverage.

## What is checked

- Actual current389 actor and unmodified HISTORY rho=.9 / receiving-wheel sigma kernel.
- Read-only raw-target errors, genuine conditional-mean gradient and frozen-initial-gradient JVP; no optimizer step.
- P02-only data must produce an exactly zero gradient in the P01 first-layer column.
- The inspected derivative scope is `actor.mlp.0.weight[:,1]` (256 scalars), not a newly implemented optimizer. P02 mean and log sigma can both respond.
- Real P01 and P03+ JVPs must be zero. A fixed, temporary P02-column perturbation must leave the entire protected same-input Gaussian bitwise equal; it is not fitted or copied to the actor.
- Same-input mathematical invariance is not unchanged future physical trajectory. Old-history one-step error/JVP does not predict the current closed-loop outcome.
- CPU actor-state copy only, with source actor/embedded metadata/Adam hash checks and RNG restoration. Original PPO optimizer is not loaded or stepped.

The CLI requires all of `--checkpoint`, `--expected-checkpoint-sha256`, `--expected-policy-decisions`, `--expected-head`, `--data`, and a new `--report` inside this directory. This explicit SHA/counter/runtime binding prevents silent fallback to an old source. Run with `CUDA_VISIBLE_DEVICES=-1`; root controls CPU-helper scheduling around the single Isaac resource.

## Verification status

25 CPU tests passed after final wiring to the moved/hash-bound candidate reader. They cover P02-only scope, protected Gaussian/JVP invariance, zero optimization and RNG preservation, synthetic actor-copy inspection without Adam load, candidate reconstruction/tolerance/partition tampering, and false direct-input/training-admission claims. The final test loaded the actual254-row candidate and exact15-row protection set against root's explicit CP211968 metadata/runtime binding, with current-actor inspection and all fitting disabled. It did not forward or modify the real CP211968 actor. The test process exited successfully after root's video had started.

No real latest-checkpoint inspection, optimization, checkpoint write, or physical run has been performed by this interface task. All new learning counters remain zero.

## Root-reviewed invocation (not executed by this task)

With the repository as working directory, after root decides to inspect:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPATH='C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\src'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' 'outputs/ppo_p05_hip_only_continuation_v1/front_rehearsal_det_v2/inspect_det_rehearsal.py' --checkpoint 'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000211968.pt' --expected-checkpoint-sha256 5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881 --expected-policy-decisions 211968 --expected-head 5fd88852bf20c94cd74405c791a13a9fd9e0a3d8 --data 'outputs/ppo_p05_hip_only_continuation_v1/det_front_rehearsal_data_v1/candidate_manifest.json' --report 'outputs/ppo_p05_hip_only_continuation_v1/front_rehearsal_det_v2/CP211968_readonly_candidate_v2.json'
```

This command only writes a new inspection report; it cannot train, admit the candidate, or publish a checkpoint. A newer source must be supplied explicitly with its own verified SHA/counter; do not silently reuse CP211968 after later learning.
