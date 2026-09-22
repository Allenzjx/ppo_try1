# Output-only finite FL conditional-mean candidate

Status: candidate only. Production changes 0; real auxiliary optimizer steps 0;
real PPO decisions/updates/optimizer steps added by this work 0. No Isaac launch.
Root alone decides whether to adopt/execute after the current CP192512 videos.

## Files and behavior

- `finite_auxiliary_mean.py`: only two temporary SGD leaves (256 last-layer FL
  mean-row weights + 1 bias), no momentum/weight decay, no PPO Adam step. Calls
  the unchanged actor forward through `torch.func.functional_call`, including
  actual 372 inputs, real HISTORY and the genuine rho=.9 chain factor .1.
- `reviewed_data.py`: hash-bound sealed real byte ranges; all 75 decisions
  372–446 (12 ramp + 63 hold) retained, exact 372 float32/hash and ACK/HISTORY
  joins, other11 unchanged, full12 permission, -6° intervention and no capture.
  Truncating an otherwise contiguous window is rejected. Reads 24 actual
  unmodified non-P05 holdout states, including explicitly verified real reset0.
- `auxiliary_cli.py`: default read-only CPU inspection; execution additionally
  requires `--execute-aux`, the reviewed same-CP/data/helper inspection receipt,
  and a new `checkpoint_aux_flmean_*.pt` name. Actual execution would use the
  official same-device full checkpoint loader and save/roundtrip. Never publishes
  a latest training pointer or overwrites a step checkpoint. No Isaac imports.
- `test_finite_auxiliary.py`: 23 CPU tests passed, recorded in
  `finite_auxiliary_tests.xml` (0 errors/failures). Two official synthetic PPO
  updates exercise source→aux→load→fresh128 PPO→save→exact-resume. They are test
  doubles, not real learning or physics evidence.

The helper does not claim CUDA→CPU official resume. CPU inspection copies only
the verified actor state for deterministic forward, checks payload/hash/policy,
never steps either optimizer, and restores inspection-process RNG. Real execution
must use the original device. Windows inspection uses `CUDA_VISIBLE_DEVICES=-1`;
setting it to an empty PowerShell string removes the variable and is insufficient.

## Independent ledger and preserved state

`task_conditioned_hip_wheel_branch.auxiliary_mean_learning` explicitly records
the independent supervision event, original CP/data/helper hashes, accepted and
attempted auxiliary steps, before/after full12 same-state means/sigmas, loss and
holdout changes. Branch ID/origin, lifetime PPO counters, stage spent and original
branch counts do not reset. Existing production `train_semantic` already carries
the complete branch dict, so this ledger survives subsequent formal PPO saves
without production edits or a new MDP migration. Tests prove this persistence.

Trunk/other23 head rows/critic/std/normalizers/full PPO Adam state/scalar LR and
RNG are byte/hash unchanged. Only selected row0 may change. Old PPO Adam momenta
are intentionally retained and can oppose the auxiliary direction on subsequent
PPO updates; there is no claim that they represent the auxiliary updates. Any
future checkpoint after execution must be described as PPO plus explicit finite
auxiliary supervision, not pure PPO. Fresh empty PPO rollout is mandatory; none
of these old diagnostic states is inserted into PPO storage.

Current data and CP192512 cross only the already-adopted quantity-budget boundary.
The CLI revalidates `checkpoint192000_quantity_continuation.json` against the
actual source checkpoint/current runtime and binds its exact source/target
contracts. It does not treat policy-name equality as MDP compatibility.

## Conservative proposed budget, not a claim of effectiveness

Current candidate defaults: at most 16 SGD attempts, independent LR .05 linearly
decaying to .003125, cumulative fixed-state FL REQUEST shift ≤1° for the training
window and ≤.25° for holdout, maximum per-state conditional Gaussian KL ≤.1.
All losses must remain finite and nonincreasing. A violating proposal is rolled
back in temporary parameters and the procedure stops; no policy sample/action is
clipped or masked to enforce these parameter trust bounds. No accepted update
means no auxiliary checkpoint is saved. No automatic loop/retry/escalation.

CP192512 immutable source SHA:
`97284bfa095cc40719b5ba0b9a309ed954dc90d8454c0af9119cb76614e8052d`.
The current read-only receipt is `CP192512_readonly_final.json`.
Earlier `CP192512_readonly_real_states.json`, `CP192512_readonly_linear_prediction.json`
and `CP192512_readonly_16x005_prediction.json` predate the final helper bytes and
are intentionally stale for execution binding; all are retained.

On these exact saved real inputs, latest FL network mean is +.08424..+.12272,
mean +.11583. Conditional mean averages -.14244 while real H averages -.17114:
negative conditional action here does not establish a self-sustaining learned
negative direction. Conditional sigma averages .12358. Latest physical REQUEST
minus recorded intervention target averages +.57981° (+.55562° in the 63 hold
decisions). These are same-numeric-state comparisons, not a natural rollout.

The inspection now measures leaf feature norm (including bias) mean 12.27039,
first true gradient norm .0407163. The original 8-step/.01 proposal predicted
only -.0393915° mean REQUEST change. Root explicitly rejected treating that tiny
change as the main fix and selected the new finite 16-step/.05 independent budget
from measured feature/gradient evidence, without changing PPO hyperparameters.

With unchanged first gradient and the revised total decaying LR .425, predicted
raw conditional shift averages -.0212130; REQUEST shift averages -.370987°
(range -.384799..-.362926°). Holdout maximum absolute REQUEST shift .354023°
**exceeds** the unchanged .25° bound: actual execution cannot simply accept all
16 proposals. It would reject/roll back the first violating proposal and stop.
Later gradients change, so this does not predict the actual accepted step count.
Current loss .000593558 would become .000113989 under the linear prediction.
First-step raw conditional shift averages -.00249565. This is a no-optimizer,
no-parameter-write frozen-first-gradient approximation, NOT a trained result,
physical evidence or a claim of restored capture/retention. The analytic first
gradient matches real autograd in a synthetic CPU test. The genuine .1 HISTORY
chain was not multiplied by 10 and the PPO LR was not changed.

## Evidence limits

All labels are `FL_AIR_gap_approach_not_capture`; the -6° run never touched or
placed FL. All 75 endpoints improve gap relative to entry and synchronous det,
but the first advantage is only .032313 mm; 36/75 local steps worsen gap and
remain included. Release/follow/after failure evidence remains in the receipt.
This is whole-window approach evidence, not 75 robust one-step gains, isolated
hip causality, capture, retention or authorization to optimize.

Other11 are never given zero targets. True negative H is not replaced by zero.
The row is shared across phases: 24 real holdouts cover P01–P04 only; they do not
prove P06–P13, on-trajectory, body stability or physical safety. Same-state sigma
and other11 equality do not imply unchanged future histories/trajectories.
Teacher-free P01 evaluation and normal formal PPO remain necessary after any
root-approved auxiliary application.

## Reproduction (CPU tests/inspection only)

From the repository root in PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' -m pytest outputs/ppo_task_conditioned_hip_wheel_v1/candidate/finite_auxiliary/test_finite_auxiliary.py --confcutdir=outputs/ppo_task_conditioned_hip_wheel_v1/candidate/finite_auxiliary -q
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_task_conditioned_hip_wheel_v1/candidate/finite_auxiliary/auxiliary_cli.py --checkpoint outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000192512.pt --selection outputs/ppo_task_conditioned_hip_wheel_v1/FL_minus6_gap_approach_selection.json --expected-head 97ecd305afb5c43b095e1206ce8917e6e742b1bb --report outputs/ppo_task_conditioned_hip_wheel_v1/candidate/finite_auxiliary/CP192512_readonly_repeat_unique.json
```

This document intentionally does not schedule or execute auxiliary optimization.
The explicit execution flags exist for root review, not automatic adoption.
