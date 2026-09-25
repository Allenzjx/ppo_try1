# Independent knee25 driver: prepared, NOT RUN

Files: independent_rr_knee25_diagnostic.py + test_independent_rr_knee25_stdlib.py.
11 stdlib tests pass; importing the script and requesting --help loads no Torch,
Isaac, PXR or production modules. This task has not launched any physics process.

The runnable protocol is narrower than the earlier candidate note:

- Explicitly selected latest compatible complete checkpoint, same strict
  production runtime. Its SHA, sidecar SHA and all three expected counters are
  required; no live pointer read or fixed CP227840 fallback.
- Entry actual FINAL is latched once; total hip delta -25deg, total knee +25deg,
  ramped over1.5s. No +30 option or repeated increment.
- First measured native TOP latches that tick's actual RR FINAL. Hold starts at
  the next existing policy decision; no mid-step write or extra physics step.
- Only channels6/7 are overridden. Other10 remain exactly the same request from
  one deterministic forward. Original FL assist remains declared; formal rear
  task assists remain OFF. This override is an explicit independent diagnostic.
- Original production core/evaluate, mapper/HISTORY/limits/atomic writes,
  native event rules and global200s limit remain in force.
- No optimizer update, save, teacher-label generation or weight modification.
  The route verifies learned-state identity; source CP/sidecar hashes checked
  before and after. A diagnostic TOP/hold is not learned PPO success.
- The callback binding is process-local and restored in finally. No production
  files or formal eval/training code are changed.

## Provenance / future review data

source/independent_diagnostic_driver.json contains script SHA, full parameters
and canonical parameter SHA, production runtime, exact CP/sidecar SHA, source
counts and all override declarations. It is included in the production source
manifest's sealed_files inventory.

source/independent_probe_steps.jsonl contains every actual step, including the
naturalP01 prefix: pre-step448 observation, selected-policy raw12, issued raw12,
after448 observation, physical evaluator/contact state, local-task state,
actual final targets and final native dispatch audit. Every row is labeled
DIAGNOSTIC/PPO0/AUX0/provisional, with no automatic successful-teacher label.
First-TOP latch includes its real native tick, eligibility and current bearing.

The normal video_policy_decisions.jsonl also carries independent_diagnostic
with script SHA, parameters, exact targets, prior ACK baseline, selected versus
issued raw and any inverse-tanh bound. This previous-ACK calculation is not
claimed to be a same-tick target guarantee; native dispatch evidence remains
authoritative. No extra policy forward or extra actuator write occurs.

The existing formal export rejects this source because mode and
diagnostic_intervention explicitly indicate a diagnostic. It must never be
renamed/repackaged as formal C.

## Deferred command (do not run concurrently with DET/train)

Run from repository root in the existing Isaac environment, only when parent
selects this optional experiment after the planned DET/new512 work. Take the
checkpoint path, both SHA values and actual counts from that completed block's
sealed run manifest/checkpoint sidecar. Do not select CP227840 merely because
the example below was prepared earlier. The planned next512 would give
CP228352/local3072/6 PPO/120 Adam if it completes; these are not claimed completed
results and must be verified against its sealed manifest.

Required count arguments for that verified new block would be:

    --expected-local-decisions 3072 --expected-ppo-updates 6 --expected-optimizer-steps 120

Historical CP227840 example ONLY (not the default next-run selection):

    & 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs\ppo_rr_capture_first_cp225280_v1\independent_rr_knee25_diagnostic.py --run-independent-diagnostic --expected-head 1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81 --checkpoint outputs\ppo_rr_capture_first_cp225280_v1\checkpoints\history\checkpoint_CP227840_local002560_lineage448_v2_g1e10d39c9a80.pt --checkpoint-sha256 81aa94a31039730c7ad87a45b53de22a12361047d324d4d5700b835d7cddbdb5 --manifest-sha256 a34999a1a2cdc963b8e7f4d1914c5564ed4a1920987293a6599fcc3313d95956 --expected-local-decisions 2560 --expected-ppo-updates 5 --expected-optimizer-steps 100 --run-dir runs\ppo_rr_capture_first_cp225280_v1\diagnostic_CP227840_hipminus25_kneeplus25_1e10

If production HEAD changes, this command must fail strict compatibility rather
than silently reinterpret old weights. Parent must review a new compatible
binding. A newer CP is deliberately not selected by reading a live pointer.
No new release gate or duplicate prior audit is introduced.
