# Student-entry direction probe and finite-AUX review

Status: **runner prepared and pure-tested; NOTEXECUTED**. No Isaac, model load,
optimizer, checkpoint write, PPO sample, AUX update, or video was produced.

## Minimal real diagnostic

`student_entry_direction_probe.py` is an isolated JSONL-only runner. It starts
one frozen checkpoint naturally at P01 and lets the student control the whole
prefix. It does not use the successful-N prefix, teleport, reset into P09, or
retry a failed prefix. At the first decision boundary that has all of:

- P09, real current RR lift qualification, active-lift history, RR AIR;
- RR inside the top XY/lateral span with nonnegative front distance and a
  positive top gap;
- real FL support plus the configured minimum number of other supports above
  the existing force-noise floor; and
- valid, nonterminal evaluator evidence,

it makes one finite raw-latent substitution. Only indices 1/3/6 change:
FL-knee positive, FR-knee positive, RR-hip negative. RR knee and the other eight
channels remain that decision's student request. The candidate is an absolute
raw value each decision, not an accumulated offset. It is released on RR
contact, loss of the safe entry, leaving P09, a terminal, or the explicit
4–6 s cap, and it cannot re-arm. The same episode continues to its real result.

The candidate enters through ordinary `core.step`, so the existing projector,
mapper, HISTORY, headroom/caps, slew, one articulation write, and independent
native readback audit remain authoritative. This is intentionally *not* the
older `direction_probe.py` filtered-residual anchor and does not hot-patch a
public module. A raw-only probe cannot guarantee an absolute knee target or an
exact final-target hold while mapped N changes; the runner therefore says only
that RR-knee raw remains the student's request and records the actual final and
measured knee on every physics tick.

### Reviewed launch template (do not run before filling exact sealed pins)

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
$python = "C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe"
$repo = "C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1"
$probe = "$repo\outputs\ppo_rl_recovery_learning_v1\student_entry_direction_probe.py"

& $python $probe `
  --checkpoint <IMMUTABLE_CHECKPOINT_PT> `
  --checkpoint-sha256 <CHECKPOINT_SHA256> `
  --checkpoint-manifest-sha256 <SIDECAR_SHA256> `
  --expected-head <FULL_40_CHAR_FROZEN_HEAD> `
  --experiment-id rr_rl_timing_policy_learning_v1 `
  --run-dir "$repo\runs\ppo_rr_rl_timing_policy_learning_v1\direction_probe\<UNIQUE_RUN_ID>" `
  --device cuda:0 `
  --fl-knee-raw <REVIEWED_POSITIVE_RAW> `
  --fr-knee-raw <REVIEWED_POSITIVE_RAW> `
  --rr-hip-raw <REVIEWED_NEGATIVE_RAW> `
  --maximum-probe-seconds 5 `
  --execute-reviewed-real-diagnostic
```

The runner refuses a last-pointer checkpoint, a dirty/different runtime, a
non-round-tripped sidecar, an existing output directory, any rear task assist,
non-all12 policy permission, a wrong candidate sign, or a magnitude above 3.
It also takes the repository single-process lock. The three raw magnitudes have
no defaults because they must be reviewed against the frozen profile and the
actual student's entry request; this document does not silently choose a
teacher waveform.

Two logs are written:

- `diagnostic_decisions.jsonl`: every decision-boundary policy observation,
  original frozen-student raw request, optional official request audit, applied
  raw request, exact override indices, step result, and eligibility blockers;
- `diagnostic_physics.jsonl`: every native tick, original/applied decision
  receipt, projected residual, atomic ACK, native readback audit, current
  physical evaluator and raw observation.

The manifest fixes PPO/AUX counters added at zero and verifies actor, critic,
optimizer, normalizer, runtime, checkpoint, and sidecar again at the end.
Rows may later be reviewed as an explicitly exogenous diagnostic dataset; they
are never inserted into the on-policy buffer by this runner.

## Existing tool compatibility

- `outputs/ppo_task_conditioned_hip_wheel_v1/direction_probe.py` is a real old
  physical diagnostic, but it binds the old configuration and anchors a
  filtered residual. It cannot guarantee current same-tick final RR-knee hold
  and must not be reused as if it were the new 422/439 student contract.
- `outputs/ppo_rr_rl_timing_policy_learning_v1/rr_direction_probe_v1/` has a
  useful pure absolute-target planner, but its runner depends on an in-memory
  headroom replacement and an exact post-mapper hook. That is a different
  diagnostic boundary from the requested explicit raw candidate. It remains
  NOTEXECUTED and is not imported here.
- `front_retention419_v1` only covers P01/P02 on historical 419-column inputs.
  Its zero-step inspection found a very small wheel-request gradient and was
  explicitly paused with zero AUX credit. It does not provide RR/RL entry data.
- `rr_mean_rehearsal_v4` is a carefully accounted finite mean-head AUX pattern,
  but it binds an old 389-column policy, CP214400, old source/runtime hashes and
  old P09–P11 data. Running it unchanged on the new student would be invalid.

## Minimal AUX recommendation after a real probe

Do not fit from the successful N's perfect entrance alone and do not treat the
three injected raw values as ordinary PPO samples. First inspect the sealed
probe rows at the *student's* visited entry:

1. bind the exact current observation layout/runtime/checkpoint and split
   distinct decision rows into a tiny train/validation partition while
   disclosing that a single episode is correlated;
2. retain the original student request, injected diagnostic request, final
   mapped target, actual response, contact/load and task outcome separately;
3. run a zero-update CPU copy/JVP inspection first, including same-input
   P01–P08 and unaffected-channel protection, with actor/critic/Adam/RNG hashes
   unchanged;
4. only if the finite physical diagnostic improves the intended geometry/load
   without losing support, request a separate, explicitly budgeted AUX update
   on the smallest reviewed parameter scope; count it separately from PPO,
   save/reload full state, then discard stale rollout and collect fresh PPO.

The successful N table supplies mechanism and ordering evidence, not a
same-contract neural teacher. If a conditional teacher is later proposed, it
must be queried on the student's actual current observation semantics and
labelled offline; an N action executed in another state cannot be stored with
the student's log probability or called on-policy.

## Pure checks performed

`test_student_entry_direction_probe.py` covers exact three-index replacement,
RR-knee/other-channel preservation, nonaccumulation, sign/finite/range refusal,
fail-closed P09/qualification/XY/AIR/gap/FL-support admission, ACTIVE release on
lost current qualification/support, contact-specific release priority, and recursive
physical-dataclass serialization. Nine tests pass
under base Python; they test wiring only, not physical usefulness.
