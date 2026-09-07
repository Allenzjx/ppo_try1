# e99 v3 candidate — device arithmetic, one CPU result snapshot

Status: **exclusive synthetic CUDA check executed and passed; still unwired**.
Preparation was source-only. After both block23 and its C84,992 evaluation
exited, root ran this reviewed candidate once (receipt below). No production
audit optimization was adopted. This is a candidate revision number, not an
environment/MDP change. The preparation details below remain historical.

The previous `actuator_target_effect_candidate_e99.py`, original launcher,
`README_e99.md` and `cuda_e99_20260906T2044049294369Z.json` are preserved.
That actual receipt remains **668/680 passed, 12 failed, timing skipped**.
Its failures are not retroactively converted into passing evidence.

## Local numerical change

`actuator_target_effect_candidate_e99_v3.py` keeps the full e99 audit API and
metadata/schema. In particular, the original request/combined bias binding,
independent headroom reconstruction and exact JSON receipt validation remain.
The actual, same-geometry zero-policy and raw-geometry zero-policy branches
still use the same frozen clamp/slew/standing/sign conversions. No mapper
advance, extra dispatch, sensor read, cache or state mutation is introduced.

All selected live buffers remain on their original device for:

- the original float32 `new_tensor` expected/counterfactual casts;
- finite checks, staged-versus-dispatch and expected-versus-dispatch equality;
- numeric changed-channel comparisons (signed zero still is not an effect);
- policy and geometry float32 target subtraction.

GPU boolean reductions remain tensors, not `torch.equal` Python booleans or
`.item()` reads. Their 0/1 values are converted to float32 **on-device**, then
packed alongside the completed numerical results. A single blocking `.cpu()`
copies 56 float32 elements without geometry or 92 with geometry. CPU performs
only slicing, reading finished flags and list serialization; it does not
recreate expected targets, compare live targets, or subtract float32 targets.
The original finite -> setter -> mapping rejection order is then enforced.

This specifically avoids the failed candidate's CPU comparison boundary. It
does **not** claim that CUDA equality/reduction implementations or subnormal
serialization are already empirically equivalent. Nor does fewer host reads
guarantee speedup: additional packing/reduction kernels have overhead.

## Narrow harness revision and exact rejection classification

`validate_cuda_candidate_e99_v3.py` is a thin, SHA-pinned reuse of the old
680-entry harness, not a new collection of fixtures. It adapts its source only
in memory and never overwrites the old file. The changed anchors are candidate
path/hash, report revision/provenance and the explicit classification below.
Every replacement requires the expected exact occurrence count. All nine old
dependency pins, fresh-output/exclusive-process checks, both stream routes,
read-only snapshots, fault injection and bounded timing code remain unchanged.

The fixtures are originally dispatched under `host_flush_denormals=False`.
For exactly the following six full fixture dictionaries, **only** their later
`host_flush_denormals=True` audit must yield the specific reference rejection:

| Exact fixture group, always channel 9 | Residual values | Entries over two streams |
| --- | --- | ---: |
| `subnormal_wheel_<value>` | `+/-1.401298464324817e-45`, `+/-1e-40` | 8 |
| `headroom_subnormal_<value>`, `headroom=True` | `+/-1.401298464324817e-45` | 4 |

Both outcomes must exactly equal:

```json
{"ok": false, "error_type": "ActuatorTargetEffectError", "error": "frozen mapping reconstruction differs from actual dispatch"}
```

The fixture match uses exact full JSON, not a name prefix or a tolerance.
For this group: reference-reject/candidate-accept **fails**; both accepting
**fails**; both rejecting with another error **fails**; any mutation **fails**.
Under flush=False the same fixtures still require both successful exact full
outputs. Normal-valued neighbors, the +/-minimum-normal headroom fixtures,
ordinary cases, and any unlisted reference error are not exempted.

Existing deliberate-fault cases still require both rejection and their exact
original full error match, except the already documented mixed-device case
which requires both rejection but permits the candidate's stricter message.
All ordinary/remaining positive fixtures require both success and exact full
JSON; a generic equal pair of errors never passes as a positive case.

Every result entry adds `expectation` and `expected_rejection`; the report also
records the six exact dictionaries, original harness hash and new wrapper hash.
With unchanged fixture generation, planned classifications are 532 ordinary/
other positive entries, 136 existing fault entries, and 12 cross-denormal
rejection entries (680 total). The 192 ordinary signed-channel entries remain
within the positive group. These are planned counts, **not passing tests**.

## Pins and remaining risks

Actual file hashes read during preparation:

- Reference audit:
  `8b2c26469c2dfc723d4ef8c89505e6272b5cc67af569cff1d15774bcf1b28b1d`.
- New candidate:
  `284f7ee524ddc28d296b73310bcc3dbca42e9e5fd4018442b4acebdb80ec5919`.
- Preserved base harness:
  `589ecc15517f9e63f94c07155a9bf4423095c592c72d8eda01734c3ee262b0eb`.
- New v3 wrapper:
  `b763417e1e03b54fbcf48cf11c1da6aefadc9b80fdcc5c8b19700215580955e3`.

No production dependency was repinned. The inherited dependency pins must
still match on any later authorized execution; a mismatch means review, not
silently substituting current bytes. The v3 wrapper itself has not been
compiled/imported/tested here. Static PowerShell reads confirmed the two
candidate-path anchors and single classification/result anchors; the candidate
has one executable `.cpu()` call and no executable `.item()` or `torch.equal`.

Deferred finite reads can change error precedence for simultaneous structural
and numeric faults; computing the third geometry branch before checking the
packed flags can similarly expose an independent metadata error first. These
operations are read-only and do not authorize accepting any invalid input.
Mixed-device inputs remain explicitly rejected. The current single-fault
harness is not a proof of every multi-fault exception ordering.

All future measurements remain synthetic CUDA tensor setters and synthetic
geometry/J fixtures, not real PhysX stream-visibility or robot dynamics proof.
Input immutability and exact full outputs/rejections are mandatory before any
timing; no successful production adoption or training-throughput claim follows.

## Optional command — root authorization and exclusive idle GPU required

**Do not execute while P10 training or another Isaac/Python process is active.**

```powershell
$qaV3Result = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\native_audit_candidate\cuda_e99_v3_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '.json'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -P 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\native_audit_candidate\validate_cuda_candidate_e99_v3.py' --run-cuda --ack-exclusive-device --device cuda:0 --warmup 5 --samples 40 --max-seconds 120 --output $qaV3Result
```

Budget is unchanged: one CPU thread, max 120 seconds after CUDA setup (checked
between cases/iterations, not an OS kill), Torch reserved memory at most
256 MiB, 680 equivalence entries and, only if all pass, four timing cases with
5 warmups + 40 measured calls per function per case. Any failed check skips
timing. At preparation time no execution or retry had been performed.

## Actual exclusive execution, 2026-09-06

Root session31030 exited0 after the training and full-P01 evaluation processes
had both exited. Receipt `cuda_e99_v3_20260906T2145368007278Z.json` reports
`TENSOR_EQUIVALENCE_PASSED`, all680 classified entries matching their exact
requirements, zero equivalence failures, and bounded GPU time9.875s. The earlier
candidate's12 failures remain preserved in their separate receipt.

Interleaved whole-function host p50, reference→candidate, microseconds:
ordinary1313.05→1042.45; geometry1687.70→1274.15;
headroom1492.20→1267.25; geometry+headroom1918.60→1475.30.
These are synthetic tensor timings, not measured real-PhysX or end-to-end
training speedups. Root did not copy this candidate into production and did not
make it a gate for the next curriculum. The next selected change is reset-only
checkpoint-policy roll-in, with the physical action/audit implementation unchanged.
