# e99 native-audit candidate — CUDA equivalence FAILED

**Current status, 2026-09-06: actually executed once by root; 668/680 comparison
entries passed, 12 failed. Timing was skipped. Do not adopt this candidate or
claim any speedup.** The actual receipt and precise failure boundary are appended
below. The following preparation record is preserved as historical planning;
its "not executed" statements describe the time before that one execution.

## Historical preparation record — before the actual execution

Status: source-reviewed, unwired candidate only. **No Python, CUDA, Isaac,
CPU test, GPU microbenchmark, or optimizer was run for this revision.** The
C82,560 evaluation continues independently. Production/configuration/tests and
all original candidate scripts/results/README remain unchanged.

## Why the old script is stale

The preserved c34262 candidate/launcher pins production audit SHA
`7ec65a2e425681d2e172caa409ae0735983d451bdf7346fa0cecba63cb268721` and has no
`policy_headroom_mode` argument or receipt verification. Its 161-passing CPU
receipt is historical evidence for that older revision only. It must not be
repinned and run unchanged against e99 or treated as headroom coverage.

New files:

- `actuator_target_effect_candidate_e99.py`: copied from the complete current
  e99 audit, changing only target materialization/comparison placement.
- `validate_cuda_candidate_e99.py`: revised finite exclusive-device launcher.
- This README. No results for these files exist yet.

Reference commit read: `e99fde1b3e8366f0ff1d484140b82877df745f0c`.
Actual reference-file SHA read once:
`8b2c26469c2dfc723d4ef8c89505e6272b5cc67af569cff1d15774bcf1b28b1d`.
Actual new-candidate SHA:
`3b0ad2fd40cc5e9951b326bc23784448c3601f86c1378d5d6487ffe052606601`.
The launcher checks these and nine explicit fixture/mapper/adapter/geometry/
headroom dependency pins before importing Torch. A later change requires
review, not blindly replacing the pins. Its result includes the launcher hash.

## Exact API and preserved verification

The candidate now matches the complete e99 keyword-only audit ABI:
`build_actuator_target_effect_audit(adapter=..., actuation=..., raw_ack=...,
previous_final_drive_servo_deg=..., source_phase_id=..., policy_request=...,
policy_headroom_mode=None)`.

With mode None, unexpected headroom receipt fields are rejected. With the
known mode, it requires the mode/evidence pair, binds original requested
controller/residual and c+r to the actuation plan, then calls the production
`project_semantic_servo_headroom` on the same geometry-adjusted native target.
The declared evidence must match the complete recomputed JSON, including
numeric-versus-bool distinctions; missing, tampered or nonfinite metadata is
not silently ignored.

All three relevant branches are preserved:

1. Actual: geometry-adjusted native + headroom-effective controller/residual.
2. Zero-current-policy: same geometry and same previous final-drive state,
   with headroom recomputed for zero residual.
3. Geometry diagnostic: raw native, zero residual, and its independently
   recomputed headroom; geometry effect is not attributed to policy.

Original request receipt values remain original requests, not overwritten with
effective values. Policy mask/request metadata, final slew/hard mapping,
standing/sign conversions, selected joint IDs, float32 finite checks, staged
versus dispatched equality, actual reconstruction, numeric changed flags,
signed-zero serialization and every output field are retained.

The only materialization change remains four canonical tensor selections /
clones followed by one blocking `(1,24)` float32 `.cpu()` call. All following
comparisons, casts, deltas and list conversions operate on that CPU snapshot.
No persistent cache, previous-state substitution, extra write/mapper advance,
background stream or nonblocking copy is introduced. Multi-fault error priority
can differ because all structures are checked before packed finiteness;
mixed-device snapshots are explicitly rejected. Neither is permission to
accept invalid inputs.

## Planned bounded coverage (not passing-test counts)

The script enumerates170 fixture descriptions, each compared under both host
denormal modes and both same-stream / explicit producer-event-wait routes:
680 comparison entries if the budget completes. Ordinary signed residuals,
quantization, signed zeros/subnormal/normal-neighbor differences, geometry,
malformed buffers, staged/dispatch mismatch and read-only state checks remain.

Added e99 cases cover all12 channels, each servo upper/lower outside-reserve
baseline with zero/outward/inward residual, the bounded-controller -59-degree
baseline, tiny requests, the real mapper -37.8 nominal / -27.8 native headroom
case, both RL/RR geometry pairs including clipped134-degree synthetic requests,
wheel signed zero/subnormal behavior, every buffer with headroom enabled,
and missing/unknown/unexpected/tampered headroom evidence, reserve, interval,
native, effective/combined values, request/controller and geometry fields.
NaN is specially represented only in read-only before/after snapshots; the
actual audit receives the original invalid input and must reject it.

Boundary fixtures warm the frozen mapper on CPU, then move all live target
tensors to CUDA before the final real adapter dispatch. This avoids thousands
of irrelevant GPU setup writes; it is explicitly synthetic initialization,
not an observed PhysX trajectory. Geometry fixtures use the actual pure helper
with synthetic Jacobians. Mutating/write/readback methods are forbidden during
the compared audit calls; byte snapshots and mapper/ACK/request state must stay
unchanged. Full sorted JSON equality is required; no allclose substitution.

Only if every required comparison passes does timing run four cases: ordinary,
geometry-only, headroom-only and geometry+headroom. It measures the complete
host-call wall time, including each call's synchronization and list creation,
with alternating AB/BA order and equal counts. Fixture preparation and JSON
comparison are outside timing. This is not a simulation/whole-training speedup.

## Command — ONLY after root clears the live barrier

```powershell
$qaResult = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\native_audit_candidate\cuda_e99_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '.json'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -P 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_semantic_v3\reports\native_audit_candidate\validate_cuda_candidate_e99.py' --run-cuda --ack-exclusive-device --device cuda:0 --warmup 5 --samples 40 --max-seconds 120 --output $qaResult
```

Do not run while Isaac, PPO, another locked-Python process or other GPU work is
active. Startup process detection is a snapshot, not a scheduling lock; root
must maintain exclusive-device ownership. No flags/`--help` import Torch.

Recommended finite budget:120 seconds after CUDA setup,5 warm-ups and40 samples
per function per timing case (360 timed/warm-up calls total). The launcher
checks time between finite cases/iterations, so one in-flight call can overrun
the boundary; it is not an OS hard timeout. It stops if Torch reserved tensor
allocation exceeds256 MiB; this excludes CUDA driver/context overhead and is
not a total-VRAM guarantee. Host/Torch threads are one. No optimizer/checkpoint
is constructed. Partial evidence is retained in the exclusively created JSON;
an equivalence failure skips timing and exits2, not a conditional adoption.

Remaining uncertainty is unchanged: GPU/CPU rounding, subnormal/FTZ and device
behavior can differ; exact parity may fail. Explicit synthetic event ordering
does not prove live PhysX producer-stream visibility. Even a pass and favorable
microbenchmark would only motivate a separate reviewed integration decision,
not prove throughput improvement or create an optimizer/task-success gate.

## Actual exclusive CUDA receipt — 2026-09-06

Root executed the pinned launcher once after the evaluation exited and the
device was exclusive. Immutable result file in this directory:
`cuda_e99_20260906T2044049294369Z.json`.
Recorded start/end: **20:44:06–20:44:17 UTC** (second-resolution receipt times,
not a performance benchmark). Status is **TENSOR_EQUIVALENCE_FAILED**;
`equivalence_failures=12`, `timings=[]`, and timing status is
`SKIPPED: exact equivalence or rejection/state checks failed`.
Root reported its process/shell closed with nonzero status1. The JSON itself
records the comparison status, not a process-exit-code field; no conflicting
exit code is inferred here.

Device/runtime: NVIDIA GeForce RTX4080 Laptop GPU, `cuda:0`, Torch2.7.0+cu128,
CUDA build12.8, Python3.11.15, one host thread, inference mode enabled. Receipt
source/candidate/launcher hashes match the prepared pins above; this update
only read the receipt and did not rehash or alter sources.

| Host denormal mode | Stream route | Entries | Passed | Failed |
|---|---|---:|---:|---:|
|Preserve|Same producer stream|170|170|0|
|Preserve|Explicit producer event wait|170|170|0|
|Flush|Same producer stream|170|164|6|
|Flush|Explicit producer event wait|170|164|6|
|Total|All routes|680|668|12|

All12 failures are canonical wheel channel9 (front-right ankle), with
`host_flush_denormals=true`. The six distinct fixture descriptions each fail
on both stream routes:

| Fixture group | Requested wheel residuals | Failed entries |
|---|---|---:|
|Legacy/no-headroom subnormal wheel|±1.401298464324817e−45, ±1e−40 rad/s|8|
|Headroom-enabled subnormal wheel|±1.401298464324817e−45 rad/s|4|

In **every failed entry**, the unchanged reference rejects with
`ActuatorTargetEffectError: frozen mapping reconstruction differs from actual dispatch`.
The one-CPU-copy candidate instead accepts, returns `verified=true` and
`actual_mapping_matches_dispatch=true`, and reports zero changed channels with
zero (possibly signed-zero) target deltas. Before/after input-state checks remain
unchanged. Therefore this is an **accept/reject correctness difference**, not
merely harmless JSON rounding, output formatting, or a changed-bit convention.
Moving the comparison/materialization onto a host with denormal flushing can
erase the distinction relevant to the original GPU-side reconstruction check.
The receipt does not isolate a single low-level cast/comparison instruction as
the sole cause, and no such additional experiment was run.

All680 entries passed their input-state-unchanged check. The192 ordinary signed
channel entries (legacy and headroom, all streams/host modes) passed, and no
other fixture names failed. Those are bounded synthetic-case results, not
universal float32 equivalence, real PhysX stream validation, or evidence that
these subnormal requests occur in the robot's actual policy. Passing ordinary
cases does not authorize dropping the failing cases, changing the reference
audit, or loosening comparison acceptance.

**Disposition: candidate remains unwired and is not approved.** No timing
samples, speed ratio, production change, fix, retry, new CUDA/Python execution,
or adoption decision was made in this receipt update. Existing production
audit and the ongoing P10 training remain unchanged. The original c34262
161-test CPU receipt also remains preserved, without being counted as a pass
for this failed e99 GPU revision.
