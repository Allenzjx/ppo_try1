# Unwired native-audit snapshot candidate — CPU correctness only

Status: **candidate only, not connected to any backend/CLI/training run**. Production, frozen mapper/infrastructure, configurations, audit frequency and existing run artifacts are unchanged. The current N1 training was not paused or replaced. No CUDA/Isaac call, GPU benchmark, additional optimizer or physical rollout was run for this prototype.

## Exact scope

Reference: `src/wlr50_clean/ppo/actuator_target_effect.py`, file SHA-256 `7ec65a2e425681d2e172caa409ae0735983d451bdf7346fa0cecba63cb268721` as read at current c34262a runtime. The adjacent tests pin that source revision, not a future changing implementation.

`actuator_target_effect_candidate.py` contains an independent copy of the one audit function. It imports the original error/schema/request/finite helpers and reuses frozen `bounded_drive_feedback_step`, `Full12Command.clamped`, `servo_limits_deg` and `build_physical_batch`; there is no duplicate mapping/sign/standing-offset algorithm and no monkeypatch of the production audit.

The only success-path rearrangement is:

1. Validate each of the four target tensors' existing rank, N1 shape and float32 requirements, and select canonical IDs with the same `detach().clone()` snapshots.
2. Require their devices to match, concatenate staged-servo8 / staged-wheel4 / dispatched-servo8 / dispatched-wheel4 into one **(1,24) float32** snapshot, and call blocking `.cpu()` exactly once.
3. Split the CPU snapshot into the same four arrays. Preserve selected-target finite checks, staged=dispatched equality, reconstructed actual=dispatched equality, same-tick zero-current-policy counterfactual and numeric-not-bitwise changed flags. Expected target construction now uses the CPU float32 slices.
4. Compute deltas and serialize every original field on CPU. The geometry branch still checks its evidence, adjusted channels/delta and reconstructs the raw-no-geometry third branch. It does not make another device copy.

There is **no cached target, live state, previous-final value, controller bias, policy request, geometry or history**. Each call re-reads all four existing buffers. There is no extra mapper advance, controller/sensor read, apply, setter, write, simulation step or phase reset. The caller's original after-dispatch/before-next-physics placement and 120 Hz invocation must remain unchanged if this candidate is ever separately integrated.

All returned keys, list orders, dtypes, actual target source, request metadata, previous-final/native/controller/combined values, delta arrays and geometry evidence remain unchanged. `+0.0 == -0.0` still means no actuator effect, while each serialized actual target retains its sign.

## Actual CPU evidence

Fixed run: **161 passed, 0 failed/errors/skipped**, JUnit `cpu_equivalence.xml`, recorded test-suite time **2.639 s**. This duration is test execution time, **not** an audit speed benchmark. Tests ran under the existing locked Python with CUDA hidden, Torch/OMP/MKL/OpenBLAS restricted to one CPU thread, and a fail-fast CUDA-lazy-initialization guard. The Python process exited with code0 before this summary was finalized.

The fixtures execute the real frozen RobotAdapter/setter float32 casts and real semantic residual dispatch. Geometry tests execute the real nominal geometry helper against an explicitly synthetic local Jacobian; they are not physical contact or kinematic validation. Deliberately extreme residual fixtures test the existing hard intersection and do not assert that the production PPO configuration permits such requests.

Coverage includes:

- Every canonical channel with both residual signs and reversed articulation joint IDs; positive/negative servo residuals1e-9/1e-12 that quantize to no effect; signed zeros and synthetic CPU wheel subnormals.
- Nonzero controller counterfactual, previous-final slew saturation, servo upper/lower limits, wheel hard clamps, and a nonzero raw request whose safe projected residual is zero.
- Both RL/RR geometry pairs, controller bias0/2, zero/positive/negative/slew-clamped/tiny residuals, and unchanged/degraded identity evidence.
- Each of all four buffer tensors independently missing, wrong dtype/rank/batch/width, NaN or infinity; duplicate/short/out-of-bounds/swapped IDs; staged/dispatch/expected mismatch; invalid ACK bias/slew/writes/native/previous state/phase request; incomplete or inconsistent geometry evidence.
- Full result-dictionary **and exact sorted JSON equality** against the unchanged reference, including geometry fields and signed-zero serialization. Single-fault rejection types/messages match. All read-only snapshots remain unchanged.
- Forbidden advance/apply/set/write/update/readback methods during positive audit comparisons; before/after target bytes, mapper state, ACK, policy metadata and write/event counts checked. Successive real dispatches produce fresh results, not stale snapshots.
- AST and a CPU `.cpu()` call spy both find exactly one candidate boundary, shape(1,24),dtypefloat32,with and without geometry. This is a Python-call observation, not a measured count of PCIe/DMA transactions or CUDA kernels.

### Deliberate failure-path distinction

Packing requires validating all tensor structures before transfer. With **multiple simultaneous faults**, a later malformed dtype/shape can be reported before an earlier tensor's NaN, whereas the reference checks each tensor's finiteness immediately. The dedicated two-fault test verifies both reject; it does not require identical error priority. No malformed tensor is accepted because of that ordering.

The candidate also explicitly rejects mixed-device snapshots, consistent with one articulation's common-device target buffers. A CUDA/mixed-device fixture was **not** run. Wrong/unsupported device behavior and error parity require future exclusive-device validation; this prototype is not permission to weaken IDs, dispatch or sensor checks.

## Remaining risks and a later exclusive measurement, if selected

1. **CPU correctness is not GPU parity.** Reference deltas/comparisons currently run on the target device. Moving subtraction/casts to CPU may expose GPU/CPU rounding, signed-zero or especially subnormal/flush-to-zero differences. The synthetic CPU subnormal tests intentionally make no GPU or physical-effect claim. Later validation must compare every returned field/serialized float and rejection on real float32 CUDA snapshots, including subnormals/normal boundaries, before deciding whether this rearrangement is acceptable.
2. **Snapshot timing and streams.** The existing N1 dispatch caller must remain synchronous and single-owner through the four clones and one blocking copy. The candidate adds no background thread, nonblocking transfer or pinned-memory lifetime complexity. A real GPU review must still establish that snapshots follow the original completed dispatch on the correct stream and precede the next physical mutation; this CPU prototype cannot prove PhysX stream timing.
3. **Allocation and total cost are unmeasured.** Four selections/clones plus a small concatenation still cost allocations/kernels. CPU reconstruction/list conversion might outweigh saved tiny synchronization calls. No end-to-end speedup or share of the approximately21.4Hz old aggregate is inferred from source call counts.
4. In a later root-authorized window with no other Isaac/optimizer work, retain one fixed already-dispatched target state and compare reference/candidate alternating order after equal warm-up. Report whole-function host wall median/p95/p99, call counts and GPU event timings with symmetric synchronization; include ordinary,geometry and fail-closed inputs. Do not benchmark only the copy or omit CPU wait time. Use a finite sample count and a fresh external result file.
5. Only if correctness and isolated measurements justify it, a separate reviewed production change and short same-checkpoint/seed/config N1 paired timing could measure dispatch/audit/sim+readback/observation/JSON/PPO/reset costs. Keep full120Hz evidence, all live safety/mapper/sensor checks, normal terminal handling and checkpoint identity. No A/B task-success prerequisite or new optimizer gate follows from this prototype.

Artifacts are limited to this output directory. They are not imported by production and must not be used to claim a faster trained checkpoint, new successful video, physical stability gain or completed future implementation.
