# N8 readback profiling and exact-value cache proposal

Read-only source review during the live N8 run. This is not an implemented
optimization or a measured speedup claim. Parent reports about 7.95 credited
decisions/s for N8 sampling/update versus about 2.9 for N1; those aggregate
numbers do not identify reset, transfer, Python sensing, or optimizer shares.
N16/N32 benefit remains unknown until the N8 boundary profile is available.

## Evidence-backed candidates

| Path | Repeated work visible in current source |
|---|---|
| `ppo/actuator_target_effect.py:108–175` | Per row/tick, success path has four selected-target finite checks ending in `.item()`, four `torch.equal` calls, four expected/counterfactual `new_tensor` constructions, and seven `.cpu().tolist()` calls for changed/actual/counterfactual/delta records. These are call counts, not measured transfer counts or timings. |
| `ppo/vectorized_isaac_backend.py:562–574` | Each row's actual joint readback separately transfers servo position, servo velocity, and wheel velocity. `apply_batch:602–625` also reads measured servo positions row by row for the unchanged mapper. |
| `sensing/sensor_reader.py:396–433,459–467` | Seven body/mass fields plus three IMU fields pass through `.detach().cpu()` per row. CUDA-resident fields incur small readbacks; already-host fields need not transfer. |
| `ppo/vectorized_isaac_backend.py:268–277` | Row-local body/root position accesses clone and subtract the origin on the source device before reader conversion to NumPy float64. Changing this operation's dtype/order can change values. |
| `ppo/vectorized_isaac_backend.py:368–418` | Contact bank already captures the full batch once per global tick, rather than eight sensor refreshes. It still has four source field conversions per each of 13 sensors. Preserve exact pair/filter and inactive-point NaN semantics. |
| `ppo/semantic_vector_env.py:234–258` | Projection, row consumption/observation/reward, and optional observers are additional CPU work outside backend stepping. Current train CLI (`semantic_cli.py:471`) does not supply the smoke `_RowProbe` observers; their JSON output is not a demonstrated training bottleneck. |

## One bounded profiling change first

Add opt-in host `perf_counter` accumulators to the new semantic vector path,
emitting one summary per rollout/update, not logging every tick. Record count,
total and sampled duration distribution for:

1. Reset: physical lifecycle, command-adapter rebuild, unchanged 180 settle
   steps, reader construction, controller/config construction, light semantic
   frame-builder construction, and each row's first read/construction.
2. Each physical tick: prepare/project; mapper + actual batched dispatch;
   native audit; actual physics + scene update; exact-pair capture; each row's
   read, controller evaluation and authoritative-frame construction; semantic
   row consume; optional observer time.
3. Decision/update: row finish and TensorDict conversion, peer-final-value
   computation/reset separately, raw/evidence serialization, official PPO
   update and checkpoint publication separately.

Do not insert `cuda.synchronize`, extra sensor reads, mapper calls, renders or
physics steps. Host timings describe observed blocking boundaries, not pure
GPU kernel duration: lazy GPU work can be charged to the first blocking read.
Include credited decisions, executed global ticks, reset count and reset-only
ticks so startup/reset overhead is not silently reported as sampling throughput.
Use the normal real run; no new success gate or extra task-completion prerequisite.

## Smallest cache candidates, only after those measurements

Two distinct capture epochs are mandatory; they must not be collapsed:

**A — actual dispatch receipt, before the next physics step.** Immediately after
the existing one `write_data_to_sim`, validate and capture the four real N-row
float32 setter/dispatch target tensors. Tag the capture with reset generation,
global command tick, actual write count, canonical joint IDs, source device and
source attribute names. Compare setter versus dispatch and reconstructed actual
mapping with the same strict equality rules. A capture is invalid after the next
write or reset; it must never be made from desired targets, an ACK list, or a
counterfactual. Keep an explicit captured-receipt API rather than replacing the
robot facade with fabricated tensors that merely look like native buffers.

The existing scalar mapper/slew/clamp/physical conversion remains authoritative.
If audit profiling dominates, compute all rows' existing scalar expected and
counterfactual physical values, cast their N-row tensors on the original device
and dtype, then compare/copy batches. Preserve float32 delta arithmetic on that
device as well; moving subtraction/casts to Python/NumPy can alter rounding or
subnormal treatment. Only validated captured slices feed the row audit records.

**B — actual kinematic readback, after the existing physics step and scene update.**
Capture each required N-row articulation field once, then let the existing
SensorReader consume read-only row slices. Required fields are `joint_pos`,
`joint_vel`, six body pose/velocity fields, `default_mass`, `root_ang_vel_b`,
`projected_gravity_b`, and `body_com_lin_acc_w`. Keep their values/dtypes and
current lazy-property semantics. Localize body positions using the existing
float32 source-device subtraction before host conversion; do not subtract a
float64 NumPy origin or share world coordinates between clones. Snapshot memory
must not alias simulator buffers that will mutate on the next tick.

Use distinct readback generation/tick checks from dispatch epoch A. Invalidate
both caches on all successful and failed reset paths, and capture settled tick
zero afresh. Do not initially change the frozen mapper's per-row measured-input
path; profiling may show that a later explicit measured-input reuse seam is
worth its additional review. No stale fallback is permitted for a missing field.

## Reset geometry option

`_make_readers` currently creates eight new USD providers and geometry caches on
every reset (`vectorized_isaac_backend.py:1341–1364`). Their first reads repeat
USD mesh traversal and exact vertex transformations (`geometry.py:117–133,
245–258,351–435`). A bounded candidate is backend-owned **per-row immutable
body-local collider points and collider-path provenance** across hard resets of
the same unchanged USD scene. Invalidate for a different stage/robot prim,
geometry/transform revision, or failed provenance check. Do not share across rows:
the original float computations and path provenance may be row-specific.

Always create fresh readers, contact backends/classifiers, guard trackers,
TaskEvaluators, mapper/filter history and finite/failure flags. Also create a
fresh `ColliderGeometryCache._shapes`: recompute bounds from the current measured
pose using exactly the old point transformation sequence. Reusing the old
world-oriented measured offsets or replacing the mesh by AABB corners changes
geometry and is outside this proposal. Immutable local-point reuse saves only
extraction, not the required current-pose point/bounds calculation; speedup is
unmeasured and must not be promised.

## Focused acceptance for any later implementation

Compare old/new consumers of the same captured real buffers: float32 target bits,
numeric +0/-0 effect semantics, ULP-edge counterfactuals, all native audit fields,
raw observations, reward, projector state and terminal/peer-bootstrap results.
Include distinct row origins, channels, signs and histories; NaN inactive contact
points; invalid dtype/shape/IDs; stale tick, reset and post-write receipts; and
attempted alias mutation. Test two hard resets with new native view identities
and unchanged immutable geometry provenance, plus a changed-stage invalidation.
Verify one write/step/capture counters are unchanged. Physical settings, masks,
action scales, task clocks, success/safety semantics and evidence retention stay
unchanged. Report measured segment times and actual end-to-end throughput; no
claim follows from source call-count reduction alone.
