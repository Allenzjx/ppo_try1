# Block03 wall-cost check (read-only)

Scope: sealed natural-P01 training run `20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff`; current training stepping/logging/render path. No physics launched, no live-process profiling, and no production/configuration changes.

## Result

No accidental per-step training video render or clearly safe, proven material logging-only speedup was identified. Keep the current block running unchanged. Existing evidence localizes most wall cost to collection (including physical stepping, synchronous sensor/controller/audit work and ordinary decision serialization), but does **not** separate those components enough to identify their individual contribution.

## Evidence

- Recorded collection/update/save loop wall time: **2014.497245 s** (33.575 minutes), for 2048 actual decisions and 16378 credited physics ticks (**136.483333 simulated seconds**, 14.7600 wall seconds per simulated second). `semantic_training.py:1871` starts this timer after environment construction/initial reset; later episode resets remain included.
- Consecutive rollout snapshots 1559–1573 average **125.643738 s per 128-decision batch**. File timestamps are coarse execution landmarks, not a component profiler.
- For all 16 updates, time from completed rollout snapshot to completed likelihood-audit JSON is **0.514628–0.657848 s**, mean **0.562346 s**, sum **8.997528 s**. The code writes the rollout before the official PPO update and the likelihood JSON near the update's end. Thus official updates plus their audit are not the multi-minute-per-batch bottleneck (about 0.45% of recorded loop wall time in these landmarks).
- Corresponding checkpoint file writes occur another **0.054672–0.062405 s** later (sum 0.939316 s). This specifically measures the checkpoint-file landmark, **not** all subsequent round-trip verification/manifest publication.
- Training decision audit is 223389138 bytes, approximately **109077 bytes per decision**. Size alone does not measure JSON serialization or disk cost. Ordinary decision lines use a buffered `write`; there is no ordinary per-physics-tick file flush/fsync here. Terminal evidence flush/fsync occurs before reset; rollout advantage fsync and update-stream flushing occur at update boundaries. Removing these durability checks is not justified by the timestamps.

## Actual call path

- `semantic_cli.py:1090` builds the scalar training core with `collect_trace=False`, not the video recorder's tick observer.
- `semantic_env.py:150` executes eight physical ticks per policy decision and preserves native tick evidence; a tick observer is optional (`:189`). This semantic path, **not** the old generic `PPOEnvAdapter.step`, is relevant to this run.
- `semantic_backend.py:249` delegates to the inherited one-write/one-step path. `isaac_fsm_backend.py:1994` calls `scene.sim.step(render=False)`; synchronous actuator readback, sensor read, controller update and authoritative frame construction follow. Same-tick target-effect evidence is constructed before the physical step (`:1974`).
- Explicit viewport rendering is isolated in `isaac_fsm_backend.py:2352` (`render_video_frame`), not called by the training collector. No policy-loop sleep or repeated source-file read was found in the inspected path.
- The actual HISTORY request audit hooks the one real actor forward and records additional CPU-readable diagnostics (`semantic_training.py:1612`); it reports zero additional model forwards/random draws. GPU-to-CPU synchronization and repeated per-tick physical/audit data processing remain plausible collection costs, **not measured causes**.

## Decision

No change recommended from this bounded evidence. Do not disable rendering that is already disabled, reduce 120/15 Hz, drop physical/event/actuator evidence, change controller order, or remove save-boundary durability for an unmeasured speedup. If performance becomes a separate authorized task, modest offline boundary timing on a future explicitly versioned run could distinguish physical step/readback from controller/audit/serialization; it is not a prerequisite for the ongoing PPO block.
