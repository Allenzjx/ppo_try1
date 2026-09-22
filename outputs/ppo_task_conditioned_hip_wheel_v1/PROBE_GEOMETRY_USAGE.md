# Optional four-hip probe evidence

`probe_geometry.py` is a standalone output-only helper. It is **not integrated**
into the active FR probe, production observations, reward, or evaluator. It does
not launch Isaac, write a file/state, step physics, or construct CoM.

At a future legal boundary, after the existing `height.start(core.frame)`:

```python
from probe_geometry import ProbeGeometry
probe_geometry = ProbeGeometry(height)
# In the same post-step sampling location as HeightDiagnostics:
geometry_row = probe_geometry.sample(core.frame)
# Caller owns any separate output stream; this helper returns only data.
```

The caller already has project `src` in sys.path; this helper resides alongside
direction_probe.py. USD is imported lazily only when resolving joint definitions.
Four hip points come from each named USD joint's authored body0/localPos0,
transformed with its current parent body_link pose. An unresolved individual hip
is null with a reason; no fallback to an upper-leg origin, base origin, or CoM.

Body AABB and lowest vertex reuse `HeightDiagnostics._bounds('base_link')`:
independently loaded enabled collider mesh at the current live body pose, not a
stale USD world extent. Bounds and the true lowest mesh vertex are both returned.
The global minimum-minus-top signed z value is explicitly a vertical lower
bound; the output also states whether the lowest vertex lies in platform XY,
whether the whole AABBs overlap in XY, and the nonnegative AABB distance lower
bound. None is labeled exact mesh-obstacle clearance; zero AABB separation does
not label a collision. Existing physical evaluator remains authoritative.

`clock_unchanged` and `frame_clock_aligned` must both be checked before overlaying
precise values. Missing/invalid measurements stay null. No unverified contact or
support is invented. Existing HeightDiagnostics still supplies raw CoM and
native joint/drive data separately.

CPU test command (no simulation/app):

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_task_conditioned_hip_wheel_v1/test_probe_geometry.py
```

Eleven tests cover four named mounts, rigid transformation, a real transformed
lowest vertex, missing/duplicate/unauthored/out-of-robot USD frames, invalid pose,
nan/bad geometry, below-top-but-outside-platform geometry, AABB overlap semantics,
and explicit missing values/clock mismatch. These synthetic tests are wiring
evidence only, not live proof that four authored mounts resolve on this asset.
