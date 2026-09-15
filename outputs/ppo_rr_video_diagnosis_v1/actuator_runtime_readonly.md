# RR actuator configuration — bounded read-only evidence

Scope: production and installed dependency source inspection; source USD opened read-only. No Isaac/SimulationApp/Torch load, live-process attachment, physics step, current video-log read, production/config edit, or asset save. This map does not establish actual RR torque saturation or the cause of tracking error.

## Configured path and values

Project: `C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`.
Production factory: `src/wlr50_clean/infrastructure/scene_factory.py`.
Actual installed editable mapping, read from `C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/__editable___isaaclab_0_54_3_finder.py:9`, maps `isaaclab` to `C:/robotics_sim/IsaacLab/source/isaaclab/isaaclab` (distribution 0.54.3). No claim of imported current-process module identity is made from that mapping alone.

| Actuator group | Configured joints | K / D | Effort limit in simulation | Velocity limit in simulation | Armature |
| --- | --- | --- | --- | --- | --- |
| `hip_knee_position_servos` | All eight hip/knee joints, including RR | 600 / 60 | 2.7 N m | `None`: inherit imported native default | 0.005 |
| `wheel_velocity_motors` | All four ankle/wheel joints | 0 / 20 | `None`: inherit imported native default | 2.0943951023931953 rad/s | 0.002 |

Both use `ImplicitActuatorCfg`; there is no special RR gain/effort group. `None` is not evidence of unlimited effort or speed. The inspected runner, video entry, semantic backend, scripts and selected config contain no extra gain/effort/armature override. The project-wide search found these actuator parameters in the scene factory, not PPO learning configuration. Controller feedback, nominal/residual target shaping and position-limit initialization are distinct mechanisms and must not be described as actuator-gain changes.

Source: factory constants lines 31–48 and `_build_robot_cfg:282–328`; `command_batch.py:74–77`.

## Installed implicit actuator behavior

Read local sources:

- `C:/robotics_sim/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:38–146`: the implicit actuator leaves input position/velocity/effort targets unchanged. Its `computed_effort = K*position_error + D*velocity_error + feedforward`; `applied_effort` is the clipped approximation. Neither is a measured PhysX motor-force readback. In particular, `robot.data.applied_torque` on this path cannot prove actual instantaneous drive saturation.
- `.../actuators/actuator_base.py:177–215`: explicit actuator-config values override imported defaults; `None` retains imported values. This resolves speed/effort/gains/armature/friction into actuator tensors.
- `.../assets/articulation/articulation.py:1708–1793`: reads matched live joint IDs, builds actuators from PhysX-derived default buffers, writes implicit K/D plus effort/speed/armature/friction into simulation. `default_joint_*` is subsequently overwritten with configured actuator tensors, so that cache is not an independent effective-PhysX proof.
- `.../assets/articulation/articulation.py:221–264,1869–1903`: `write_data_to_sim` processes the actuator then sends target buffers to PhysX; `update(dt)` refreshes data timestamps. This is an implicit PhysX drive, not an explicit torque-command PPO policy.

A configured 2.7 N m limit makes insufficient available drive torque one hypothesis for large tracking error, but does not establish that torque was at the limit, that contact caused it, or that increasing the limit is appropriate. No physics change is proposed here.

## Actual source USD joints, not live PhysX configuration

Locked asset path is `C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd`; factory declares SHA-256 `e8a2a2b1485a32a50e851a07b9dd8ac4945b78ec49b7fada2b61c3eeb1e18892`. This review did not repeat the whole-asset hash. The 45,607,185-byte file has USDC header. It was opened with the installed standalone `pxr.Usd`, `LoadNone`, without importing Isaac or starting a simulator. All three joints were found; layer dirty list stayed empty; no export/save was called.

Asset prim prefix: `/wlr_robot_isaac/joints/`. All three RR joints are `PhysicsRevoluteJoint`, with authored local axis `Z`, localRot1 `(1,0,0,0)`, and angular drive type `force`.

| Joint | Parent -> child | Authored localPos0 | Authored localRot0 (w,x,y,z) | Source lower/upper degrees | Source max velocity degrees/s | Source angular drive K / D / maxForce |
| --- | --- | --- | --- | --- | --- | --- |
| rear_right_hip | base_link -> rear_right_upper | (-0.17017756,-0.24299191,0.11001223) | (0.00066369283,0.00066375465,0.7071065,-0.7071065) | -89.9543685913 / 89.9543685913 | 286.4788818359, approximately 5 rad/s | 27.0088615417 / 0.01080354396 / 2.70000004768 |
| rear_right_knee | rear_right_upper -> rear_right_bot | (0.14779,0,0) | (0.7224549,0,0,0.69141805) | -179.9087371826 / 0 | 286.4788818359, approximately 5 rad/s | 7.96543788910 / 0.00318617537 / 2.70000004768 |
| rear_right_ankle | rear_right_bot -> rear_right_wheel | (0.1559,0.02,0.0784) | (-0.0000029365256,0.8111971,-0.5847728,0.0000021168719) | -inf / inf | 1145.91552734375, approximately 20 rad/s | 0 / 1.77165329456 / 1.0 |

All localPos1 values were `(0,0,0)`. USD angular-speed units were verified in installed `omni.usd.schema.physx-107.3.26+107.3.3.wx64.r.cp311.u353/plugins/PhysxSchema/resources/schema.usda:1512–1526`. The source-drive K/D values above are literal authored USD attributes, not the configured IsaacLab SI tensors or final effective PhysX parameters; do not compare them without the correct unit convention. Runtime wheel speed is explicitly overridden to approximately 2.0944 rad/s rather than the source 20 rad/s. Runtime servo speed is expected to inherit the imported approximately 5 rad/s, but current-run effective readback remains to be checked.

`Z` is the local **joint-frame** axis, not world vertical. Joint-frame rotations and body pose matter. Both RR hip/knee logical command signs are -1; RR ankle forward sign is +1 (`command_batch.py:52–71`). A negative logical RR knee action cannot by itself prove world-space downward foot motion. Production nominal-geometry calculations instead use the actual articulation Jacobian and matched joint IDs.

## Existing position-limit initialization and fixed step order

The source USD position limits are deliberately not the final runtime limits. This is inherited behavior, not a new repair:

1. Factory creates PhysX from the locked source asset/config, calls `sim.reset()` and `robot.update(0)`.
2. `RobotAdapter` resolves exact joint IDs and captures actual standing angles. It authors only the eight servo lower/upper pairs into the live session layer, deriving physical bounds from the standing angle and canonical sign/limits; it synchronizes diagnostic caches. Hip command bounds are [-135,135] degrees and knee [-60,210] degrees. It does not save the source asset.
3. The original 180 zero-command settle ticks execute. `verify_authoritative_servo_limits_adopted()` then compares actual `root_physx_view.get_dof_limits()` against all authored servo bounds at 2e-6 rad tolerance; reset evidence records this verification. Therefore there is already an effective position-limit readback gate, but not an equivalent structured gain/speed/effort readback in the inspected project streams.
4. Reused-episode reset removes the exact old session-limit opinions before hard PhysX reset, then reauthors them through the new adapter. The current fresh natural-P01 video follows the first-scene order, not an indexed state snapshot.

Normal episode tick order (`isaac_fsm_backend.py:1946–2037`, `robot_adapter.py:288–300,380–383`):

`source controller frame -> nominal mapper + independent residual/feedback -> position-target setter for 8 servos + velocity-target setter for 4 wheels -> exactly one write_data_to_sim -> read-only staged/dispatched target audit -> exactly one sim.step(render=False) -> robot.update(1/120) -> sensor read with dispatched command -> controller step using that new measured state -> next authoritative frame`.

No extra physics advance is introduced by the same-tick target audit. Its target equality proof is not a gain/force-cap proof. Position/velocity target buffers are verified before the physical step; actual q/qd is read afterward. The mapper's 150 deg/s commanded slew, 4-tick tracking feedback, gain 8 and maximum 10-degree compensation are target shaping, not the underlying servo PD K/D or physical velocity cap.

Source: `robot_adapter.py:105–150,180–195,391–451,454–569`; `isaac_fsm_backend.py:739–785,6172–6301`.

## Frozen A comparison and remaining readback

Read-only `git diff --no-index` comparisons found no differences for scene_factory.py, robot_adapter.py and command_batch.py between this project and `C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1/src/wlr50_clean/infrastructure/`. This is source equivalence, not a claim of an already verified matched live A/B/C actuator state. Semantic B/C reuse this backend/adapter, with selected semantic control/measurement changes above it.

First reuse existing evidence: installed Articulation initializes the actuator configuration and then calls `_log_articulation_info()` (`articulation.py:1580–1591,1952–2027`). Its INFO table headed `Simulation parameters for joints in /World/WLRRobot` reads actual `root_physx_view.get_dof_stiffnesses`, `get_dof_dampings`, `get_dof_armatures`, friction properties, `get_dof_limits`, `get_dof_max_velocities`, and `get_dof_max_forces`. A bounded excerpt from each **completed** B0/C0 launcher may already prove effective initialization values, if logging retained it. It is rounded text at initialization, and its position limits precede the later adapter session authoring; combine it with existing post-settle limit verification rather than treating that early table as final limits. This review did not inspect active launcher logs.

If those existing logs cannot establish the effective fields, the **one minimal proposed capture** is a single read-only JSON snapshot from the existing live backend immediately after the ordinary settle and limit verification, before the first P01 action. Record PID, run identity, episode tick 0, physics dt, module/class paths, joint-name/index map; selected configured actuator tensors; and one call each to the seven relevant `root_physx_view` getters above (friction as the version-5 tuple). Record values keyed by all 12 exact joint names so RR can be compared with the other legs. Reuse the existing reset receipt for standing/sign/position-limit mapping. This capture must not call a setter, reset, forward, update, render, step, or stage-save operation. It is a proposed instrumentation read only, **not performed or implemented here**; do not interrupt the media runs just to add it. A single getter snapshot proves effective configuration at that boundary, not instantaneous motor torque or contact impulse later in P09.

Status: report only. No gain, damping, torque, friction, speed, action range, reward, or physics modification; no new simulation or checkpoint load.
