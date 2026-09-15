# P01–P03 reference / runtime: bounded read-only review

Scope: 2026-09-14 task, latest `bf884f92.../pasted-text.txt` read completely. This review owns this file only. No production, reward, tests, checkpoint, physics, or active B0/C0 logs were changed/read. Root supplied current HEAD `4b2c038`, production-identical to d4; actual fresh process identity and video results remain root-owned. No parent/project AGENTS.md existed at the three checked paths. This is not a new whole-Recording replay, source re-hash, training result, or proof of full tracking.

## Reference retained, known gaps already fixed

Reused `outputs/ppo_fsm_reference_p09_stable_v2/first_execution_divergence.md`, selected reference JSON, immutable Trial043 manifest and **one first command row**, plus current production callers. Source directory is `C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1`; source run is `runs/trial_043_20260902_clean_v010`. Prior verified source identities and physical readjudication remain historical evidence, not a new A run. The report records actual source remote as ppo_try1, not the differently named user URL.

The source compact Full12 sequence is still in `configs/recording_motion_contract.json`:

| Phase-local time, s | Source request in canonical coordinates (degrees; wheels rad/s) |
|---|---|
| P01 0 / .066667 / .133333 / .266667 | RL hip +9 / +17.5 / +33.4 / +37.6; all other servos 0 |
| P01 .333333 → 13.266667 | Four wheels +.3, then an explicit four-wheel stop; RL hip held +37.6 |
| P02 0 / .066667 / .266667 / .4 / .466667 | FR knee +7.8 / +30 / +41.6 / +44.8 / +45.9; source phase-entry RL hip +37.6; RR hip/knee 0; source wheels 0 |
| P03 0 | Atomic FL knee −22.9, RL hip +17.5, FL wheel −.79, RL wheel +.61; FR knee +45.9 and RR pair 0 retained |
| P03 1.2 | Explicit four-wheel stop; servo request retained |

The .61 RL wheel value above is the untuned compact request: existing source StateSpec's P03 logical correction fraction −.149 is applied by the actual MotionExecutor. It must not be mistaken for an absent correction by inspecting JSON alone. `NominalMotionProvider._start_source_motion()` now forwards StateSpec normal correction and time scale, and `_source_normal_bias()` handles finite source post-mapper bias. Prior P03 correction loss, P06/P07 blank handoff/pre-slew, and P05 late-capture wheel gap are already fixed; **none is a newly discovered defect here**.

The full12 contract explicitly uses zero-order-held authored requests, all simultaneous channels per native write, and the mature mapper's 150°/s (1.25°/120-Hz tick) slew/tracking. Waypoint target jumps are not actual physical angular velocities. Runtime FSM does not chase a Recording cursor or apply an arbitrary playback multiplier. First actual A command confirms P01 tick0 nominal RL hip 9°, mapped/final 1.25°, RL-hip tracking enabled.

## Confirmed remaining N/A semantics to measure, not new bug verdicts

1. Original successful A manifest: P01 0→13.333333 s, P02 13.333333→13.866667 s, P03 13.866667→15.133333 s. Its lifecycle includes the source endpoint plus result verification. Current N advances P01 on live `transfer_ready_FR`, not A's exact reference endpoint/pose gate. `TaskStageSupervisor` also accepts already-qualified current FR AIR as an entrance. Historical **video11 only** entered P02 at episode tick16/.133333 s; that fact is not a new B0/C0 result and is not proof that RL preparation disappeared.
2. N explicitly retains unfinished P01 source owners during P02. Each layer owns only channels it actually changes; a new FR knee owner does not automatically replace continuing RL hip or wheel ownership. Thus early P02 may overlap the RL transfer and finite P01 wheel timeline; A's P02 follows A's completed P01. This is a real scheduling difference, originally intended to allow continuous physical task entrances, not an instruction to restore A's 13-second wait.
3. N's P02 `_approach_assist_required()` can add the P01-derived four-wheel +.3 suggestion after physical FR active-lift history, current clearance ≥15 mm above top, front distance <−5 mm and ≥2 other supports. A's compact P02 has no such extra approach. Its current implementation does not use an explicit `air` boolean in that helper. This is an existing feedback choice to inspect in measured windows, not evidence that it fired in a new run or caused RR failure.
4. Capture-owned retirement and measured semantic handoffs are still deliberate differences from original A. Same phase label does not mean same state or source-owner age. Zero and PPO share N; no zero-policy shortcut to A was found in this bounded review.

First useful comparison: verify reset observation/standing mapping and first issued Full12; then inspect episode ticks 1–64 (.008333–.533333 s) around RL hip onsets, first FR unload/Q, P01→P02, and first source wheel launch. Expand only up to the first sustained RR tracking deviation. The historical video11 t256 sample already had target −3.305823° vs actual −11.095250°, so reviewing only its t1317 hard-limit tail misses onset. Do not force A and N's later equal-tick trajectories into a residual-only attribution. Align further A windows on measurable FR unload/AIR/Q, geometry and support, and retain each run's own source age.

## Critical clock / coordinate join

- Original A `full12_commands_120hz.jsonl` row `control_physics_tick=t` is issued **before** physics t→t+1; join its tracking result to `observation_120hz.jsonl.physics_tick=t+1`. Same-tick A observation is the pre-command state. Verified in `infrastructure/app_runtime.py` command-write→sim-step→readback→sensor-read order.
- New `source/native_tick_audit.jsonl.episode_physics_tick=t+1` is already the post-step endpoint. Join with `source/physical_observations.jsonl.physics_tick=t+1`. Its nested `native_audit.physics_tick` belongs to the adapter-native clock and includes settle offset; do not plot it as episode time.
- `native_drive_target_full12` is **canonical** degrees/rad/s after mapper shaping, despite its name. `actual_native_targets.servo_position_rad` is real joint-coordinate radians. Convert with each run's captured standing pose and sign: canonical actual = (degrees(q_native) − standing_deg) / sign. RR and RL servo sign −1; FL and FR +1. Do not apply the offset twice. Wheel signs FL−/FR+/RL−/RR+.
- Define signed `e_tracking = q_actual_canonical − final_target_canonical`; physical observation's existing `joints[name].error_deg` is **the opposite sign**, command−actual. Make this explicit in CSV.
- Define requested `e_residual = final_target − same-run mapper-N target`, but separate `controller_drive_bias_full12`, geometry adjustment and final slew/headroom. This difference is not automatically pure learned residual. For direct same-tick PPO target effect, existing `actual_native_targets − counterfactual_native_targets` removes only current PPO residual from the **same pre-tick state and previous final target**. That is not the independent B0 trajectory, and it does not remove earlier-policy or other-channel physical effects.
- `e_reference` must be separately labeled event-window diagnostic. Do not call post-divergence A/N differences the direct current residual.

## Existing synchronized measurements and missing drive evidence

`semantic_legacy_evaluation.py:39–48` already serializes full 120-Hz raw physical evidence, shared by current video modes:

| Need | Existing field / source |
|---|---|
| Canonical joints actual/velocity/final target | `physical_observations.joints.<exact_joint>.position_deg / velocity_deg_s / command_deg`; also actual_full12 / commanded_full12 |
| Four wheels target/actual | `wheels.<exact_ankle>.command_rad_s / velocity_rad_s`, geometry center/bottom world coordinates |
| Raw / projected / target effect | native audit `raw_policy_action_full12`, `phase_mask_full12`, `projected_residual_full12`, actual/counterfactual native targets, headroom evidence |
| Nominal / mapping | native row `nominal_full12`; nested native audit `native_drive_target_full12`, `controller_drive_bias_full12`, `combined_post_mapper_bias_full12`, `previous_final_drive_servo_deg`; underlying ACK additionally has mapper tracking compensation/active flags and physical target buffers |
| Actual body state | `base.position_w_m`, orientation_wxyz, world linear/angular velocity; `imu.projected_gravity_b`, body angular velocity; derive labeled Euler angles if necessary |
| Full-body CoM | `center_of_mass.position_w_m / velocity_w_m_s / total_mass_kg / included_bodies / source / valid / reason`; mass-weighted from live body_com_pos_w, body_com_lin_vel_w and default_mass, **world frame**, not the normalized actor feature |
| Contacts / forces | `contacts.<body>.ground / obstacle` exact pair records: active, force_w_n, normal_force_n, pair_verified, histories; do not treat unverified/missing load as zero |
| Task events / quality | `stage_transition_evidence` plus current semantic task evaluator history/current_legs; per-decision semantic task contains role contexts, load validity, FR geometry and lift/cross/place events |

Body/leg identity comes from exact joint names and prim mapping in RobotAdapter/geometry, never image left/right. Existing recorder retains initial observation tick0 but ordinarily encodes first post-control endpoint; root's media provenance/pre-roll treatment is separate and this review does not relabel it.

Configured drive path: `infrastructure/scene_factory.py:29–45,282–330`, `configs/environment_lock.json:40–54`. A direct no-index comparison of current vs frozen `scene_factory.py` was empty (exit0): identical source, not a physics-run equality claim. Hip/knee use `ImplicitActuatorCfg`, position target, stiffness600, damping60, effort_limit_sim2.7 N·m, armature .005; servo velocity_limit_sim=None. Wheels use velocity target, stiffness0, damping20, armature .002, effort_limit_sim=None, velocity_limit_sim2.0943951023931953 rad/s. Mapper slew150°/s is a target shaping limit, not evidence for a separately set physical servo speed cap. Root source USD path is `C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd`; no fresh asset hash here.

RobotAdapter captures all eight standing angles from actual joint_pos and resolves exact joint IDs. Its initialization-only live session overrides the eight authoritative command-space limits using exact RevoluteJoint names under `/World/WLRRobot`, with PhysX limit readback; this is the frozen source behavior, not an episode-internal state write. A's saved environment_initialization has limit/mapping records but does **not** prove effective per-joint stiffness/damping/effort readback. Local joint axes, local rotations, collision joint flags, effective gains/limits and force saturation should come from actual loaded prims/articulation if needed. They were not present as measured fields in this bounded review and must remain unknown until inspected. No production src fields for computed_torque, applied_torque, joint effort or solver joint reaction were found; none may be fabricated from target error or configured 2.7 N·m.

The reused live backend order is: one RobotAdapter write_data_to_sim → optional read-only target audit → one sim.step → RobotAdapter.update_readback (robot.update(dt)) → sensor read with final target → evaluator. ACK/target equality is useful mapping evidence, **not** evidence that a finite-force joint actually followed target.

## Reward availability / hypotheses, not causal conclusion

Current `reward_config.yaml` is `role_transfer_body_motion_applied_only_v1`: task potential5, terminal±40, time−.02/s; body family .4, contact .2, applied smoothness .1; residual magnitude cost0. Transfer motion blend is `1 − .8*physical_transfer_fraction`, applied to attitude, Euler rates and angular acceleration. Scales .5 rad /2 rad/s /20 rad/s². Costs integrate real executed dt and saturate normalized squared terms; no FSM-angle imitation or nonzero-action bonus.

`TaskStageSupervisor` selects measured role `motion_fraction` for eligible not-yet-placed target legs; current role version overrides the older load-only fraction. `TransferRoleTracker` keeps .5 s physical windows and independently marks unknown loads, with a .066667 s evidence scale. AIR alone does not simply force maximum attitude cost. Existing motion_fraction can nevertheless decline if measured activity/continuation disappears or capture grows; actual values and signed cost integrals must be inspected before saying required held transfer was penalized. `pending_capture` exists but is not alone the motion weighting formula.

`SemanticRewardCalculator.evaluate()` returns actual signed `families`, `unweighted_families`, `potential_before/after`, `potential_shaping`, `terminal_event`, elapsed_physics_s and integrated `cost_components`. Per-decision logs can support signed family integration; finer 120-Hz role/cost reconstruction may require replaying the pure evaluator from initial state with exact stream, explicitly labeled derived rather than measured. Do not equate lower raw episode total across different lengths with quality, or offline reward changes with new learned actions. No reward conflict, torque deficiency, wrong axis, stale sensor, or new nominal omission has been established by this static review.

Conclusion: keep videos-first priority. Most informative next evidence is short synchronized new B0/C0 onset window with the clock conversion above, then A event-window if needed. Preserve historical −3.37° target /−60.018° actual as a genuine historical tracking failure, not a command to −60° or proof of the latest checkpoint's behavior. No additional optimizer gate or speculative physical/reward change is proposed.
