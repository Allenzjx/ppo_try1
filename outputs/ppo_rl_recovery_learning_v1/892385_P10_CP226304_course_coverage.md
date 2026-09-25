# 892385 P10 course from CP226304 — actual sealed coverage

Run `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260924T1717242165088Z_g892385cba8a7_afdc21482bbc480a9692fb63fe8ad858`. Actual 1024 learner decisions / 2 PPO updates / 40 official Adam steps.
Saved CP227328 / PPO1729 / Adam34580; roundtrip true.
Request phases: `{'P01': 0, 'P02': 0, 'P03': 0, 'P04': 0, 'P05': 0, 'P06': 0, 'P07': 0, 'P08': 0, 'P09': 0, 'P10': 5, 'P11': 5, 'P12': 1014, 'P13': 0}`.
Prefix excluded: `{'decisions': 3835, 'physics_ticks': 30680, 'learner_starts': 5, 'PPO_credit': 0}` (zero credit).

| Episode | Learner n | Handoff | End phase/time | Terminal cause | RR placement ownership | RL fresh qualifications |
| --- | ---: | --- | --- | --- | --- | --- |
| 0 | 59 | 6136 / 51.133333s | P12 / 55.050000s | central body/obstacle collision | prefix 6133 | [] |
| 1 | 111 | 6136 / 51.133333s | P12 / 58.491667s | hard joint limit: front_left_knee | prefix 6133 | [6194] |
| 2 | 269 | 6136 / 51.133333s | P12 / 69.033333s | hard joint limit: front_left_knee | prefix 6133 | [6190] |
| 3 | 452 | 6136 / 51.133333s | P12 / 81.266667s | INCOMPLETE_CONTROLLER_BLOCKED | prefix 6133 | [] |
| 4 | 133 | 6136 / 51.133333s | P12 / 60.000000s | nonterminal budget tail | prefix 6133 | [] |

Current-course event counts: `{'student_RR_first_capture_events': 0, 'student_RL_fresh_qualification_events': 2, 'student_RL_qualified_endpoints': 3, 'inherited_or_unresolved_RL_qualified_endpoints': 0, 'student_RL_cross_events': 0, 'student_RL_placed_events': 0}`.

| Physical endpoint window (nonexclusive) | Samples |
| --- | ---: |
| RR_reachable_AIR_preparation | 179 |
| RR_actual_bearing_front_preparation | 126 |
| positive_FR_axis_CoM_body_projection_with_RL_fraction_decline | 69 |
| qualified_RL_edge_recovery | 0 |
| RL_qualified_AIR_capture_region | 0 |
| RL_actual_TOP_bearing | 0 |
| RL_placed_history | 0 |
| P13_actual_stage | 0 |
| all_task_complete | 0 |

Positive local fixed-FR-axis projections of both CoM and body displacement, plus declining RL load fraction while RR and a front leg currently bear. NOT verified lateral/rightward transfer, NOT absolute RL force decline, NOT qualified RL unloading. Forward movement alone can make the projection positive; a load fraction can decline because other legs load more. World xyz components and current absolute forces are reported separately; yaw/body-frame decomposition is not inferred from the projection.

| Episode | RR legal TOP endpoints | TOP losses | AIR->TOP recontacts | Post-ground TOP restoration |
| --- | ---: | --- | --- | --- |
| 0 | 16 | [6240, 6288] | [6256] | [] |
| 1 | 38 | [6264, 6704] | [6640, 6912] | [] |
| 2 | 23 | [6208, 8120, 8208, 8248] | [8152, 8216, 8272] | [8112] |
| 3 | 40 | [6208, 6296, 6400, 6592, 7024, 7056, 7152, 7200] | [6280, 6312, 6544, 7008, 7032, 7096, 7192] | [] |
| 4 | 11 | [6224, 6376] | [6368] | [] |

## Official optimization and separate replay

Five official likelihood exposures per optimized raw index; max independent old Gaussian logp error 2.60072484e-06. Actual replay exposures=1280; added on-policy samples=0, separate AUX Adam=0.
- Completed PPO1728 / global226816: 640 replay exposures; heldout after `{'rows': 99, 'mean_kl': 0.007436131592839956, 'max_kl': 0.014256834983825684, 'phase_counts': {'P01': 1, 'P02': 32, 'P03': 2, 'P05': 32, 'P06': 32}}`.
- Completed PPO1729 / global227328: 640 replay exposures; heldout after `{'rows': 99, 'mean_kl': 0.011718149296939373, 'max_kl': 0.033050477504730225, 'phase_counts': {'P01': 1, 'P02': 32, 'P03': 2, 'P05': 32, 'P06': 32}}`.

Collection actor semantics: each sample uses the actor active before its consuming update; later rollout can use prior completed update
Checkpoint `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\cp225280_front_preserved_v1\checkpoints\history\checkpoint_step_000227328.pt`; SHA `5b871af2df8d6b9c0180f7a51863b18b2153148d8dc2a94876f21f5bd73791b4`; sidecar SHA `0200daa9a22652e1f1c64e5c5ff82a0c977f0418c508158455b3a5ce85bac308`.

## Branch actual totals

{"branch_origin": 225280, "actual_policy_decisions": 2048, "ppo_updates": 4, "optimizer_steps": 80, "request_phase_counts": {"P01": 0, "P02": 0, "P03": 0, "P04": 0, "P05": 0, "P06": 0, "P07": 6, "P08": 6, "P09": 430, "P10": 7, "P11": 31, "P12": 1568, "P13": 0}, "endpoint_physical_windows": {"RR_reachable_AIR_preparation": 294, "RR_actual_bearing_front_preparation": 202, "positive_FR_axis_CoM_body_projection_with_RL_fraction_decline": 109, "qualified_RL_edge_recovery": 2, "RL_qualified_AIR_capture_region": 0, "RL_actual_TOP_bearing": 0, "RL_placed_history": 0, "P13_actual_stage": 0, "all_task_complete": 0}, "prefix_decisions_excluded": 9259, "prefix_physics_ticks_excluded": 74072, "prefix_PPO_credit": 0, "replay_exposures": 2560, "fresh_student_RR_capture_events": 2, "fresh_student_RL_qualification_events": 3}

- Training lifecycle success is not physical task success.
- Teacher prefix forms real states but has zero PPO credit; no snapshot teleport is inferred.
- Fresh versus inherited events are attributed against each actual handoff tick.
- RR TOP recovery/phase labels do not themselves prove a renewed qualified traversal.
- Each optimized raw index has five likelihood exposures verified from official JSON receipts; no tensor/model loaded.
- Offline front replay is counted separately; heldout KL is not physical front-retention proof.
- CP226304 formal video report remains unchanged; new checkpoint is not evaluated by this coverage report.

## 封存补充：尾段与RR恢复口径

末段133学生决策为global227196–227328，tick7200/60.0s/P12，**非terminal预算尾**，不是第五次失败或成功。两份512官方return receipt都把env0列为nonterminal bootstrap；数值来自既有official compute_returns last_values，JSON没有独立记录该数值，本审计没有重新运行critic。末态FR TOP10.737N、FL AIR0N，RR/RL ground3.484/11.353N，RL未qualified。

本课1024条raw/mean/sigma/oldlogp逐条一致；每条5次官方likelihood曝光，共5120次、40个官方minibatch/Adam。前段回放1280 exposures是在这些更新中的单独正则项，不增加PPO样本，不是1280新物理数据或独立AUX优化。

第三回合RR的8112 TOP恢复不能直接叫新捕获：先有6328 ground撤销、7792 TOP但XY外，再8112合法TOP1.078N而current_lift_valid=false；8136才出现连续unsupported AIR新建立（free-rise8.758mm），8152合法TOP8.197N且valid。后者是真实功能性再接触/加载，但历史cross/placed仍为教师6133；合法TOP已先于新AIR发生，所以不计新独立RR主动越沿/placed事件。本课程RR学生首次placed=0；保护分支累计2次学生首次capture来自此前P07课，不能与再接触混算。

**分支账本分开：** 当前保护分支CP225280→CP227328实际新增2048决策／4 PPO／80 Adam，回放2560 exposures。旧 `ancestor220544_recapture_v2` 的CP231168→CP231680收尾512／1 PPO／20 Adam单独保留（见59e_P10_CP231680_512_coverage），对当前保护分支计数贡献0；不借用较大的旧CP编号。最新heldout mean/max KL=0.011718149/0.033050478，仅证明此固定保留集上的分布漂移量，不等价于自然P01前段或整任务已通过。
