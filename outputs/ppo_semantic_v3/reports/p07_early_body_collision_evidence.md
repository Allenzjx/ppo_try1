# P07 课程：episode 1、2 的早期 BODY_COLLISION 证据边界

只读 [run 20260907T0313083674401Z_gf1a9bbf650b1_98167177c0ff4cd6b15dd57e491fa8c2](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0313083674401Z_gf1a9bbf650b1_98167177c0ff4cd6b15dd57e491fa8c2) 的前三个 completed records、episode 1/2 各11个信用点，以及 episode 0 的前11个信用点。未读取 ep3 或其后数据、optimizer流、PT或checkpoint。

**结论：日志记录了两次有效物理观测下的真实安全终止路径，不是传感器无效/接口异常分类；但训练 compact 未保存终止原始碰撞力、接触点或穿透量，无法在离线记录中独立重建这两次碰撞判据。** 不重分类既有 BODY_COLLISION，也不把现有 summary 冒充原始接触证明。

## 1. 精确终止边界

| 项目 | Episode 1 | Episode 2 |
|---|---:|---:|
| 首信用 tick /时间 | 5952 /49.6 s | 5952 /49.6 s |
| 信用决策数 | 11 | 11 |
| Terminal global | 108751 | 108762 |
| Episode tick /任务时间 | 6034 /50.2833333333 s | 6035 /50.2916666667 s |
| 当前信用物理 ticks | 82 | 83 |
| 最后决策实际 ticks | 2（6033–6034） | 3（6033–6035） |
| Native command tick | 6213 | 6214 |
| 阶段 /正式结果 | P09 /BODY_COLLISION | P09 /BODY_COLLISION |

两者前一个信用点10均止于 tick6032，nonterminal；首次终止发生在下一决策内。native时钟与episode时钟仍差+179。相对11×8，分别短6、5 ticks，原因是运行发现安全终止后及时停止物理循环，不是丢失日志。

两次 `physical_evaluator` 均 valid=true、termination_reason=`TASK_FAILURE_BODY_COLLISION`、reason=`central body/obstacle collision`，source=`current_episode_live_joint_geometry_exact_contact_history`。finite fallback=false、time_outs=false、bootstrap=false、terminal event=−40、next potential=0。scope仍为`teacher_initialized_suffix`，不是full-P01成功/失败评估。

终止区间2/2、3/3 native ticks全部verified且有效下发；末tick setter/dispatch和actual mapping相等，四类in-episode状态写入均0、no-state-write=true。**Actuator native验证证明控制目标下发，不证明碰撞传感器力值。**

## 2. 哪个 body、哪些原始值可核验？

当前 [BodyCollisionDetector](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/sensing/body_collision_detector.py) 仅接受 exact **`base_link` ↔ `/World/Obstacle`**，不是任意上腿/下腿或轮子接触。条件是pair_verified且active、sensor_body与other_body精确匹配，并满足连续至少2个physics ticks，或live collider penetration至少.001m。原始contact与geometry由 [SensorReader](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/sensing/sensor_reader.py) 输入检测器；[TaskEvaluator](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py) 将`body_collision.detected=true`记录为上述通用BODY_COLLISION原因。

因此 **base_link/Obstacle是由当前检测器实现推得的唯一允许来源**，不是本文在终止raw接触行里独立读出的body pair。保留下来的 terminal_info/physical_evaluator没有以下字段：

- `body_collision.real_pair_active/persistent/geometry_penetration_m/reason` 原始状态；
- base_link/Obstacle的normal/tangential/world force、contact point、逐tickactive/history；
- 触发时的live penetration数值，以及到底由持续接触还是几何分支触发。

该训练目录的JSONL只含prefix证据、decision compact与completed summary，没有对应终止120Hz raw-sensor/contact流。prefix-start/接管raw快照是**更早时刻**，不能代替tick6033–6035的碰撞证据；rollout PT也没有被加载。因而这两次的**测得力N、接触点xyz、首次exact-pair接触tick、实际持续长度/penetration分支均不可恢复**，不填猜测值。

现有数据没有“传感器无效/NaN/接口失配”指示，支持按运行当时的安全检测保留终止。另一方面，仅valid=true和检测器实现也不能离线排除错误pair绑定、几何或底层sensor异常。独立确认具体力/点/分支需要该时刻原始pair及geometry记录；本次未重放、未新增实验或门禁。

## 3. 少量相同信用点对照（不作唯一因果结论）

下表是三个episode的**信用点10、同tick6032/P09**，均尚未终止。角度是actual filtered canonical **驱动目标**，不是实测joint q；FL均AIR/load0。

| Episode | FL hip/knee目标 (°) | RR hip/knee目标 (°) | FR轮目标 (rad/s) | RR当前GROUND/load |
|---|---|---|---:|---|
| 0 | 48.915643 /−22.838013 | 47.840316 /−.725101 | −1.195194 | true /.115214 |
| 1 | 54.403652 /−36.250465 | 43.508749 /−9.286713 | −1.358940 | true /.131833 |
| 2 | 63.443134 /−32.356507 | 42.347489 /−14.090767 | −1.080000 | true /.116083 |

三者此刻RR nominal均 `[48.2,0]°`；projected RR residual分别 `[-1.609684,-1.679450]`、`[-5.941251,-10.241063]`、`[-7.102511,-15.045117]°`。这些真实目标/残差存在差别，但没有保存对应body碰撞点/力及完整实测q，不能从角度差直接得出哪一项导致碰撞。

在信用点1，episode0 FL仍TOP/load.151409，episode1/2已经AIR/load0；到点10三者都AIR。因此“FL卸载”并不单独区分这两次短回合和episode0。Episode0的点11在tick6040仍nonterminal，RR仍GROUND；episode1/2的点11提前于6034/6035终止，RR已AIR/load0、但均无hard Q。AIR不是合格抬升，也不能凭同时AIR说明碰撞由RR或某腿造成。

相同信用序号11因提前终止而不是同一物理时刻；不能把episode0的6040目标当成6034/6035实际动作反事实。Episode0后来仍以P09 incomplete结束，不把它当成功对照。

仅新增本报告，不改production/config/tests/master、不哈希、不运行Python/PT/GPU/Isaac，不读ep3或后续。完成后停止。
