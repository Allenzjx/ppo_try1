# P06 block9 首回合：RR 连续接管后回地、再抬升，最终高度 FALL

**固定范围。** Run `train/20260910T1155110470637Z_g7db0d17f398d_94db61a60fb54319aef3eb1d71d97e00`，仅单次解析 audit 第 1–148 行（global145921–146068），另核对 completed_episodes 第 1 行：episode0、seed1001、148 决策、FALL、task_success=false。未读取后续前缀或策略。148 行规范化 LF 文本 SHA256 `67952109e9096f2990e88dafa76693e7c259073fc9f4977106f6933e51bccb98`（11,556,143 bytes；不是原 CRLF 文件的 byte hash）。

策略从教师已实际到达的 P06 tick3584 / 29.866667 s 接管，执行 1178 ticks / 9.816667 s，到 tick4762 / 39.683333 s。**P06=136、P07=1、P08=1、P09=10；RR 新 I=2、Q=2、回地撤销=1、C=0、P=0。** 前缀 FR/FL 放置（FL P3583）不计策略成绩。按主控已保存状态，前 128 条已更新保存146048/1106；本范围末20条尚未优化，不计入该 checkpoint，也不预支后来更新。

| 实际 tick / 时间 | 当时动作阶段 | 真实事件 |
|---|---|---|
| 4672 / 38.933333 s | P06→P07 | rear_approach=1，当前物理目标满足 |
| 4673 / 38.941667 s | **P07** | RR I，向上 excursion 3.043 mm |
| 4678 / 38.983333 s | **P07** | RR Q，excursion 8.935 mm，距台面仍 -41.241 mm |
| 4680 / 39.000000 s | P07→P08 | 当前有效抬升触发 continuous takeover；role_prepared_RR=1 |
| 4688 / 39.066667 s | P08→P09 | 同一次已成立抬升连续接管；transfer_ready_RR=1 |
| 4742 / 39.516667 s | P09 | 实际 GROUND 撤销此前 Q/current validity；不是阶段切换人为清零 |
| 4751 / 39.591667 s | P09 | 新 I，excursion 3.125 mm |
| 4754 / 39.616667 s | P09 | 新 Q，excursion 10.614 mm，距台面 -39.735 mm |
| 4762 / 39.683333 s | P09 | FALL；仍 AIR，但当前受控有效性 false，无 C/P |

**全身运动与历史分开。** 两次 I 的 whole-body 实测关节变化分别13.716°/18.330°；Q 时 RR 自身实测运动6.829°/2.299°。这支持有主动关节与全身运动的实际抬升，但不足以确定唯一致因，也不能把 FL 悬空说成 FL 承载。第一次尝试跨 P07/P08/P09 保持，直到真实回地才撤销；第二次 Q 不能用 history 首次 event_tick4678 的旧时间代替。全范围9个决策末态 current_lift_valid=true，是离散末态样本数，不是固定悬停验收或保证后续越沿。

末态 RR AIR、有效载荷0、台面净空 **+9.216 mm**，却仍在前缘 **-139.941 mm**，within_top_xy=false；大净空不等于过沿。FL 同样 AIR/载荷0；实际 FR TOP 支撑、RL GROUND 支撑，载荷比例约0.569/0.431且验证有效。FR/FL 历史 placed 不能证明当前 FL 承载。首个未完成后腿任务仍是 RR 保持有效抬升并前送越沿、受控放置；保留安全中止，不把新 Q 当完整后缀成功。

**FALL 的独立实测依据。** terminal_observation 的 policy/critic 均372维，finite_fallback=false。按当前 schema 固定 scale/offset 解码：gravity（offset66）=[-0.061700,0.152944,**-0.986307**]；base 世界线速度（offset75）=[0.045544,0.008187,**-0.197171**] m/s，模长0.202528；角速度模长1.914794 rad/s。relative obstacle bottom/top（offset93，组内零基索引4/5）=-0.0143429935/+0.0356570072 m；`sensor_reader.py:346` 使用 `locked_obstacle_planes()`，`geometry.py:73` 的 bottom=0（environment_lock同值），故回解 **base_z≈0.0143429935 m < 0.015 m**。top=0.05反算也吻合。这里是实际观测经float32编码回解，不声称重新读取了原始双精度位置。

`guard_state.py:376` 的高度下限分支明确满足；重力 z 远低于翻转阈值-0.30、线/角速度低于5/20，未显示这些其他分支触发。因此是身体过低的 FALL 证据，不是因为 RR AIR、阶段名称或仅仅姿态倾斜。终态RR实测hip/knee约36.163°/3.866°，不能用命令或历史Q覆盖这一身体安全失败；不放宽阈值。

**执行与保存边界。** 1178/1178 native effect tick verified、own-phase request1175；差额3为普通交接 hold，禁止的 episode 内 state write=0。`semantic_env.step`逐tick延续 bridge/projector及 previous residual/applied history，未发现新的下发失败或普通切换reset证据。终态RR target[39.108,3.671]°确有实际响应，但跟踪响应不能单独证明稳定。该回合跨过一次128更新，因此也不是一条固定策略性能比较。

终态为第二个128 rollout的第20拍，**不是 rollout 尾拍**。现有 `semantic_training.py:1208` 的终态回调先写 audit/completed 并 fsync，再进入普通 eager reset；本次终态证据已在新前缀前持久化。它不能称为 tail-reset deferred 分支的实训验证。仅新增本报告，未改四主报告、生产、阈值、checkpoint或进程。
