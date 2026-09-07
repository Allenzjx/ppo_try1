# Nominal 越沿前下降时序：只读诊断

2026-09-06；生产 HEAD `9d70aae58243`；仅 PowerShell 读取既有源码、完整训练数据及历史报告。没有执行 Python、Isaac、FK/Jacobian 计算、额外仿真或生产修改。本检查不是当前 P10 补样的前置门禁。

## 结论与边界

确认一个**调度语义缺口**：当前后腿 nominal 的关节建议仍按录制时钟推进，并不检查当前轮是否已进入可放置区域。完成 46464 的训练 run 中，episode 1 的 RL 尚在前沿之前、已低于顶面并触墙，P12 建议仍继续到历史落脚尾段。它不是“计时结束后建议自动消失”：有限变化结束后，已取得通道所有权的末目标仍被保持。

但这不是实际机器人被软件硬门槛强制下降，也不能据此宣称本回合失败完全由 nominal 造成。全 12 维 residual 仍开放；实际轨迹同时受 sampled residual、mapper、伺服响应、浮动基座与接触影响。现有数据证明时序与当前几何不匹配，**没有执行保持其它条件相同的真实动作 counterfactual**。不能恢复旧版“膝角开始减小就暂停整段”条件，也不能等已经跨线后才允许产生跨线的运动。

主规范第 230 行明确要求：未进入可放置区域时，不应仅因 nominal 时间到达而提前下落；同段也明确不能固定唯一轨迹或替 PPO 完成全部困难。补充规范同样要求保留任务区域与 residual 自由，而不是用历史关节构型、静止载荷模板或新的完整任务成功门槛代替学习。

## 实际调度是什么

- `semantic_supervisor.py:773–805`：每个连续 source layer 都无条件 `motion.tick()`；所有曾改变的通道进入 sticky `touched`，最后一个 owner 提供其当前采样值。没有后腿 servo 的当前几何/可放置性调度。
- P09/P12 的物理反馈只在本段对四轮追加 approach prior：当前有效 lift、净空至少既有 15 mm、尚在 approach 前沿之外。它不调度 hip/knee 的落脚建议；净空下降后该前送辅助会退出。
- `fsm/motion_executor.py:224–276` 按量化后的录制时间发 waypoint；endpoint 后仍返回末值。外层 nominal 的 slew/hard bound 仍存在；独立 post-mapper residual 不因此关闭。
- `recording_motion_contract.json` 的 P12：先升膝至 35.3°，后 hip 从 0.5°升至 31.2°；约 3.2–3.933 s 的 knee 35.3→−18.7°属于 carry 过程，4.0–4.667 s 的 hip 31.2→−10.1°进入后续段。末 RL 为 −10.1/−18.7°。这些是录制建议，不是普适的 lift/place 角度定义。

## Episode 1：高位 AIR、前送、触墙、仍播放尾段

来源：`runs/ppo_semantic_v3/train/20260906T0946289788164Z_g49749aa527a4_d866c10e684d4248b10521cb7d32bf09/residual_and_projection_audit.jsonl`，及该 run 的 completed episodes / prefix evidence。这里“46464 ep1”指产生 46464 checkpoint 的在线训练回合，**不是固定 46464 权重的评估**。该回合 PPO credit 为 global 39074–39895，共 822 decisions，另有 448 teacher decisions；P12 起于 tick 6560。下表为实际 decision-end 观测，事件 tick 另由 history 给出，不把 15 Hz 采样点冒充 120 Hz 首事件。

单位：front/clearance 为 mm；nominal 与 final 为 canonical RL hip/knee 度数，**不是实测关节姿态**；接触/几何来自真实观测。

| tick / P12 时间 | RL nominal | RL final command | front / clearance | 当前事实 |
|---|---:|---:|---:|---|
| 6936 / 3.133 s | 30.2 / 35.3 | 30.877 / 39.495 | −152.133 / +86.801 | 第二次有效 lift，AIR；无 cross/place |
| 6944 / 3.200 s | 31.2 / 35.3 | — | −142.491 / +83.756 | AIR，开始后续 knee carry |
| 6984 / 3.533 s | 31.2 / 1.4 | 29.396 / 3.095 | −99.549 / +10.146 | AIR；前送 prior 条件失去 15 mm 净空，轮 nominal 正在退至零 |
| 6992 / 3.600 s | 31.2 / −2.8 | 27.323 / −5.105 | −77.525 / −7.868 | 仍 AIR，但轮底已低于顶面；无 cross/place |
| 7008 / 3.733 s | 31.2 / −6.0 | 31.734 / −8.305 | −48.962 / −32.642 | 首个此采样序列的 wall=true；load fraction 0.389755；非 AIR、非 GROUND |
| 7040 / 4.000 s | 31.2 / −18.7 | 30.333 / −13.005 | −50.245 / −37.750 | 触墙、尚未跨线，轮 nominal 已为零 |
| 7048 / 4.067 s | 27.0 / −18.7 | 23.473 / −17.005 | −50.569 / −40.355 | 时钟继续发 hip 尾段，没有等待当前 place 区域 |
| 7128 / 4.733 s | −10.1 / −18.7 | −11.156 / −18.055 | −50.332 / −43.820 | 已到录制末目标，仍触墙、无 cross/place |
| 7232 / 5.600 s | −10.1 / −18.7 | — | −51.129 / −50.675 | GROUND+wall；有效 lift 已在 tick 7225 撤销 |
| 10160 / 30.000 s | −10.1 / −18.7 | −10.964 / −19.464 | −52.816 / −48.798 | P12 INCOMPLETE_CONTROLLER_BLOCKED；仍无 RL cross/place |

tick 6936 实际 residual 为 −0.573/+2.945°，native mapped 为 31.45/36.55°，final 为上表数值；tick 7048 residual 为 −3.527/+2.945°，native mapped 为 27/−19.95°。因此不能把 nominal 直接当机械关节位置，也不能忽略 residual 与映射历史。

RR 的历史事件为 Q6297/C6536/P6543，本窗口 RR 与 FL 起初确实 TOP；例如 tick 6936 的 RR/FL load fraction 为 0.490825/0.509175，tick 7008 为 0.247712/0.362533。到 tick 7232 RR 仍 TOP，不能把 RL 早期下降全部归因于 RR 已经回地；RR 后来在 tick 7520 采样首次出现负 front，7616 已 GROUND。另一方面，**RL 第一次 Q6729→GROUND 撤销6799 发生时 nominal 一直是 0.5/35.3°**，早于上述 knee/hip 尾段。这是反例：本回合所有 lift 失败不能都归因于时钟下降。

## 为什么不能恢复角度符号暂停

已记录的 A 对照显示：P09 的 knee 0→−37.8°在 hip 保持 55.6°时使 RR 空中前送约 75.97 mm，末时仍在前沿前 64.16 mm、净空约 95.56 mm；随后 hip 段继续产生跨线。旧成功 A 的 P12 在 hip 31.2°保持、knee 35.3→−18.7°期间，RL 空中前送约 115.86 mm，前沿距 −144.774→−28.919 mm、净空 119.464→54.794 mm，之后约 +4.242 s 才跨线。

历史来源/既有核对：`training_report.md` 的 5036 修复记录、`rr_nominal_overlap_readonly.md`；P12 原数据为 `runs/ppo_phase_v1/baseline-fsm-eval/20260905T010448442021Z_g36a0d57eb96a_c5dc85cd3cb31_s2001_n1_baseline-fsm-eval-fresh-process/episode_000_seed_2001`。这里复用此前已核对的历史反例，未重跑 A，也**不把旧 A 与当前 v3 当同 MDP 配对因果实验**。

所以，错误不宜被定义为“第一个负向 knee/hip waypoint”。真实 ep1 的净空损失已经出现在必要 carry 过程内；只暂停后面的 hip 尾段也未必足够。

## 最小可行调度方向与现有工具

候选是**只修正当前 nominal 建议中的局部向下分量**，不是冻结完整 P09/P12、不重放唯一峰值姿态、不自动完成落脚。使用真实当前 base/joints 与轮/障碍几何，估计下一小 nominal 关节增量的 world 位移；在尚未进入实际可放置 XY 区域、该增量会消耗所需净空时，衰减/移除其向下分量，尽量保留前送分量和其它通道。进入真实可放置区域后允许下降/capture。residual 在该 nominal 修正之后照常作用，不能被这个调度投影或封锁。

只读找到的可复用资源：

| 现有资源 | 可以复用什么 / 不能宣称什么 |
|---|---|
| `sensing/sensor_reader.py`，`sensing/geometry.py:93–177,302` | 当前 base/body/joint 状态、轮底、前沿距、净空、collider extent；不是 joints→body FK 求解器。当前轮底由实测 center 加缓存 extent 构造，不能把缓存平移模型当任意旋转 collider 的精确预测。 |
| `actuator_target_effect.py:49`，`tools/audit_semantic_workspace.py:42` | 同一真实 mapper native 与上一实际 final 下的 bounded target 重建/对照；可借鉴无第二次 mapper.advance/write 的 command preview。工具自身明确 `uses_forward_kinematics=False`，不是物理位移 counterfactual。 |
| 本机 `C:/robotics_sim/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py:74–82,143–154` | 已安装 IsaacLab 通过 `Articulation.root_physx_view.get_jacobians()` 取 world Jacobian；代码明确 floating base 的 joint 列加 6，fixed base 的 body 行减 1。已有 world→base 变换示例；不用升级环境或实现新 FK 框架。 |

项目自身未找到已接到本机器人上的 FK/Jacobian wrapper。技术上可以以真实当前状态的 wheel body world Jacobian 估计 `Δp ≈ J Δq`，但必须先验证本机器人 body/DOF 索引、rad/deg 与 canonical/native 符号、浮动基座列、轮底 offset，以及 mapper/slew 后真实关节增量含义。不能直接把 full12 nominal 度数差乘 Jacobian。可以研究受原 hard/slew 界约束的最小局部投影同时保留前送、削减向下；在奇异位形或约束冲突时，这两者未必同时可行，不能承诺总能保持前送。

最低正反例：高净空且前方尚远时仍允许产生前送的负向膝角；低于可用净空且未到区域时不纯按时钟继续向下建议；已到合法区域时允许不同关节构型的真实落脚；GROUND 撤销后不凭历史资格自动回放旧峰值姿态，也不夺走 PPO 重试自由。旧 A 的 carry 片段必须保持可通行，而不是增加“先到终点才能发生成终点动作”的循环依赖。

以上只是最小可行性边界。固定当前 base 的运动学一阶方向不能预测接触动力学、全身支撑转移和实际下一 tick 位移；对 command buffer 的 counterfactual 审计也不是这些物理效应的因果证明。尚未实现或验证该局部投影，不增加任务判定、奖励、范围或训练成功门槛；当前 P10 真实采样无需等待此方案。
