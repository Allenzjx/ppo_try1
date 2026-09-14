# 首个 P06 课程回合：实际事件链（只读审计）

范围固定为新训练运行 `20260910T0542011958077Z_g28609010db4e_b88e6f85c2e1480ea33d2fbcf28d6139` 的 `residual_and_projection_audit.jsonl` **第 1–82 行**，全局策略决策 **141057–141138**；以及 `completed_episodes.jsonl` 第 1 行。运行位置为 `runs/ppo_fsm_reference_p09_stable_v2/train/`。本报告不修改生产、训练配置、结果标签或运行设施，也没有启动 Torch/Isaac。

## 结论与计数边界

这是一次真实 PPO 策略采样的 **P06 教师前缀初始化后缀失败**，不是 P01 完整 PPO 成功，也不是环境/执行链损坏的证据。82 个新策略样本中 P06=77、P07=1、P08=1、P09=3，其余阶段=0。普通阶段转换没有 terminal；终止仅发生在最后一行。最后一次只读检查（2026-09-10 05:59:56 UTC）`optimizer_updates.jsonl` 尚有 **0 条非空记录**；因此这里不能把采样说成已完成一次新更新或已保存新的训练 checkpoint。

第一段教师前缀接受于 tick 3584 / 29.866667 s，448 次教师决策全部不入 PPO storage。随后 82 次策略决策执行 656 个物理 tick（每次 8 tick），后缀长 5.466667 s；全物理回合终止于 tick 4240 / 35.333333 s。教师前缀时间计入 200 s 任务时限，但不计策略决策或 optimizer 更新。

RR 在 P06 内已建立具有当前控制证据的抬升，P07/P08/P09 接管同一尝试，没有“落地后重新抬一次”。然而 RR 没有推进过前缘或完成放置；末端机身碰撞、前腿当前承载丢失，当前抬升有效性被撤销。大高度、AIR 和历史 Q 都没有被误判为完整成功。

## 事件、任务交接与当前状态

表中 Q=同一尝试历史抬升建立，V=当前抬升可用，C/P=越前缘/放置历史。事件 tick 是已保存历史事件时刻；决策末状态不是事件瞬间完整传感器样本。`RR gain` 是原实际地面参考上的抬升，不是高出台面量。

| 证据位置 | tick / 时间 s | 事件或阶段 | RR gain / 台面间隙 mm | Q / V / C / P |
| --- | --- | --- | --- | --- |
| audit 第 75 行，g141131 | 4184 / 34.866667 | P06，AIR 但尚无 initial/Q | 0.513 / -49.520 | 0 / 0 / 0 / 0 |
| history.lift_attempt_events | 4189 / 34.908333 | RR whole_body_initial_clearance，仍在 P06 | 3.061 / 未存事件级值 | 0 / 未存事件级值 / 0 / 0 |
| history.lift_attempt_events | 4195 / 34.958333 | RR qualified_measured_upward_lift，仍在 P06 | 8.358 / -41.675 | 1 / 未存事件级值 / 0 / 0 |
| audit 第 77 行，g141133 | 4200 / 35.000000 | P06→P07，连续接管 | 13.770 / -36.263 | 1 / 1 / 0 / 0 |
| audit 第 78 行，g141134 | 4208 / 35.066667 | P07→P08，连续接管 | 34.404 / -15.628 | 1 / 1 / 0 / 0 |
| audit 第 79 行，g141135 | 4216 / 35.133333 | P08→P09，连续接管 | 65.930 / +15.898 | 1 / 1 / 0 / 0 |
| audit 第 80 行，g141136 | 4224 / 35.200000 | P09，继续 AIR | 99.213 / +49.180 | 1 / 1 / 0 / 0 |
| audit 第 81 行，g141137 | 4232 / 35.266667 | P09，继续 AIR | 133.134 / +83.101 | 1 / 1 / 0 / 0 |
| audit 第 82 行，g141138 | 4240 / 35.333333 | P09，BODY_COLLISION | 164.699 / +114.666 | 1 / 0 / 0 / 0 |

连续 takeover 的保存理由都是 `qualified downstream motion already active; continuous takeover`。P06→P07 的 rear_approach 仅 0.532106，并非普通狭窄位置入口已经达标；已有有效下游动作使其接管。P07→P08 `role_prepared_RR=1`，P08→P09 `transfer_ready_RR=1`。三次转换后 RR 的 Q event tick 都保持 4195，没有新增 RR retry、re-ground 或重新资格事件。

g141133–141137 共 **5 个决策末快照** V=true，不能把这写成 5 个物理 tick 或用稀疏边界样本证明区间每 tick 全部有效。其依据不只是 AIR/高度：实际全身关节响应、地面参考抬升、已有去噪确认以及当下 FL/FR 顶面与 RL 地面的已验证支撑和安全包络共同成立。`body_control_evidence` 明确是短时功能证据，不是静态稳定证明，也不保证后续 C/P 成功。早先多次不足毫米或约 1.35 mm 的孤立 AIR 都没有得到 Q。

## 抬升附近：实测关节、wheel、CoM 与载荷

initial 事件记录实际全身关节路径运动 11.084350°、命令路径运动 11.719959°。Q 事件记录 RR 自身关节路径运动 4.664096°；这是既有测量窗口内的路径量，不是一次瞬时关节差。下面以相邻决策末实际关节给出独立佐证。关节来自当前 `transfer_roles.*.receiver_workspace_state.joint_range_margin_deg` 与已冻结真实硬限的反算，并用上下余量一致性核对，不是 commanded/drive target。

| 决策末 | FL hip/knee ° | FR hip/knee ° | RL hip/knee ° | RR hip/knee ° |
| --- | --- | --- | --- | --- |
| tick 4184，initial 前 | 34.19566 / 22.82933 | 7.95216 / -23.36502 | 39.41123 / 8.79570 | 3.99019 / 9.35052 |
| tick 4192，initial 后/Q 前 | 34.11998 / 23.06060 | 6.37173 / -20.18145 | 38.54707 / 6.29684 | 2.13664 / 8.40719 |
| tick 4200，Q 后 | 32.96730 / 23.18451 | 5.93282 / -16.65510 | 33.40522 / 5.28635 | -0.01669 / 8.42613 |
| tick 4216，P09 接管 | 40.48080 / 22.67813 | 6.17071 / -9.41791 | 18.82350 / 6.18636 | -3.85998 / 7.28455 |
| tick 4240，碰撞终止 | 63.44407 / 22.60890 | 5.12228 / -6.77290 | 18.41885 / 0.77483 | 6.44496 / 2.20638 |

4184→4200 四腿均有响应；同期 FL/FR/RL/RR 的实际 dispatch wheel 请求由 `[0.161458,-0.230326,0.520047,0.779730]` 变为 `[0.360238,-0.261551,0.699171,0.789555]` rad/s。每行都保存了实际 native setter 效果，全部 82 行的 8 tick effect 校验通过，且没有 episode 内 root pose/velocity/force/gravity 写入。这里能证实主动全身运动与 RR 抬升同时存在，**不能从单条轨迹隔离因果贡献、认定 RR 是纯被动抬升，或把任一腿角度指定为必需方法**。

RR 载荷占比从 tick 4168 的 0.120977 降到 4176 的 0.036571，4184 后为已验证的 0。FL 并非此段悬空：4184/4192/4200 均有真实 TOP 接触与承载，力分别 3.138053/3.559951/2.707332 N，载荷占比 0.098254/0.119539/0.087650；FR、RL 分担更多载荷。不能把该轨迹归纳为“先抬 FL 必然抬 RR”或“RR 的载荷全转到 FL”。

RR→FL 角色使用的 0.5 s 滑动测量窗口中，CoM 向 FL 侧的投影位移在 tick 4184/4200 为 +68.631/+43.625 mm；但相应当前 CoM 速度投影为 -0.012343/-0.111693 m/s，说明窗口净位移与瞬时趋势不同。P08 末 4216 当前投影速度再次为 +0.067720 m/s。终止时 CoM 向 FL 侧投影速度仍可为正（+0.044372 m/s），**但 FL 已 AIR、承载为 0**；不可以用方向趋势虚构接触或承载。

## 跨阶段没有清零；source 请求跳变不是 mapper 状态重置

三次桥接的 previous residual 与 carried residual 一致，无禁止通道丢弃、无 phase-scale 截断；residual 浮点差不超过 8.9e-16° / 5.6e-17 rad/s，wheel 请求跳差不超过 1.2e-16 rad/s。新阶段首 tick `handoff_hold_used=true`，保留旧残差；随后 7 tick 执行当前阶段策略请求。全部三次阶段转换均 `terminal=false`，没有 episode/回报截断。

交接并非所有逻辑目标都数值相等：source 新 owner 的 held target 真实更新产生 P06→P07 FL hip +14.8°，P07→P08 RL hip +7.4°，P08→P09 FL hip +8.5° / RL hip +4.2° / RR hip +1.6° 的请求差。实际 drive 仍走唯一成熟 mapper 的 1.25°/tick slew；例如连续三个新阶段的 FL 最后 dispatch 均仅比前一 tick drive 增加 1.25°，8 tick 决策边界间最多增加 10°，不是物理瞬跳。

last-dispatch 审计中的 previous ACK 始终逐 tick 相邻，独立验证 true，bootstrap tick 始终 180，未重新初始化。源码 `semantic_env.py` 每 tick 更新而不重置动作历史；`phase_action_masks_v2.py` 仅真实安全停止等条件清空存储，否则带着上次 residual 继续；`semantic_supervisor.py` reference 模式普通换阶段仅启动/组合新的 source motion owner，继续使用既有 mapper。**未发现普通切换 reset residual、filter 或 mapper 的证据。**正常 owner 请求更新本身不构成恢复双重 pre-slew 的理由，本报告不提出该修改。

P09 的当下 nominal geometry 审计均为 `identity_within_descent_allowance`，保存的 nominal correction 12 通道均为 0；当前样本没有证据表明几何投影把本次前送/下降强行截掉。AIR 可用时 `rr_carry_continuation` 保存 `current_AIR_safe_approach`、`source_joint_owners_continued=true`，明确无固定抬升计时和 15 mm 台面门槛。它是动作建议，不保证实际前送成功。

## 碰撞、首个未完成任务与可用证据边界

末行评价器 `valid=true`、`run_validity=VALID`、`physical_evidence_status=VERIFIED`，结果 `TASK_FAILURE_BODY_COLLISION`，原因 `central body/obstacle collision`，`termination_source=BODY_CONTACT`，外层终止 `BODY_COLLISION`。终止时 FL/FR 都是 AIR、TOP=false、bearing_force=0；唯一当前支撑 RL 为真实地面接触、10.107896 N。RR 仍 AIR、有 164.699 mm 地面参考抬升、历史 Q 保留，但 `current_lift_valid=false`、`body_control_evidence=false`、`motion_continuation_allowed=false`、reason=`physical_safety_abort`。

这是有有效碰撞评价证据的真实任务失败，不是仅任务超时、录像损坏或历史 Trial043 代替本次判断。所读紧凑训练审计没有保存终止瞬间 base_link/Obstacle 的原始接触点与力矩/冲量，因此本报告不臆造碰撞冲量、精确接触面、碰撞前 roll/pitch 或关节因果归因。82 个决策末实际关节对硬限最小剩余距离 17.511683°，并非 command-only 核对；这不替代运行时每 tick 真实硬限/数值/跌倒保护。

**第一个未完成任务是 P09 的 RR 持续前送、越过前缘并受控捕获/放置。**在 Q 后首个边界到终止，RR front distance 从 -0.311956 m 变为 -0.324824 m，反而远离前缘约 12.868 mm；C/P 始终 false，P09→P10 未发生。虽然台面净空变大，仍距前缘约 0.325 m，不能称后腿越障成功。这是应继续纳入正常 PPO 失败学习的数据，不构成重新要求 A 5/5 或非零探针全程成功的 optimizer 门禁。

## 前缀文件实际可读性（非训练计数）

此前目录信息显示 `prefix_evidence.jsonl` 长度 0 时，直接读取已得到 606410 个字符、573 行，故不能据目录缓存/活跃写入期长度认定文件为空。2026-09-10 05:57:11 UTC 的另一次一致快照是 769532 bytes/字符、781 非空行，781 行均解析成功；05:57:40 UTC 又读到 792793 bytes、810 行，显示文件在持续追加。

05:57:40 快照由 806 条 `reset_only_prefix_decision`、2 条 `reset_only_prefix_start`、1 条 accepted `reset_only_prefix_result`、1 条 `policy_credit_start` 元数据组成，全部 `policy_credit=false`。其中首段已明确 accepted=448 决策/3584 tick、目标 P06；余下 358 条是第二次前缀重采到 tick 2864 / P05，尚不是策略样本。`policy_credit_start` 是边界元数据，不是额外的一次策略决策。以上为带时间点的只读快照，不是此活跃运行的最终全量统计，也不证明目录长度滞后的具体底层原因；未修改文件设施。
