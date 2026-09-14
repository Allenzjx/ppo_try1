# Block17：P07 教师入口后首 9 条策略样本（固定范围）

结论：真实信用入口是 P07/tick 5912，不是已抬起 RR 的快照。P07→P08→P09 两次普通切换连续保留 residual；RR 全段无 I/Q/C/P。68 个物理 tick 后因真实低高度 FALL 中止，并非关节硬限、任务成功或已证 reset 损坏。没有从这 9 条记录发现新的下发/判定缺陷。

## 范围与信用

源 run：`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1839014552998Z_gd4e46006b382_3bf26c40dc774ea38185c4b77938fded`。
2026-09-10T18:54:15.424Z 固定读取 audit 前 9 行、字节 `[0,763865)`，以及 completed_episodes 首行 `[0,80760)`；不读取第二前缀、后续策略或大 physical 流。

episode 0 / seed 1001；global **154881–154889**，来源样本 **P07=1、P08=1、P09=7**。实际 `curriculum_start` 为 `teacher_initialized_suffix`、requested/actual=P07、tick **5912 / 49.266666667 s**、prefix attempt 0。末态 tick **5980 / 49.833333333 s**；新增策略物理时长 **0.566666667 s**，不是 49.833 s 全部由 PPO 控制。教师 FR Q23/C1665/P1695、FL Q2417/C3115/P3583 均为继承证据，`prefix_teacher_data_in_ppo_storage=false`，不加新学习信用。本报告不读取 optimizer/CP，不提前声称这 9 条已优化。

## 真实交接、接触与首个未完成任务

| 决策末 tick | 源→末阶段 | RR 当前证据 | FL 当前接触/承载 |
|---|---|---|---|
|5920|P07→P08|GROUND；I/Q/current/C/P 均 false；front −208.933 mm|TOP，support=true，5.478810 N，verified|
|5928|P08→P09|GROUND；仍无 I/Q/C/P；front −208.344 mm|TOP，support=true，3.389776 N，verified|
|5936|P09|GROUND，无 I/Q；front −206.510 mm|AIR，support=false，0 N，verified|
|5976|P09|GROUND，无 I/Q；front −200.452 mm|AIR，0 N|
|5980|P09，done/FALL|AIR 但 lift=−0.142714 mm，无 I/Q/C/P；front −201.627 mm，top clearance −50.687562 mm|AIR，0 N；末态实际支撑为 FR TOP 13.636527 N、RL GROUND 14.525094 N|

5920 的 completion 是 RR/RL edge_proximity=1、role_prepared_RR=1；5928 是 RR edge_proximity=1、transfer_ready_RR=1，均 entry valid、FR/FL placed 历史成立。两次 `continuous_takeover=false` 指走正常物理目标分支，**不表示动作被截断**；普通阶段 `done=false`。入口时 FL 确实在承载，而不是仅以历史 P 冒充当前支持。RR 准备/转移诊断有实测 CoM/动作证据，但它不是 RR 已抬升的证明：例如 5920 窗口内 CoM 向 FL 侧位移 +3.835346 mm，同时当前该方向速度 −0.074160 m/s，不能简化为始终向 FL 侧运动或稳定性证明。

首个未完成的物理任务是 **P09 获得并维持 RR 有效抬升，继而前送越沿/放置**。末拍 AIR 只是失去接触，既没有正向抬升量，也没有新初始/建立事件；不能记成功抬起。整个事件历史在信用区间内无新增 lift 事件，RR C/P 始终 false。

## 动作确实继续，但实际控制未完成任务

两次普通交接的 `handoff_hold_used=true`、12 通道 carried residual 保留、forbidden/clipped 列表为空，servo residual 差仅浮点量级（最大 2.67e−15°）；wheel residual 差近零。源 owner 请求的 servo 跳变为 7.4°、4.2°，不能误称 residual/filter reset。tracking 的 bootstrap tick 持续为 180，previous-ack 与 dispatch 连续相邻；其绝对命令计数和 episode tick 不是同一时钟。

RR hip/knee，单位°；N 是 raw nominal，R 是 projected residual，F 是最终 target，A 来自实测角色 joint-margin 反算，不以 N 当实测：

|tick|N|R|F|A|
|---|---|---|---|---|
|5920|0 / 0|−4 / −4|−5.349903 / −3.045650|−2.133458 / −0.086814|
|5928|0 / 0|−7.5 / −0.5|−8.849903 / +0.454350|−4.316411 / −0.481212|
|5936|+1.6 / 0|−11 / +0.836938|−10.65 / +1.791288|−6.487708 / +0.547815|
|5980|+45 / 0|−16.697337 / −14.817108|+25.8 / −13.862759|+13.202862 / −8.628999|

四轮顺序 FL/FR/RL/RR：5920、5928、5936 的 N 均 `[.3,.3,.3,.3]`，没有普通切换清零。5960 起记录 N 为 `[.3,−.63,.3,.3]`；末帧 R 为 `[−.822955,+.017504,−.271578,+.347641]`，F 为 `[−.522955,−.612496,+.028422,+.647641]`，实测速度为 `[−.524884,−.759448,+.131408,+.645234] rad/s`。这说明有不同方向的真实响应，不等于已经完成协同控制，也不单因归咎某一轮。

9/9 末端 mask 全开、headroom clipped=[]；RR hip 在最后 3 个记录端点受到最终 slew 约束，候选减 F 分别 +0.154596/+3.902663/+2.502663°，knee 未在这些端点被该最终 slew 裁剪。**68/68 native ticks verified 且 actual-effect，own-phase=66**（两次正常 handoff hold），所有已记状态写入检查为零；末端 target dispatch equality/mapping 均真。记录只支持这些末端 N/F/A 与逐 tick 有效性，不虚构完整物理动作曲线，也不凭 hip 角度方向推断轮端高度。

## FALL 的具体依据与入口限制

terminal policy observation 为 372 维、`finite_fallback=false`。按当前 schema 的 obstacle relative-bottom 分量（locked bottom=0）回解 **base_z=0.014719717205 m < 0.015 m**；这是持久化 float32 张量的几何回解，不是额外读取的原始双精度传感流。projected gravity 为 `[.070754305,.077547796,−.994474828]`，线速度 `[.065860815,.004056428,−.199857906] m/s`，角速度 `[.633847788,.168341007,.052254102] rad/s`：支持低高度 FALL，不支持翻转或速度爆炸分支。规范关节末态 `[63.971335,−2.040696,40.921713,7.089521,26.337728,−13.078674,13.202862,−8.628999]°`；9 个记录端点全部实际关节在硬限内，不能把本次命名为 HARD_JOINT_LIMIT。原结果仍为 FALL / SAFETY_ABORT / PHYSICAL_SAFETY，physical VERIFIED、full_task_success=false。

交接后确实很快丢失高度并失败，但范围内没有 tick 5912 **动作前**的完整高度/姿态张量，不能断言教师结束已贴安全边界。最早策略末端 5920 仍 physical valid、四腿真实接触，线速 0.158713 m/s、角速 0.444180 rad/s，实际关节不接近硬限；随后 FL AIR 和高度下落是观测事实，而不是已证明的 reset 损坏或单一动作因果。保留这次正常失败供持续 PPO 学习，不新增成功门槛、不改安全阈值/范围/超参。
