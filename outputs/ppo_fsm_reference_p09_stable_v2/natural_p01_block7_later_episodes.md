# Natural P01 block7：已完成 episode 4–9 的有界诊断

结论：六回合全部保留原失败结果；episode 7、9 有新的 RR Q，但没有任何 RR/RL C 或 P，没有 P10 样本，也没有全程成功。当前证据未发现新的明确执行或判定缺陷；不把采样轨迹差异称为学习改善。本报告不改变训练门槛或参数。

## 固定证据范围

生产 `69aeaca777dcc8653e60f19da56ae1cd002e5271`；run `runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e`。只分析 `completed_episodes.jsonl` 的 episode index 4–9，以及决策 audit 第 966–2503 行：1538 策略决策，global 143814–145351。此前 0–3 仅用于累计行界，不重审 episode 1；不读取后续策略行。

Audit 固定字节范围（零基、含端点）`67792468–175817309`，108024842 bytes，SHA256 `1e894d2c4814f4ecc06e14e94ec44e58db04091e1afcf05fc3c4c8b3e866c85a`。每回合均为 seed 1001、自然 P01，无教师前缀；I/Q/C/P 均属该物理 episode，不继承其他 episode。

## 真实样本与结果

阶段计数按发出策略的 `phase_id`，不是末端 `end_phase_id`。未列阶段均为零。

| episode | audit 行 | decisions | P01/P02/P03/P04/P05 | P06/P07/P08/P09 | 终态 tick / 秒 | 原终止 |
|---|---|---:|---|---|---|---|
| 4 | 966–1318 | 353 | 2/127/3/1/203 | 17/0/0/0 | 2817 / 23.475 | P06 FALL |
| 5 | 1319–1448 | 130 | 2/101/5/1/21 | 0/0/0/0 | 1038 / 8.65 | P05 FALL |
| 6 | 1449–1635 | 187 | 2/150/3/1/31 | 0/0/0/0 | 1491 / 12.425 | P05 FALL |
| 7 | 1636–1975 | 340 | 2/115/4/1/148 | 66/1/1/2 | 2715 / 22.625 | P09 HARD_JOINT_LIMIT |
| 8 | 1976–2186 | 211 | 2/105/4/1/99 | 0/0/0/0 | 1681 / 14.008333333 | P05 FALL |
| 9 | 2187–2503 | 317 | 2/103/3/1/168 | 30/1/1/8 | 2534 / 21.116666667 | P09 FALL |

合计后腿前驱/后段样本：P06=113、P07=2、P08=2、P09=10、P10–P13=0。RR 新 I=4、Q=2、Q-ground-revocation=0、C=P=0；RL 新 I=7、Q=C=P=0（RL I 大多为 P01 初期动作，不是后腿越障成功）。

- episode 4：RR I1115，没有 Q；末端 RR GROUND。FL 已 C2196/P2682，P05→P06 在 2688；首个未完成任务仍为 P06 后腿前驱准备。
- episode 5/6/8：均没有 RR I/Q；FL 未 C/P，首个未完成任务为 P05 FL 前送/越沿/放置。episode 8 FL Q950 后于 1643 地面撤销，末端已在地面。
- episode 7：RR I2625 后尝试中断；决策末端 2632 AIR/4.320 mm、2640 AIR/0.201 mm 已非 active attempt，不能从末端 AIR 排除中间触地，也不补造具体撤销 tick。新 I2677/Q2681 为另一尝试，没有已成立 Q 的 ground-revocation 事件。FL C2085/P2160，P05→P06=2160。
- episode 9：RR I2449/Q2451，没有撤销事件；FL C1980/P2211，P05→P06=2216。

## RR 成立、连续接管与未完成之处

episode 7 的同一 Q2681 连续经过 P06→P07=2688、P07→P08=2696、P08→P09=2704；对应离地 14.739/24.565/37.794 mm，前缘距离 −320.942/−313.023/−304.878 mm。2712 仍 current-valid（离地 52.968 mm、台面净空 +1.930 mm、前缘 −299.646 mm），2715 因实测硬限当前有效性撤销。Q 的自身关节运动证据 2.570°；I2677 的全身实测运动 16.058°，是不同 tick 的证据，不相加或假定单关节归因。

episode 9 的同一 Q2451 连续经过 2456/2464/2472 三个后腿准备交接。Q 自身关节运动仅 1.697°，I2449 的全身实测运动 14.382°，没有强制等待 RR 自身大角度动作。其 RR current-valid 决策末端共 10 个（P06/P07/P08 各 1，P09=7）；episode 7 为 4 个（各 1）。两回合均为 AIR，无 RR EDGE 接触。新 Q 是功能抬起证据，不是静态稳定、越沿或未来可控放置保证。

episode 9 在 2456 最近前缘仍为 −464.774 mm，此后到 2528 后退至 −538.408 mm，离地反而升至 181.019 mm；终态 2534 离地 200.227 mm、前缘 −538.342 mm。安全终止使 `current_lift_valid/body_control_evidence/motion_continuation_allowed=false`，即使保留 I/Q 历史且仍 AIR，也不能称为可控成功。两回合的首个未完成后腿任务均为 P09 保持可用抬起、前送越沿及受控放置，而不是已达到 P10。

episode 7 终态实测接触：FL/FR TOP、RL GROUND、RR AIR；三条支撑载荷均有效。episode 9 为 FL TOP、FR GROUND/OBSTACLE_AMBIGUOUS、RL GROUND、RR AIR，`CONTACT_BEARING_UNVERIFIED`，所有 load_fraction_valid=false；保留接触几何但不把未验证的比例解释为实际载荷分配。CoM 与 body 是独立测量；此报告未以 CoM 位移替代 body 位置，也未把历史 FL 放置当成必然持续承载。

## episode 7：命令未越界不等于实测未越界

实际越界关节是 `front_right_knee`，不是 RR：规范硬限 −60°，2° 命令余量下最终 target 为 −58.00000000000001°。同一关节实测值从 tick 2680 的 −55.421276°，经 2688 −56.876226°、2696 −57.717890°、2704 −58.544513°、2712 −59.640124°，到 2715 **−60.00254031222839°**；上述末端 target 均为 −58°。

实测值依据同 tick `transfer_roles.RL.receiver_workspace_state.joint_range_margin_deg.front_right_knee.negative_deg` 加下限反算（RL 角色的对角接收者为 FR，不能误标为 RL 关节）。终态 372 维实际关节观测独立 float32 回解为 −60.00253915786743°，速度 −17.488327°/s，与负 margin −0.002540312° 一致。不是将 target 或 nominal 当成实测，也不是仅有浮点 −58.00000000000001° 的命令舍入问题。

`sensing/guard_state.py:365` 按 live logical joint.position_deg 与闭区间逐关节判断，无命令替代；`ppo/semantic_backend.py:290` 独立保留该实测硬限安全信号。这里可确认是受限 target 下仍发生实际动态越界，不能仅凭这些数据把原因单独归给驱动跟踪、接触力或策略某一项；没有发现 setter 未下发的证据，不降低硬限或改写失败。

## FALL 分支：终态观测确实保存了可解码测量

六个终态 audit 均有 `terminal_observation.policy/critic`（372 维、两者相同），`terminal_observation_finite_fallback=false`。按本实验 `observation_schema.json` 与 `semantic_observation.py` 解码：actual joints offset38×90、gravity offset66×1、body velocity offset75×1、body angular velocity offset78×5、obstacle relative planes offset93×1（均为零基，相关字段未 clip）；固定 chassis 四元数 identity。这是已保存 float32 特征回解，不是原始双精度物理流。

当前 `scene_factory.py` SHA256 `02df57b900fc4208ff4959e9db6fb9b39116210a4fe58d9848e6e1aa06b7fca1` 与该 run runtime_contract 相同；其障碍中心 z=.025、height=.05，锁定 bottom=0/top=.05。由 bottom−base_z 的已存实测相对几何推导下表高度，且以 top−base_z 交叉核对到 float32 精度；没有重新 raw-stream 复核几何。

| episode | gravity z | 推导 base z (m) | 终止分支 |
|---|---:|---:|---|
| 4 | −0.984572947 | 0.014257297 | 高度 <0.015 m |
| 5 | −0.974151373 | 0.014727108 | 高度 <0.015 m |
| 6 | −0.970230341 | 0.014935791 | 高度 <0.015 m |
| 7 | −0.971287251 | 0.054899864 | 非 FALL；实际 FR knee 硬限 |
| 8 | −0.966955006 | 0.014631487 | 高度 <0.015 m |
| 9 | −0.919131279 | 0.014878109 | 高度 <0.015 m |

各 FALL 均不满足 gravity_z>−.30 的姿态分支；因此不能笼统描述为翻倒。episode 9 body-frame 线速度 [−.004887,+.010081,−.070776] m/s、角速度 [−.321880,+.217892,−.051895] rad/s；身体持续下降与足端大抬升可以同时出现。

## 执行连续性、更新上下文和边界

1538 个决策覆盖 12276 个真实物理 tick，全部 native effect verified，未发现 forbidden state writes；own-phase request tick=12243，差额 33 个均为已验证的 handoff-hold tick，不是漏下发。33 条交接审计 residual step 最大 servo=1.421e−14°、wheel=1.110e−16 rad/s，未丢弃通道、未因阶段尺度裁去历史 residual、无 hard-safety-modified 交接。owner 的 nominal request 可以跳变，不能据此认定 mapper/filter 被 reset，也不建议恢复双重 pre-slew。

该有界 global 区间内有 12 次真实 PPO 更新 1089–1100（global 143872–145280，每次20 optimizer steps）；全部记录 finite_nonzero_gradient_observed=true、actor_parameters_changed=true。更新跨越 episode，故不是六次固定同策略配对试验，也不能由某回合更晚阶段证明稳定性改善。未把范围以外更新或下一块教师 P10 后缀算入本报告。

本次仅新建此输出报告，无生产、训练配置、checkpoint、manifest 或进程改动；未运行测试、Torch 或 Isaac。
