# Block09 首回合：P09 真实 BODY_COLLISION

只读本 run 的首 484 条 learner 记录及首条 completed episode；未重放物理、策略前向、优化或修改生产。

## 判定与证据边界

首回合在 **tick 5982 / 49.85 s / global decision 214884** 终止，最后一步实际 6 physics ticks。终止来源为 `BODY_CONTACT`，原因 `central body/obstacle collision`；不是“几何质量代价非零”触发的停止。该回合是 **P04 checkpoint-policy-initialized suffix**：learner 从 tick 2112 接管，484 决策不包括前缀；不属于从 P01 开始的完整策略成功。

生产 `BodyCollisionDetector.evaluate` 仅接受 **/World/WLRRobot/base_link ↔ /World/Obstacle** 已核实的真实 active 接触对，并另要求至少 2 个 active 物理样本或 ≥1 mm 实时 AABB 交叠。腿／轮接触不属于 BODY 失败。当前 evaluator 保存 valid=true / run_validity=VALID；CONTACT_BEARING_UNVERIFIED 不能改写这个已触发的 BODY_CONTACT 为几何代理误报。

本日志没有保存 base 原始力、contact point、具体 collider 子路径、首触点 tick 或 detector reason/streak，JSON 明确为 null。可确定 offending rigid body/pair，**不能定位到某块 base 子碰撞体或具体表面点**。当前姿态 body AABB 终态交叠仅 **0.002069 mm**，小于 1 mm：按已读判定代码，持续接触分支与记录相符；这是代码+已存几何的推论，不是重新测得的原始力/持续计数。

## 碰撞前身体／腿状态

在已存 learner 的逐物理 tick 质量样本中，base 聚合 AABB 的 20 mm 净空软代价首次非零于 **tick 5482 / 45.6833 s**，当时 separation=19.6360 mm，交叠=0。首次记录正 AABB 交叠则是终态 tick 5982。这些量是保守 AABB，不是 mesh 接触位置或穿透深度。

| Tick | base 下界高于障碍顶面 (mm) | FL 净空 (mm) | 当前记录支撑 |
|---|---:|---:|---|
| 5944 | 11.2290 | 136.2360 | FR TOP；RL/RR GROUND |
| 5960 | 6.2860 | 146.6602 | FR TOP；RL/RR GROUND |
| 5976 | 1.8876 | 153.4344 | FR TOP；RL/RR GROUND |
| 5982 | −0.002069 | 153.5930 | 当前 leg-pair bearing/support 全零 |

最后 6 ticks 的 base–障碍 AABB separation 为 **1.49067 → 1.00486 → 0.530665 → 0.058584 → 0.002950 → 0 mm**。最后一次完整非终态端点 5976：FR bearing 16.275 N、RL 11.822 N、RR 4.743 N；FL 已不承载。终态身体线速 0.09039 m/s、角速 0.31537 rad/s；不是用“历史 FL/FR placed”虚构当前支撑。

RR 并非从未抬起：**tick 5384** 取得连续 unsupported AIR 的真实 8.01563 mm lift qualification；**tick 5450** 在尚未 cross 时落地，资格被撤回。最终 RR 无 cross/place，轮底绝对 z=**2.45368 mm**，相对本次 AIR 起点 gain=**2.63549 mm**，AIR 仅 2 ticks、lift_established=false。不得混称“已完成抬升/后腿越障”。

## Capture-assist 与终态动作

FL placed=3307，P05→P06=3312；P06→P07=5256、P07→P08=5264、P08→P09=5272。assist 第一个保存的 RELEASE 端点是 5264（前端点 5256），第一个 RELEASED 是 **5352**（前端点 5344）。这些是保存端点／时间区间，不伪称未保存的准确 mode-onset tick。

因此距碰撞至少 630 ticks / **5.25 s** 前，assist 已完全释放；终态 mode=RELEASED，owner 为 nominal_plus_policy，assist correction 全零、owner_indices=[]、policy_request_unchanged=true。不能将后续碰撞直接归因为当拍 assist 接管。

终态实际下发（每腿 hip/knee 单位 °；wheel 单位 rad/s）：

| 腿 | hip target | knee target | wheel target | 实测 wheel |
|---|---:|---:|---:|---:|
| FL | 20.1814 | −39.9092 | −1.162013 | −1.162121 |
| FR | 10.9216 | −10.1553 | +0.003754 | +0.003942 |
| RL | 23.9492 | +4.0578 | +0.276724 | +0.276697 |
| RR | 79.7202 | −11.1430 | −0.113163 | −0.112665 |

最后 actuator tick 的 native target effect verified=true、own_phase_request_effect=true、handoff_hold_used=false。目标与实测接近不等于动作正确，也不证明某个单通道独立导致碰撞。

终态任务惩罚 −40、Phi .456875→0，PBRS −2.284375，合计 reward −42.286800，bootstrap=false；这是真实采样失败，不是未完成预算尾段或几何成本独自终止。无建议放宽安全条件。

详尽选取值、原始单位与首 484 行哈希见 [JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_p05_hip_only_continuation_v1/block09_first_episode_failure.json)。原始 run：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_p05_hip_only_continuation_v1\train\20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413`。

