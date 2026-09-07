# 当前 capture 区域保持：真实回退证据与只读数学候选

2026-09-06。生产 HEAD `4d268fc547b7`；本报告仅整理此前 PowerShell 只读数据与源码分析，未运行 Python/Isaac、未修改生产/奖励/判定、未做物理消融。主线程已完成本块至 global 54656 / PPO update 392，并在运行自然 P01 重载评估；本报告的详细回合窗口在此前已完成的 **53888 / update 386** 边界内，不能把后续尚未结束的评估预记成功。

来源 run：`runs/ppo_semantic_v3/train/20260906T1308472858273Z_g4d268fc547b7_dcefe730da37448599db863a5f261d47`；主要文件为 `residual_and_projection_audit.jsonl`、`optimizer_updates.jsonl`。对象是第二个真实回合 episode 1，不是固定最终 checkpoint 的确定性评估。

## 1. 先验证 7048 放置，不把快速回退当成错误判定

RR 历史为 qualified tick 6112 → crossed 6536 → placed 7048。global **53807** 恰好记录 decision-end tick **7048**：

- RR front distance **+0.427309 mm**、clearance **+0.669518 mm**；
- `within_top_xy / top_geometry / top_contact = true`，`obstacle_pair_active = true`；
- `GROUND / AIR = false`，连续 TOP 计数 **2**，达到现有 `minimum_top_samples=2`；
- RR 当前 load fraction **0.598101517**，历史首次置 placed，并发生 P09→P10。

所以这里有**当前共享判定要求的 TOP/contact＋XY 证据**，不是“轮已经明显在台下/纯 wall 几何状态，却直接沿用旧 placed 标签”。但日志标明接触证据采用 `verified_body_pair_and_live_wheel_geometry_no_contact_point_classification`；本训练流没有保留此 tick 的完整 contact point/normal，不能独立宣称接触 patch 已被复核为纯竖直顶面。当前判定不因本报告而重分类或放宽。

| global / tick | RR 当前 front / clearance（mm） | 当前接触 | RR placed 历史 | 全局 phi |
|---|---|---|---|---:|
| 53806 / 7040 | −3.784 / +3.278 | AIR；无 obstacle pair、无 TOP；XY 容差内 | false | 0.615253294 |
| 53807 / 7048 | +0.427 / +0.670 | TOP、obstacle pair；load 0.598102 | 首次 true | 0.676000056 |
| 53808 / 7056 | −5.673 / +1.627 | AIR、无 TOP | true | 0.657909298 |
| 53809 / 7064 | −19.427 / +3.235 | AIR、无 TOP | true | 0.678472147 |
| 53814 / 7104 | −86.224 / −50.234 | GROUND；load 0.493929 | true | 0.678560724 |
| 53818 / 7136 | −79.261 / −50.052 | GROUND；load 0.443366 | true | 0.680000000 |
| 53823 / 7176 | −73.167 / −50.582 | GROUND；load 0.468281 | true | 0.676779260 |

离开 TOP 的 7056 采样距 placed 仅 8 tick（0.0667 s）；7104 采样已回地，早于最初报告的 7176。表中“首次观察回地”等仅为该 8-tick 决策末采样精度，不冒充未保存的 120 Hz 首接触时刻。7176 的 FL 仍真实 TOP、load **0.531718793**；RL AIR，clearance **−49.028 mm**、front **−257.891 mm**。历史 placed 不等于当前仍在台面承载。

## 2. strict RR-only 区域变化的直接 phi 信号确实缺失

当前 `semantic_supervisor.py:552–578` 的 global potential 对已 placed 腿在第 559 行直接追加 `1` 并 `continue`。第 577 行只在四腿都 placed 后启用 finish。这里 RL 未 placed，故：

`phi = 0.85 × (3 + RL_current_progress) / 4`

**固定 history、其它腿及所有参与的 support/load 等量，只改变 RR 自身当前 front/clearance/区域几何，RR 的直接份额仍为 `0.85/4 = 0.2125`；该 isolated phi 差严格为零。** `predicate('placed_RR')` 也保持历史完成值 1。actor observation 仍看得到 RR 当前状态，缺的是这项 dense reward 的直接敏感性，不是传感器信息消失。

| tick | RR 原始 workspace / unload / capture（仅诊断） | RR 实际进入 phi 的 leg value / 份额 | 由实际 phi 反解的 RL leg value |
|---|---|---|---:|
| 7040，未 placed | 1 / 1 / 0；initial=1、hard lift/cross=true | 0.8 / 0.1700 | 0.095309617 |
| 7048，placed | 1 / 0.502373 / 1 | 1 / 0.2125 | 0.181176732 |
| 7056，已 AIR 退回 | 1 / 1 / 0 | 1 / 0.2125 | 0.096043755 |
| 7104，GROUND | 1 / 0.632588 / 0 | 1 / 0.2125 | 0.193226935 |
| 7176，GROUND | 1 / 0 / 0 | 1 / 0.2125 | 0.184843578 |

诊断列是按当前源码公式用记录值作 PowerShell 标量计算，**placed 后实际代码跳过这些 RR 分量**。7048→7056 全局 phi 确实下降，但来自 RL 可用 support/unload 等其它项，而不是对 RR 自身区域损失的直接分项；之后 RR 回地时全局 phi 还可以因 RL 进展回升。不能反过来把这种回升解释为“reward 专门奖励 RR 倒退”。

整个 reward 仍有间接依赖：RR 当前接触/载荷可改变 RL 的 support/unload，另有身体运动、接触质量、动作平滑、时间和终止事件。7048 的 task-progress reward 为 **+0.285500475**；7056 为 **−0.108234854**；7104 为 **+0.002538192**；7176 为 **−0.021782433**。符号本身不证明 RR 区域分项存在或反号，且原 shaping 使用 `5 × (0.995 × phi_after − phi_before)`，不是每 tick 重复发 placed bonus。

## 3. 实际动作不能缩写为 nominal，也不能据此作单因果归因

以下均为 canonical command 度数，不是实测机械关节位置：

| tick | RR nominal hip/knee | mapper native | PPO residual | 实际 final target |
|---|---|---|---|---|
| 7048 | −6.9 / −37.8 | +3.1 / −27.8 | −5.0192 / −12.2 | −16.6694 / −48.75 |
| 7056 | −6.9 / −34.6 | +3.1 / −34.6 | −6.4931 / −8.7 | −6.6694 / −43.3 |
| 7104 | −6.9 / −27.2 | +3.1 / −17.2 | −3.6682 / −6.4683 | −0.5682 / −23.6683 |
| 7176 | −6.9 / −27.2 | +3.1 / −17.9621 | −3.5328 / −13.8164 | −0.4328 / −31.7785 |

final 仍受真实 mapper 补偿与上一实际 final 的 slew 影响，不能简单用 nominal+residual 代替。所查 7048–7176 的四轮 nominal 均为零；7176 实际四轮为 residual **[−0.146858, −0.026337, −0.130201, +0.081733] rad/s**。这里不是此前那种“永久残留 rolling prior”证据。相关行的 no-in-episode-state-writes 均 true。

这些数据说明 nominal、策略动作、映射历史与全身接触同时作用，未隔离出哪项导致回退。几何限降 nominal 的线性 target preview 同样不是实际位移/净空保证。

## 4. proposal-only：复用 capture 的 0.2，而非新增 bonus

候选仅修改 **global physical_potential 的已 placed 腿分支**：

`leg_value = 0.8 + 0.2 × R_current_region`

0.2 使用原 capture 权重，未 placed 腿的公式不变；append-only placed/lift/cross 历史、阶段完成 predicate、RL 顺序、成功判定、奖励五族和总边界均保留。不是撤销合法完成事件，也不让已完成阶段重走。

`R` 只依赖当前几何，不能使用必须承载、必须 TOP 接触、连续 TOP 帧数、AIR 惩罚、固定支撑组合、关节姿态或停稳要求。一个可讨论的连续数学候选是：

- `d_xy`：当前轮 center 到**既有 XY tolerance 扩展台面矩形**的外侧距离（可按最大轴向外距定义，矩形内为 0）；需要保留当前几何中已有的 front/back/left/right 数据或其派生外距，不能只把 `within_top_xy` bool 当二值惩罚；
- `R_xy = clip(1 − d_xy / 0.25)`，0.25 复用现 workspace/carry 的外侧衰减长度；
- `R_z = k / (k + max(0, top_gap_min − current_clearance))`，复用 `k=minimum_lift_gain=0.008 m` 与现 `top_gap_min=−0.015 m`；
- `R = min(R_xy, R_z)`，范围 [0,1]。台面上方 AIR 不因超过原 top gap 上限而扣分；这是几何保持进度，不是要求当前接触的 placement 判据。

这只是最小既有成分重分配候选；公式、尺度复用是否合适仍须实施前审查，不是已验证的 reward 或效果承诺。尤其 0.25 m 外仍有现有 clipped 函数的平坦区，本提案不宣称解决任意远距恢复。

### 同状态正反例

| 仅替换已 placed 腿的当前几何，其余量固定 | 旧分支 | 候选结果 / 边界 |
|---|---|---|
| 7048：XY 内、clearance +0.6695 mm | leg=1 | R=1，仍 leg=1；原合法放置进度不减 |
| RR 改为 7176：front −73.167 mm、clearance −50.582 mm | 仍 leg=1，直接 phi 差=0 | 若其它矩形边界内，Rxy=0.727331、Rz=0.183562，R=0.183562；leg≈0.836712 |
| FL 等腿暂时 AIR，但 XY 内且在台面上方 | leg=1 | R=1，满保持信用；允许卸载、协同和不同姿态 |
| 几何相同，只切换接触/载荷标签 | 不提供直接区域差 | 候选也不增加该差；仍只有原 reward 的间接 support/load 依赖 |
| 从区域外恢复到区域内/上方 | 原 placed 份额不变 | R 连续增加，提供恢复信号，不发 reset、不回旧 pose |

7176 数值例中，旧 RR phi 份额为 **0.2125**，候选份额约 **0.177801367**，isolated phi 差 **−0.034698633**。保持该行其它量不动，实际记录 phi **0.676779260** 在候选数学下为 **0.642080627**。这是 PowerShell 标量反例计算，**未运行新 reward、未获得物理因果或学习效果验证**。

该规则若采用，应对每条历史 placed 腿 FR/FL/RR/RL 一致，而不是只给 RR 定制保护；但因为 R 不要求接触，合法 AIR 支撑转移不会被硬锁。共享 TaskEvaluator 仍负责真正违规/成功判定，不能用 reward 候选补写成功历史。

## 5. 与旧反例一致，但不把所有 RL 失败归为 RR 回退

此前 [placed_history_current_region_reward_readonly.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/placed_history_current_region_reward_readonly.md) 已定位同一个直接依赖缺口：历史 38272 自然 P01 回合 RR placed tick 5829，5832 的 TOP/front+8.553 mm/clear−0.737 mm 有真实证据，随后 6144 退到前沿后、6424 GROUND；C28032 另有 qualified/cross/placed 与接触点/力的既有核对。新回合不是这一缺口的唯一依据，也不是旧 A/旧 MDP 与当前权重的配对因果试验。

RL 失败还可能发生在 RR 尚未完成、RL 自身准备/净空/前送不足、全身状态或策略探索不同的情形。本报告没有证明“所有 RL 失败都是 RR 退回造成”，也没有证明此候选能修复后腿任务。当前 2048 块不混入新 reward；主线程先完成重载自然 P01 评估，再决定是否实施。若采用候选，应作为显式新 MDP revision 重新采样 rollout，保留旧 checkpoint 与证据，不追加新的 A/B 完整成功训练门禁。
