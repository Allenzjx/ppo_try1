# placed 历史与当前前沿位置：P12 奖励依赖只读审查

2026-09-06。当前 HEAD `49749aa527a4a00901e434104821d7e8241ea8da`；`git diff 64abc..HEAD` 确认本报告涉及的 supervisor、reward、observation 与全部 v3 config 未变。仅 PowerShell 源码/既有报告核对；未读当前未完成训练为新结论、未运行 Python/Isaac、未改生产。当前 policy-only 训练继续。

## 确认的有限结论

**当 FR/FL/RR 已合法 placed、RL 尚未 placed 时，global task potential 没有 RR 当前前沿距离/最终区域的直接保持或恢复项。** 这是真实的局部目标信号缺口，不等于整个 reward 为零，也不证明它是历次 P12 失败主因。

`semantic_supervisor.py:553` 对任何历史 placed 腿直接 `values.append(1.); continue`，跳过该腿当前 workspace、unload、净空、carry、capture。`:565` 又将 finish 项在 all-placed 之前置0。因此这个历史状态下：

`phi = .85 * (3 + current_RL_progress) / 4`

在有效且非 terminal、其余参与项相同的前提下，仅 RR 当前 front distance 从正变负，或其当前 top_geometry/top_contact 变差，不会降低 RR 那一份 phi；回到更安全的位置也不增加这一份 phi。`:632–633` 确认 v3 实际 reward 使用上述 global phi，不是临时计算后丢弃；P12 的 `goal_features/completion_predicates=[placed_RL]` 也没有另一条 RR current-region reward。

`final_region_valid` 实际每 tick 计算（`:432–433`），并参与成功稳定时间（`:442–444`），所以最终要求并未被遗漏或放宽；**只是 all-placed 之前，它没有通过 finish 项进入 dense task shaping**。`whole_task_success:484–485` 的 forward 还采用各腿/身体的 clipped minimum：即使未来单纯提前打开 finish，尚在前沿后方的 RL 也可能让 forward=0，不能未经检查就声称这一改动会给 RR 提供独立反馈。

## 仍然存在的控制反馈与合理历史

- RR 历史不是伪造成功：`:407–408` 在真实合格 crossing 后、达到连续 loaded 条件才记 placed，并保存事件 tick。后来退回不应抹掉已经发生过的有效事件，也不能把追加式历史误读成“当前仍在TOP承载”。
- 这一历史确实用于顺序/入口语义：RL crossing 要求 RR 已有 placed；P12 有效起点检查 `placed_RR`。撤销它会改变合法任务历史/课程含义，不是本报告建议。
- 当前 RR contact/load 仍能**间接**影响 RL unload 的 support count、归一化载荷（`:525–531`）；整机运动也会影响 RL 几何。不能声称 RR 对 task reward 完全无控制依赖。这里缺的是在其他相关量固定时，对 RR 自身位置损失的直接差分信号。
- reward 还有实际身体姿态/角速度变化、接触运动质量和 applied-command smoothness 成本，以及时间/终止事件。已保留的历史 potential 本身仍产生普通折扣项；不是每帧重复发正的 placed 奖励，也不能仅因单步 task reward 为负判定有反号 bug。
- 324维 actor observation 仍读入当前每腿 clearance/front_distance/load_fraction（`semantic_observation.py:230–238`）及另外的历史位：信息未消失，奖励敏感性与可观测性是两件事。

## 已有物理证据，而非当前新 run 的失败归因

`p01_block_38272.md` 的首个已完成自然 P01 回合记载：

| RR实际记录 | 时刻 | 当前状态 |
|---|---|---|
| 合法 placed | event tick5829；首决策末tick5832 | TOP，front+8.553mm，clearance−.737mm，load .543490 |
| P12入口 | tick5856 | TOP，front+23.689mm |
| 首决策末退到前沿后方 | tick6144 /51.2s | front−.970mm，非TOP；RL AIR |
| 首决策末GROUND | tick6424 /53.5333s | front−53.439mm，clearance−47.706mm |

同回合 RL 在 tick5939 获得真实 qualified，tick6240 AIR 净空+115.489mm但 front仍−200.830mm；最终未cross/placed，并在 tick6632 GROUND 撤销其未完成资格。以上位置/接触“首次”为8tick决策末采样精度，不伪造120Hz精确发生时刻。

另一条独立既有证据是 C28032 的 RR Q5820→C5998→P5999：`top_contact_evidence_readonly.md` 记录5999接触力几乎竖直向上+14.205N、接触点在顶面附近，支持真实放置后退回，而非用15mm tolerance直接判成误放置。它不证明以后始终保持支撑，也不构成某个 reward 修改的因果实验。

## 边界

append-only 完成历史可以保留；它与尚未到终点时的当前、可恢复物理位置进度是不同概念。任何以后讨论都不应自动惩罚所有已placed腿的 AIR：例如 FL 暂时抬起参与整机协同必须仍可合法发生，更不能恢复固定支撑组、唯一姿态或 dwell 成功门槛。

本次仅定位函数依赖和实际失稳实例。没有配对 reward 消融，不能量化这一缺口对 RL crossing 或退回的因果贡献；不建议在正在运行的分布迁移块中同时改 reward，不新增训练门禁或生产补丁。
