# Capture retention：首 1024 已优化样本的有界真实奖励核验

状态：**固定已完成窗口；训练仍运行。未实测到保持度下降/恢复分支。**

本报告只读 PowerShell 解析真实日志并做标量重算，没有运行 Python/Isaac，没有修改生产、奖励、判定或训练状态。窗口以实际 `optimizer_updates.jsonl` 中 global **55680 / PPO update 400** 为截止，不含随后采集或已更新的数据，不把本次计划 8192 决策记作完成。

## 1. 来源与实际优化分账

运行目录：[20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189)。当前 capture revision，P01 / N1 / seed 1001，source global 54656。

- 连续审计范围：global **54657–55680**，恰好 **1024** 条，物理 ticks 合计 **8192**。
- 完成优化：updates **393–400**，8 次 PPO update、160 个 optimizer steps；8 行均 `finite_nonzero_gradient_observed=true` 且 `actor_parameters_changed=true`。
- 回合 0：939 决策，global 55595 / tick 7512 / **62.6 s**，P06 `INCOMPLETE_CONTROLLER_BLOCKED`，task success=false。
- 回合 1：窗口内只有 85 决策；global 55680 / tick 680 / P02，未终结。这里不借用窗口外的后续 P06 状态。

| 起始阶段 | P01 | P02 | P03 | P04 | P05 | P06 | P07–P13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 已优化决策 | 2 | 270 | 4 | 1 | 147 | 600 | 0 |

阶段数取每行 `applied_audit.phase_id`，不是结束阶段，也不是 nominal 内部通道来源。

## 2. 新 phi、旧反事实与实际 shaping 全窗口核验

当前 opt-in 为 `capture_retention_semantics=current_platform_region_after_placement`。对已经实际 placed 的腿，重算：

`Rxy = clip(1 − top_xy_outside_distance_m / 0.25, 0, 1)`

`Rz = 0.008 / (0.008 + max(0, −0.015 − clearance_m))`

`R = min(Rxy, Rz)`，`leg_new = 0.8 + 0.2 R`。

对未 placed 腿，独立重算原 workspace、其他腿 support/load-ready、initial、hard/soft lift、carry、capture 与前驱条件；全局仍为 `min(1, .85 sum(leg)/4 + .15 finish)`。旧反事实只把已 placed 分支恢复成 `leg_old=1`，其余同一实际观测、历史和公式完全不变。它不是另一次物理 rollout。

每个非终止决策核验 `5 × (.995 × phi_next − phi_prev)`；真实终止时按原有限任务约定，next potential=0、不 bootstrap。两次 fresh P01 首决策保留日志的 initial potential：其当前历史没有任何 placed，故首状态的新旧 capture 差为零；其余 **1022** 个同回合相邻 before 值与上一真实 after 状态重算相接。

| 核验项 | 首 1024 行最大绝对误差 |
|---|---:|
| 独立新 phi vs `semantic_task.task_progress_potential` | 0 |
| 考虑终止归零后的 phi vs `reward_breakdown.potential_after` | 0 |
| 同回合 `potential_before` 连续性 | 0 |
| `5(.995 phi_next − phi_prev)` vs `potential_shaping` | 0 |
| shaping + terminal event − `.02 × elapsed_physics_s` vs task-progress family | 0 |

这里的 0 是这次 PowerShell double 重算与落盘数值的实际最大差，不代表连续物理模型证明。

**新旧反事实 phi 不同的行数为 0；新旧 shaping 差也全为 0。** 原因是本窗口里所有已 placed 腿的 R 都等于 1，而不是实现跳过了 AIR 或载荷状态。不得把整体 reward 的正负变化误认成 capture 保持项的响应。

## 3. 实际保持度与 AIR / load 分开统计

下表以“决策结束时该腿历史已 placed”的腿-决策为分母；不把 RR/RL 尚未放置的台下位置算作已放置保持惩罚。

| 腿 | 历史 placed 行数 | R<1 | AIR 且 R=1 | 已 placed 时 clearance 范围 (mm) | 最大 XY outside (mm) |
|---|---:|---:|---:|---:|---:|
| FR | 749 | 0 | 8 | −2.253772 ～ +4.240363 | 0 |
| FL | 601 | 0 | 550 | −0.829923 ～ +46.580290 | 0 |
| RR | 0 | 不适用 | 不适用 | 不适用 | 不适用 |
| RL | 0 | 不适用 | 不适用 | 不适用 | 不适用 |

FR 真正 qualified/cross/placed ticks 为 **45 / 1504 / 1524**；FL 为 **1611 / 2640 / 2709**。事件 tick 与决策结束 tick 有区别。

| 实际行 | 状态 | outside / clearance (mm) | 实际 load fraction | 新旧 phi | 新旧 shaping |
|---|---|---:|---:|---:|---:|
| g54847 / t1528 / P03 / FR | TOP，placed event t1524 | 0 / −1.787842 | .394186744 | .255 | +.312375 |
| g54995 / t2712 / P05 / FL | TOP，placed event t2709 | 0 / +.738209 | .277261767 | .444613884 | +.299454073 |
| g54997 / t2728 / P06 / FR | AIR，无 TOP / GROUND | 0 / +2.171881 | 0 | .425 | −.105789848 |
| g54999 / t2744 / P06 / FL | AIR，无 TOP / GROUND | 0 / +.058716 | 0 | .44625 | +.09509375 |

这些 AIR 行的 R 均为 1；FR 的总体 shaping 为负和 FL 的总体 shaping 为正，都不是新增 capture 罚/奖：同一实际状态下旧 `placed=1` 反事实得到完全相同结果。FL 已 placed 样本的最高净空为 +46.580290 mm，超过旧 top-contact 几何上界 +25 mm，保持度仍为 1；另外明确记录到 550 条 AIR 满保持行。当前保持不是 contact/固定支撑组/停姿约束。

没有 R<1 的实际样本，因此本窗口没有“退回使 phi 降低”或“恢复使 phi 增加”的真实序列可展示。此分支只已有专项 CPU 正反例，不能把单测或先前版本的回退轨迹说成本次已经 exercise 的 live 奖励。

## 4. 当前未完成处与终止约定

首回合终止时 P06 `rear_approach=.05312039307878991`；RR/RL 均没有 initial/qualified/cross/placed 事件，均为 GROUND。RR front=−395.201337 mm、clearance=−50.156555 mm、load=.042494402；RL front=−456.719902 mm、clearance=−49.712231 mm、load=.490894620。两者尚未 placed，因此新 capture 保持项本来就不应该因其台下位置扣减历史完成信用。

终止行 g55595 / t7512：原始物理 phi=.4537366947155732；reward before=.4539061320569652，reward after=0，shaping=−2.269530660284826，terminal event=−40，`terminal_bootstrap_allowed=false`。这保持原终止语义，不是新的 capture 终止条件。

结论限于：该真实已优化窗口公式接线一致，并实际保留合法 AIR 满信用；退回/恢复响应尚未实测。没有证明 capture revision 已改善策略或后腿可达性，不将随机训练前腿事件与 C54656 固定均值评估作因果比较，不将 P06 incomplete 或运行完成称为全任务成功。训练仍按主线程计划继续；本报告到 55680 停止统计与写入。
