# Capture retention：固定 3072 决策窗口

**有界只读结论：截止 global 57728 / PPO update 416，仍未出现 R<1。训练后续不计入本报告。**

运行：[20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189)，capture revision / P01 / N1 / seed 1001，source global 54656。本报告只使用 PowerShell 流式读取，未运行 Python/Isaac，未修改生产或主报告。

## 1. 实际更新链与阶段样本

固定连续审计范围 **54657–57728**，恰好 **3072 决策**。`optimizer_updates.jsonl` 中对应 **393–416** 共 24 次 PPO update、**480 optimizer steps**；逐行 global 增量均为 128、update 增量均为 1、每次 optimizer steps 均为 20。23 个相邻 update 的 actor before hash 与前一次 after hash 全部一致；24 次均 finite nonzero gradient=true、actor parameters changed=true。

- 首次 update 的 actor before：`cca90f60744f0e00068068c9160cd06648a129a881df9cb010380a17dc716f5d`
- update 416 的 actor after：`4853f86fd4f86d296305333dab4258940abb4769354bffb785357308d12d5476`

这两项是 update 日志的 actor 参数 hash，不是 checkpoint 文件 SHA，也不在这里额外宣称独立重载检查。

| 起始阶段 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10–P13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 已优化决策 | 4 | 701 | 14 | 4 | 590 | 1743 | 4 | 1 | 11 | 0 |

阶段按每行 `applied_audit.phase_id` 计数，合计 3072，不以结束阶段或计划份额替代。窗口物理 ticks 合计 **24569**。3072 行的 native audit `all_ticks_verified` 和 `no_in_episode_state_writes_verified` 均为 true。

## 2. 三个完整回合与第四回合的已优化尾段

| 回合 | global 范围 | 决策 / physics ticks | 实际结束 | 结果 | return |
|---|---|---:|---|---|---:|
| 0 | 54657–55595 | 939 / 7512 | 62.6 s，P06 | INCOMPLETE_CONTROLLER_BLOCKED | −53.5871461315 |
| 1 | 55596–56272 | 677 / 5409 | 45.075 s，P09 | BODY_COLLISION | −49.5469740462 |
| 2 | 56273–57203 | 931 / 7448 | 62.0666667 s，P06 | INCOMPLETE_CONTROLLER_BLOCKED | −53.6507499019 |
| 3（未终结） | 57204–57728 | 525 / 4200 | 截止 35 s，P06 | 窗口内无终止 | 不预填最终 return |

前三回合 task success 均 false。回合 1 的最后一个决策 g56272 只执行 **1 physics tick** 即 BODY_COLLISION；native verified tick count=1，`time_outs=false`、`terminal_bootstrap_allowed=false`。因此总 tick 数比 `3072×8` 少 7，不是丢失 7 条样本或同伴截断。

| 回合 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 186 | 4 | 1 | 147 | 600 | 0 | 0 | 0 |
| 1 | 1 | 167 | 3 | 1 | 148 | 341 | 4 | 1 | 11 |
| 2 | 1 | 178 | 3 | 1 | 148 | 600 | 0 | 0 | 0 |
| 3 已优化尾段 | 1 | 170 | 4 | 1 | 147 | 202 | 0 | 0 | 0 |

三次完成回合的前腿硬事件（qualified / crossed / placed，episode-local physics tick）：

| 回合 | FR | FL | RR / RL |
|---|---|---|---|
| 0 | 45 / 1504 / 1524 | 1611 / 2640 / 2709 | 均无硬 qualified/cross/placed 事件 |
| 1 | 66 / 1348 / 1368 | 1450 / 2488 / 2559 | 均无硬 qualified/cross/placed 事件 |
| 2 | 59 / 1445 / 1455 | 1534 / 2579 / 2642 | 均无硬 qualified/cross/placed 事件 |

终止处当前状态不能由历史 placed 推断：

- 回合 0：FR 当前 TOP/load .466611；FL AIR/load 0。RR/RL 都 GROUND，front 分别 −395.201 / −456.720 mm，P06 `rear_approach=.0531203931`。
- 回合 1：FR、FL 都 AIR/load 0；RR AIR、front −198.322 mm、clearance −45.091 mm；RL GROUND/load 1。P09 `placed_RR=.2652685262` 是未完成进度，绝非放置事件。结果保持原 BODY_COLLISION，不另行重分类。
- 回合 2：FR TOP/load .443386；FL AIR/load 0。RR AIR、front −317.697 mm、clearance −49.511 mm；RL GROUND、front −360.738 mm。P06 `rear_approach=.4370486324`，尚未完成后腿准备。

## 3. Capture 分支是否实际 exercise

只对历史已经 placed 的腿，逐行从实际 `top_xy_outside_distance_m` 与 `clearance_m` 计算当前 R；未 placed 的后腿仍属于原 workspace/lift/carry 进度分支。本次没有重复首 1024 的完整 phi/shaping 重算；其核验收据见 [capture_retention_live_reward.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/capture_retention_live_reward.md)。

| 腿 | 已 placed 腿-决策行数 | R<1 总数 | 新增 2048 行中的 R<1 | AIR 且 R=1 |
|---|---:|---:|---:|---:|
| FR | 2357 | 0 | 0 | 57 |
| FL | 1763 | 0 | 0 | 1462 |
| RR | 0 | 不适用 | 不适用 | 不适用 |
| RL | 0 | 不适用 | 不适用 | 不适用 |

所有实际已 placed 样本的最小保持度仍为 **1**。没有真实保持度下降、低谷或恢复序列，因此不拼接 CPU 反例或别的版本轨迹来补造“完整正反序列”。合法 AIR 满保持仍被实际覆盖；但新增保持扣减及恢复响应 **截至本窗口仍未 exercise**。

窗口里进入 P09 的 11 条样本不等于 RR 抬升/放置完成，P06 回合超时也不能解释为 capture 惩罚造成——此窗口没有任何 active capture 扣减。这里没有比较其他 checkpoint 的固定均值轨迹，不声称策略进步或物理因果。后续训练保持主线程原计划，本报告到 57728 停止统计。
