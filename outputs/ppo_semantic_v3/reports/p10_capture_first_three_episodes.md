# P10 capture：前三个已完成回合的有界核验（训练仍运行）

本报告仅核验运行 `20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc` 的 episode 0–2。源 checkpoint 为 global 62848 / PPO update 456 / optimizer step 9120。读取时已确认优化日志到 global 66176 / update 482；以下三个回合的最后一行 global 65232 已在这一完成边界内。**不统计后续回合，也不把本块 4096 的请求预算当作已完成。**

证据来自该 run 的 `completed_episodes.jsonl` 前三条、`prefix_evidence.jsonl` 前三条结果、`residual_and_projection_audit.jsonl` 的固定 global 62849–65232，以及 `optimizer_updates.jsonl` 的完成边界。前 512 决策的策略分布与 exact-resume 证据另见 `p10_capture_initial_512.md`，本报告不重做该扫描。

## 1. 优化样本与教师前缀分账

| 回合 | PPO global 范围 | PPO 决策 | PPO 物理 ticks | 阶段样本 P10/P11/P12/P13 | 终止 tick / 仿真秒 | 正式结果 |
|---|---|---:|---:|---|---|---|
| 0 | 62849–63815 | 967 | 7736 | 1 / 1 / 65 / 900 | 15320 / 127.666667 | P13，INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 63816–64779 | 964 | 7712 | 1 / 1 / 62 / 900 | 15296 / 127.466667 | P13，INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 64780–65232 | 453 | 3617 | 1 / 1 / 451 / 0 | 11201 / 93.341667 | P12，INCOMPLETE_CONTROLLER_BLOCKED |
| 合计 | 62849–65232 | **2384** | **19065** | **3 / 3 / 578 / 1800** | — | 3 个未完成，无任务成功 |

前三次实际 prefix 均 accepted，分别执行 948 个教师决策 / 7584 ticks，并明确 `policy_credit=false`。每次在 P10、episode tick 7584 / 63.2 秒接管；`from_P01_current_policy=false`。因此教师共 **2844 决策 / 22752 ticks**，不计入上表 PPO 样本。三次完整物理历程合计 41817 ticks，不能把它们全部称作策略学习 ticks。

固定 2384 条策略审计逐行核验结果：`prefix_teacher_data_in_ppo_storage` 全为 false；native verified ticks 合计 **19065**，等于实际 PPO ticks；全部决策 `all_ticks_verified=true`；四项 in-episode root pose/root velocity/force-or-impulse/gravity writes 均为 0，`no_in_episode_state_writes_verified=true`。

## 2. 后腿历史事件与当前状态必须分开

三次教师 RR 事件均为 Q6938 / C7109 / P7579，均早于策略接管 tick 7584；这些不能算作本回合 PPO 新获得的 RR 能力。RL 的下列事件发生在接管之后：

| 回合 | 策略段 RL qualified | RL crossed | RL placed | 终止时实际 RL | 终止时实际 RR |
|---|---:|---:|---:|---|---|
| 0 | 7959 | 8070 | 8113 | AIR，净空 +7.508 mm，load 0；历史 placed=true | TOP，净空 +0.212 mm，load 0.467820 |
| 1 | 7969 | 8059 | 8089 | TOP，净空 −1.304 mm，load 0.028772，当前 TOP 连续 4 ticks | TOP，净空 −0.058 mm，load 0.502175 |
| 2 | 7961；8054 撤销 | 无 | 无 | GROUND，前沿 −252.466 mm，净空 −51.362 mm，load 0.548419 | GROUND，前沿 −233.655 mm，净空 −48.397 mm，load 0.086997；仅历史 placed=true |

Q/C/P 是日志中的资格/越沿/放置事件 tick，不是新的重分类。回合 0 的 RL 在终止时是 AIR，不能因已有 placed 历史称它当时承载；回合 2 的 RR 历史已放置，也不能据此称它仍在台面。

第三回合的 RL 于 tick 7961 获得资格，tick 8054 出现 `qualification_revoked_ground_before_cross`，之后没有 crossing 或 placement，终止时 `active_lift=false`、`placed=false`、`placed_RL` 完成度为 0。这一回合第一未完成的后腿环节是 RL 重新获得并保持资格以完成越沿/放置，而不是已经放置后的 P13 停止。

第三回合终止时 FL 实际 AIR / load 0 / 净空 +18.769 mm；FR 实际 TOP / load 0.364584。此处仅报告实际接触与载荷，不从固定支撑组合推导新的成功条件。

## 3. 第三回合的 7-tick 差额来源

唯一短决策是 **global 65232 / episode decision 453**：终止 episode tick **11201**，该决策 `physics_ticks=1`、`terminal=true`、native verified tick count 为 1。其余 2383 个决策均执行 8 ticks。

因此第三回合为 `452 × 8 + 1 = 3617` PPO ticks，较 `453 × 8 = 3624` 少 **7**；前三回合总计 `2384 × 8 − 7 = 19065`。这是已完成决策内的真实提前终止，不是丢失日志或未优化 rollout。

第三回合正式 `INCOMPLETE_CONTROLLER_BLOCKED`，终止 P12 stage age **30.0083333333 秒**，总任务仍余 **106.6583333333 秒**。独立 physical evaluator 的 termination_reason 为 null，日志没有将此终止标为碰撞；也不是 200 秒全任务超时。`time_outs=false`、`terminal_bootstrap_allowed=false`，终止奖励项 −40。保留原判定，不因该回合较短而重分类。

## 4. 前两个 P13 未完成的实际末状态

两回合均在 P13 stage age 60 秒终止，`final_controlled=false`、`final_stable_for_s=0`、`task_success=false`。回合 0 当前 region/support 为 true/true；回合 1 为 false/true。日志完成度 0.723643 / 0.772597 只是进度，不等于任务成功。

| 回合 | 末状态四轮 nominal | 实际 filtered canonical 四轮命令（rad/s） | 最大命令幅值 | 最大实测轮速 | body / angular speed |
|---|---|---|---:|---:|---|
| 0 | 全 0 | +0.179704 / −0.254246 / −0.330504 / −0.142333 | 0.330504 | 0.359397 | 0.099242 m/s / 0.176296 rad/s |
| 1 | 全 0 | +0.165566 / −0.284693 / −0.461021 / −0.211315 | 0.461021 | 0.421291 | 0.008314 m/s / 0.127552 rad/s |

表中命令来自 `actual_drive_target_full12` 的 canonical 逻辑应用 double 向量，**不是经过 native 左轴符号变换的 float32 target readback**；真实 native 下发核验来自独立 actuator audit。以上仅为实际末状态，并非确定性 mean 的完整轨迹，也不据此推断成功概率或应重置策略噪声。

结论：固定前三回合中，两次策略接管后 RL 实际获得 Q/C/P，但未完成 P13；第三次 RL 资格在越沿前落地撤销，P12 未完成并贡献唯一 7-tick 差额。该结论不代表仍在运行的训练块已经完成，更不把教师初始化后缀结果标为自然 P01 全任务成功。
