# 72e P10 块：已封存的 512 个真实 PPO 样本

本块在 **2026-09-24 08:34:12 UTC** 正常停止于完整更新边界：**512 个优化决策 / 4 次 PPO / 80 次 Adam**；global **226049–226560**，updates **1732–1735**。已规划但本块未消费的 1536 个决策不计入。本报告只读取此 sealed run，不读取下一课程。

请求阶段样本：**P10=4、P11=3、P12=505**，P01–P09/P13=0。四段成功 nominal 真实 P01→P10 前缀各 767 决策、6136 physics ticks，合计 **3068 决策 / 24544 ticks，全部 0 PPO credit**。所有 learner 样本维持 all12 许可和后腿补全 OFF，原 FL assist 单独保留。学习率 1e-5，gamma=0.9985，lambda=0.99；每个完整 128 样本 rollout 有 20 Adam 步，尾部正常非终结 bootstrap。

## 真实回合，不以 phase 代替结果

| 回合 | learner 样本 | 末时刻 | 终态 | RL qualified 端点 | RL 越沿/放置 |
| --- | ---: | ---: | --- | ---: | --- |
| 1 | 314 | 72.0417 s | HARD_JOINT_LIMIT：FR knee | 2 | 0 / 0 |
| 2 | 62 | 55.2083 s | HARD_JOINT_LIMIT：RR knee | 25 | 0 / 0 |
| 3 | 135 | 60.1167 s | HARD_JOINT_LIMIT：FR knee | 6 | 0 / 0 |
| 4 | 1 | 51.2000 s | 非终结，完整 update 后合法停机 | 0 | 0 / 0 |

这三个终止是实测安全中止，不是只因 P12 超时。第四条不是失败或成功，也未人为将普通 phase 变化设为 done。

前三回合 RL 首次真实资格分别在 tick **6198 / 6219 / 6175**，均晚于 learner handoff **6136**。第二回合另有 6529 的重新资格，第三回合另有 6622；共 **5 段可见资格、33 个 qualified 决策端点**，均随后回到 GROUND，尚未越沿。JSON 分开记录 evaluator 资格事件 tick 与首次观察到失资格的决策端点，不把 15 Hz 端点伪称精确 120 Hz 掉载时刻。

## 用户物理窗口覆盖

以下为不互斥的真实决策窗口计数，不是持续秒数或成功率；四个回合的第一个输入没有相邻 learner 行可对齐，输入窗口缺失计 4。

| 物理窗口 | 对齐输入 | 决策端点 |
| --- | ---: | ---: |
| RR 合法 reachable AIR 的四腿准备 | 161 | 161 |
| RR 真实承载＋前腿准备 | 175 | 176 |
| 固定 FR 轴 CoM/body 正投影＋RL 载荷占比下降 | 48 | 49 |
| 当前 qualified RL 边缘恢复 | 8 | 8 |
| qualified RL AIR 已到顶部捕获区域 | 0 | 0 |
| RL 真实合法 TOP bearing | 0 | 0 |
| RL placed / P13 / 全任务完成 | 0 | 0 |

FR 投影窗口**不等于已经完成侧向转移或 RL 合格卸载**；载荷占比下降也不等于绝对力下降。JSON 提供各回合一个独立 CoM/body xyz 与当前 RL 实测力样例，并记录短窗 reference tick。首个样例的短窗含 nominal 前缀运动，不能将全部投影归因于 learner。

## RR 接触的来源与恢复

四回合 RR 首次 `placed` 均在 **6133**，属于零 credit 的 nominal 前缀；learner 首端点 6144 继承真实 TOP 接触。因此不能称本课程让网络学会了首次 RR 放置。

learner 期间，按当前 sensor TOP 接触/bearing false→true 的决策端点分别观察到 **8 / 1 / 7 / 0** 次恢复；不代表持续稳定捕获。合法 TOP bearing false→true 则为 **8 / 2 / 8 / 0** 次，两者不可混同。例如第二回合 tick **6504** 只是重新回到合法几何，当拍之前 sensor TOP 仍承载；tick **6520** 才是此前失去 sensor TOP 接触后的恢复。第三回合亦有回区但接触未断，以及接触恢复却仍在区域外的不同情况。所有回合 RR 的 GROUND 决策端点为 **0**：本块没有补到“RR 落地后重新起抬”的端点样本，不声称已验证那部分 reward 修复。

## 封存模型与范围

Checkpoint：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000226560.pt`

CP226560，累计 **1735 PPO / 34700 Adam**，保存重载 round-trip=true。checkpoint SHA：`868981b1a31c3bba8bc97f747e4e01a7c9d5b1963dccc8fcec2e725c19be87c5`；actor `435d95f5…` → `626143c5…`。只使用元数据/真实日志，未在本分析中加载模型。完整绑定和逐回合细节在同名 JSON。

本块证明真实续训、RL 卸载尝试与部分 RR 接触恢复覆盖，**不证明 reward 导致了确定性提升，也不是自然 P01 全程成功**。当前首个未完成后腿任务是：维持 RL 合格卸载并实际越沿、形成顶部捕获；随后仍需完成放置与 P13。下一 P12+8 课程由主任务继续，此只读报告不阻挡它。

