# P13 stop progress 平区：已完成真实窗口的只读核验

2026-09-06。仅使用 PowerShell 读取既有报告、生产公式和已完成 audit；没有 Python、Isaac、参数修改或新实现。**本次明确不改 P13 control、entropy 或 std**。以下仅保留为后续单因素候选，不是当前训练门禁，也不是已证明的失败主因。

## 固定证据范围

源运行：`runs/ppo_semantic_v3/train/20260906T1211309264074Z_g9d70aae58243_42ace02dcac84aaa83ee7e1ee578f58e`，runtime `9d70aae58243b9fb2248d64561959e6a67901b44`。该 P10 teacher-initialized suffix 块已经完成 2,048 decisions / 16 updates / 320 optimizer steps，最终 checkpoint 为 **52,608 / 376 / 7,520**；执行完成不等于任务成功。两个完整 episode 均为 P13 `INCOMPLETE_CONTROLLER_BLOCKED`，第三个为未终止尾段。

本报告直接重读 `residual_and_projection_audit.jsonl` 的前 970 行，只保留 episode 0 的 **global 50,903–51,530，共 628 行**。它们均是已完成优化范围内的 P13 请求，decision-end 全 12 维 nominal 为零；对应物理 tick **10,328–15,344**。最后一行为真正终止，其余 627 行非终止。不使用后续运行的未优化 tail，不把 teacher 历史、suffix 或局部进度称为完整 P01 成功。

读取字段：`applied_audit.semantic_task.physical_evaluator.goal_features` 的实测速度、同 evaluator 的最大 command / region / support，以及 `applied_audit.actual_drive_target_full12[8:12]`。这些是实际 audit 记录，不是 actor mean 推算。辅助来源为同目录 `p10_block_52608.md` 与 `entropy_stop_shaping_readonly.md`；后者的旧 64abc 数据不是本报告 628 行窗口，二者没有混为同一批样本。

## 现公式与真实阈值覆盖

`semantic_supervisor.py` 的 `TaskStageSupervisor.predicate("whole_task_success", ...)` 保留三个物理速度项：

`q_speed = clip(1 - speed / tolerance, 0, 1)`。

因此 **speed ≥ tolerance 时该隔离项为零**，包括减速但两端仍超阈值的情况。现有 command 项另为 `q_command = 1 if max_abs_command <= .02 else .02 / max_abs_command`，没有相同的超阈值零平区。

| 物理量 | 未改变的硬阈值 | 628 行中超阈值 | 满足阈值 |
|---|---:|---:|---:|
| 最大实测轮速 | .25 rad/s | 281 | 347 |
| 身体线速度范数 | .05 m/s | 405 | 223 |
| 身体角速度范数 | .30 rad/s | 84 | 544 |
| 最大实际轮 command 绝对值 | .02 rad/s | 628 | 0 |

这些是相互重叠的边际计数，不是互斥失败分类或成功概率。command 最大值在此窗口为 .072670099–.366911288 rad/s。全 900 行 P13 的既有报告记录 controlled 始终 false，采样到的 stable duration 最大为零；15 Hz 摘要不冒充完整 120 Hz 子决策最大值，实际终止结果仍证明没有完成硬要求的连续 .5 s 稳定。

## 真实相邻减速，隔离项仍为 0→0

下列每对均是相邻的实际非终止记录，间隔 8 个 physics ticks；两端 region/support 均为 true。

| 隔离物理项 | global / physics tick | 实际 before → after | 当前隔离项 |
|---|---|---|---|
| 最大轮速，T=.25 | 51,354→51,355 / 13,936→13,944 | .6653692722320557 → .3016962707042694 rad/s | 0 → 0 |
| 身体线速度，T=.05 | 51,464→51,465 / 14,816→14,824 | .20550923449558836 → .07370776886108263 m/s | 0 → 0 |
| 身体角速度，T=.30 | 51,404→51,405 / 14,336→14,344 | .5666096501840356 → .3461716138985146 rad/s | 0 → 0 |

这证明公式对这些真实减速没有该项的连续改善，**不证明总 reward 为零或减速受罚**。例如第一对的身体线速度同时从 .029346893 上升到 .103081070 m/s，角速度从 .418326223 降到 .113161135 rad/s；其它状态及 body/contact/smooth 成本也在变化。不是配对控制实验，不能把总 reward 的符号归因于单个速度。

正例边界：进入阈值以下后原公式有信号。以轮速 .20→.10 rad/s 为纯数学例，该项 .2→.6。command 的既有连续项同样有信号：最大 command .16→.08→.04→.02 时，该项 .125→.25→.5→1。后两组是公式例，不是声称上述真实轨迹逐项执行过这些数值。

## `max` 对非最大轮的信号遗漏

实际 global **50,903** / tick **10,328** 的四轮 command，按 FL/FR/RL/RR 排列为：

`[.15392489696406916, -.18488621010542575, -.12323237071913808, -.1339810112310154]` rad/s。

最大绝对值是 FR 的 .18488621010542575。若在数学反事实中只将 FL 的 .15392489696406916 降到 .02 或零，而其余 command 和物理状态不变，则最大值不变，`q_command=.02/.18488621010542575` 完全不变。**修改后的向量不是实际执行记录**；这只隔离证明非最大轮改善在该聚合项中不可见，直至最大轮切换或最大值下降。

最大实测轮速也有同样的数学聚合性质，但本次紧凑 evaluator 字段仅保存它的最大值，未读取逐轮实测速度向量，故不伪造对应的实测逐轮反事实。最大值作为“所有轮必须达标”的硬控制条件仍有合理意义；信号是否稠密与硬条件是否正确是两个问题。

## 权重、计奖与成功条件边界

当前 stop 是四项平均，finish 中占 .2，global physical potential 的 finish 占 .15。因此每个 stop 子项对 global φ 的系数为 **.0075**，没有第二次添加独立 bonus。finish 仅在四腿历史 placed 后进入；普通 phase 标签不改变这条全局公式。

奖励仍为 `F = 5 * (.995 * phi_next - phi_before)`，真正 terminal 的 next phi 置零；其余 reward families、事件和按真实 dt 积分的成本另计。以仅 command 改变、非终止的数学例，.16→.08→.04→.02 的隔离 shaping 分别为 +.004640625、+.009281250、+.018562500。保持相同进度不是每帧重复发放进度值，而有折扣项。以上数字不能直接当真实整步总 reward。

本窗口终止 global 51,530 的实测最大轮速 .180182397 已达标，但身体线速度 .124574697、角速度 .335342168、command .178885413 仍未达标。因此这里不仅存在命令噪声问题。成功仍要求真实速度/command、region、当前 support、安全及连续 .5 s 全部满足；历史 placed 和 AIR 合法协同不等于最终停稳。

没有发现 entropy 混进环境 reward、重复乘物理频率或维数、终止 bootstrap、potential 符号/折扣接线的具体错误。entropy 是 PPO loss 正则，不能拿它的标量 loss 与环境 reward 直接相减，也不能根据这些平区推断 sigma/entropy 是失败主因。

## 后续候选，不是本次实施决定

证据支持将“现有 control 权重内、超阈值仍连续单调的当前物理进度”保留为单因素候选；可参考既有 command reciprocal 形式。若要补非最大轮的减速信号，还可以单独审视逐轮进度聚合，但硬成功里的最大值条件必须原样保留。不得以 shaping 替代稳定时间，不能硬清轮命令、关闭 PPO 或据此操纵 std/entropy。

候选应复用原控制份额，经同一全局 potential 只计一次；正反例应区分超阈值减速、加速、非最大轮变化、阈值连续性、普通阶段无重复信用，以及 region/support/稳定时间仍不可跳过。它属于后续显式 reward/progress 语义修订，而非无声修改或策略网络重置。

**本次结论是局部信号缺口，不是已证实失败主因。当前选择明确不改 P13 配置、entropy 或 std，也不把此候选设为优化器门禁。** 继续既定训练/评估，是否在后续固定边界选择单因素修订另行决定。当前 capture-retention 改动不属于这个历史窗口，亦未由本报告改动或评判其训练效果。

仅新增本报告；未改任何现有报告、生产文件、配置、测试或脚本。写入后停止。
