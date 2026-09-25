# 72e P12+8：1024 样本封存覆盖

**实际新增1024个优化决策 /8次PPO /160次Adam**，global226561–227584，updates1736–1743。2026-09-24 **09:37:22 UTC** 正常停止于完整更新边界；原1536计划中未消费512不计credit。前两个回合的旧68/9待优化尾段现已进入完整更新，不重复计数。全部1024个请求阶段为P12。

三个successful_nominal真实连续前缀各777决策、6216 ticks（51.8s），共 **2331决策 /18648 ticks，全部零PPO/teacher credit**；无fallback。三次RL初始资格都是prefix tick6212，RR首次placed均为prefix6133，不能记为网络发现。

| 回合 | 已优化样本 | 终点 | 结果 | learner新RL资格段 | RL qualified端点（含继承） |
| --- | ---: | ---: | --- | ---: | ---: |
| 1 | 452 | 81.9333s | P12有限恢复耗尽，非安全中止 | 2 | 28 |
| 2 | 453 | 81.9583s | P12有限恢复耗尽，非安全中止 | 3 | 26 |
| 3 | 119 | 59.7333s /tick7168 | 非终结，正常update边界停机 | 2 | 30 |

RL共有 **84个当前qualified端点**：其中16个继承三次prefix6212，68个对应7次learner新资格。RL越沿、顶部捕获/承载/放置、P13、全任务成功均0。第三段的新资格6362、6650分别18/7端点，随后6512、6712观测到GROUND；前5个qualified端点仍是nominal继承。

## 物理窗口，不用P12标签代替任务

| 非互斥决策端点窗口 | 数量 |
| --- | ---: |
| RR合法reachable AIR的协同准备 | 72 |
| RR真实承载＋前腿准备 | 159 |
| 固定FR轴CoM/body正投影＋RL载荷占比下降 | 36 |
| 当前qualified RL边缘恢复 | 9 |
| qualified RL AIR进入顶部捕获区 | 0 |
| RL真实合法TOP承载 / placed / P13 /完成 | 0 |

第三类只是投影与占比条件，**不是已证明向FR侧运动或绝对卸载**，部分短窗含nominal前缀。已保留的第二回合6688→6752确有body前送、RL实测力10.59→0N，但CoM主要向前，不据此宣称完整FR侧移或越障。样本计数不是持续时间/成功率。

## RR地面恢复与合法TOP恢复严格分开

全块RR-currentGROUND **675**端点；sensor TOP承载211端点，满足合法XY的TOP承载 **164**端点。后者包含继承前缀支持及掉载后的短暂恢复，不是164次新放置。

前两回合RR首次GROUND后均没有新的有效抬升/TOP恢复。第三段RR在 **6840** 首次GROUND；此前合法TOP回归端点为6280、6360、6544、6632，最后合法TOP为6760，均不能冒称地面后重捕获。

第三段有**有限真实地面后再离地**：

| 端点 | 当前资格/接触 | 相对最后实测接触抬升 | 前缘距离 | 轮底相对台面gap |
| --- | --- | ---: | ---: | ---: |
| 7120 | current-qualified AIR，0N | 9.01mm | −67.84mm | −41.29mm |
| 7128 | current-qualified AIR，0N | 11.58mm | −75.70mm | −38.72mm |
| 7136 | current-qualified AIR，0N | 7.44mm | −76.95mm | −42.87mm |
| 7144 | 已回GROUND，资格撤销 | 0 | −78.66mm | −50.13mm |

这是从实际contact参考tick7097开始的连续AIR上升，不是旧active_lift5427/placed6133刷历史；但三端点均outside XY、非reachable、无TOP承载。**全块地面后TOP接触/合法承载恢复为0**，不能叫重捕获成功，也不能将这三个训练端点当最新确定性策略已学会恢复。

第三段截至7168末端仍RR/RL ground、FL AIR、FR TOP；首learner端点至末端body forward−135.06mm，固定FR轴CoM净投影−110.95mm。非终结不等于成功或已发生第三次任务失败。

## 已封存模型与分析范围

Checkpoint：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000227584.pt`

累计 **227584决策 /1743 PPO /34860 Adam**，roundtrip=true。checkpoint SHA `7969bdd11f60acb2aa1aaf09185aa6a76b3951ae2e4f6dd2f1989e11e4630df5`；sidecar SHA `24ca80245bd9208094f329bbcb545e280c5c686bacbec387845f23073b172f7d`。哈希与重载由主任务独立核验，本分析不加载模型。actor由626143c5…更新为98bce2e9…；更改参数不等于已经证明确定性任务进步。

8个rollout末尾均使用正常nonterminal bootstrap；最后update由第二回合末9样本＋第三段119样本组成，没有把普通阶段切换或人工停机当done。

复用前两份已核对报告，仅新解码第三段119行、其内必要相邻端点和封存更新/manifest。未扫描其他历史、未查看新自然P01运行、未改生产/奖励/阈值。详细绑定及逐回合数值见同名JSON。

