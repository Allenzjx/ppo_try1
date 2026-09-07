# P06 161114 ep0：有限 nominal 尾段退出的真实适用性

固定来源：`runs/ppo_semantic_v3/train/20260906T1611147430138Z_g68cd9f5fca6c_846e7b65ab224b86a53905c55fa11f85`，只分析已实际终止的 ep0。没有等待整块训练，没有 Python/Isaac、生产修改或实施 `p06_nominal_tail_feedback_proposal.md`。

## 完整边界

- ep0 是 seed1001 的随机 PPO、teacher-initialized P06 suffix，不是天然P01固定mean评估。计入策略前的真实物理起点 tick3584 / 29.8666667s，actual/requested phase均P06；前缀不属于本回合 credited budget。
- ep0 credited global66945–67545，共601 decisions，600个完整8tick加最后1tick，共4801 credited physics ticks / 40.0083333s。
- 最终tick8385 / 全物理时间69.875s，P06 `INCOMPLETE_CONTROLLER_BLOCKED`，task/full-task success false，evaluator valid=true、physical failure=null；末下一状态phi=0、terminal bootstrap=false。tick8384的age=39.99999999999999尚未终止，下一tick达到40.0083333s才按原deadline结束；本报告使用实际终止记录，不将前一行近似40s当done。
- 审计时已经存在完整 update493 / global67584：本块已优化640 decisions、5 updates、100 optimizer steps。它覆盖ep0的601及ep1最初39；本报告不把ep1混入ep0，也不把其后未优化尾段或计划预算计为完成优化。

## P06 目标没有被接受完成；两腿最近点不是同刻

601个 credited decision末均为P06，无P06→P07交接；记录中rear_approach从未达到1，两后腿各自也没有在记录边界进入原workspace[−.22,+.06]m。最大rear_approach只有0.448608238。supervisor仅在8tick边界接受普通阶段完成，因此这里能确认没有发生接受完成的P06目标；没有完整120Hz几何序列，不能把“边界未到达”扩大成每个未记录中间tick都绝对未越界。

| 事件 | global / decision | tick / P06 age | RR front，m | RL front，m | rear_approach |
| --- | --- | --- | ---: | ---: | ---: |
| 首credited边界 | 66945 / 1 | 3592 / .066667s | −.494230 | −.509556 | 0 |
| RL最近、整体进度最大 | 67180 / 236 | 5472 / 15.733333s | −.257706728 | **−.357847940** | **.448608238** |
| RR最近 | 67235 / 291 | 5912 / 19.400000s | **−.241226759** | −.361421507 | .434313971 |
| 原finite时间附近 | 67327 / 383 | 6648 / 25.533333s | −.272597369 | −.360776004 | .436895985 |
| 首个全零nominal末端 | 67329 / 385 | 6664 / 25.666667s | −.278751057 | −.363947239 | .424211044 |
| 真终止 | 67545 / 601 | 8385 / 40.008333s | −.445090782 | −.557032839 | 0 |

RL最近时距下界仍137.848mm，RR最近时仍差21.227mm；两最近事件相隔440tick / 3.666667s，不能拼成一个更好的同时构型。两次最近点都出现在nominal四轮仍为+.3期间。之后已有回退发生在finite尾段退出之前；不能把所有回退归因于停止nominal。

## finite .3尾段退出时，实际目标与residual没有消失

以下顺序均 FL/FR/RL/RR；单位rad/s。command是ACK应用logical target，不冒充实测轮速。

| tick / global | nominal四轮 | projected residual四轮 | ACK实际command四轮 |
| --- | --- | --- | --- |
| 6648 / 67327 | [.3,.3,.3,.3] | [+.215026,−.342312,−.327332,−.068306] | [+.515026,−.042312,−.027332,+.231694] |
| 6656 / 67328 | [.125,.125,.125,.125] | [+.153467,−.222312,−.380008,−.188306] | [+.278467,−.097312,−.255008,−.063306] |
| 6664 / 67329 | [0,0,0,0] | [+.273467,−.293541,−.368985,−.095121] | 与residual相同 |
| 8385 / 67545 | [0,0,0,0] | [+.248342,−.247039,−.473788,−.050660] | 与residual相同 |

末端audit的zero-PPO native drive基准在以上各行分别与nominal一致、controller wheel bias=0。6664真实float32 native targets为 `[-.273466915,−.293540925,+.368985444,−.095120542]`，8385为 `[-.248341709,−.247038841,+.473788440,−.050660435]`；左侧native方向换号遵循原映射。这证明policy残差仍经实际dispatch作用，没有因nominal归零而被清零或mask。

g67329–67544共216个完整decision的**末端**nominal均全零；这些行residual没有一行全零。末端样本actual/residual均值均为 `[+.182984496,−.246680219,−.350991930,−.098900978]`。这个均值是216个末端采样平均，不是120Hz时间积分；首个零末端decision内仍可能包含最后的nominal slew。另有终止1tick单列，不混入等时均值。

这些具体值也说明，即使+.3 nominal仍在，策略已能抵消部分轮建议：6648的FR/RL实际command已经为负。不能假设延長+.3会产生固定前向body位移，更不能将wheel命令乘任意半径当真实平移。

## retirement peak：本prefix日志缺失，必须分清记录与推断

本次601条task audit都没有 `nominal_provider_diagnostics.p06_rolling_retirement`。具体来源是 `semantic_prefix.py:136–137` 的 `ResetOnlyPrefixController.task_snapshot` 返回 `supervisor.snapshot`，没有透传内部 `_semantic.task_snapshot` 中的provider诊断；后者在 `semantic_supervisor.py:975–978` 才装饰诊断。当前正常功能并非未创建P06层；这是日志透传分辨率缺口。本次没有修改它或增加新门禁。

因此不能照旧天然P01报告写成“本次实测peak0/gain1字段已逐行验证”。可以核验/推断的范围是：

1. 记录中原+.3一直保持至6648，随后按finite尾段时间与原slew退出，所有涉及的当前RL front都离retirement开始的−.22m甚远；这不是记录中已到workspace而触发的退出。
2. 生产P06层只做 `source_sample * (1-peak)`，peak单调不回退；没有其它阶段层在本ep0抢走wheel。退出前实际+.3与原+.3 source相同，支持当时没有产生可见retirement减幅。这是代码与实际target的交叉推断，不是已存储的peak读取。
3. source变零以后，`0*(1-peak)`始终为0，无法再从目标反推出peak。15Hz边界geometry不足以排除未记录中间tick的所有短暂事件；不得伪称本次整个后段120Hz peak精确已知。旧天然P01报告的peak字段不能借用到本次suffix。

源退出机制与已有proposal吻合：finite wheel-stop建议到时后变零，乘法retirement没有能力在source零尾继续提供rolling。`.3→.125→0`时序与原25.533333s尾段及限速相符，且任务尚未接受完成。不是P13新stop-progress模式要求P06停轮，也不是PPO通道关闭。

## 对既有 tail-feedback 提案的适用性

本次再提供一例真实“目标未完成时finite名义滚动结束，随后策略仍继续动作直至deadline”的数据；因此 `p06_nominal_tail_feedback_proposal.md` 的局部问题设定在本次仍适用。它只说明值得保留为单因素nominal候选，不证明延伸尾段就能完成P06、避免回退或提高稳定性。

限制尤其明确：两腿最佳进度本就早于尾段退出且未进workspace；RL lag显著，部分实际command在退出前已反向；本回合为固定teacher prefix起点的随机训练，不能当同一状态下old/new paired因果对照。延伸nominal可能改变支撑、墙接触、侧滑和BODY碰撞暴露，必须保持原hard安全/任务条件以及独立residual，不与reward/entropy/std改变混成一项。

结论仅基于已完成ep0，到此停止。未实施tail反馈、未修改production/config/tests、未启动第二Isaac或Python、未提交；后续运行由根按当前固定版本继续。
