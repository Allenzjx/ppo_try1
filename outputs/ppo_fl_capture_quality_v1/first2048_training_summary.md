# 首个 2048 决策训练块：已完成更新核验

五块真实续训为 **512+896+384+128+128=2048 policy decisions**，从 **178432 / 1359 updates** 到 **180480 / 1375 updates**，新增 **16 PPO updates、320 optimizer steps**。决策区间连续；教师 prefix 和未优化尾部未计入。两个中途停止块均在 verified update boundary 封存。

| 实际请求阶段 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10–P13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 已优化样本 | 6 | 488 | 9 | 18 | 988 | 226 | 6 | 13 | 294 | 各0 |

P03–P06 共 **1241/2048=60.596%**。记录到39次普通阶段切换，均未设 terminal。六个已终止 episode 为2次 BODY_COLLISION、4次 FALL；非终态采样段不冒充完整成功。没有 P10–P13 学习覆盖。

## 前段质量子状态的真实覆盖

这是实际物理子状态，不是按请求阶段简单推算；来源为已封存 receipt 的逐物理质量记录聚合。

| 物理子状态 | 样本数 | 时长s | 实际β范围/s | 累计质量成本 |
| --- | ---: | ---: | ---: | ---: |
| P01 / 未资格准备 | 45 | .375 | .015–.030 | .00007900 |
| P02 / 未资格准备 | 20 | .1667 | .015 | .00014347 |
| P02 / 有资格抬升或转移 | 117 | .975 | .015 | .00158641 |
| P02 / 功能性携带 | 3768 | 31.400 | .015 | .04994610 |

成本确实非零并进入实际 reward；这不是稳定性已经改善的证据。

## block05 的128个样本直接核验

- 实际请求P01=2、P02=126。128/128决策的全部质量记录均β>0；1024/1024物理样本β在.015–.03。
- 逐物理质量成本合计 **+.012341244002**，reward中body_stability贡献 **−.012341244002**；逐决策负和误差最大1.09e−19。总实际reward合计+.3589901660，范围[−.0604266301,+.1171695888]。
- 保存rollout的reward、raw action、old μ/σ/logp与实际请求/审计全部零差异；float64分解到float32 reward误差最大3.63e−9。
- 首次真实optimizer minibatch为32个样本，ratio范围[.9999980330,1.0000019073]，偏离1最大1.97e−6，strict clipping=0。**这不是全部128样本的首次未更新权重ratio检查**；20个真实minibatch中128个样本各被使用5次，全部ratio有限，额外forward/RNG抽样均0。
- update1375记录有限非零梯度，KL=.028895、value loss=.093747、学习率1e−5。

## 最新 checkpoint

`outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000180480.pt`

SHA256 `ec48951dbc345d196c7ffa14fc6fa7aba7c33723292c11e8d500ef640ca2b05a` 与旁侧manifest一致；CPU读取后embedded infos、manifest和分支计数一致，actor/critic张量有限，manifest记录save/load round-trip通过。累计optimizer steps=27500；本分支新增2048/16/320。runtime为`f2e552406ea74a5aae2803347d1c8bebe261910d`，rho=.9、temperature=.25。

本摘要仅确认真实采样、更新和落盘完整性，不宣称完整越障成功或稳定性优于基线。正式固定checkpoint视频评估另行进行；本工作未读取其活动大日志、未改生产、未启动仿真或重算GAE。完整小型结构化汇总见同名JSON。
