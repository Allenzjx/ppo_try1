# 72e P12+8 首回合窄诊断

**真实结果：INCOMPLETE_CONTROLLER_BLOCKED，非安全中止。** 回合在 tick 9832 / 81.9333 s 结束；来源 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。P12 elapsed=30.6667 s，当前有限额度=30.0+0.66007 s；safety evaluator termination=null，stall_diagnostic=false。没有调整时限或任何阈值。

452 个已采集 learner 决策（226561–227012）。读取时完整更新到1738/global226944：**384 已优化、68 未完成 update 尾段**，不提前计优化 credit。本报告只读这个已结束回合，不读第二个运行中的前缀。

## RL：继承一次、learner 新增两次，均未越沿

handoff=6216 / 51.8 s，真实 successful_nominal P12+8，无 fallback，前缀零 credit。

| 资格事件 tick | 来源 | 可见 qualified 端点 | 首次看到失资格 | 最高轮底相对台面 gap | 最近前缘距离 |
| --- | --- | ---: | ---: | ---: | ---: |
| 6212 | nominal 前缀 | 5 | 6264（端点 AIR；不否认期间有 ground 子步） | +15.08 mm | −78.34 mm |
| 6294 | learner suffix | 12 | 6392，GROUND | −11.41 mm | −120.12 mm |
| 7124 | learner suffix | 11 | 7216，GROUND | +51.66 mm | −50.92 mm |

总计 28 个 qualified 端点，其中 5 个继承前缀；RL cross、TOP bearing、placed、P13 均为0。第三次抬得更高也未到前缘，不能称越障。新事件是 learner 控制期间的物理证据，不是已证明的 PPO/reward 学习收益。

## RR/FL 支撑丢失的可证实顺序

| 端点 | 物理事实 |
| --- | --- |
| 6368 / 53.0667 s | RR 退出合法落脚 XY，front=−7.95 mm；但 sensor TOP 仍承载13.13 N，FL14.00 N。不能称RR已经AIR。 |
| 6448 / 53.7333 s | 首次可见 RR 与 FL 同为AIR、各0 N；FR TOP15.09 N，RL GROUND15.23 N。 |
| 7328 / 61.0667 s | 本回合最后一个RR合法TOP bearing端点，12.12 N，说明此前并非永久失去全部接触。 |
| 7432 / 61.9333 s | RR 首个GROUND端点，10.13 N，gap=−51.56 mm。 |
| 7624 / 63.5333 s | 最后一个FL合法TOP bearing端点，0.91 N。 |
| 9832 / 81.9333 s | RR GROUND、FL AIR gap75.0 mm、FR TOP11.81 N、RL GROUND13.34 N。 |

RR-ground 共295个已采集端点，其中227个已进入1736–1738完成更新；另68尚未优化。自7432首次ground后，没有新的RR qualified或sensor TOP恢复端点。本块终于覆盖了地面重捕获任务，但尚未完成恢复，不能把覆盖当能力。

## 方向与 wheel：分层记录，不能单因果归责

从首 learner 端点6224到终端9832，mass-weighted CoM世界Δxyz = **(−170.91, −71.36, −26.85) mm**；以首端点固定FR轴(0.84064,−0.54159,0)投影 **−105.02 mm**。body forward变化 **−139.35 mm**，RR前缘+44.90→−135.20 mm。初始短窗从6164开始、包含nominal前缀；不将该初始正投影归为learner造成。

四轮顺序 FL/FR/RL/RR，单位rad/s；直接使用同期source N、FINAL和实测，不独立重算nominal相减归因。

| 端点 | source N | FINAL | 实测 |
| --- | --- | --- | --- |
| 6368，RR先退离合法XY | −.3 / −.3 / −.3 / −.3 | −.271 / −.291 / −.433 / −.476 | −.099 / −.290 / −.432 / −.710 |
| 6448，RR/FL首次同AIR | −.3 / −.3 / −.3 / −.3 | −.614 / −.244 / −.295 / −.318 | −.613 / −.291 / −.243 / −.321 |
| 6480，首个记录source全零端点 | 0 / 0 / 0 / 0 | −.636 / +.159 / +.186 / −.014 | −.636 / +.087 / +.276 / −.014 |
| 9832，终端 | 0 / 0 / 0 / 0 | −.958 / +.130 / +.095 / −.082 | −.957 / +.238 / −.366 / −.469 |

早期掉支撑发生在源仍为有限反向段时，不能全部归因于stop后的policy。后段FL持续负向的FINAL/actual存在，但终端FL是AIR，不计为地面反向牵引。实际wheel与目标的差异还含接触负载/动力学，不等价于mask丢失。all12许可、后腿辅助OFF均保留。

下一处真实未完成任务：在RR/FL接续支撑下保持RL有效摆动并前送越沿，同时避免RR先退离台面；本回合后段还出现了RR落地后无法重捕获。这里不提出新阈值或隐藏控制干预。

