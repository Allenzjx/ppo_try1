# 下一2048决策：最小课程建议与后续真实入口核验

保持当前训练 seed1001、最新兼容权重/Adam/Identity/RNG/LR1e−5，128决策一更新；如采用新REQUEST-history版本，先完成其显式迁移，不混旧未完成rollout。以下只是课程分配，不增加reward/caps/温度或发布门禁。

| 真实学习决策预算 | Stage | FromPhase | PrefixSource | TeacherOffsetDecisions | 用途 |
|---:|---|---|---|---:|---|
| 256 | full_episode | P01 | frozen_fsm（默认值，此处没有prefix） | 0 | 自然P01/P02质量与任务能力维护；其后的阶段继续学习 |
| 768 | phase_suffix | P04 | checkpoint_policy | 0 | 策略自己创造FL准备→摆动→捕获入口；保留真实P05→P06后果 |
| 768 | phase_suffix | P05 | checkpoint_policy | 150 | 已越沿但尚未触地、源动作末端附近的捕获学习；不直接从placed状态开始 |
| 128 | phase_suffix | P06 | successful_nominal | 0 | 明确标注N前缀的后腿及其准备维护；不冒充策略自行解决FL |
| 128 | phase_suffix | P10 | successful_nominal | 0 | 真实N前缀后的RL/收尾维护；N完成的RR不计PPO成功 |

合计2048（16个完整128 rollout），每块续用当时**实际最新**checkpoint。不保留旧策略专门制造固定入口；checkpoint prefix在每块开始时从已加载actor独立冻结，块内保持不变。依据实际P10–P13尚无样本，将维护预算拆开，合计5块，不新增自动调度器。上表参数均为现有CLI合法组合，使用`-SemanticVersion v3 -ExperimentId fl_capture_quality_v1 -NumEnvs 1 -Seed 1001 -Checkpoint <最新兼容CP> -CheckpointIntervalUpdates 1`；统一生产HEAD和迁移文件由实际版本决定，不复制旧HEAD到新命令。

## 为什么选这些入口

1. 首2048已优化实际覆盖：P01=6、P02=488、P03=9、P04=18、P05=988、P06=226、P07=6、P08=13、P09=294，P10–P13=0。P04课程确实能生成大量P05：两块实际896/384决策分别含P05=580/295，但也有225/25个P09尾部样本。下一块保留P04，并加入局部P05捕获以缩短大量准备重复；不能仅从课程名声称已有后腿完整覆盖。
2. 已封存CP178944的真正checkpoint-policy前缀曾自然P01→P04成功，195次prefix决策，tick1560交给学习；原始HISTORY和时钟连续，prefix零optimizer credit，无N替代。这证明接口可执行，不保证之后任意权重仍可达。
3. **当前CP180480确定性实跑能到P05源末端，但不能到P06。** P05首次tick1616；offset150对应tick2816（23.466667s），保存的同拍证据：FL crossed=true、placed=false、AIR、bearing=0N、gap26.973815mm、front92.015389mm。源末端推导tick2785，比入口早31ticks。这里是“近捕获但未接触”，不是成功快照，也不是从已placed状态训练。当前没有单独P05/offset150的prefix receipt，以上是真实完整评估的可达性证据，不能冒称该课程已经运行。
4. CP180480随机评估能FLplaced2995→P06，再于P09低高度FALL；这证明随机任务能力可出现，**不使当前确定性checkpoint prefix获得随机roll-in能力**。当前FrozenCheckpointPrefixPolicy明确`stochastic_output=False`，故不建议P06/checkpoint_policy作为已知可达课程。
5. P06/successful_nominal已实际接受过：335次真实前缀决策、tick2680交给学习，128个已优化样本中P06/P07/P08/P09=73/2/9/44。这是可复用的后腿维护能力，不证明P10–P13可达，也不把N完成的FL算作PPO capture学习。

## 覆盖目标与失败处理（不新增optimizer门禁）

- 目标：P03–P06仍占多数，约900–1200个实际P05请求作为方向性目标；P05需包括未触地approach、实际capture或失败后果，而不只统计request_phase。全身P05→P06及安全失败回报连续保留。数值只是采样目标，不是预报或完成凭证。
- 256自然P01保留真实P01/P02非零质量成本；前缀中的P01/P02不计入学习覆盖。后腿256只是最低维护预算，按已优化实际阶段和首个未完成任务报告，不声称覆盖了尚未到达的P10–P13。
- P05 offset150随新权重或新kernel可能提前完成P05、在前段失败，或进入更差状态。现实现遇到prefix miss会**仅一次fresh-P01 fallback**；实际run必须记录miss/credit_start并按真正P01计数，不能按请求标签计成P05。不得静默改用N造同名入口。
- 若第一完整update显示P05课程只在fallback P01采样，保留该实际128和checkpoint，在合法保存边界把**剩余预算**转为P04/checkpoint_policy/offset0；若只是P05更快成功导致offset晚了，可在下一块使用P05/checkpoint_policy/offset0，保留新的成功入口证据。无需重复可达性扫参，不杀当前episode或丢已有update。
- P05近末端仍约27mm gap，未必一次rollout就能接触；P06仍有强负FR knee均值、全身支撑变化及低高度风险。REQUEST-history候选仅改变cap扩大入口首拍中心，不能据课程或候选本身认定这些风险已修复。最终仍需新CP自然P01完整确定/随机评估，各自标清结果。

证据：`first2048_training_summary.md/json`、`p04_CP178944_first_prefix_audit.json`、`C_CP180480_P05_diagnosis.json`、确定性source决策352/tick2816、`CP180480_stochastic_handoff.json`、block04真实`prefix_evidence.jsonl`。本建议只读/输出目录工作；未启动仿真、未优化、未改生产。

## block07 已实现的 P05/offset150 入口（仅前缀与首个学习决策）

Run `20260918T0650463688089Z_g3a50657a96c9_630e9161d7094af18eb29dbbf6df35b9` 的已完成 prefix receipt：CP181248 确定性 checkpoint_policy 从自然 P01 实际运行354个前缀决策；P05 首见 decision204，offset150 后 **tick2832 / 23.6s** 交给 learner，`accepted=true`、`miss=null`、actual/requested phase 均P05、attempt_index=0，**没有 fallback，也没有以N替代策略前缀**。这是策略前缀初始化的后缀训练，不是自然P01完整学习episode。

首学习记录 global181249 为P05→P05，结束tick2840 / 23.666667s：FL crossed=true（event tick2676）、placed=false、AIR=true、support=false、bearing=0N，gap **21.098748mm**、front **61.452392mm**；总support_count=3。这些物理数值属于首拍结束tick2840，前缀outcome未单独保存tick2832的腿部几何，不混称精确入口测量。所查回执及首条记录没有明确的P05 `source_endpoint_issued` 字段，未把时间推导写成已记录端点证据；已确认不是从FL已接触/placed状态开始。

CP181248 provenance SHA `49617e25e4ec1bb4636e43f2790ccb81bfaa5f2ca91c50df2cfa2dc99e834027`；source/effective policy 均为正式 REQUEST-history quarter kernel，runtime 均 `e1e2a003f6711a77c463eb743282c250c67748dc1467e819ab90d9959e2d7ca4`。冻结prefix参数hash与源actor一致（`655f2370…6868b693`），独立storage与块内冻结均true，使用 `stochastic_output=False`。首学习观测中的previous raw与最后前缀raw全12维**误差0**；previous filtered REQUEST与最后前缀REQUEST最大差2.318e−7（float32编码尾差），时钟连续且P05 age=10s，未重置HISTORY。354条prefix决策均`policy_credit=false`，teacher/checkpoint-prefix data in PPO storage均false；首learner decision_count=1、physical_core_decision_count_including_prefix=355。以上不预支后续optimizer计数或捕获结果；仅只读前358条前缀证据及首条learner记录，未额外前向、优化或仿真。
