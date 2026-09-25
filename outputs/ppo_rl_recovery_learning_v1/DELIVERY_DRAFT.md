# RL recovery learning — delivery draft

> 状态：**草稿，不是发布门禁或完成声明。** 截至 2026-09-24 最后完整更新为
> global `229632` / PPO `1759` / Adam `35180`；相对 CP225280 为
> `+4352 / +34 / +680`。65a 的 front-retention439 AUX 仍为独立
> `32 accepted / 32 attempted`，不是 PPO；其后 P07 learner suffix 已自然封存
> `+512 / +4 / +80`，没有新增 AUX。该训练后缀真实取得 RR qualification/cross/place，
> 随后又丢失当前 RR/FL 支撑，故不是稳定捕获或整机成功。natural-P01 1536 与 CP229120 正式
> DET 均已自然封存。最新 DET 仍是57.866667 s /868帧的 P05 前驱失败；不是
> RR/RL detail、成功或媒体错误。三片均已完整解码 QA。59e 的 collection512
> 边界已零更新发布并重载；首个真实512块仍在运行，未封存前不增加计数或视频结论。

## 当前可交付事实

- 最新**正式自然 P01 确定性视频**是 72e 的 CP229120：
  `video_review/CP229120_deterministic_rr_retention_72e_review/`。物理/显示时长
  57.866667 s、868帧，停在P05，未进入RR/RL窗口；终态
  `INCOMPLETE_CONTROLLER_BLOCKED` / `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。
  full、同回合前驱失败detail与历史N对照片均已完整decode/PTS/黑帧QA。这是任务未完成，
  不是 BODY_COLLISION、媒体失败或数值安全中止。CP226048/f6d保留为上一正式视频。
- 训练回合中的 `HARD_JOINT_LIMIT`、真实中央机身 `BODY_COLLISION`、以及 P12
  `LOCAL_BOUNDED_RECOVERY_EXHAUSTED` 是不同回合的实际终止。其中 BODY_COLLISION
  明确来自 f6d 训练 run
  `20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7`
  的第二个采样回合（45.375 s/P09），不是泛化推测。不能用其中一个替换 CP226048
  的正式视频结论，也不能把有限恢复耗尽写成碰边硬终止。
- 第一检查块重排为：P07 `768` + P10 `512` + P12 `1024` +
  natural-P01 `1536` = `3840`，现已全部在完整更新边界封存：
  `3840 / 30 PPO / 600 Adam / new AUX 0`。随后 65a/AUX32 后的独立 P07 学生后缀
  亦已封存 `512 / 4 PPO / 80 Adam / new AUX 0`；本轮合计到 CP229632 为
  `4352 / 34 / 680`。CP229120 DET 已封存并如实交付P05失败，不能借用后续权重改名。
- 封存的 natural-P01 训练尾段在 tick10211/85.0917 s 出现真实短暂 RR
  qualification，共11个决策端点；tick10301/85.8417 s 回到 GROUND，未 cross/place，
  也未形成持续或恢复后的 capture。102.4 s 尾点为 RR GROUND、FL AIR；FR/FL
  `placed` 只表示历史事件，不代表该窗口两条前腿都在当前承重。

## 1. 对照 N，当前缺少什么？

成功 N 的关键不是某个固定角度，而是连续、重叠且有实测支撑的机制：RR 已在合法
区域承载后，P09 late 同时启动 FL knee/wheel 与 RL 准备；P10 展开 RR knee；
P11 保持 FR knee 约 +30° 的接收构型并允许 FR 暂时 AIR；P12 继续 RL knee
正向变化、四轮短暂退让和明确 stop。该窗内 FL knee 实测约 −12°→−33°，FL
载荷约 5.93→14.45 N，RR 约 14 N；RL 随后卸载、cross、place。

当前 PPO 的缺口是**保持真实 RR/FL 支撑的同时完成前右侧转移，并把 RL 的短暂资格
变成越沿与捕获**。72e 的 P10 后缀已出现 RL 短资格与 RR 接触恢复样本，但仍是
0 次 RL cross/TOP/place；这些 P10 回合开头的 RR placed 来自零 credit nominal
前缀，不能记成 learner 学会了首次捕获。P12 第二回合的一段 learner 窗有 body forward
+35.54 mm、固定 FR 轴投影 +19.92 mm，但横向分量方向相反；这是前送/卸载尝试，
不是已完成向 FR 侧转移。最新 natural-P01 learner 又取得一次短 RR qualification，
但很快回到 GROUND且未越沿/放置；新增事实把缺口收窄为 sustain/recover→cross/capture，
而不是“完全没有 RR 抬升”。缺口也不能缩写成“把 FR knee 复制到 +30°”。

## 2. RL 接触以前是否真的导致终止或冻结？修了什么？

未发现“RL wheel/linkage 一碰边就直接 done”的通用硬门，因此不能声称删除了一个
不存在的终止条件。实际修复有两处：

1. `rear_dependency()` 在 edge-v4 下允许**同次、当前 qualified、非 GROUND**的
   RL 前缘/立面接触继续安全恢复；当前回到 GROUND 会撤销资格。真实机身碰障、
   wheel-only 过程判定和原安全边界仍保留。
2. RR 支撑丢失时，已经下发的 P09-late FL/RL 或 P12 RL 远端 servo owner 不再
   无条件追逐旧目标；它们使用公开 entry FINAL/request anchor 做有界 suspension /
   projection。显式 wheel stop 仍执行，独立 policy 恢复通道仍保留。

这是控制接口修复，不是 PPO 学会了恢复。CP226048 没有进入 RL edge，所以也没有
物理验证这条 edge 恢复路径。

## 3. FL 发力是否真的形成前右运动和 RL 卸载？

在成功 N 中有：FL 与 RR 均真实承载时，P09 late→P12 的 FL knee/wheel、FR
接收姿态和 RL knee 重叠；四轮退让开始时实测 body vxy 约
`(+0.180, −0.120) m/s`，负 y 为该 run 的 FR 侧，且 RL 随后从 GROUND 到
AIR/cross/place。但这只是多通道共同响应，不能单因果归给 FL。

当前 PPO 尚无同等级证据。CP226048 在首次 RR 抬升前失败，终端 FL 为 AIR，不能把
轮转记成地面反作用。后续 P12 learner 窗只证明有限前送与短暂减载，横向分量未朝
固定 FR 方向；natural-P01 learner 的 RR 短资格也在未 cross/place 前丢失，尾点
RR GROUND/FL AIR。因此当前答案是：**出现了前送、短暂抬升/卸载尝试，尚未证明
持续支撑或完成前右侧横向转移。**

## 4. RL knee 正向／短退是否改善碰边，RR 是否保住？

成功 N 的顺序支持该候选：RL knee 先正向变化；在 FL/RR 仍承载时四轮短退；随后
明确 stop、RL cross 与 capture。它证明应检查实际响应，不证明任何固定 knee 角或
固定倒退时长必然成功。

当前 72e 回合中 RL 有多次 learner-owned qualification，但都回到 GROUND，尚无
cross/place。P12 首回合 RR 先退出合法 XY，随后 RR/FL 同为 AIR；虽曾短暂恢复
RR TOP，最终 RR 落地且未重捕获。故当前不能声称 knee/退让已经改善碰边，也不能
声称 RR 支撑保持。下一次评估必须同时报告 RL clearance/contact、RR 当前 TOP/force、
FL 当前支撑和 source wheel stop，而非只报历史 placed。

## 5. 无后腿实时辅助的确定性 policy 到了哪里？各类贡献是什么？

- CP226048 确定性自然 P01：FR/FL 前段完成，停在 P09 首次 RR 抬升前。FL assist
  ON；后腿 capture/geometry/forced-forward task assist OFF；issued-owner
  suspension/projection ON。它不是完整 RR/RL，也没有 teacher 拼接。
- **控制/接口贡献（49eb→b0ff→f6d）**：新增可观察 owner439 状态、edge-v4 恢复许可、
  已下发 owner 的 suspension/projection、修正 FL range / RL space proxies；f6d 仅修复
  零 credit prefix 对显式 owner439 schema 的接收，不改变物理、reward 或控制。
- **PPO 贡献**：CP225280→CP226048 实际 `+768 decisions / +6 PPO / +120 Adam`。
  72e publication 零 credit 后先完成 `+3072 / +24 / +480`；再从独立 AUX32 候选
  完成 65a P07 学生后缀 `+512 / +4 / +80`，本轮截至 update1759 合计
  `+4352 / +34 / +680`。正式DET仍只覆盖 CP229120，结果是在P05前驱阶段失败；
  CP229632 尚无正式自然P01 DET，不能把训练后缀结果冒充确定性整回合表现。
- **reward-only 72e**：仅替换“RR 历史已 placed 后掉载恢复”的局部 retention 信号；
  actor/kernel/HISTORY/caps/physics/439 输入不变。它在 CP226048 的首次抬升失败窗应为
  零差异，不能声称修复了该视频。
- **AUX**：72e 正式训练与 CP229120 DET 时本轮新增仍为 `0`。随后 65a
  front-retention439 已独立完成 `32/32` 有限 AUX 并官方 save/reload；AUX发布当时
  PPO 计数仍为 `229120/1755/35100`，未部署 teacher，也未宣称物理成功。其最大 REQUEST 变化为
  joints `0.035654°`、wheels `0.000918925 rad/s`，KL `0.0122388`，
  `|Δlogσ|=0.0561200`。它必须作为独立 AUX ledger 展示，不能借旧 AUX 或 PPO 计数。
  其后真实 P07 学生后缀虽取得一次 RR place，但随后 RR/FL 当前支撑丢失；这是一条
  后续训练物理证据，不把 AUX 单独升级为成功归因。

## 6. 最新／最佳模型、训练量和下一项真实任务

- **最新完整更新候选**：CP229632，global229632 / PPO1759 / Adam35180，65a runtime，
  save/load round-trip=true；继承独立 AUX32/32，后续新增PPO `512/4/80`、new AUX `0`。
  P07后缀真实 RR placed 后又丢失当前支撑，因此不是任务成功；尚无该权重的正式自然P01 DET。
- **collection512 边界**：59e 从 CP229632 做零更新 publication，固定输入
  mean/sigma/value、完整状态与RNG重载一致；它不增加学习。首个真实块必须按
  `+512 decisions / +1 PPO / +20 Adam` 计数，当前仍为 PENDING，不能把历史128块重记。
- **最新正式DET模型**：CP229120，global229120 / PPO1755 / Adam35100，72e runtime；
  57.866667 s P05 incomplete。它不能代表 CP229632 的新权重。
- **上一正式DET**：CP226048/f6d，84.8 s P09 incomplete；保留原身份，不改名为72e。
- **当前可复现的 PPO 后段最好几何证据**仍是 CP225280 的 RR qualified/crossed AIR，
  但 RR 0 N、未取得 TOP 支撑；RL 已有前缘/准备上下文，却没有 qualified、cross、
  place 或完整完成，故不是任务成功。成功 N_ref 是机制参考，不是 PPO checkpoint
  或同版本净学习基线。
- 下一真实任务仍是取得能越过P05并进入后腿窗口的同版本真实评估。验收顺序是 RR 当前承载→
  前右侧转移→RL 有效卸载/碰边恢复→cross/place→P13/整机收尾。

## 实际生产变更与测试边界

`49eb→f6d` 的生产路径为该实验的 `curriculum_plan.json`,
`execution_profile.yaml`, `observation_schema.json`, `reward_config.yaml`,
`stage_task_spec.yaml`，加下列 runtime 组：

- 执行/证据：`actuator_target_effect.py`, `semantic_backend.py`,
  `semantic_residual_adapter.py`, `semantic_checkpoint_prefix.py`,
  `semantic_checkpoint_prefix_policy.py`；
- 状态/策略：`semantic_observation.py`, `semantic_policy_distribution.py`,
  `semantic_rear_owner_{actor,profile,recovery,migration}.py`；
- 任务/控制：`semantic_cooperative_preparation.py`, `semantic_rear_policy_timing.py`,
  `semantic_supervisor.py`, `semantic_reward.py`；
- 载入/训练：`semantic_cli.py`, `semantic_migration.py`,
  `semantic_rear_policy_timing_migration.py`, `semantic_training.py`。

`f6d→72e` 严格为六个 runtime 路径：`semantic_reward.py`、新增
`semantic_rr_retention_migration.py`，以及 `semantic_cli.py`,
`semantic_migration.py`, `semantic_rear_policy_timing_migration.py`,
`semantic_training.py` 的窄接线。未改 config、actor、分布、HISTORY、控制或物理。

定向 CPU 测试覆盖：edge/owner/contact/source-stop，rear-owner439 actor/Adam/迁移，
prefix439 接口，proxy/adapter/actuator 前段回归，以及 72e reward eligibility、终态、
迁移/CLI、full-state save/reload 与普通后继 receipt carry。它们验证接口和状态保持，
不是物理成功次数；真实结论仍以封存 episode 和正式 DET 为准。

## 后续边界

- **LATEST_DET_COMPLETE**：CP229120 source
  `20260924T1008401445075Z_g72e63592bdf4_3f1b5877a524484aab1eecca97694515/source`
  已封存；三视频、source/run hashes和decode/PTS/黑帧QA见唯一review目录receipt。
- **PENDING_FINAL_COMPARISON**：只比较同一封存模型真实结果；历史 N 继续标为参考，
  不据不同版本/不同时长失败片宣称稳定性优于 N/FSM。
