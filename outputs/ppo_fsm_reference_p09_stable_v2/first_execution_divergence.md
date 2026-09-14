# 成功 FSM 来源与首个执行分歧

## 2026-09-10 补充：P05 当前放置到策略边界的 wheel 建议缺口

后续在 HEAD `7db0d17f398d393ce026b6990bd2566d53366407` 确认一个局部软件缺口：P05 的 pending-capture helper 在真实 FL 放置的 120 Hz 观测当拍退出，但监督器到下一 15 Hz 边界才进入 P06。如果源 P05 停轮事件已经过去，旧零值会重新露出至多 7 个 physics ticks。N0 的 P2677→P06 2680 和 video6 的 P4604→P06 4608 提供对应事件与决策端点证据。端点中的 N 属于最后一个物理步的旧 source frame，不能误称为新 P06 首请求为空；实际 residual/mapper 没有整组清零。没有证据证明这几拍就是后续未完成的唯一原因。

已在完整更新边界正常停止并核验 checkpoint152192/1154/23080，随后提交 `d4e46006b382093b960f2c428081d8a1c595bb6a`。只让已在使用的滚动建议在当前 FL 顶面区域、真实接触/承载有效、既有安全条件成立时延续到正常交接；新源 wheel 事件、接触失效和边界到达仍可结束它。不改普通阶段 done、P09 功能性抬升、reward、动作范围、控制频率或物理参数。

该变更是明确 nominal MDP 边界，不是 instrumentation-only：保留网络/learned std/critic、372维 HISTORY、identity normalizer、RNG、计数和预算；只重置 Adam moments，保留源实际 LR=1e-5 与其他 Adam options，丢弃旧 rollout。相关 CPU 回归190、专门交接34、迁移113全部通过。实际新版本从自然 P01 续训，来源和加载凭据见 [迁移与实训记录](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p05_handoff_revision/actual_training_load_receipt.json)。

首个真实交接为 FL P1535→P06 1536，窗口 N 四轮保持+.3、非零 residual 继承、24/24 native ticks verified；但其放置早于旧源停止281拍，**只验证早期交接，不冒充晚捕获缺口的 live 覆盖**。后续 FL knee 实测−60.037767°越过−60°下限，虽然 target−48.202920°仍在界内，仍保留 HARD_JOINT_LIMIT。详见 [实训交接证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block15_first_p05_handoff_live_receipt.md)。晚捕获正反例已有 CPU 验证，不另设成功探针或 A5/5 门禁；后腿训练继续。

以下保留最初来源与执行链修复调查，2860901是其历史修复版本，不是当前训练 HEAD。

本报告基于用户指定源目录中的实际 Trial043、冻结发布 manifest、原始 Full12/observation telemetry，以及修改前 PPO HEAD `042e6e69302ebda7bfc83d09f30bf1800b157bc7`。修复已进入生产 HEAD `28609010db4e57c5b34304a4ae2563c69f9d00b9`。此处不是本轮 PPO 完整成功声明；真实运行结果由新 run manifest 单独给出。

## 1. 来源成立，但当前源仓库 remote 与用户线索不同

实际成功目录：`C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1/outputs/final`。

实际原始 run：`C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1/runs/trial_043_20260902_clean_v010`。

- `selected_trial_id=trial_043_20260902_clean_v010`、`v010_20260806_220745_363972_manual`、`RR_FIRST` 均从实际文件核对。
- 源目录当前为干净 `main`，HEAD `7d6bfda0da593e2cace2accd8bc81d300bdd9288`；其 origin 实际是 `https://github.com/Allenzjx/ppo_try1.git`，不是用户给出的 `fsm_base_on_recording` URL。不能宣称 remote 相同。
- 冻结发布记录的历史 commit 是 `b870ccd0bed8e458a0b39c6af5d16950929c4d43`。当前 git HEAD 不同不等于控制输入不同：实际比对的 23 个配置、FSM、sensing、adapter/mapper/infrastructure 文件，在源目录、PPO 目录与冻结 manifest 中均逐字节 SHA256 相同；不是仅换行归一化相同。
- 发布后的 selected trial、物理 evidence、reclassification、原始 trial manifest 和共享 robot USD 的本次只读哈希均与对应发布记录相同。哈希清单在同目录 `fsm_reference_and_runtime_identity.json`，未重复哈希视频或大轨迹。
- 原始 run manifest 保留 `INCOMPLETE_CONTROLLER_BLOCKED`，没有改写。发布的独立物理重审结论是 `TASK_SUCCESS_WITH_REFERENCE_DIVERGENCE_WARNING`：P01–P13 连续完成，物理越障、四腿主动抬升和无机身碰撞/纯轮违规成立；最大参考差异约 36.4586%，不再以相似度否决物理成功。
- 该 run 为 12,952 个控制 physics ticks，约 107.9333 秒，180 个 settle ticks。不能将发布的物理重审结论套用到另一条失败的 A 运行。

USD 只读依赖枚举显示一个本地 USD layer，没有外部 mesh/USD layer。独立 USD 解析器未解析 `OmniPBR.mdl`（材质），其真实 Kit 搜索路径留待新 live manifest 记录；这不是已证明物理资产不一致，也不能假装材质解析已核验。机器人 USD 本体哈希相同。

## 2. 数据一致，不代表旧 N 执行等价

旧 N 实际调用的是 `SemanticControllerAdapter` / `TaskStageSupervisor` / `NominalMotionProvider`，不是 `SensorFsmController` 生命周期。

成功源拥有：有限的 source Full12 请求及重叠、原始 `MotionExecutor` 时序、成熟 `ServoTargetMapper` 持续载荷误差补偿、FSM normal tuning 和有限 recovery。旧 N 虽然复用同一 motion contract 和 mapper 文件，却在其外面加了一层逐 tick logical nominal 限速，以及一个保留旧 nominal/旧 tracking 的交接 tick；同时没有向 MotionExecutor 传入源 normal correction/time scale，frame 中 normal/drive feedback 均写为零。

其中 FL hip 在 P06–P13 的额外 nominal slew 还是 60°/s；源 mapper 是 150°/s。二者不是同一个对象：逐 tick 修改逻辑请求会重置成熟 mapper 的目标到达/补偿状态。Residual 的 60°/s 限速与此不同，修复没有修改 residual 限速。

## 3. 已有成功观测的控制计算证明

用原始 telemetry 的 6,712 对连续 command/observation 做了离线控制计算，没有启动 Isaac、没有导入 Torch、没有重播 Recording：

- PPO 目录内冻结 `SensorFsmController` 对源观测重算的 nominal 最大差值为 **0**；tracking 名单不一致为 **0**。
- 同目录成熟 mapper 的 native servo target 重算最大差值仅 `5.684341886080802e-14°`，属于浮点往返误差。
- 实际 Python、sys.path、所有关键类的 `__module__` / `__file__` 在 JSON 中记录，均来自当前 PPO worktree，未从同名旧安装包导入。

然后在同一源 measured observation 和同一重建 mapper prestate 上，对旧 N 的单一新 owner 交接做受控比较。这个比较刻意对齐 phase 以隔离包装层，不声称是完整 semantic supervisor replay，也不要求不同 live 仿真逐浮点复现接触。

| 首个比较 tick | 成功源请求/状态 | 旧 N 请求/状态 | 实际计算出的 native 差异 |
|---|---|---|---|
| P06，3576，29.8 s | 四轮立即 +0.3 rad/s，tracking 为空 | 交接 tick 仍四轮 0，仍保留 FL hip tracking | FL hip 21.55° 变为 20.3°；四轮 0.3 变为 0 |
| P07，6648，55.4 s | FL hip 请求 37.6°，tracking 立即启用 | 请求仍 22.8°，tracking 仍为空 | FL hip 22.8° 变为 21.55° |

P06 这是“保持上一拍旧值”，不是代码显式给所有 wheel 清零；但在该真实 prestate 中，结果确实延后了源 +0.3 的派发。P07 后续额外 0.5°/tick nominal ramp 又进一步改变了源 held-request 语义。

离线 probe 的 placement history 现使用实际源 `leg_top_loaded_latched` 字段，修正后重新计算与保存结果完全相同。尤其源 FL TOP latch 在 tick 3583，晚于源 P06 开始的 tick 3576；不能把“P06 phase 已开始”当成 FL 此时已经承载。这里的 phase 对齐是控制隔离条件，不是新的 entry 接受规则。

## 4. 不把未触发的历史反馈当成成功原因

对实际源 P06–P10 Full12 ledger 的一次限定统计：P06 3,072 ticks、P07 208、P08 56、P09 872、P10 26。

- P09 的全部 872 ticks 中，实际非零 drive-feedback 请求数为 **0**。旧 RR knee 精确探针/回弹反馈在源代码中存在，但这条成功 run 未使用它；不能据此声称缺失此反馈导致当前失败。因此没有把它或其精确角度/速度/相似度判据接回 N。
- P10 在 tick 7794（64.95 s）起，确实有 **15 ticks** 的 RR knee `+0.75°` post-mapper normal tuning，首拍 native `-29.05°`、最终 `-28.30°`。源 state 的 `normal_time_scale=0.875`；N 先前两项都丢失。
- 源 P03 另有 RL wheel 的 logical correction fraction `-0.149`；旧 N 同样未传入。此次使用原 StateSpec 的值和 MotionExecutor 算法，而非另写一套常数序列。
- 成功 run 的 P06–P10 没有 recovery。只观察到 P13 在 89.1333 s 进入 recovery；不能宣称当前 N 已逐字复用了源 P13 recovery。

## 5. 已完成的最小生产修复

新 opt-in 为 top-level `reference_nominal_semantics: successful_fsm_derived_v2`。没有此字段的历史配置保留旧行为。

1. `NominalMotionProvider` 接收同一冻结 FsmSpec；统一 `_start_source_motion()` 为源 owner 传入 normal logical corrections 和 normal time scale。恢复 P03 logical tuning、P10 的源时序。
2. `_source_normal_bias()` 使用源公式 `.5 * phase.delta * state.normal_correction_fractions`，恢复 P10 15-tick（含 endpoint）post-mapper tuning；finite tail 不永久保持该 bias。它是 nominal 建议，不是 residual 15% 上限。
3. 新 N 直接发出当前 source owner 的 held logical requests；去掉额外 logical pre-slew 和空白 handoff tick。物理 servo slew/追踪仍由同一成熟 mapper 执行，residual slew/最终执行器硬限不变。
4. 按实际 source owner 重建 tracking：结束或捕获的旧 owner 不再因为 phase 变动多吃一次反馈；真正尚未结束的 predecessor 保留 target 与 tracking，并与新 owner 同 tick 合作。未被新 owner 改变的通道保留，不将整条 Full12 恢复为旧 phase.start。
5. `SemanticControllerAdapter` 将同一 N 的 normal tuning 传给既有 post-mapper seam；zero 与 nonzero 不分叉到不同 controller。
6. `semantic_prefix.py` 修正一个下游消费者：READY 后不再用已归零的 teacher taper 覆盖 N 的 source normal bias；TAKEOVER 期间只保留旧 teacher taper，不与新 bias 相加。精确 receipt handoff tick 不变，TAKEOVER 仍无策略信用。

有意保留的差异：任务语义交接、当前 capture/continuous ownership、全通道 residual、受控几何建议和独立物理验收；不用 A 的精确姿态、回弹、15%/30% 入口/成功门。因此新 N 应称为“成功 FSM 派生的最小修改版本”，不是原 A 的完整逐位复现。

P13 当前仍是既有 continuous nominal + PPO + 物理受控收尾语义；没有恢复源基于旧 reference completion 的重试流程。P13 后续实际可达性/恢复效果尚需新真实运行，不能把该差异藏在“完整 FSM 已复现”的表述中，也不将其设置为当前后腿训练门禁。

## 6. 验证与边界

`test_successful_fsm_derived_nominal.py`、全部 `test_all_stage_nominal_transitions.py`、全部 `test_semantic_prefix.py` 合计 **112/112** 通过，0 errors/failures/skips，19.008 s。第一次发现一个旧 v3 测试仍将已存在的 372 维 role schema 写成 324，只修正该测试的两个断言，保留 v2 的 324 测试与生产维度不变。

覆盖源 request/native-target 受控正反例、P07 held request/单 mapper、未完成前驱并发、完成 owner tracking 退役、P03 correction、P10 时序/bias 有限窗口、N 输入等价及 prefix READY/TAKEOVER 无双重 bias。

这些是控制和接口证据，不是 live P09 放置或自然 P01 完整成功。P09 功能性抬升/edge/carry 的当前规范、live N smoke、续训决策与更新、模型重载及完整评估由 root 的独立新实验 manifest 汇总；此报告不增加任何 policy credit，不命名成功 checkpoint 或成功视频。
