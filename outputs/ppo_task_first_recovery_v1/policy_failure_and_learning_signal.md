# 旧策略故障与学习信号：一次集中诊断

范围：只读已封存的 CP172544→174592、2048 decisions / 16 updates。读取全部 16 个原始 rollout 张量和同批实际决策记录；未执行 Isaac、未优化旧 rollout、未改成功 N、物理或训练生产配置。本文件分析的仍是**旧目标**，不是新恢复 profile 的训练成绩。

## 结论先行

**确认错误：**本次未发现 HISTORY 错位、温度漏用、latent/projection 混淆、12 维概率聚合、loss 符号、normalizer 漂移或跨 reset GAE 的明确实现错误。不能把这一结论当作策略有效。

**合理但不利的设计／数据事实：**质量成本几乎吃掉了保持净空并前送的任务收益；128-step rollout 的信用经常依赖仍然很负的 critic；随机数据中的前段动作获得了混合信用，而没有直接覆盖本块确定性 P02 硬限失败。整体更新确实小幅减轻错误差速，却远未把均值移出原来的偏置区域。支持先采用独立 epsilon=0 任务恢复 profile、全新数据及保存后闭环评估，不支持继续仅凭梯度／步数宣布恢复。

**尚不确定：**哪个轮速／关节组合是 RR 跟踪退化的唯一动力学原因、改变质量目标能否单独恢复闭环、旧中间更新的逐样本概率轨迹。没有补造缺失的旧权重或旧 minibatch logp。

## 1. 原始数据核验，不再以“缺 advantage 日志”为由停下

数据：`runs/ppo_fsm_reference_p09_stable_v2/train/20260915T0633266376940Z_g4a58c0190ef7_d19a81607e6044a281a010de29469635`。

| 检查 | 实测结果 |
|---|---|
| 2048 个 observation / raw sample / stored μ、σ、logp / value、return、advantage | 均读取；372 输入有限，最大绝对值 20，符合 schema clip |
| 存储 HISTORY 对上一实际 raw（clip ±20） | 最大差 **0**；5 次自然 reset HISTORY 均为零 |
| 12 维 Normal(raw; stored μ,σ) 的 logp 重算 | 最大差 **3.815e−6**；不是 tanh、slew 后动作 |
| 完整 128 样本、sample-std 的优势标准化 | 最大差 **4.768e−7**；不是逐阶段／逐 minibatch 标准化 |
| 原始 GAE 内部递推，使用保存的旧值和 returns | 最大差 **5.722e−6**；4 个 terminal return=reward 最大差 0 |
| 精确对应冻结 actor：CP172544→1314、172672→1315、173824→1324 | 384 样本 ratio **[0.99999237, 1.00001240]**；CPU/GPU float32 微差范围 |
| 相同 actor 打乱样本后重新计算 HISTORY μ；train/eval 模式 | 最大差均 **0**；Identity normalizer，未重新拟合 |
| 官方 actor stochastic forward 的 likelihood 与独立同公式重算 | 差 **0**；半温度作用于 σ 和 likelihood，未改变 deterministic mean 定义 |

16 个 rollout 尾均未终止，保存的 return 含原 collector 的 bootstrap。可从 tail return/reward 代数反推 last value，但没有独立保存原 tail critic forward；本诊断明确将其标为“推导”，没有用当前 critic 替换旧值来声称验证。普通阶段切换不产生 done，4 个真实终止阻断后续 reset 的 GAE。有限任务超时是任务终止；普通采样块结束不是终止。

三块以外的逐 update actor 未全部保存；不将每个 episode 的动作归到最终 CP174592。episode 跨 update 是实际事实。

## 2. 有效前段的奖励和优势并不等于同一件事

以下“前送端点”仅用于描述：P02、FR 当前 AIR、top gap ≥15 mm、相邻同 episode 决策端点前距增加 >0.1 mm；不是新增门槛或每 tick 单调要求。“非前送端点”可包括必要等待／回撤，不能自动叫失败。

| 实际分组 | 样本 | task_progress 总和 | body 总和 | smoothness 总和 | raw GAE 均值 | 更新 advantage 均值 |
|---|---:|---:|---:|---:|---:|---:|
| 全部 P02 | 726 | +0.946250 | −0.166195 | −0.928082 | −0.236522 | +0.105092 |
| AIR、有净空、前送端点 | 432 | +0.686542 | −0.100134 | −0.544453 | −0.264708 | +0.071634 |
| AIR、有净空、非前送端点 | 247 | −0.883318 | −0.059724 | −0.314772 | −0.131592 | +0.246983 |
| P02 净空 <15 mm 端点 | 47 | +1.143025 | −0.006338 | −0.068858 | −0.528893 | −0.333062 |

432 个前送端点的其他 contact quality 合计 −0.000559；全部质量成本 −0.645146，几乎抵消 +0.686542 的任务项，净奖励仅 +0.041396。全部 P02 净奖励 −0.148942。这里 task_progress 含正确的 potential 与时间成本，不能把该局部总和当成完整任务回报。

5 个 FR 越沿事件当拍奖励均正，raw GAE 全负，标准化 advantage 为 **1 正 / 4 负**；5 个真实放置事件当拍奖励均正，标准化 advantage **5 正**。典型真实样本：

| decision / episode / update / rollout row | 物理事件 | reward | 旧 value | return | raw GAE | 更新 advantage |
|---|---|---:|---:|---:|---:|---:|
| 172695 / 0 / 1315 / 22 | FR 越沿（tick1202 在本区间） | +.037589 | −19.497614 | −26.523390 | −7.025776 | +.460330 |
| 172701 / 0 / 1315 / 28 | FR 放置（tick1254） | +.046775 | −21.620216 | −27.257748 | −5.637531 | +.763640 |
| 172917 / 1 / 1316 / 116 | FR 越沿（tick1006） | +.033233 | −20.566488 | −21.158865 | −.592377 | −.798348 |
| 172922 / 1 / 1316 / 121 | FR 放置（tick1046） | +.162491 | −21.866095 | −21.388727 | +.477367 | +1.265535 |

所有完整 episode 后来都在 P05 失败；第五条是未终止尾，不是第五个完整成功。credit 是相对旧 value 和 rollout 分布，不是动作的道德标签：不能要求所有前段有效动作都正优势，也不能据此将 FR 越沿后的任何采样一律加概率。不同状态、不同后果的分组均值不能证明“原地停滞优于前进”的同状态因果结论。

P02 的平均条件 wheel μ 为 `[-1.128393,-.475516,-.403789,+1.057855]`，平均实际 raw 为 `[-1.132207,-.474731,-.397311,+1.057638]`；五条随机轨迹通过前段时仍以该偏置附近采样，并非已经学出四轮名义前送。平均 effective σ 为 `[.101796,.041233,.083428,.034387]`。12 通道扰动、实际 raw HISTORY 与全身反馈让随机轨迹和确定性均值轨迹很快不同，不能将随机轨迹的 return 归给没有执行过的确定性动作。

## 3. 更新有方向，但没有足够改变故障均值

在**同一组 726 个已保存 P02 observations**上比较 CP172544 与 CP174592：

- base wheel mean 平均变化 `[+.237070,+.038945,+.016839,−.128385]`。
- 条件 wheel mean 平均变化 `[+.023707,+.003894,+.001684,−.012839]`，由于固定 HISTORY 核对基础 mean 乘 0.1。
- 这表示对原 FL 反向、FR/RL 抵消、RR 增强有一定减轻，不是“完全没更新”；相对原偏置仍很小。固定旧 HISTORY 的离线结果不代表新闭环 HISTORY 被固定，也不证明 rho 是故障根因。

唯一拥有紧邻完整 before/after CP 的首 update1314：126 个 P02 样本的 `sum(advantage * Δlogp)=+4.123735`，68/126 样本同向；432 分组在该块的84个有效前送端点中，49/84 同向，组和 +2.961397。这符合共享网络优化总体目标，不要求每个样本概率单调。其条件 wheel mean 平均变化仅 `[+.001811,−.000531,−.002553,−.002558]`，不构成真实 P02 恢复。

原始历史 update 只保存 aggregate KL/clip；没有逐 minibatch logp。JSON 中记录的 `post_block_same_obs_ratio` 是**最终 CP 的离线重算**，不是当时 optimizer 的 ratio/clip，也不能用累计16次更新后的 ratio 超范围宣称 PPO clipping 失效。关键样本明确保留这一字段未知。

## 4. 针对性修复建议已被证据限定

1. 不改未查出错误的 actor likelihood、HISTORY、τ=.5、mask、映射和 Identity normalizer。保住成功 N。
2. 新版本先 epsilon=0，消除上述实测质量竞争；新目标只消费新 rollout，旧 critic 需靠新数据校准，不预设旧值正确。是否充分由保存后自然 P01 确定性评估决定。
3. 本块有726 P02样本却没有P02 terminal；其中有效随机片段不等于确定性失败已获直接信用。新块追踪实际阶段和任务后果，不把“P01课程”理解成全部早期样本。
4. 若首块仍保留同类故障均值，按用户授权建立单一显式 mean-head 恢复候选，而不是无限重复相同小更新；本诊断没有自行重置任何权重。

## 5. 定向测试与新审计的无扰动证明

新增 `tests/unit/test_task_recovery_learning_signal.py`：**10 passed**。用独立新造小型合成 RSL rollout 验证：

- 开启新的逐 minibatch likelihood 记录后，actor、critic、Adam、storage、RNG 及 update summary 与未开启路径逐位相同；没有额外 actor forward / RNG。
- 20 minibatch 中每个样本恰好出现5次，索引由本样本 observation+raw 绑定；首 minibatch ratio≈1（后续 batch 已经经过优化，不能再要求全部为1）。
- 正负 advantage × 正负 sample 的4组局部梯度方向正确；clip 分支依赖 advantage 符号，不等于所有 band 外 ratio 都选 clip。

测试只有接线／优化数学证据，不算物理成功；没有拿旧成功样本进行 optimizer 更新。

## 复核文件

- `analyze_saved_learning_signal.py`：CPU-only 只读封存数据分析脚本。
- `saved_learning_signal_evidence.json`：2048 行带 episode/update 的紧凑关联，关键样本含完整372 observation、previous raw、基础／条件 μ、σ、sample、N／有效残差／最终目标、实测 native wheel velocity、接触／净空、带符号奖励、旧值／return／GAE／advantage及离线概率。
- `learning_signal_tests.xml`：10 项定向测试结果。

本文件不宣称 M1/M2/M3 已恢复。对应物理证据仍由新任务优先版本的保存后重载评估提供。

## 6. 新 epsilon=0 首个封存 update 的有界检查（独立于上述旧目标）

只读取新 run `20260916T0342024652780Z_gb0438f66ec63_a11da4badf6f4422b0af0decfc19053c` 的 `rollout_001330.pt`、`update_001330_likelihood.json`、CP174592／174720，以及第一条薄优势审计；**没有扫描仍活动的大决策 audit，没有重复优化旧数据**。实际 reward revision 为 `task_first_recovery_epsilon_zero_v1`，五族结构保留、四个次级质量权重为0。

该 update 的128条样本含 P01=2、P02=126，无 terminal，也尚未到 FR 捕获。P02 raw GAE 全正，均值 **+.480069**，标准化后 **65正／61负**、均值 **−.023834**；这是128条整体相对标准化，不是符号错误。其旧 critic 均值仍为 **−16.899504**，不能将新 raw GAE 全正当作 critic 已适应新目标或确定性已恢复。

新保存的真实 optimizer 调用记录补上了过去缺失的证据：

- 20 minibatch、640 次样本访问；每条样本实际访问5次，保存的 advantage 与 old logp 和 rollout **精确一致**。
- 对保存的当时 μ／σ／raw 重算12维 logp，最大差 **3.815e−6**；ratio重算最大差 **1.220e−7**。首 minibatch ratio **[.99999237,1.00000381]**。
- 严格 clip 分支 **102/640**；ratio 超带 **144/640**，二者没有混为同一个统计。
- P02 全更新后 `sum(advantage × Δlogp)=+10.540513`，**82/126** 样本同向；75条“结果 AIR、有净空、前送”的可见端点中 **48/75** 同向、组和 **+5.658533**。这些是新目标在实际被优化的证据，不是任务成功证据。

**首 update 并没有纠正错误轮均值。** CP174592→174720 在同一组 P02 observations 上，条件 wheel mean 平均变化为：

`FL −.00387985；FR −.00049985；RL −.00360057；RR +.00031506`。

方向上仍略增强此前的负向三轮／正向RR偏置，幅度很小。不能说“去掉水平成本立即让均值恢复”；也不能仅凭第一128条、没有捕获或失败后果的窗口断言整个恢复块无效。应等待已安排的固定训练块结束及正式重载闭环评估，再决定是否使用显式 mean-head 恢复候选。

与旧目标 update1314 的比较是不同 checkpoint、不同实际状态轨迹的描述性比较，不是同轨迹 reward-only 因果重放；没有把后来第2个 update 的随机 FR 捕获归给本 update。最后一拍的 next observation 未保存在本 snapshot，端点功能分组排除该拍，未补造真实状态。

新证据：`first_epsilon0_update_learning_signal.json`；可复核脚本：`analyze_first_task_update.py`。本次检查止于这一个封存 update，不扩大审计门禁。
