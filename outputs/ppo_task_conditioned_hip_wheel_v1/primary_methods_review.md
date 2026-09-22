# R1–R5：本轮采用边界

2026-09-20。完整读附件 `07768750`，复用本地方法报告，并核对下列一手论文及作者代码。只作方法选择审查；未复制外部源码、安装库、改生产、运行物理或记入训练步数。审查时本地 HEAD 为 `73c128d0f43c628a0d0c3bdb8e60c5f3414a6233`；以下本项目函数是现有连接点，不宣称新方案已实现。

## 标题、采用与不采用

| 来源核验 | 采用到当前工程 | 本轮不采用 |
| --- | --- | --- |
| **R1 PM-FSM**：[论文及 §III-C–F](https://arxiv.org/html/2109.12696v2)，与附件标题匹配。 | 接触事件、摆动与支撑任务显式可观测；现有 `TaskEvaluator` / `TaskStageSupervisor` 提供物理语义，`NominalMotionProvider` 提供结构，policy 闭环调节。 | 不替换 FSM、不复制论文角度到位守卫/频率参数。§III-E 的 upstairs/downstairs reflex 是手工模块；不引入“触地失败就自动下压”后称为 PPO 收益。 |
| **R2 PMTG**：[PMLR 2018](https://proceedings.mlr.press/v87/iscen18a.html) 与 [arXiv 2019 §3](https://arxiv.org/html/1910.02812v1) 同题同作者，年份不同不是错链。 | 保留已有运动结构及其可观测状态，学习任务兼容反馈；对应 `NominalMotionProvider.evaluate`、`history_conditioned_head`。 | 不引入 CPG，不强制周期步态，不把 Full12 全换低维幅值，不把 nominal 当唯一姿态标签。 |
| **R3 Smooth Exploration**：[PMLR 2022](https://proceedings.mlr.press/v164/raffin22a.html) / [arXiv 2005.05719](https://arxiv.org/abs/2005.05719)，标题匹配。 | 核查探索与 likelihood 一致、按状态分配幅度；沿现有 `physical_innovation_effective_log_std` / actor `forward` / `audited_history_policy_request` / `audited_ppo_update` 作最小版本化修改。 | 当前 Gaussian 状态 sigma 不等于 gSDE；不叠加 gSDE、HISTORY、有色噪声或迁移 SB3。减 sigma 不等于已改善确定性行为。 |
| **R4 Keep Rollin’**：[论文 §II-A、§IV-A / Table II](https://arxiv.org/html/1809.03557v2)，标题匹配。 | 轮驱滚动、真实接触、浮动机身和摆腿空间必须共同检查；对应 `_verified_p06_rolling_source`、`TaskStageSupervisor.physical_potential` 及真实几何/轮速证据。 | 不移植其分层 WBC/QP/ZMP 规划，不照搬“摆轮转速最小化”到所有阶段，不把四轮均速或轮速均值当净牵引；论文不保证本机负向 hip 一定抬高机身。 |
| **R5 DAPG**：[论文 §IV-C](https://arxiv.org/html/1709.10087v2)、[hand_dapg 作者入口](https://github.com/aravindr93/hand_dapg)、[mjrl 算法库](https://github.com/aravindr93/mjrl)，标题/仓库归属匹配。 | 仅作为方向已获实测、正常学习长期不吸收后的有限辅助候选：对真实历史/接触/gap 状态的相关条件均值监督；辅助数据、loss、步数单独记账。 | 不移植 NPG/DAPG，不导入手部任务数据，不把示范混为 PPO 行为采样，不监督所有其他通道为零，不部署 teacher。当前没有启用该辅助分支。 |

## 已核对作者函数与许可

- [SB3 `StateDependentNoiseDistribution`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py)：`sample_weights` 采样噪声矩阵；`proba_distribution` 由特征和标准差形成方差；`sample` / `get_noise` 生成动作；`log_prob` 使用同一分布，启用 tanh 时有变换修正。`DiagGaussianDistribution.log_prob_from_params` 同样配对采样与概率。[`OnPolicyAlgorithm.collect_rollouts`](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py) 负责噪声重采样及采样动作/old log-prob 入库，而不是把环境 clip 后动作替换为原 Gaussian 样本。代码为 [MIT](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE)；本轮只参考，未复制。
- [mjrl `DAPG.train_from_paths`](https://github.com/aravindr93/mjrl/blob/master/mjrl/algos/dapg.py) 显式构造带衰减权重的 demo 项，再作 NPG 更新；这不是标准 PPO buffer 用法。[`BC.mse_loss` / `mle_loss` / `fit`](https://github.com/aravindr93/mjrl/blob/master/mjrl/algos/behavior_cloning.py) 是独立监督目标和优化循环。两作者仓库均为 Apache-2.0（[mjrl](https://github.com/aravindr93/mjrl/blob/master/LICENSE)、[hand_dapg](https://github.com/aravindr93/hand_dapg/blob/master/LICENSE)）。不照搬 BC 的全通道 MSE、数据方差初始化或 normalizer 变更。公开 master 随时间变化；以上是本次访问内容，不是安装版本声明。

## 本地设计推论与最小检查

1. **状态 sigma**：从当拍可观测物理状态计算正的逐通道倍率，进入原始 Gaussian 的 log-sigma 后再 `distribution.update`。保留 12 维、动作容量和合法 stop；变更均不作用于 nominal mask。训练 old/current likelihood、熵/KL、checkpoint loader 和随机评估共用真实内核。固定 sigma 调整只改变双向概率，不能偷偷在 policy 外加负向偏置。
2. **概率坐标**：当前 buffer 存 raw Gaussian action，则 new/old 概率都在 raw 空间算；tanh、限速和物理投影属于后续执行映射，不给非可逆投影伪造 Gaussian 密度。若将来采用可逆满秩 wheel 变换，必须显式保留变换/概率坐标；目前没有证据要求同时改表示。
3. **HISTORY**：现有 `mu=(1-rho)*base+rho*observed_history`，直接均值导数含 `1-rho`；但 log-likelihood 梯度还含 `(raw-mu)/sigma²`。仅见 `rho=.9` 不足以判定梯度过弱或许可乘十。比较固定真实样本的 base、history、conditional mean、sigma、GAE/标准化 advantage 及逐通道更新，不能把联合概率上升当 hip 均值学会下放。
4. **任务空间 reward**：沿 `_current_capture_progress`、`_current_capture_retention`、`physical_potential`、`SemanticRewardCalculator.evaluate` 复用真实 gap/接触/推进分工，质量项按 dt 累积。几何接近不等于 placed，历史 placed 不等于当前承载；P06 接触保持不应永久压制后续 FL 开空间。只在够用净空以下计成本，不奖励无限抬高、无限下压、空转或固定角度。具体项/系数由真实数据决定，不宣称出自这些论文公式。
5. **若以后启用有限辅助**：只用真实可持续进展的有标签片段，保持原始历史和数据来源；在相关通道上有限监督条件响应，不将“25→20”文字造标签、不把负历史改成零历史。单列辅助步数/权重/退出预算，明确不是纯 PPO 梯度；辅助更新后重新采集 rollout。最终由同一网络自然 P01 运行，无 teacher/fallback；普通阶段仍非 done。

上述研究是实施依据，不是新门禁；不证明任何本机改进或任务成功。四项 hip/轮驱方向、时机、持续接触和质量收益仍由本轮真实运行决定。
