# Exploration temperature 后备只读审查

结论：若新正式 C171264 再次呈现 stochastic 训练可完成 FR crossing/placement、deterministic mean 却在 P02 失败的差异，常数 `0 < tau < 1` 可作为**更靠近当前条件均值的采样覆盖候选**。这不是已证实的 sigma 根因，也不是控制链修复；未选定 tau，未实施、未启动试验。本审查不预判正在运行的 C171264 结果。

## 已有证据与边界

- 已封存 2048 块：P02 516 决策，4 个完整 stochastic episode 都有 FR crossing/placement、0 个 P02 terminal；P02 raw GAE 均值 -0.38302，395 负/121 正，完整 rollout 标准化后的 P02 advantage 均值 +0.21637。标准化符号不等于某通道因果。见同目录 `training_2048_reward_credit_summary.json`。
- 新四个自然 P01 ×256 块：1024 决策/8 updates，P02 730 决策；3 个完整 episode 均 P05 FALL，0 个 P02 terminal；4 个预算尾分别为非终止 P02/P05/P02/P02，保留 bootstrap。没有 teacher prefix。最新为 CP171264/1303 updates。证据仅复用已封存 `training_P01_four256_actual.json`，未扫描新活动 C。
- 因而不能说“P02 没样本”。可确认的不足是本批没有覆盖正式 C 中那种 P02 终止信用；短预算尾也不是成功或失败 episode。新 C 是否仍有该差异，等待其正式终态。

## 当前实现支持与缺口

1. `src/wlr50_clean/ppo/semantic_history_actor.py:28–62,67–115`：条件均值为 `(1-rho)*base_mean + rho*clipped_previous_raw`，rho=.9，HISTORY 来自现有 372 observation 的 195:207；std head 未缩放。构造器无 temperature。stochastic 分支更新 distribution 再 sample；deterministic 分支直接返回原均值，不更新 distribution cache。新增温度可只作用于 stochastic 分支的 log-std 输入，保留 mean 原表达式、权重形状、HISTORY 与原 deterministic 分支。
2. `src/wlr50_clean/ppo/semantic_policy_distribution.py:47–102,145–180`：当前 HISTORY v1 contract 明写 `conditional_std="unchanged_learned_sigma_as_innovation; no_stationary_rescaling"`，且 loader 精确核对完整 contract、actor/distribution 类型组合及重建的完整 runner_config。**不能只在现有 YAML/runner 配置加数值，然后沿用同一个 v1 contract。** 当前版本解析仅由既有 class/std_type 组合推导；如果新版本复用 actor 类，解析器必须明确且严格地区分新旧 contract，不能静默接受额外键。
3. `src/wlr50_clean/ppo/semantic_training.py:104–138` 固定构造 runner 配置；`461–482,518–541` 校验 policy version/contract、空 storage、完整 runner 配置后严格恢复 actor/critic/Adam/normalizer/RNG。当前没有温度开关或允许此变化的精确迁移因子。
4. 安装的 RSL `modules/distribution.py:239–297`：`init_std` 只用于初始化 std head；已加载 checkpoint 的 learned head 将覆盖它，故修改 `init_std=0.15` 不是已训练 actor 的温度机制。实际分布 `update` 使用 `Normal(mean, exp(log_std))`。同文件 `199–225` 的 std/params/log_prob/entropy/KL 都读取该实际缓存分布。
5. 安装的 RSL `algorithms/ppo.py:139–148,254–298`：采样时保存 action、old logprob、old mean/std；更新时再次以 `stochastic_output=True` 重建分布，再算已存 action 的 logprob、entropy、KL 和 ratio。因此只在 `sample()` 后乘噪声或只改数据收集端，会造成错误 likelihood；所有这些入口必须使用同一有效 std。

上述项目源位置相对于 `C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`。RSL 源相对于 `C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl`。未改安装库。

## 最小候选语义与迁移

候选核为 `mu_tau(o)=mu_v1(o)`、`sigma_eff(o)=tau*exp(log_sigma_learned(o))`；tau 为显式、有限、严格正的常量，无逐腿 mask，无缩小 tanh 后物理 caps/slew，无 rho 改动，无教师轨迹。tau=1 应保留旧路径；tau<1 只改变创新噪声和其完整概率分布。不要将 deterministic mean 动作伪装成 Gaussian 随机样本入 PPO。

需要新增明确的 policy/distribution contract 版本（或等价的显式版本化 contract 修订），记录 tau、sigma_eff 定义、适用全部 stochastic 调用、deterministic 不变和 source contract。把 exact tau 接通 actor、runner config、CLI/evaluation loader、checkpoint manifest 的同一严格配置链，不能只作为未落盘的训练包装器参数。输出审计宜分开 learned sigma 与 effective sigma，避免把缩放误报为网络学习。

可做参数拓扑不变的窄迁移：逐项保留 actor/critic 权重、Adam moments/step、实际 LR、identity normalizer/固定 observation schema、RNG、累计 decisions/updates；明确条件采样核改变、物理 MDP/reward/N/12 action 通道/caps/HISTORY rho 不变，迁移新增 updates=0；旧未完成 rollout 不继承，合法自然 P01 reset 后重新采集。保留 Adam 不代表下一次梯度或优化轨迹相同。

**不能直接复用旧 warm-start 路径**：`semantic_migration.py:752–809` 的旧 STATE_DEPENDENT→HISTORY 转换只支持旧源版本，且其通用结果写明 Adam `reset_all_moments`、LR=3e-5；`semantic_policy_distribution.py:304–324` 的旧 scalar→heteroscedastic 转换也建 fresh Adam。这两者违反本候选的保留 Adam 要求。应增加独立、窄边界且精确 source/target contract 校验的 shape-preserving 因子，不泛化已有 code/config 豁免。

## 必需 CPU 回归（设计，未执行）

- 固定保存再重载的 fixture actor、372 observation/HISTORY/normalizer，在 tau=1 与 tau<1 下 `torch.equal` 比较 deterministic 输出；同时确认输入、state_dict、normalizer、RNG 和既有 distribution cache 未被 deterministic forward 修改。冻结 mean 不变是同权重同 observation 的不变量，不意味着后续 PPO 更新后的 mean 不变，也不是物理成功证据。
- tau=1 与旧类路径固定 RNG 的 stochastic action/logprob/params 逐项一致；tau<1 固定 RNG 的创新标准化 `(action-mu)/sigma_eff` 与对照相符。既有投影/caps/mask/HISTORY 回写不变。
- 采样和更新重建分布都与解析 `Normal(mu,tau*sigma)` 的 logprob/entropy/KL 一致；更新前同权重同 stored observation 的 likelihood ratio≈1，storage 保存的是 effective std，而非未缩放 sigma。不要为了验证再 forward critic 或改变存储。
- 保存/迁移/重新加载保持参数及 Adam/normalizer/LR/RNG 值，累计计数不增，storage fresh；拒绝 tau 非有限/≤0/超出候选允许区间、missing tau、contract/config 不一致、未知旧版本及旧 rollout 混用。保持旧版本 checkpoint 可验证加载。
- 继续保留自然 phase switch 非 done、终止 bootstrap=0 等现有测试；无新增奖励或观察位。

## 风险与下一步选择

降低 tau 可能更频繁覆盖均值附近的失败，也可能减少原有成功随机路径，不能保证提高结果。固定 learned sigma 时 Gaussian entropy 总量改变 `12*log(tau)`；KL 对均值差的尺度也改变，现有 adaptive LR/clip 行为可能随之变化。learned std 可以在更新中部分补偿 tau，因此需要同时记录两种 std，不能仅宣称探索已缩小。HISTORY=.9 下闭环 observation 会变化，不能把单步温度解释为已知稳定状态方差。

先看正式 C171264；若同类差异仍在，温度是可比较的**采样修订**，不是已定位的 drive/load/control 缺陷修复。保持 frozen deterministic mean 的对照可验证迁移没有偷偷改控制，但新训练的作用仍需随后正式 P01 all12 deterministic 重载评估。此文件仅审查设计，不构成新的训练门禁，也不要求零残差 5/5。
