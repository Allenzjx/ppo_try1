# Gain10 有限 AUX 后备候选：未实施

仅依据已封存 update8/CP229376 输出及当前源码，只读分析；未读取活跃 continuous2048 数据，未导入 Torch/PXR、修改生产或执行 AUX。当前继续 PPO；只有新完整块及后续 DET 仍证实均值/闭环缺口时，才考虑本候选。若采用，父模型是届时实际最新兼容 checkpoint，不回退 CP229376。

## 现有事实与 identity 限制

Update8 的485个相同合格AIR输入，RR条件均值 hip −0.0171384、knee +0.00801537，说明真实PPO有有效输出变化。CP229376 DET仍无RR TOP：最低gap47.141mm、末gap51.160mm，实际hip约−4.37°、knee约+3.38°的变化尚不足；这不是下一2048块的结果。随机成功轨迹不等于确定性能力，也不能把旧11条15.7mm末段样本当作约59mm激活入口的闭环证明。

`semantic_rr_capture_local_aux.py:211` 拒绝非identity是必要防护：v1独立SGD直接拟合序列化RR末层W/b，而actor在gain10下先把其输出乘10，再应用一次HISTORY。原样移除guard并沿用学习率时，梯度坐标及输出坐标各带一个gain，同一输入/损失的一步有效均值变化约有 **g²=100** 的尺度效应，并非只放大10。v1还硬绑定旧source、41条首机会/11条固定窗口及配方，不能静默用于新source。

## 最小可审计改动（仅方案）

1. 新增显式gain-aware AUX配方/包版本，保留旧v1拒绝gain10。仍只允许RR mean末层两行、独立SGD、固定小LR和1…64实际步数上限；不得自动继承旧64步/.005配方。优先拟合**有效均值行增量** `ΔU/Δv`（零初始化），实际actor替换行使用 `W₀ + ΔU/g, b₀ + Δv/g`，而不是直接优化gain10坐标W。这样初始函数不发生乘除往返漂移，优化尺度不因gain暗增100倍。最后写回现有Parameter对象；不修改actor前向或σ。

2. loss继续在真实actor输出的条件raw均值上计算：`μ = .1*(prior + g*local) + .9*真实history`。teacher只能是兼容真实连续成功窗口的**实际issued raw**，不是FINAL、用户角度、`10×`反解的无界head标签或历史placed。保留完整448观测、全12实际动作/接触/投影归因；只监督RR不表示其余10动作可忽略。新数据必须封存、绑定实际collector checkpoint；多update块不能把run初始CP冒充所有后续rollout的采样CP。优化父CP另绑定最新完整state/hash。

3. 信任域仍检查固定真实参考集上的 `max |(μ_new−μ_parent)/σ_parent|`，累计相对父模型，不逐步重置、不用head/参数范数替代；参考应覆盖实际激活入口、AIR下降、TOP保持/掉载及inactive前缀，AIR与TOP分别报告。参考观测不是新增成功标签，有限集合约束也不是全状态安全保证。若没有更完整成功证据，可保留小末段示范，但必须明确仍不能证明入口闭环能力。触发裁剪/其他轮动作不兼容的片段不得直接当可执行标签。

4. **不再次迁移mean坐标或缩放原Adam矩**：当前gain10的原PPO Adam/m/v/step/LR、critic、prior、trunk、std、其他10行、RNG和Parameter身份/顺序均需不变；仅新增独立AUX事件/状态和实际步数，PPO credit=0。拒绝步/异常保留候选与receipt、实际步计账，不能默默归零。新源码需严格冷rebind并保留历史AUX64与coordinate receipt；当前坐标validator把receipt destination绑定当前HEAD，必须显式允许有据的后继源码绑定，不能重写旧迁移事件或绕过hash。现有视频对应validator同步严格接受该透明后继关系。

## 冷边界最小验证与验收

- 实际identity/gain10 actor：零增量前后raw均值、σ、logp、inactive前缀一致；一次HISTORY证据一致。等效有效行的小SGD步在两种参数坐标下函数变化一致（数值容差明确），不能凭参数梯度非零验收。
- 真实AIR/TOP参考上的loss和条件μ信任域正反例；超界/非有限/σ或其他10漂移必须拒绝、保留attempt与正确实际步数；接触后hip方向需求分开报告。
- 原PPO Adam全部state及LR/RNG/critic/冻结部分、参数身份、旧ledger哈希保持；只有RR两行及新AUX记录允许变化。错误gain、父CP、collector/source/包hash、旧v1配方混用一律拒绝。
- 唯一新checkpoint严格重载，保持gain10/AUX总账/源码与视频披露一致；空旧rollout、重新采集fresh PPO，再用同一学生自然P01 DET验证。拟合误差下降、离线信任域通过或后缀TOP均不自动称为物理成功，不启用后腿实时补全器。

来源：`RR_learning_signal_update8_gain10.json`、`train_gain10_fresh512_complete.json`、`CP229376_gain10_aux64_DET_result_readonly.md`；代码 `semantic_rr_capture_local_aux.py:74,108,207,252,283`、`semantic_rr_mean_coordinates.py:77,132`、`semantic_rr_capture_local_actor.py:277`。本说明不是实施授权、AUX结果或新数据标签。
