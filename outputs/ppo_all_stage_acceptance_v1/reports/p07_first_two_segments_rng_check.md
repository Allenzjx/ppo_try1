# P07 两段已落盘策略动作的有界 RNG 核对

读取范围：当前训练 run `20260909T0405165532990Z_ga8b148463115_14d21d83af52454f94eb0eabd4cc01ce` 的固定前 27 条策略 audit，global 139905–139931；未读取之后的行。第一回合第 14 条为 139918 / tick 6017 / FALL；第二回合只有前 13 条（至 tick 6016，nonterminal），不预判其终止结果。

- 对齐两回合前 13 个决策：raw 完全相同向量 0/13，156/156 通道值不同，最大绝对差 3.998119354248047（raw latent 单位，不是关节角）。
- mean/std 只有第一对向量完全相同；该第一对 12 个 raw 采样值全部不同。因此这些实际记录不支持重置后重播同一随机动作串的猜测。
- 源码核对：`SemanticRslAdapter.step` 在真实终止后调用 `core.reset(seed=self.seed)`；prefix/episode/backend reset 路径没有显式 global RNG 重设或 checkpoint RNG restore。`seed_training_rngs` 位于 CLI 启动，checkpoint RNG 恢复位于加载及保存 roundtrip 边界，不在所检查的逐 episode reset 路径。
- 未独立捕获 CUDA RNG 状态或检查 Kit 内部全部实现，因此这不是对第三方全部随机性的证明。没有从相同/接近的终止时刻推断单一原因；没有修改源码、种子、优化器或训练判据。

本核对不把 prefix 算入 PPO，也不构成成功或稳定性改善证据。
