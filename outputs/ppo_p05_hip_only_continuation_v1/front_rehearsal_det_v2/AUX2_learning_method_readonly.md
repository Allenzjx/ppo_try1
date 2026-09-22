# AUX2 后续方法：只读、有界结论

本检查没有训练、更新权重、生成预算、启动物理实例或修改任何 hash-bound helper。输入为已保存的 `CP211968_actual_detP02_AUX_execution.json`、源 CP211968、当前生产 actor/训练代码，以及 block05/06 封存审计。当前确定性运行的 P01/P02 对比由 root 提供；其未封存结果不替代最终任务结论。

## 已证实的限制，不夸大原因

- AUX2 32/32 接受，停止原因是有限预算用完，不是 trust 拒绝。训练 raw half-MSE `0.0001800300 → 0.0001161097`（下降约35.5%）；验证 `0.0001806770 → 0.0001165937`。因此不能说它“没有学到历史动作”。
- 验证 REQUEST MAE：FL knee `.7266 → .5065 deg`、FR knee `.1636 → .0754 deg`、FL wheel `.02625 → .02140 rad/s`；部分后腿通道误差反而增大。12通道平均 raw MSE 是一拍回归目标，不保证所有物理相关通道都改善。
- 累计最大 REQUEST 变化：FL knee `.30346 deg`、FR knee `.10266 deg`、FL wheel `.005948 rad/s`；不是最终驱动、实际关节变化或物理效果。最大双向 KL `.33166/.34133`，最大 `|Δlogσ|=.14875`；尚未触发 `.5/.25` 界限，但也不能称分布变化可忽略。
- AUX 的目标是 `0.5*mean((μ-a_recorded)^2)`，没有 NLL/entropy 项，sigma 不是降低这项 loss 的捷径。其变化来自共享 trunk。末层 sigma/mean 权重 Frobenius 比为3.059；mean 另有真实 HISTORY 导数 `.1`，sigma 没有这个衰减。然而最大 old→new KL 的 mean 项为`.23505`、sigma 项`.10629`，不能按数值大小或权重范数就认定“sigma 主导学习/物理失败”。

## 为什么相同32步不应重复堆叠

仅改 `W0[:,P02]` 等于给该阶段所有状态同一组首层预激活偏移；状态依赖只能借助固定后续层及已有非线性，不能独立纠正每个动作通道，也会联动 mean/logsigma。与此同时 `.9` 的 history 来自旧记录状态：在这些固定输入上，小一拍误差下降并不修复当前闭环 history/接触/载荷分布。

Root 的同时间记录表明：旧 CP201728 与当前 AUXDET2 在 P01 decision2 的 FL knee raw 已为 `-.05793` 对 `-.11969`，FL wheel 为 `-.02988` 对 `-.09965`；P02 的 history 已接收该差异。P02-only AUX 证明“不改变当前 P01”，不证明“当前 P01 可行”。把目标改成 `(a_recorded-.9*h)/.1` 的 innovation MSE，在固定输入/权重下仅相当于原 loss 乘100，不是解除耦合的新学习方案。也不能把自生成 history 填入旧物理状态后称作真实闭环数据。

## 最小下一步（选此项，不增加实施门禁）

从实际已保存/重载的 CP211968_AUXDET2 继续 root 指定的普通 PPO 2048 decisions，保留现有算法、权重/Adam、真实事件及无教师的自然 P01。并行仅补旧 CP201728 **P01 decision2/input tick8** 的重建与源 actor 复核，之后才考虑它和既有 P02 数据的有限局部监督；本报告不建议立即第三次32步 AUX。

Data agent 已确认 tick8 的字段链可候选：decision1 的上一 raw/task endpoint；native8/7 的 REQUEST；capture8/7 的 applied/nominal；capture9 的 mapper pre-state 由 ACK8 绑定；physical7 提供有限差分状态。不需猜 reset0。但这只是字段可用性，仍需原 actor μ/σ及历史核验、当前语义准入。一个 P01 点没有独立同阶段 holdout，也不能证明首拍/全部入口的泛化。block03 两条 direct P01 是实际随机 raw，不能把其未执行 stored μ 当作历史确定性成功动作。

普通 PPO block05/06 均是 P01=14、P02=2034，后续阶段全0；再跑相同自然 P01 **不能承诺**摆脱 front-only。应按新块实际过沿/放置和阶段样本计数报告。若仍 front-only，不能把加总 PPO credit 或历史 suffix 结果称为后腿学习。仅改采样名称、增加相同片段步数、从已越沿快照开始，都没有解决入口。当前无无需新准入就保证后腿覆盖的课程捷径。

## 若之后确需改变 AUX 子空间（备选，不实施）

比继续共享首层偏移更直接的是：固定 trunk 及 logσ 输出行，仅允许现有末层 mean 行/偏置做有限历史 raw 回归；所有当前权重作为起点，不重置 head，不替换算法或部署教师，原 PPO Adam 保留、清空旧 rollout。它在相同输入上精确保留 sigma，并允许逐通道状态相关均值调整；代价是 **P01/P03+ mean 也可能变化**，现有 phase-only 的全状态严格不变证明不再成立，真实 holdout 只能约束所采状态。因此若严格 P03+ 不变仍是硬要求，这不是可直接执行的兼容方案；新增 phase-gated mean 容量则需要新 policy/schema、优化器新增参数迁移和独立 ledger，绝非“编辑现成兼容权重”。

仅更换现有参数子空间/loss 不需伪造环境 MDP 迁移，但必须新的 AUX 方案/预算/作用域记录及明确保护语义；不覆盖 event1/2，不冒计 PPO。上述备选不能绕过 P01 数据证据或当前真实物理评估，也无成功保证。
