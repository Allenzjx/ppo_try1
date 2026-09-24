# 首个真实已完成 PPO 更新：只读日志闭环

审计时刻 2026-09-23 19:15:41 UTC。只检查自然P01 run `20260923T1908118345669Z_gfa4b98ed506e_1e77c2dd353745ebb86708c5095a34c9` 的**首个已完成128决策更新**。训练仍运行；此报告不是最新总量，不把后续采样或预更新GAE记录计成已优化。没有导入项目模型、Torch/CUDA/Isaac，也没有改生产、测试或原始日志。

## 计数与实际覆盖

| 项 | 初始已发布兼容分支 | 本次封存更新 |
|---|---:|---:|
| global policy decisions |220544|220672（+128）|
| PPO updates |1688|1689（+1）|
| optimizer steps |33760|33780（+20）|
| auxiliary updates新增 |0|0|

已保存 `checkpoints/history/checkpoint_step_000220672.pt` 对应manifest的计数和actor hash与optimizer日志一致。实际LR为1e-5；actor hash从09cb1ce…变为9b743204…；finite_nonzero_gradient=true。KL均值0.0184048，clip fraction0.2890625，数值有限。

实际request-phase分布是 **P01=2、P02=126；P07–P13全部0**。因此可确认真实前段PPO更新，不能声称本块已经训练或验证RR捕获、向FR转移或RL越障。也不能凭这个早期更新宣称稳定性已提高。

## 后腿控制权与高斯审计

- 全128条实际policy request均为 `rear_task_assists_enabled=false`，RR assist14维为WAIT全0；一次原始Gaussian采样，无额外forward或随机抽样。raw/old mean/old sigma/old logp与当次request记录逐值相同。
- 128个decision末拍的native target audit全部verified、setter/dispatch/映射一致，mask全12开放；没有RR补全owner或RR wheel投影证据。FL assist的owner位没有占用6/7。RR hip/knee两通道每条均有同拍native counterfactual差异；这证明这些前段决策中残差到实际下发目标的作用，不是后腿捕获物理成功，也不是把每一个中间物理拍都重新独立审计。
- 对保存的原始12维raw，直接以JSON mean/sigma重算Normal logp：128个old最大误差2.50e-6；20个实际优化minibatch共640个样本访问，每个样本恰好5次，current最大误差2.95e-6。minibatch old logp与采样记录完全相同；`exp(current−old)`与记录ratio最大误差1.62e-7，符合float32记录与float64重算误差。没有用applied target替代原始raw计算PPO概率。

## 跨阶段与尾部

220546决策实际P01→P02，`terminal=false`、bootstrap许可保留；全128样本terminal数量0，teacher-prefix samples=false。尾部env0为nonterminal，记录使用既有官方compute_returns的last_values。日志没有另存该tail value，审计不补造值或伪称重新计算GAE；本报告只确认实际非terminal路径与保存的preupdate审计相容。

机器可读明细见 `real_block_audit.json`，只读复核脚本 `audit_real_block_readonly.py` 只输出stdout。结论：**首个已完成更新的采样—执行—优化—保存证据通过；后腿任务阶段尚无覆盖。**
