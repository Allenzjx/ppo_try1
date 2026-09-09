# HISTORY optimizer 只读诊断

结论：本次未发现需在边界修复的 mean/std 接线或 rho 尺度错误。已见到持续参数更新与有限梯度；这不证明后腿任务已充分优化，也不能仅凭探索失败认定实现故障。未选择调参或物理修订。

## 固定证据范围

2026-09-08 03:03:02 UTC 单次读取以下两个 `optimizer_updates.jsonl`，未扫描轨迹、读取PT或启动Python/Torch/Isaac：

- Run7：`20260908T0230082471288Z_gdb991aa68103_b9b719b139e44396ab233876cbdbf87b`。
- Run8：`20260908T0242228047566Z_gdb991aa68103_ee0cd73bdf6a44138b554c8841f8b901`，仅当时已发布的4次更新；后续live样本不计入。

| 更新日志范围 | 实际新增dec / update / optimizer | KL均值（各update范围） | likelihood clip fraction均值 | 更新结束LR=1e-5 |
|---|---:|---:|---:|---:|
| Run7 up984–995，g130432–131840 | 1536 / 12 / 240 | .018315（.010251–.025315） | .219531 | 10/12 |
| Run8 up996–999，g131968–132352 | 512 / 4 / 80 | .018972（.014513–.022462） | .235938 | 4/4 |

16/16行均报告actor参数变化、有限非零梯度；两段内部actor哈希链各自连续。Run7另两次结束LR为6.75e-5及2.25e-5。这里只使用已存日志，不重新验证权重。日志没有每个mean/std头的参数差或独立梯度，不能宣称两个头每次都有效更新。

## 核函数与PPO接线

[semantic_history_actor.py](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_history_actor.py:24) 明确使用观测195:207的上一raw动作（schema clip ±20），不是投影残差或私有缓存：

`mu = 0.1 * base_mean + 0.9 * previous_raw; sigma = exp(log_sigma)`。

采样、确定性输出和训练重算使用同一个核；log-sigma保持原值，不乘0.1或平稳方差修正。sigma是**条件创新标准差**，不是跨时刻边际标准差；不保证最终物理目标平稳。当前372只追加列，未移动历史切片。

[semantic_env.py](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_env.py:163) 在执行后更新raw历史；[training.py](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:1044) 保存执行前观测产生的raw、条件mean/std与logprob，并逐项核对storage；官方PPO再用同一保存观测重算条件分布。没有把当前动作当成其自身采样前历史，也没有把投影动作代入likelihood。

固定保存的历史h时，`dmu/d(base_mean)=.1`；单通道score梯度为`.1*(a-mu)/sigma²`，log-sigma梯度为`((a-mu)/sigma)²-1`。这是所选核的真实非对称敏感度，不是断梯度；Adam、共享隐藏层和梯度裁剪使“mean一定学习慢10倍”并不成立。相同h下旧新mean差也自动含`.1`，官方KL无需再手乘rho补偿。失败率不足以判断该核是否适合任务。

## KL、LR、entropy的准确含义

本地官方RSL `algorithms/ppo.py:268–313` 对保存条件高斯做`KL(old || new)`，先求12通道之和再求minibatch均值。adaptive目标.01；**每个minibatch** KL>.02则LR÷1.5、下限1e-5，0<KL<.005则×1.5、上限1e-2。actor与critic共用该Adam LR。日志KL是20个minibatch的平均，而LR是update结束值；均值<.02与最后LR触底并不矛盾，也不能由结束LR证明每个内步均处下限。

日志clip fraction是`abs(exp(new_logprob-old_logprob)-1)>.2`的比例，不是物理动作饱和率，也不等于这些样本全部无梯度。记录的合并梯度norm可约1.414，因为官方对actor、critic**分别**clip至1，审计随后合并两者；不是max_grad_norm=1失效。

[现有调度](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:1095)为`beta=.005-.004*min(lifetime_global/210000,1)`，不随新实验/P06重置：本窗口Run7约.00251558→.00248876，Run8约.00248632→.00247901。它加在PPO目标`surrogate + value_loss - beta*H`，按policy样本/minibatch平均，不按120Hz dt积分，也不是环境reward family或PBRS奖励。

H是12维raw条件高斯的微分熵，负数合法。Run7范围−8.24359至−3.09781，Run8−8.74068至−4.72636；它不能恢复各轮sigma或比较不同物理状态的同态std趋势。熵对mean无直接梯度，对各log-sigma有向增大的直接梯度；与任务收尾存在潜在权衡，但不能从这些汇总值认定熵主导失败或应立即修改。

## 最重要的证据边界

当前仍是128步rollout、5×4优化，v3 gamma=.9985/lambda=.99。Run7 value loss从.03054到300.115，Run8从.31827到414.284，说明这些批次拟合误差差异明显；没有固定状态回放或分头梯度/均值/std统计，不能据此断言critic损坏、std冻结或mean方向错误。已有数据支持“更新在发生、学习受KL限制且样本组成不同”，不支持唯一失败根因、确定的调参修法或新增启动门禁。
