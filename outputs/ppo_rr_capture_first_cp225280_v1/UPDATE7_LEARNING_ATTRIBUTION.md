# Update7：真实 PPO 已更新；确定性能力待本次评估

依据唯一已封存 `RR_learning_signal_update7_after_aux64.json`。本摘要未加载 `.pt`，未读取正在运行的 DET 视频，也未再次运行 Tensor 分析。

CP228352 AUX64 → CP228864 AUX64：新增 **512 个实际 PPO 样本 / 1 次 PPO / 20 次原 Adam step**；2992 个真实前缀决策全部 credit0。AUX维持64，本块新增AUX=0。当前本地总计3584决策、7 PPO、140 Adam；不能将冻结prior的225280历史决策计作本地新增训练。

| 真实机会 | 本块样本 | 实际结果 |
|---|---:|---|
| 1 | 423 | TOP端点1个，最大已存 native hold 0.008333s；最终未完成 |
| 2 | 46 | TOP端点9个，8344tick完成当前合格RR承载保持0.55s；仅局部成功 |
| 3 | 43 | 无TOP端点，末8320tick gap42.080mm；尚非episode终止，在完整rollout边界保存 |

第三机会不是新增失败，第二机会的采样成功也发生在update7 optimizer之前，不能归因为update7的收益。接触数字是决策端点与已存native-observer字段，不是独立120Hz完整复核。

实际入库阶段为P09=166、P10=339、P11=2、P12=5（合计512）。20个真实minibatch中每行曝光5次，合计2560次；对应阶段曝光830/1695/10/25。P12标签不等于新RL卸载已经执行。

## 能归因于这次 PPO 的变化

对相同的 **163个“当前合格、合法XY内AIR”输入**，CP前后条件Gaussian均值平均变化：

- RR hip **−0.003417623**，163/163均向负变化；平均条件均值0.604293→0.600876。
- RR knee **+0.003209849**，163/163均向正变化；平均条件均值−0.628633→−0.625424。
- 对应raw local head平均变化−0.034176194 / +0.032098460；不能把未经过HISTORY的head量与最终条件均值混用。这些数值均为无量纲raw Gaussian，不是关节度数。

方向与捕获候选相容，但幅度较小；相同输入比较不能证明实际落脚更好，更不能替代正在运行的确定性验证。

## 剩余学习限制

144/512（28.125%）样本末拍RR knee被headroom裁剪，这些FINAL均为−58°；平均requested−26.993°、effective−20.499°。这里裁剪的是执行请求，不能说144个实际关节都已精确达到−58°，也不能用提高噪声替代有效动作范围分析。

实测全12通道KL均值0.020460503；other10贡献 **95.2058%**，RR两通道约4.7942%。最终实际LR为 **1e−5**（本块入口1.5e−5）。这说明全局分布变化主要不在RR两通道，不证明改动KL机制就会解决任务；本摘要不提出新控制/reward更改。

Tensor审核报告显示obs/raw/old-logp/mean/std/reward/done与storage差异均0；Gaussian logp最大误差3.815e−6、重载旧策略同输入mean误差2.384e−7；HISTORY一次、采样一次、额外随机抽样0、全12实际raw未干预，冻结prior哈希前后一致。

成功机会46行整段的GAE未在该报告单独分组，**N/A，不据其他cohort推测其全正或全负**。现有总体/状态cohort GAE仅是保存returns−values及归一化检查，不是独立bootstrap递推证明或每个动作的因果标签。

最新checkpoint：`outputs/ppo_rr_capture_first_cp225280_v1/checkpoints/history/checkpoint_CP228864_local003584_aux000064_lineage448_v2_g0ff03eafeeb7.pt`（SHA `2996c9886b68ce716828fe7be84c7dcd28089ddb49f233699fa86bd3b18b8ee5`）。它具有有限RR AUX训练谱系，后腿实时辅助关闭；本摘要不提前声明其DET或完整越障结果。
