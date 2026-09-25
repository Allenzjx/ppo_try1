# tracking 修复后：首次随机机会的真实 RR 捕获

固定读取 `train_tracking_fixed512_1e10d39/decisions.jsonl` 初始 **7,315,514 bytes**，1,076 个完整行，无未完整尾行；没有追读 live writer。该快照 updates 文件为 0 bytes / 0 个新 update。

**41 条新的 PPO_credit=1 样本后，RR 于 tick8312 / 69.266667 s 实际达成 local capture hold。** 本次机会前有 998 条 PPO_credit=0 的 frozen-prior 前缀；随后已经自然进入第二回合前缀，快照到 tick296 / P02，另 37 条 credit0。没有诊断或 AUX 数据。

本条是**随机、headless 局部成功**，不是录像、确定性成功、全程越障或新 optimizer update 的收益。沿用已训练 local 权重；不声称均值零初始化，也不把历史累计计数加入这次新样本。

## 短窗口实际目标与响应

RR 数值顺序 hip/knee；raw 无量纲，N/目标/实测为度。REQUEST 来自真实执行审计，不由不同 mapper 状态重算。

|tick / 请求→结束phase|raw sample|source N / mapped N|generic RR bias|REQUEST|FINAL|actual|gap mm / N / hold s|
|---|---|---|---|---|---|---|---|
|8240 P09→P09|0.070/-0.057|-6.900/-37.800 / -8.150/-39.050|0.000/0.000|3.620/-0.520|-4.530/-39.570|-0.164/-37.453|7.704 / 0.000 / 0.000|
|8248 P09→P09|-0.040/0.094|-6.900/-37.800 / -8.150/-39.050|0.000/0.000|-0.380/3.384|-8.530/-35.666|-3.600/-37.196|2.784 / 0.000 / 0.000|
|8256 P09→P10|-0.506/0.352|-6.900/-37.800 / -8.150/-39.050|0.000/0.000|-4.380/7.384|-12.530/-31.666|-7.067/-35.255|-1.040 / 6.790 / 0.033|
|8264 P10→P11|-0.584/0.638|-6.900/-29.300 / -8.150/-32.100|0.000/0.750|-7.880/10.884|-16.030/-21.666|-11.041/-31.892|-1.063 / 14.211 / 0.100|
|8272 P11→P12|0.127/0.803|-6.900/-27.200 / -8.150/-27.200|0.000/0.000|-4.380/14.384|-12.530/-12.816|-13.152/-29.404|-1.097 / 13.271 / 0.167|
|8312 P12→P12|1.575/0.074|-6.900/-27.200 / -8.150/-27.200|0.000/0.000|15.120/4.540|6.970/-22.660|0.374/-38.275|-1.503 / 12.606 / 0.500|

首个 TOP 端点为 8256；各接触端点的 native-observer consecutive_TOP 计数一致反推出 **first TOP tick8252**；真实 placed 事件8253；终端8312累计61个TOP samples、hold0.5s、12.605516N、合法XY、非GROUND、current_attempt_capture_eligible=true。此次 `load_fraction_valid=true`，与旧 v1 成功的数据状态不同。

接触保持来自运行时 native observer 的计数/hold/当前传感器载荷；JSON 中 dispatch ticks8249–8312连续且 verified。没有独立重读每一物理 tick 的接触测量文件，故不把派发连续性冒充独立接触验证。

## 分工、概率与终止

- 首次 TOP 时 RR source 仍是 P09 [−6.9,−37.8]°、generic RR bias=0；这段实际 hip 负向、knee 展开与下降是在 sampled policy、已有 HISTORY/filter/物理限幅链下发生。不能从单次轨迹推定某单轴为必要或充分原因。
- 接触后 P10 source knee 到−29.3°、随后−27.2°。8264只有一次既有 generic knee +0.75°修正，其余40个端点 RR generic bias=0；不是 RR capture helper，亦不能把全部接触后展开归为网络。
- P09 late始终未消费。P12标签出现时 `local_RR_capture_holding_new_P12_start` 实际保持新RL lane pending，RL仍GROUND；必要P10动作没有被整组冻结。P09→P10、P10→P11、P11→P12均非终止、可bootstrap；只有真实hold成功才终止，最后bootstrap=false。
- 41/41 条原始 selected raw == issued raw；各条 `old_logp[0]` 精确等于 selected_raw_log_probability，HISTORY一次、sampling_draws一次、extra_random_draws为0。以已记录全12维mean/std/sample进行stdlib Gaussian密度复算，最大误差2.432e−6；没有拿 FINAL 当 policy sample。例：8256 old logp=12.247411，8264=12.569175，8312=15.832510。
- 后腿capture-assist ownership/correction均无，all12执行许可保留，无episode state write。尚无本run optimizer update，因此 minibatch saved-likelihood审计是N/A，不伪造storage/训练更新证明。

终端奖励事件 +40；含potential终止修正及时间项的最后reward为35.132。它表示RR局部捕获完成，不表示RL越障完成。主线可继续新512训练；本分析没有触碰运行、模型或生产代码。

全精度及固定字节证据见 `tracking_fixed_first_stochastic_capture_readonly.json`；计数快照见 `train_tracking_fixed512_first_contact_snapshot.json`。
