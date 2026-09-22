# Block07：RR回地信用与512预算边界

**结论：RR回地已产生负奖励，不是被历史placed完全遮蔽；但只损失现有捕获保持份额，且本块未收集任何真实episode终止。** P10–P13没有另一个直接惩罚机身后退/接触丢失的激活质量项。不能据此断言奖励实现错误，也不能把预算停止改判成失败。

## 确切函数与实际信用

`TaskStageSupervisor.physical_potential`（semantic_supervisor.py:1150）对已placed腿使用`.8+.2*retention`，再乘`.85/4`。`_current_capture_retention`（:1288）依据**当前**轮端平台XY范围和低于顶部容差的程度；RR回地并非保留满分：

- 6408：RR retention=1，RR全局Phi份额=.212500。
- 6808接地：retention=.186008，RR份额=.177905；这部分Phi损失乘reward权重5约−.172973。
- 9552：retention=.184040，RR份额=.177822，仍不是满分。历史事件对应的`.8`保留，避免抹掉真实越沿/放置历史。

该规则刻意不要求已放置腿永久接触：合法平台范围内、未向下穿出容差的临时AIR仍可retention=1。RR已placed后，前腿P06专用rolling-contact修正不再启用。`current_lift_valid`不是已placed腿这一分支的直接reward输入；它仍影响真实任务判定/动作建议。不能把保留合法AIR信用改读成给GROUND承载满分。

`_current_finish_progress`（:1198）的body区域、真实速度、TOP支撑及观察保持只在**四腿历史均placed**后取得`.15`的finish份额；本回合RL未placed，此项始终0。因此P10–P12机身后退没有独立的base-x负奖励，而通过RR/其他腿的当前几何、未完成RL的workspace/transfer等间接反映。总体Phi可被RL短期进展抵消：6784→6792 RR份额下降.000339，但RL份额增加.001847，当拍总reward仍+.001029，不能只看总reward判断“没有看见RR退步”。

`SemanticRewardCalculator.evaluate`（semantic_reward.py:436）实际为`5*(.9985*Phi_after-Phi_before) + terminal_event − .02*dt + active_quality_costs`。本配置front质量只P01/P02，几何质量到P09；故以下P10–P12窗口的body/contact/smoothness/regularization实际贡献均0，不把被记录但权重为0的diagnostic当作训练惩罚。

|实际保存窗口|决策数/时长s|Phi起→末|总reward和|potential项和|时间成本|terminal项|
|---|---:|---:|---:|---:|---:|---:|
|6408→6808，RR放置后退回地|50 /3.3333|.661198→.642794|−.416688|−.350021|−.066667|0|
|8528→9552，最后完整rollout|128 /8.5333|.630854→.618759|−.832859|−.662193|−.170667|0|

第一窗base真实后退103.451 mm；Phi来自相同evaluation经原函数离线复算，与日志逐点一致。没有增加评分或重放仿真。以上是含PPO更新的训练样本，不是固定策略评估。

## 512决策预算如何结束

`train_semantic`（semantic_training.py:1840）每个完整128步rollout调用现有`runner.alg.compute_returns(obs)`。本块4个rollout全部完成并更新到CP190464，但封存点9552/P12的`terminal=false`、`termination_reason=null`、`time_outs=false`、`terminal_bootstrap_allowed=true`。P12 age26.0667 s尚小于当拍真实任务上限30.6925 s，`completed_episodes.jsonl`为0行。因此**确实尚未采到这个物理回合的成功/失败/超时terminal成本**，不是丢失已发生的终止，也不是自动获得成功。

`_rollout_advantage_audit`（:1618）记录最后rollout nonterminal_env_indices=[0]。实际保存的`rollout_001453.pt`也验证128个done全0；尾reward=−.031879567、return=−28.808279、old value=−28.756937。安装版RSL `PPO.compute_returns`（ppo.py:187）使用`r+gamma*V(next)`作非终态尾目标，保存float32数值反推V(next)≈−28.819629；这是由已存return推导，不是另跑网络或直接记录的独立critic读数。此边界没有把Phi清零或加−40失败事件。

末rollout raw GAE有127/128为正、官方整rollout标准化后67正/61负；即刻负reward不等于advantage必负，这由critic基线与bootstrap共同决定，不能据此单独断言优化器奖励失败。若后来真实task terminal被采到，既有实现会done=true、Phi_after=0、失败−40且不bootstrap；本块尚无这类样本。

## 有界建议

先保留当前规则和已经继续的训练，后续用真实自然回合或足够后段采样取得任务终止，再核对回地前后advantage与terminal传播。新进程自然P01不是继续block07的物理状态，不能把后来另一回合的结局倒填本块。现有证据足以确认“保持损失可见但有限、终局反馈尚缺”，不足以选择新reward、永久TOP接触或固定支撑组合；不建议立即增加这些约束。

机器可读窗口见`block07_reward_credit_readonly.json`。仅源码/封存数据/CPU tensor读取及outputs写入；未改生产、配置或启动Isaac，不是训练门禁。
