# Block09：真实 RR 捕获与接续的 PPO credit

**RR 真捕获本身得到正向 PPO credit；不是“放置奖励为正、实际却惩罚捕获动作”。后续接续的 credit 不一致，不能把捕获成功推广为整段动作都在被强化。没有发现普通阶段切换切断 GAE、奖励符号或实际 PPO 接线错误。**

范围：旧 runtime `336b7c56d2f0`、已封存 rollout **1648/1655**，只取第二回合 **215369–215404（36 步）**、第三回合 **216225–216288（64 步）**及各一个前端点。没有模型 forward、优化、物理重放或新学习信用。100 条 action/reward/old-logp/done 与对应 storage 逐项相符；每条在真实官方更新中出现 **5 次**，共 500 次 exposure。两次更新均已完成 20 Adam steps、参数 hash 改变，实际 LR 均 `1e-5`。

## 第三回合：捕获被强化，但不能混称保持成功

RR cross 的真实历史 tick=**5707**，placed=**5759**。表中 tick 是决策末端，不冒充逐物理 tick 事件时刻。

| 决策 / 末端 tick | 真实状态或交接 | reward | raw GAE | 实际标准化 A |
|---|---|---:|---:|---:|
| 216238 / 5712 | 首次 cross 所在步，AIR | +0.103642 | +2.086775 | +1.256485 |
| 216244 / 5760 | 真 TOP/placed，P09→P10 | +0.200658 | +2.205402 | +1.382467 |
| 216245 / 5768 | TOP，P10→P11 | +0.001046 | +1.347740 | +0.471634 |
| 216246 / 5776 | TOP，P11→P12 | +0.011523 | +1.097811 | +0.206210 |
| 216247 / 5784 | P12、TOP | −0.006433 | +1.022181 | +0.125891 |
| 216248 / 5792 | 仍 TOP | −0.025909 | +0.557930 | −0.367141 |
| 216253 / 5832 | 首个保存的 TOP 丢失端点；history placed 保留 | −0.030506 | +0.627163 | −0.293616 |

cross→capture 的 7 个样本全部 A>0（均值 **+1.307508**）；35 次实际 exposure 只有 2 次 surrogate clipped。捕获步自身 5 次均未 clipped，其最后一次已记录的联合 raw12 likelihood ratio=**1.059386**（更新中、该 minibatch optimizer step 之前，不是最终新策略重评）。这是直接的实际 PPO 使用证据。

P10/P11/早期 P12 当前 TOP 的 8 步，raw GAE 全正，但标准化后 **3 正/5 负**；均值 A=−0.171781。随后选取的 36 步为 **9 正/27 负**，A 均值 −0.522218。后段包含 AIR/回落和几何变化，不能称作稳定保持示范。

原因可定位到真实价值基线及整块归一化：rollout1655 raw GAE 均值 **0.903639**。例如 216247→216248，old V 从 −17.572781 升到 −17.121370，而 return 从 −16.550600 变到 −16.563440；raw GAE 降至 +0.557930，低于该块均值，所以 A 变负。**这不是把正 reward 硬改负，也不能据此认定 RR TOP 这个状态被专门惩罚**：A 针对完整采样动作及后续回报，全局 Phi 还含其他腿。

## 与第二回合 hover 对照

所选 cross 所在步及随后 35 个 hover 样本 raw GAE 全正，但实际 A **2 正/34 负**，均值 −0.618651；该块 raw 均值 0.848394，高于所选窗口均值 0.567431。所选 12 个下降转移的 A 全负（均值 −0.669530），说明“当前下降方向局部有利”不保证整个采样动作比 rollout 平均更好。复用既有 `block09_rollout8_RR_v2_readonly.md`：这里 v2 新增 Phi 差为零、加权质量成本为零；下降局部 capture shaping 存在，但不等同全局回报。未重复 evaluator 审计。

## 奖励与跨阶段 GAE 的实际接线

- 100 步质量 family 均为零，terminal event=0；每步时间成本 −0.00133333。总奖励精确为 `5*(.9985*Phi_after-Phi_before) − .02*dt`。捕获步 Phi **.633507757→.674918339**；首次保存失 TOP 步 **.6746875→.669857747**，全局进展损失有可见负 shaping（−0.029173），并非全被历史 placed 吞掉。当前 usability 仍 true 不表示当前承载。
- P09→P10→P11→P12 三个交接 **done=false**；两份完整 128 storage 都无 terminal。用已保存 reward/value/return 核对内部 GAE 递推，最大误差 **1.91e−6**；整块标准化最大误差 **2.39e−7**。捕获步 raw GAE 可分为当前 TD **+0.873142** 和后续 GAE 项 **+1.332261**，后续接续确实回传到捕获。
- 非终态 rollout 尾部使用官方 critic bootstrap，不是零回报；尾部后没有已观测的 GAE trace。尾观测/last value 未单独保存，不能把代数反推当独立测量。这两块不能证明后续完整任务的价值预测准确，但没有证据说 bootstrap 缺失或阶段归零。
- 正 A 是增加**联合 raw12 采样动作概率**的 surrogate 压力；负 A 相反。clip 实际是否生效以逐样本 `clipped_branch_strictly_active` 为准，不以汇总 `clip_fraction` 代替。共享网络、其他样本、entropy 和 Adam 可能改变净效果：例如 P10 正 A 样本最后已观察 ratio 仍为 **0.894597**。不能声称正 A 保证每个状态/关节 mean 正确移动。

结论边界：目前证据支持“真 capture 已收到正信用，保持/后续动作仍未一致学好”，以及部分有利局部动作在全局/归一化 credit 中被压低；**未定位到必须修复的 reward/GAE 实现错误**。未提出改权重、改优势标签或新增辅助训练；本报告不构成当前训练门槛，也不声称从 P01 完整成功。

[逐样本证据及两份 rollout/likelihood SHA256](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_p05_hip_only_continuation_v1/block09_RR_capture_credit_readonly.json)。CPU helper 已自然退出。
