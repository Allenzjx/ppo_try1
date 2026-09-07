# P06 rolling retirement：首次真实运行的只读 native 核验

## 范围与结论

运行：`runs/ppo_semantic_v3/train/20260906T0545269489491Z_gcae1d6e7cfdd_1813c27a4d0e408aab8fde1f0c4010c6`，runtime `cae1d6e7cfdd8383fec67e8f919cc823f9e65448`，P01 N1 训练，由实际 checkpoint 17792 权重继续。

本报告固定在完整 optimizer update **110 / global 18560**：读取 global **17793–18560，768 个连续决策**，不统计后来未完成 rollout 的优化量。对应 6144 个实际 physics ticks，全数 native audit verified；各行均记录无 episode 内状态写入。运行在报告时仍继续，这不是最终训练/任务结果。

**修正已实际改变 P06 继承滚动的原生目标，不是仅改变诊断字段。** workspace 外权重保持 1；进入后连续降低，P09 抬升初段 FL/RL/RR 的 nominal 从 .3 降到 .0486123663 rad/s。P07 的 FR −.63 调整保持，后续 P09 自己的 .3 滚动也保持。没有将所有阶段的轮速或残差清零。

**不能据此声称 RR 修好或成功。** 此固定窗口无 task success；决策末记录中 RR qualified lift 为 true 的样本为 0。44 个决策末记录 RR initial clearance，但 initial 不等于 qualified，更不等于任务成功。当前证据未实际覆盖 qualified-lift carry override 的激活，不能拿后面的源 P09 rolling 冒充该分支。

## 实际覆盖及两个时钟边界

| 请求 phase | 完整决策数 |
|---|---:|
| P01 | 1 |
| P02 | 169 |
| P03 | 4 |
| P04 | 1 |
| P05 | 145 |
| P06 | 255 |
| P07 | 1 |
| P08 | 1 |
| P09 | 191 |

实际 transition events：P05→P06 为 tick2560 / 21.333333s，P06→P07 为 tick4600 / 38.333333s，P07→P08 为 tick4608 / 38.4s，P08→P09 为 tick4616 / 38.466667s。阶段改变并未停止或 reset episode。

诊断 `semantic_task.nominal_provider_diagnostics.p06_rolling_retirement` 明示 `current_nominal_suggestion_before_slew_not_applied_target`。它来自**决策末帧、新计算的名义建议**，不是最后已执行命令。449 行拥有真实 P06 层的诊断中：

- `source_observation_tick == applied_audit.physics_tick`，source time 与该末帧时间一致，零不匹配。
- 用显式 double 算术重算 `clip((min(xRL,xRR)+.22)/.005,0,1)`（横向无效则0），与记录的 `measured_fraction` 最大偏差为 0。
- `peak_fraction` 在本 episode 不降低且不小于当前 measured fraction；零违例。由于峰值可来自两个决策之间的真实物理 tick，不能要求它等于仅在 15Hz 边界采到的最大值。

`applied_audit.nominal_action_full12` 则由 `SemanticEpisodeEnv` 的 `samples[-1].nominal` 写出；其完整 `actuator_target_effect_audit` 对应**同一个最后已执行物理 tick**。该窗口 native command tick 为 episode completed tick +179（既有 settle 偏移）。每个决策还保留全部 8 个 tick 的 verified/effect 摘要，但不保存每个 tick 的全部目标向量；以下向量是确实记录的决策末已执行 tick，不伪称完整 120Hz target stream。

特别是 global18112：请求仍 P05、last-applied nominal wheels 为0，末帧已计算 P06 层、gain1。这是合法的下一建议时间语义，不能把它当作已执行 P06 rolling 的首行。

## 从外侧到实际淡出的数据

轮序为 FL, FR, RL, RR；logical nominal 用 rad/s。表中 gain 是上述末帧下一建议字段，nominal 是上一实际执行源，因此切换边界不要求 `.3 × gain` 逐位等于同一行 nominal。

| Global / tick / 时间s | 最后请求→末 phase | 诊断 gain | 实际 nominal wheels |
|---|---|---:|---|
| 18366 / 4592 / 38.266667 | P06→P06 | 1 | .3, .3, .3, .3 |
| 18367 / 4600 / 38.333333 | P06→P07 | .995516811 | .3, .3, .3, .3 |
| 18368 / 4608 / 38.4 | P07→P08 | .775986944 | .259448300 ×4 |
| 18369 / 4616 / 38.466667 | P08→P09 | .503152643 | .162629495 ×4 |
| 18370 / 4624 / 38.533333 | P09→P09 | .218987499 | .073787580 ×4 |
| 18371 / 4632 / 38.6 | P09→P09 | .162041221 | .048612366 ×4 |
| 18379 / 4696 / 39.133333 | P09→P09 | .162041221 | .048612366, −.63, .048612366, .048612366 |
| 18391 / 4792 / 39.933333 | P09→P09 | .162041221 | .048612366, 0, .048612366, .048612366 |

外侧样本18366的 RL/RR front distance 为 −.221064445/−.222965416m，仍完整 .3。18367两者为 −.219703343/−.219977584m，刚进 −.22m 边界，才开始退火；没有在外侧把滚动速度按距离误差压成趋零。

峰值抗回摆也实际生效：18371 当前 measured fraction 已回落至 .269664254，而 peak 保持 .837958779；18379 RR front distance 后摆到 −.260704395m、当前 measured fraction=0，gain仍 .162041221，没有重启旧 .3 滚动。

## 原生目标与其他 ownership 保留

以下 native 值取真实 float32 setter/dispatch audit，左侧负号来自冻结 side mapping，不代表 logical 反向行驶。counterfactual 是同 tick 去除当前 policy residual 的目标，**不是重跑一条 zero-policy 轨迹**。

- Global18371：nominal 四轮 .0486123663；counterfactual native 为 `[-.0486123674, .0486123674, -.0486123674, .0486123674]`；实际 native 为 `[-.0617596060, .0987994894, .0015865677, .1302597374]`。残差仍真实作用，nominal 退火没有屏蔽它。
- Global18379：nominal `[.0486123663, -.63, .0486123663, .0486123663]`；counterfactual native `[-.0486123674, -.6299999952, -.0486123674, .0486123674]`；实际 native `[-.0440978035, -.6289035082, -.1458470821, .0653584227]`。P07 的 FR owner 没被 P06 gain 再乘一次。
- RR nominal hip/knee 此时仍发到 **55.6/0°**（18379），之后发 **55.6/−37.8°**（18386）。同期 FL hip49.2°、RL hip31.2°建议持续，后续 FL回38.6°也实际发出；没有新增等待历史姿态的门。
- Global18399 / 40.466667s：P09自己的滚动源开始取得四轮 ownership，实际 nominal 为 `[.2486123663,.2,.2486123663,.2486123663]`；18400 / 40.533333s达到四轮 .3。对应 counterfactual native 已是冻结符号下的 ±.3，未被 P06 gain .162041221 压低。
- **gain首次到0的边界是18411 / tick4952 / 41.266667s**，但此时 P09 owner 已生效，实际 nominal仍四轮 .3，counterfactual native仍±.3000000119。不能将“P06层退休”误写为“实际四轮全部停止”。
- 最后固定边界18560 / 51.2s，有限 P09 源尾后的 nominal四轮为0，counterfactual native为0，但实际 native为 `[-.0743001550,.0715386420,-.0121926898,-.0110639483]`，说明残差继续工作。

## 与 C16768 的有限比较

C16768 的 P09 RR升髋55.6°时，nominal是 `[.3,-.63,.3,.3]`，native counterfactual对应 `[-.30000001,-.629999995,-.30000001,.30000001]`。本次相同源建议段记录为 `[.0486123663,-.63,.0486123663,.0486123663]`，实际原生值如上，证明所修的 P06 贡献确实改变，而 FR owner没有改变。

两次训练/评估的策略采样方式、持续更新的权重、自然轨迹和入口状态不同，本次是随机训练而非 C16768 的确定性评估。因此这里只确认**实现的实际目标效果**，不量化配对因果收益，不声称通过 RR 或 full-task，不把 optimizer 成功当 task success。合格-lift carry 分支只有先前 CPU回归的保持证据，本窗口没有其真实激活证据。

本报告只读取已完整写出的边界数据并新增此独占报告；没有 Python、Isaac、生产代码修改、提交或训练控制操作。
