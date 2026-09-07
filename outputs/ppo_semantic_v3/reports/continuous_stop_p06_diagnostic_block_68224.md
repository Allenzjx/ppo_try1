# P06 诊断短块最终账本：在 verified update 边界停止至 68224

运行 `20260906T1611147430138Z_g68cd9f5fca6c_846e7b65ab224b86a53905c55fa11f85`，HEAD `68cd9f5fca6c9ba06c38666fb55024b9ba04f9b0`，source66944 / PPO488 / optimizer9760，P06 offset0 / N1 / seed1001 / NewMdpWarmStart。

最终 `run_manifest.json` 与 result lifecycle 均为 **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**，不是完成原 4096 决策计划。主线程通过内置 `stop_after_update.request.json` 请求完整更新后交接；原运行路径未中途换版本，也不是按任务成功与否丢弃 rollout。

## 1. 实际完成与剩余预算

| 字段 | 实际值 |
|---|---:|
| 原启动计划 / planned_requested_policy_decisions | **4096** |
| actual_policy_decisions | **1280** |
| 最终结果 requested_policy_decisions | 1280（停止后实际阶段量；不是覆盖原计划证据） |
| unconsumed_requested_policy_decisions | **2816** |
| rounding_overrun | 0 |
| 本块 PPO updates / optimizer steps | **10 / 200** |
| 最终 global / PPO / optimizer 累计 | **68224 / 498 / 9960** |
| wall time | **830.2300018998794 秒** |

最终来源链从66944增至68224，无额外 initial 优化信用。阶段预算为 full_episode **25088**（未增加）、phase_suffix **33024**（31744+1280）、smoke0，origin10112 保留。`25088+33024+10112=68224`。计划未消费的2816未写入实际账本，下一实验是否使用它不属于本报告的完成计数。

最终 [checkpoint68224 sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000068224_manifest.json) 为 `save_load_round_trip=true`。主线程已核验 checkpoint SHA `0bc86a834f7baeb6fb35a5c27d9d31dab6738f7656f528dc8cf101866af26897` 与 sidecar/pointer 实际文件绑定；本报告不重复大文件 hash。初始网络/std、critic、normalizer、RNG 与 Adam reset 的证据保留在 [initial 67072 报告](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/continuous_stop_p06_initial_67072.md)，该有界报告不被回写为最终结果。

## 2. 完整策略样本、两次终止与第三回合尾段

实际 audit global **66945–68224** 共1280条，连续无缺号/重复，**全部属于 P06**，P01–P05/P07–P13优化样本均为0。读取的是停止后最终文件，不包括推测的下一更新。

| 回合 | global 范围 | PPO 决策 | PPO ticks | 最末 episode tick / 秒 | 正式状态 |
|---|---|---:|---:|---|---|
| 0 | 66945–67545 | 601 | 4801 | 8385 / 69.875 | P06 INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 67546–68146 | 601 | 4801 | 8385 / 69.875 | P06 INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 68147–68224 | 78 | 624 | 4208 / 35.066667 | **P06 非terminal尾段** |
| 合计 | 66945–68224 | **1280** | **10226** | — | **1202终结回合样本 + 78已优化非terminal样本** |

两次真正终止的最后决策 global **67545、68146** 都仅执行1个物理tick；每回合 `600×8+1=4801`，各比完整8tick决策短7。因此完整策略ticks为 **1280×8−14=10226**，差额不是漏采样或未优化信用。

两次 P06 stage age 均为40.008333秒，独立 physical evaluator termination_reason=null，总任务仍余130.125秒；原结果是阶段未完成，不重分类成碰撞或200秒总任务超时。第三回合在第三次真实prefix后只执行78个策略决策，stage age5.2秒、remaining164.933333秒、termination_reason=null，不能因训练块交接而称该尾段物理失败。

## 3. 教师排除与 native/state-write 对账

三次 prefix 均 accepted、无miss/fallback，各 **448教师决策 / 3584ticks**；总计 **1344决策 / 10752ticks**。逐条prefix证据中 `policy_credit=false`，raw与projected residual全为0，10752/10752 native ticks verified，no-in-episode-state-writes标记全true。策略1280条中 `prefix_teacher_data_in_ppo_storage` 全false。

因此完整物理core为 **2624决策 / 20978ticks / 3回合**，其中教师1344/10752与策略1280/10226严格分开。所有策略行 `all_ticks_verified=true`、verified ticks合计10226且逐回合等于实际ticks；四项root pose/root velocity/force-or-impulse/gravity写计数均0、no-state-write verification全true。教师前缀的真实FR/FL完成历史不能作为PPO重新学得样本。

## 4. 本块实际第一未完成环节

三回合接管均为P06/t3584，经真实A roll-in，不是RR已经AIR的历史快照。最终历史仍只有教师FR Q71/C1665/P1695与FL Q2461/C3115/P3583；**本块各回合RR/RL均没有qualified/crossed/placed事件**。

| 最末状态 | RR真实状态 | RL真实状态 | rear_approach |
|---|---|---|---:|
| episode0终止 | AIR，front−445.091mm、clear−43.718mm、load0 | GROUND，front−557.033mm、clear−50.008mm、load0.536903 | 0 |
| episode1终止 | AIR，front−544.209mm、clear−48.141mm、load0 | GROUND，front−639.969mm、clear−50.659mm、load0.518840 | 0 |
| episode2尾段 | AIR，front−362.026mm、clear−48.450mm、load0 | GROUND，front−479.900mm、clear−50.079mm、load0.513904 | 0 |

RR的当前AIR不等于达到越障资格；上述净空仍低于台面，且没有Q/C/P事件。两个完成回合的第一未完成阶段仍是P06前驱workspace准备。其回报分别−51.850572、−51.670324；第三回合尾部累计−1.084328，不是最终回报。

**本块没有P13样本，因此未实际检验新continuous-stop shaping的主动分支或最终停止能力。** 此账本也不把P06结果全部归因于某个nominal或策略噪声；stop请求里的后续nominal因子研究理由是实验计划，不是本报告新增的物理因果证明。

## 5. 十次已完成真实更新

optimizer log精确覆盖update **489–498**，global **67072–68224** 每次增128、每次20 optimizer steps，合计200。十次 actor-before/after链连续，全部 `actor_parameters_changed=true`、`finite_nonzero_gradient_observed=true`，末actor-after与checkpoint68224记录一致。首次source/initial权重绑定已由initial报告核验。

initial Adam LR为3e−5，十个已记录update边界/结束LR均为1e−5；未逐minibatch记录LR，不作全程常数断言。保存的是完成第十次更新后的actor/critic/optimizer状态，不是从未完成rollout选取的结果。

停止结论：本次短诊断块实际1280/10/200，已保存至68224/498/9960；剩余2816明确未消费，两个P06未完成加一个已优化非终结尾段，无任务成功声明。新启动的自然P01评估是独立后续工作，本报告不预填其结果。
