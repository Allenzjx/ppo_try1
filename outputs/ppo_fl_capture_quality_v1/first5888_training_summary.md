# 首5888个真实训练决策：封存汇总

从 **178432 / 1359 / 27180** 续训至 **184320 / 1405 / 28100**（policy decisions / PPO updates / optimizer steps），累计新增 **5888 / 46 / 920**。本次仅在既有5120汇总上加入 block12 的 **768 / 6 / 120**，实际区间183553–184320连续，全部已优化。

本块原计划2048决策，在第6次完整更新后按请求封存为 `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`；剩余1280只是未执行请求，不计信用。该状态不是任务成功，也不是宣称原2048预算已全部完成。

## 实际覆盖

| 请求阶段 | 前5120 | block12 | 累计5888 |
| --- | ---: | ---: | ---: |
| P01 | 12 | 0 | 12 |
| P02 | 1090 | 0 | 1090 |
| P03 | 20 | 0 | 20 |
| P04 | 24 | 0 | 24 |
| P05 | 1903 | 181 | 2084 |
| P06 | 1441 | 467 | 1908 |
| P07 | 11 | 2 | 13 |
| P08 | 18 | 2 | 20 |
| P09 | 473 | 116 | 589 |
| P10 | 1 | 0 | 1 |
| P11 | 1 | 0 | 1 |
| P12 | 126 | 0 | 126 |
| P13 | 0 | 0 | 0 |

P03–P06共4036/5888（68.546196%），P05已实际优化2084决策；**P13仍为0**。旧quarter kernel（block01–05）2048/16/320；CP180480显式迁移后的 REQUEST-history kernel（block06–12）3840/30/600，不能把两个时期直接当受控单因素对比。46次更新均记录有限非零梯度；累计87次普通阶段变化没有被切为terminal。

## block12真实物理结果与前缀排除

课程是冻结 **CP183552** 从自然P01运行至P05后再经过200决策的真实policy前缀，随后由持续更新的learner接管；不是nominal教师或历史状态注入。三次前缀均在tick3208 / 26.733333s接管，合计**1203决策 / 9624物理tick全部排除**；本块仅计之后768决策 / 6140物理tick。前缀出现的P01/P02不算新质量训练样本。

| episode | learner决策 | FL真实placed tick / s | P05→P06 global决策 | 后续结果 |
| --- | ---: | --- | ---: | --- |
| 0 | 365 | 3748 / 31.233333 | 183620 | P09 tick6124 / 51.033333s BODY_COLLISION |
| 1 | 333 | 4018 / 33.483333 | 184019 | P09 tick5872 / 48.933333s BODY_COLLISION |
| 2 | 70 | 3295 / 27.458333 | 184261 | 更新边界tick3768 / 31.4s仍P06，无terminal |

三次FL捕获均在前缀结束后，属于真实学习后缀成果；但不是3次冻结模型从P01完整成功。前两次的物理理由均为 **central body/obstacle collision**，RR均未crossed/placed。第三次末状态仍有效/VERIFIED、bootstrap=true、task_success=false；FL历史placed=true，但**当前AIR、support=false、top_contact=false、0N、gap=8.050685mm**；RR当前GROUND并承载约1.838741N、尚未越沿。不能把历史放置写成持续支撑，也不因合法后续FL离地另造失败规则。

累计真实终止13个：**8次BODY_COLLISION、5次FALL、完整任务成功0次**。第三次更新边界截断不加算失败。

## 奖励与更新：不把前缀质量算入学习

本块P01/P02新增learner决策与front-quality样本均为0，body/contact/smoothness/regularization四族实际贡献均为0；这是课程覆盖事实，不是配置回退或质量系数被mask。此前首5120中的非零前段质量贡献仍保留，但本次没有新增。P05带符号task贡献+0.2328409883，P06 −2.2562433452，P07 −0.0092714728，P08 +0.0172270823，P09 −85.3780768684（含两次真实失败）；这些求和不是折扣回报或单一成本的因果梯度。

更新1400–1405均有有限非零梯度。实际optimizer LR在1400为2.25e−5，之后1401–1405均1e−5；不能写成本块每次都是1e−5。最后KL=0.02208508147、value loss=47.93907013；含失败批次的value loss明显增大，如实保留。未重算GAE、未重跑optimizer，也不由非零更新推断稳定性改善。

## 当前checkpoint及边界

[checkpoint_step_000184320.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000184320.pt)

pointer与实际文件hash匹配：CP `4f752923182de4fefb7ab9219c3d3bfbd4ffd1a3b896786da6e3633e21fdc545`；manifest `f18733884220c92bc50d48397857f304e5394789e2598d6d320a2624bf2231f5`。manifest计数184320/1405/28100、分支5888/46/920一致，save/load round-trip=true；本次只检查文件，不重新调用actor加载。物理环境状态没有保存，不能把末姿态称为可续接快照。

policy保持372维／12通道、rho=.9、temperature=.25、Identity normalizer，版本 `cap_transition_request_history_heteroscedastic_log_temperature_quarter_v1`；runtime SHA `e1e2a003f6711a77c463eb743282c250c67748dc1467e819ab90d9959e2d7ca4`。不包含CP184320正式评估结论，不宣称完整成功、优于N或泛化。

来源：[前5120汇总](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/first5120_training_summary.json)、[block12收据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/block12_768_receipt.json)、本块training_manifest、两条completed_episode、末条audit及checkpoint pointer/manifest。仅新增本MD与[结构化汇总](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/first5888_training_summary.json)，未改生产、DELIVERY或RECOVERY。

