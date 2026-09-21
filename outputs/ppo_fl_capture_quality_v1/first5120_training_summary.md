# 首5120个真实训练决策：封存汇总

已从 **178432 / 1359 / 27180** 续训至 **183552 / 1399 / 27980**（policy decisions / PPO updates / optimizer steps）；累计新增 **5120 / 40 / 800**。本次只在既有4096汇总上加入封存block11的1024/8/160，区间182529–183552连续、全部完成优化、无未消费尾部。训练预算执行SUCCEEDED不等于越障成功。

## 覆盖与版本

| 实际请求阶段 | 前4096 | block11 | 累计5120 |
| --- | ---: | ---: | ---: |
| P01 | 8 | 4 | 12 |
| P02 | 698 | 392 | 1090 |
| P03 | 13 | 7 | 20 |
| P04 | 22 | 2 | 24 |
| P05 | 1596 | 307 | 1903 |
| P06 | 1173 | 268 | 1441 |
| P07 | 10 | 1 | 11 |
| P08 | 17 | 1 | 18 |
| P09 | 431 | 42 | 473 |
| P10 | 1 | 0 | 1 |
| P11 | 1 | 0 | 1 |
| P12 | 126 | 0 | 126 |
| P13 | 0 | 0 | 0 |

P03–P06合计3388/5120（66.171875%）；**P13仍为0**。block01–05旧quarter kernel共2048/16/320（HEAD f2e）；block06–11经CP180480显式迁移的REQUEST-history kernel共3072/24/480（HEAD 3a）。不能把两个时期当成受控的单因素因果对比。观测372维、12通道、rho=.9、temperature=.25、Identity normalizer、有效LR1e−5维持。40次更新均有有限非零梯度记录；累计78次普通阶段变化未切为terminal，真实prefix不计学习覆盖。

## 新增块的真实物理结果

block11为自然P01的full_episode学习，非冻结模型评估；网络在同一运行中持续更新。

- 第一episode：661个学习决策，FR placed1536，FL crossed2276、placed2848（23.733333s）；global182884发生P05→P06交接。随后global183189 / tick5283 / 44.025s在P09真实 **BODY_COLLISION**；物理评价器原因为central body/obstacle collision。RR未crossed/placed。
- 第二episode：363个学习决策，FR placed1688，FL crossed2786、placed2847（23.725s）；global183545发生P05→P06交接。预算末global183552 / tick2904 / 24.2s仍P06，无terminal、task_success=false。FL虽历史placed=true，**当前AIR、support=false、top_contact=false、bearing=0N、gap=7.949018mm**，不能写成持续承载或完整成功。后续准备中FL暂时AIR本身也不另造失败标准。

累计11个真实终止episode：6次BODY_COLLISION、5次FALL，完整任务成功0次。预算截断不另计终态失败。

## 质量项确有非零贡献，但不宣称稳定性已改善

本新增块的4种前段物理子状态共3166样本，beta范围.015–.03/s全部为正。封存收据中实际quality cost合计 **0.036705387638033**，带符号body_stability reward合计 **-0.036705387638033**，成本与负贡献相抵误差小于1e−12。按请求阶段分为P01 -0.00006237893306374、P02 -0.03664300870497；P03–P09此质量项为0，其他三个质量族也为0，不能将它们说成已得到非零质量优化。

上述是原实际奖励分解与8次已完成更新的证据；本汇总不重算GAE，不从标准化advantage正负推断单一成本的梯度，更不把非零成本称为质量提升。先前block08的storage逐样本核验继续保留在first4096报告，不虚构本次再次做过该核验。物理子状态归属与整拍请求阶段分组可能不同，总和一致。

## Checkpoint核验与边界

[checkpoint_step_000183552.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000183552.pt)

当前pointer及两个实际文件各核验一次：CP SHA256 `9f42f6eff0795eefa857756d4746a4170167966606e6ff3b5c231f2a7420fb7d`；manifest SHA256 `f733e1e9e4c6bf709c30415b06416c9c6545580edd5df9520b435d9e0c0228c2`，均匹配pointer。manifest总计数183552/1399/27980、分支5120/40/800与汇总一致，记录save/load round-trip=true；本次未调用actor或重新加载训练网络。物理环境状态未保存，不把末姿态称作连续恢复的快照。

来源：[first4096_training_summary.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/first4096_training_summary.json)、[block11_1024_receipt.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/block11_1024_receipt.json)、block11 training_manifest/首个completed_episode/末条audit，以及CP pointer/manifest。未扫描历史原始run；当前CP183552正式评估不在本报告内。未启动Isaac/CUDA、未改生产/DELIVERY/RECOVERY。

结构化结果：[first5120_training_summary.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/first5120_training_summary.json)。

