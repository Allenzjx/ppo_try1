# 首4096个真实训练决策：封存汇总

已完成 **4096 policy decisions、32 PPO updates、640 optimizer steps**，从 **178432 / 1359 / 27180** 续训至 **182528 / 1391 / 27820**。十块分别为512+896+384+128+128+768+768+256+128+128，决策区间连续，prefix和未优化尾部不计入。本次复用十个封存receipt，不重扫全部原始日志。

## 版本与实际覆盖

block01–05：旧quarter HISTORY kernel，2048/16/320，HEAD f2e552406ea7。block06–10：经CP180480显式迁移后的REQUEST-history kernel，另2048/16/320，HEAD 3a50657a96c9。两版不能混称，且不是仅改变kernel的配对因果实验（权重与课程也在演进）。372维观测、12动作通道、rho=.9、temperature=.25、Identity normalizer和有效LR1e−5保留。

| 实际请求阶段 | 旧kernel样本 | 新kernel样本 | 累计已优化样本 |
| --- | ---: | ---: | ---: |
| P01 | 6 | 2 | 8 |
| P02 | 488 | 210 | 698 |
| P03 | 9 | 4 | 13 |
| P04 | 18 | 4 | 22 |
| P05 | 988 | 608 | 1596 |
| P06 | 226 | 947 | 1173 |
| P07 | 6 | 4 | 10 |
| P08 | 13 | 4 | 17 |
| P09 | 294 | 137 | 431 |
| P10 | 0 | 1 | 1 |
| P11 | 0 | 1 | 1 |
| P12 | 0 | 126 | 126 |
| P13 | 0 | 0 | **0** |

32次更新均记录有限非零梯度、LR1e−5。65次普通阶段变化均未设terminal；10个真实episode终止为5次BODY_COLLISION、5次FALL，**没有训练中完整任务成功记录**。非终态预算结束不改写成任务失败终局，也不冒称成功。

## 已观察到的能力与限制

- block07从冻结CP181248真实P05/offset150策略前缀接管，三次FL placed分别tick3010/3236/3634；P06交接时都有真实TOP承载，分别2.7095/4.8560/5.2892N。前两段随后在P09机身碰撞，RR未crossed/placed；第三段预算末仍P06，FL已重新AIR、gap24.014mm、0N。捕获存在不等于持续支撑，更不是固定checkpoint自然P01成功。
- block08补充自然P01/P02：四种真实物理子状态共1695个质量样本，β=.015–.03/s全部非零。成本+.0145646430967与实际body reward −.0145646430967逐决策对应；256条总reward与PPO storage精确一致，40个实际minibatch各样本使用5次，hook oldlogp/advantage与保存张量逐样本一致。**这是成本进入实际学习的证据，不是单项梯度方向或稳定性改善证明。** 详细边界见block07_summary.md追加节。
- block09的successful_nominal P06前缀后，128个学习样本全部P06；不能根据课程名称宣称RR摆动覆盖。
- block10的successful_nominal P10前缀后，真实P10/P11/P12=1/1/126。预算末global182528、tick7184/59.866667s仍P12，无terminal、task_success=false、RL尚未placed。RR虽然历史placed=true，**当前ground_contact=true、top_contact=false**，bearing4.5451N来自当前地面支撑；gap−50.043mm、front−54.600mm，不能写成RR保持台面放置。源记录的obstacle contact_surface为NONE，与ground_contact标记应分别解读。**P13学习覆盖仍为0。**

## 最新checkpoint与后续评估边界

[checkpoint_step_000182528.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000182528.pt)

本次实际重算并匹配：checkpoint SHA256 `81f392a726b425a0bd8e004db11513a4acdb5e2413aec676f8a1b5b835ebc47e`；manifest SHA256 `3e61d497dd8e44207fc8caa8d9d60539b025df3db4d15ac34231d81242d06e6a`。manifest计数与上述4096/32/640相符，记录save/load round-trip=true；未另调用actor或优化器。物理episode状态未保存，不能把预算末姿态当下一run无缝续接的物理快照。

正式固定CP182528自然P01确定/随机视频评估单独记录，本汇总未读取活动评估日志，**不包含新完整越障或优于zero稳定性的结论**。未新增发布门禁，未启动Isaac/CUDA，未改生产、DELIVERY或RECOVERY。

结构化证据：[first4096_training_summary.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/first4096_training_summary.json)。来源为block01–10 sealed receipts、已有REQUEST_KERNEL_REAL_AUDIT/block07_summary及block10末条真实审计。
