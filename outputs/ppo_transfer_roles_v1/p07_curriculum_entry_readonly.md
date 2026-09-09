# P07 / offset0：入口、信用与采样边界（只读）

**可用现成P07课程做有限补样，但不能预称它是“RR首次离地前的GROUND入口”，也不能替代P06/P01学习。** 当前仅核源码、完成run3/6/8的小摘要及各首个真实prefix handoff/start；没有扫描策略原始轨迹或Run10，没有运行Python/Isaac或修改生产。

**真实入口。** [semantic_prefix.py:175](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_prefix.py:175) 从自然P01用原FSM教师滚入，共享语义supervisor从tick0观察。offset0是在目标**语义**P07首次出现后、仍在P07且物理valid的首个8tick网格交接（不是教师FSM标签、不是固定保存姿态）。最后真实ACK提供nominal/tracking/bias；有bias则继续reset-only takeover，至当前及上一实际dispatch的bias均0才READY。READY不保证仍P07；`actual_phase`、`requested_phase_still_active_at_credit`如实记录。目标提前离开、教师失败或限额用尽走已有miss及一次fresh-P01 fallback；没有新的GROUND/抬腿门。

P06完成条件是两后腿edge proximity；P07要求两腿edge proximity与RR role preparation，P08要求RR proximity/transfer readiness，P09目标才是RR placed。**这些条件不排除此前RR已有initial AIR甚至硬Q。** 首次AIR、initial-clearance、硬qualified是不同证据，不能互相替代。所选三个run都是P06接管，未提供本版本P07目标的实际入口；因此其RR ground/AIR/Q/C/P目前未知，不能用旧版本教师时钟或P06当前GROUND外推。

**保留什么。** 同一物理episode、mapper/final-drive、真实q/contact、supervisor的Q/C/P及role滑动窗口不在handoff重置；`handoff.semantic_task.history`保留带tick的既有事件，`entry_observation`与`credit_start_observation`分别记录交接/开信用时的真实状态。core历史在prefix每tick更新；教师raw/REQUEST为0，但实际drive/nominal不是无条件0。P07→P08→P09继续同core/bridge，无phase done或额外reset，允许GAE穿过非terminal阶段。

**不保留成PPO信用的部分。** 教师P01→P06全过程（含bias takeover）均排除storage；P07起的下游回报不能直接给教师P06动作credit。`from_live_prefix/from_handoff`新建语义nominal provider，以最后实际nominal/tracking为种子，并非复制教师完整动作队列或虚构一个旧P06 continuous layer；此后实际产生的P07/P08层连续延续。教师既有Q/C/P不是本策略新增事件；suffix成功不等于自然P01 full success。

| 完成块 | 实际决策 | P06 / P07 / P08 / P09信用样本 | 接受prefix / 排除决策 | RR硬Q新增 / C / P |
|---|---:|---|---|---|
| run3，324列 | 1024 | 990 / 1 / 1 / 32 | 4/4，1792 | 1 / 0 / 0 |
| run6，324列 | 1536 | 911 / 7 / 7 / 611 | 13/13，5824 | 10 / 0 / 0 |
| run8，372列 | 1024 | 984 / 3 / 3 / 34 | 8/8，3584 | 3 / 0 / 0 |

三块合3584信用样本中P06=2885，P07/P08各11，P09=677；均无policy rear C/P。Q次数是事件尝试、非不同成功回合。各首handoff/credit均t3584=29.8667s，上一控制tick3583对应native ACK3763，bias全0；RR ground=true、AIR=false，front−494.884mm/gap−50.950mm/load.228028；RL亦GROUND，front−510.247mm/gap−49.513mm/load.298524。这证明所选P06入口的实测状态，**不是P07状态**。prefix各25次均接受并不证明未来P07教师一定可达。

**预算建议边界。** 同HEAD、同372布局固定 `phase_suffix / frozen_fsm / FromPhase P07 / offset0 / N1`可普通resume保Adam，不需再NewMDP；1024=8个128更新，1536=12个更新，仅是请求预算，实际以最终manifest为准。倾向先1024作有限P07/P08→RR补样，1536也只是多512信用样本而非成功保证。较晚入口可减少花在远端P06的采样，但教师准备分布、较短剩余horizon、快速跳过P07/P08及已有AIR/Q都可能发生；不能保证每阶段正样本或消除FALL/collision。应保留自然P01/P06课程以让策略自己形成前驱入口；本结论不要求新probe、A成功或固定姿态门，不等待它才继续训练。
