# N＋0 与 video5：按 P05 物理进展对齐

2026-09-10 14:49:18 UTC固定核查。两个runtime contract逐字段相等、HEAD `7db0d17...`、seed4001。N＋0是`semantic_prior_eval`（不是冻结FSM A），video5为checkpoint149888 deterministic C。各审计只扫一次：N＋0仅前335决策，读到已完成P05的tick2680即停止；video5读800决策/P05终止。不评价N＋0后续阶段或整回合成功。

## 真实事件，而非相同时间/姿态

| 证据 | N＋0 | video5 PPO |
|---|---|---|
| FL I |1550 /12.916667s|1989 /16.575s|
| FL Q |1559 /12.991667s|1992 /16.6s|
| FL C |2468 /20.566667s|2587 /21.558333s|
| FL P |**2677 /22.308333s**|未发生|
| 首个C后决策端点 |2472：AIR，顶面gap25.668mm|2592：AIR，gap62.582mm|
| 首个C后top_geometry端点 |2672：AIR，gap9.731mm，0N|3104：AIR，gap18.263mm，0N|
| 已放置/最终端点 |2680：**TOP/support=true，8.531N**，gap0.012918mm|6396：AIR/support=false，0N，gap11.249mm|

N＋0在P05共146决策；C后端点AIR26/TOP1。video5在P05共559决策；C后477端点全AIR，未出现P。N＋0的2680端点已经包含P05→P06交接，其四轮目标0不能倒推成“2677放置前先停轮”；2672的四轮目标实际仍均+0.3rad/s。精确P来自历史事件tick2677，接触/目标向量在上表明确标的是所采决策端点。

## 第一处可证实的行为差异

本窗口包含的P04→P05前驱端点已经不同：PPO residual FL hip/knee约+3.127/+19.713deg、四轮−0.558/−0.197/−0.152/+0.382rad/s，并连续继承到P05；N＋0均为0。P05首个动作的nominal恰好相同（FL hip/knee0/−26.1deg，四轮均+0.3），但：

- N＋0@tick1520：FL实际目标0/−27.35deg，四轮+0.3；FL仍GROUND、front−135.415mm、承载12.249N。
- PPO@tick1936：residual FL+3.191/+19.763deg；实际目标3.191/−7.411deg，四轮−0.266/+0.101/+0.147/+0.686rad/s；FL仍GROUND、front−96.511mm、承载14.417N。

因此可以确认策略在进入P05时已实质改变髋/膝目标及四轮分配，不是无控制效力；但两个真实入口及滤波/支撑历史明显不同，不能把目标差简单当作唯一失败原因或要求PPO复制N＋0关节姿态。

同类“已Q且刚C”窗口中，nominal又都为FL48.2/−36.7deg、四轮+0.3。N＋0@2472目标49.45/−37.95deg；PPO@2592叠加FL+5.880/+21.856deg，目标55.330/−16.094deg，四轮−0.316/+0.054/+0.206/+0.763rad/s。随后N＋0靠近并真实接触顶面；PPO虽然到达top_geometry，仍未承载。这是动作与结果的可测差异，不是“某个膝角导致/防止接触”的因果证明，也不推断未测CoM运动。

## 审计范围与结论

两者P05全12通道mask开放，native逐tick verified分别1168/1168和4468/4468。此前video5简单`native_base+residual`差额0.143315现在定位为FL hip@tick1984（deg）；当时base26.05、residual3.578068、final29.484753、controller_bias0。该差异也存在于含combined-bias的简单求和，本次未细分其动态投影环节；它发生于Q前，不能推成导致后续不放置的缺陷。对应native审计仍verified；此处不掩盖该差异，也不把它升级成故障或门禁。

本对比未证实控制实现、概率存储或reward符号缺陷，也未重新计算整回合reward。现有证据支持：**同一规范下P05真实放置可达；当前已学策略有效改变动作，却未学到本次自然入口下的放置结果。** 将其作为学习结果继续PPO，不要求N＋0全程成功、不恢复旧精确姿态/相似度门槛、不以零残差替换学到的策略。

[固定JSON证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/n0_vs_video5_p05_progress_comparison.json)保留两个audit绝对路径、事件及各进展端点。未启动Isaac、扫描physical/native大流、重算媒体/模型哈希，未改生产或主报告。
