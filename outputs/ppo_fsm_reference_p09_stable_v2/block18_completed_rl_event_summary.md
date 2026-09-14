# Block18 完成：RL事件补充摘要

已完成run：`train/20260910T1921276505520Z_gd4e46006b382_2f570e3174ba4fe3a1035ca1c6162b8b`。主固定快照：[20260910T202117929903Z.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T202117929903Z.json)。

实际512决策/4 PPO更新/80 optimizer steps，阶段P10=3/P11=17/P12=492/P13=0；334条BODY＋109条RR膝HARD＋69条已学习非终止尾＝512。全4090 native/actual-effect verified、4084 own-phase；四类episode状态写入0，teacher误计0。正常训练exit0并非物理成功。

## RL新事件与教师历史分开

在已有前95、96–276和episode1硬限报告之外，本次只补充一次已完成audit的512行RL标量/历史事件提取；source字节前后均50,056,873，无活动后续、physical大流、Torch或CSV读取。事件按episode/tick/类型去重，tick≤各回合信用起点7584算继承。汇总器本身只统计RR，未更改其框架。

|回合|策略决策|RL新I/Q/回地撤Q/C/P|结果|
|---|---:|---|---|
|0|334|4 /4 /3 /1 /0|P12/t10256，BODY|
|1|109|4 /2 /1 /0 /0|P12/t8450，RR膝HARD|
|2|69|1 /0 /0 /0 /0|P12/t8136，非终止采集尾|
|合计|512|**9 /6 /4 /1 /0**|没有RL放置|

RL教师继承仅I6（每回合tick7/1722），无继承Q/C/P，不计作本段新事件。

- ep0：I8369/8819/9042/9149，Q8374/8822/9051/9153，回地撤Q8633/9019/9111；新C9256/global155217。
- ep1：I7621/7634/7909/8272，Q7946/8279，回地撤Q7996；无C/P。
- ep2：I8065/global155512，无Q/C/P。

全512个决策末态，RL task top_contact=true为0，history placed=true为0；AIR261、GROUND183，两者并非完备或互斥分类规则，不能由余数推断接触类型。最向前RL front=+16.005400mm，仅短暂越沿，不能称完成放置。

## 终态与非终止尾的真实RL状态

- ep0 BODY：RL历史Q/C=true、P=false；当前AIR、无GROUND/TOP/support，bearing_verified=true且承载0，front−47.603127mm、top clearance+61.823010mm。全局load_fraction_valid=false，不补造载荷比例；历史C不能冒充当前仍在前缘前方。
- ep1 HARD：RL历史Q=true、C/P=false，AIR/无TOP/support，承载0、front−264.434662mm、clearance−21.195436mm。硬限是RR膝，不把RL悬空写成有效支撑。
- ep2尾：RL GROUND/support=true、bearing verified、8.147947N；front−50.151762mm、clearance−51.089041mm，当前initial_clearance/active_attempt=false、历史Q/C/P=false。69条已学习、terminal=false/reason=null，不是第三个完成回合。

RL没有RR专用的current_lift_valid/established定义；本摘要不把缺席字段补成false。

## RR与结论边界

主快照RR新I/Q/C/P全0，继承教师I/Q/C/P各3；512行历史C/P成立不能当作PPO新学出的RR放置。RR有2次current_lift_revoked_ground，current-valid端点158；ep0末RR混合接触/承载不确定，ep1末GROUND，ep2末真实TOP/current-valid。历史放置不等于始终当前TOP承载。

本块已证实的是教师P10入口之后的真实PPO RL新越沿C1，**没有RL放置、P13样本或完整后缀成功，更不是自然P01完整成功**。不建立成功门禁、不改判据或生产，四主报告暂不更新。

原有诊断保留其各自固定截点：[前95连续性](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block18_first95_p10_p12_continuity.md)、[RL越沿至276](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block18_new_rl_edge_crossing_through276.md)、[episode1硬限](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block18_episode1_hard_limit.md)。其中旧“443采样/384优化”的状态已由本次512完整更新保存解决，不能重复计信用；原诊断不改写。

