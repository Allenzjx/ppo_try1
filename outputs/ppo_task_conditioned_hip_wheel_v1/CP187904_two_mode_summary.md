# 固定 CP187904：两种采样模式的真实结果

两次均实际加载同一 checkpoint，actor/critic/optimizer/normalizer 哈希一致，评估均无 optimizer 更新。stochastic 是 `training_style_conditional_gaussian`，policy seed 4101。每种模式仅一条轨迹，不是成功率统计；新同版本 B 尚未配对，本次未重扫旧 B。

|项目|deterministic|stochastic|
|---|---:|---:|
|整次结果|50.825 s，P05 未完成；semantic 有限恢复耗尽|42.050 s，P09 机身碰撞失败|
|独立物理结果|success=false；termination=null|success=false；TASK_FAILURE_BODY_COLLISION|
|FR placed|tick1620 /13.500 s|tick1611 /13.425 s|
|P01→FR 捕获 rate RMS|0.104231 rad/s|0.105738 rad/s|
|同窗峰值合成倾角|0.290952 rad|0.297549 rad|
|同窗 collider 最低世界 z|92.512 mm|89.727 mm|
|同窗 FR 最大轮底净空|99.293 mm|108.524 mm|
|FL 越沿→捕获|tick2728 越沿后一直未捕获|2726→2790，0.5333 s 完成|
|P06 body 净前送|未到达，null|2792→4704，15.9333 s，+295.279 mm|
|RR 结果|未到达|曾获资格5001；当前有效性随后丢失，落地5018撤销；无越沿/放置|

FR 相同任务窗口中 deterministic 的角速率/倾角略低，但它随后未完成 FL 放置；不能据此选成任务成功策略。RMS 仍是 `sqrt(∫rate²dt/T)`，没有再除时长。

FL 捕获不等于持续承载：stochastic 从 FL placed2790 到 RR 准备入口4704，有1915个120 Hz真实接触观测，FL AIR1643、障碍接触272；240个保存的 evaluator 端点中只有37个 TOP/verified-bearing/support、203个AIR。捕获是真实事件，保持却明显不足；不能用历史 placed 给悬空 FL 记载荷。deterministic 越沿后3372个物理观测全为AIR，最后净空8.078 mm，确实没有落脚。

stochastic P06 的 body 前送295.279 mm，RL/RR轮中心分别前送298.782/301.896 mm；这是实际位移，不是根据轮速推算。该观察不单独证明轮驱承担主要推进力。P06窗口 FR/RL持续接触，FL多数悬空；进入 P07 后的343个物理观测 FL 全AIR。

RR时间线尤其需要保留“资格曾发生”和“当前有效性”的区别：

- 4993初始free-air lift，5001 qualified（free-air rise8.662 mm）。
- 5008仍AIR/current_lift_valid=true，当前rise8.026 mm。
- 5016仍AIR，但current_lift_valid=false，rise仅1.679 mm。精确丢失tick未保存，只有(5008,5016]区间证据，不能补造。
- 5018真实GROUND（exact pair verified），同时出现资格撤销事件；距首次qualified仅0.1417 s，但**当前有效资格并未保持满这段时间**。
- 最后history active_lift.RR=false，旧event_ticks仍留5001；没有RRcross/placed。5046的真实base_link/obstacle接触持续，物理失败终止；最后6tick决策未返回，独立manifest保留终态。最后RR又AIR一点也没有重新取得资格。

RR qualified→失败窗口RMS0.553634 rad/s，body collider最低49.934 mm；碰撞结论来自真实exact pair持续，不是单凭几何低点。文件：`CP187904_stochastic_event_windows.json`、`CP187904_stochastic_RR_retention_appendix.json`。本分析只读封存数据/写outputs，没有改生产、没有启动仿真，也不影响新B录像。
