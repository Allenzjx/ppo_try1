# CP187904 对本轮同版本 B：物理事件窗口对照

结论：本轮 B 从 P01 完整成功（8857 tick /73.808333 s）；CP187904 deterministic 未完成 FL 放置，stochastic 在 RR 准备后发生机身碰撞。局部 FR 角速率较低不等于任务能力或整体稳定性优于 B。

三条 manifest 的 experiment、物理 seed4001、源码提交ee5a965、runtime-content hash及全部6个配置hash一致。两条 C 实际加载参数hash一致的CP187904，评估均0次更新；stochastic policy seed4101。每种模式仅一条轨迹，不是统计成功率。

## FR：均完成自然 P01→真实捕获

|指标|本轮 B|deterministic|stochastic|
|---|---:|---:|---:|
|捕获 tick|1502|1620|1611|
|时长 s|12.5167|13.5000|13.4250|
|roll/pitch rate RMS rad/s|0.117889|0.104231|0.105738|
|峰值合成倾角 rad|0.309464|0.290952|0.297549|
|body collider 最低世界 z mm|91.272|92.512|89.727|
|body/障碍 AABB 分离下界最小值 mm|217.320|227.336|229.414|
|FR 轮底相对顶面 gap 最大值 mm|102.132|99.293|108.524|

相对当前 B，det/stoch 的该窗 RMS分别低11.586%/10.307%，但捕获分别慢0.9833/0.9083 s；stochastic 的机身最低几何点反而低1.545 mm。不能只取较低 RMS 宣称无代价改进。

## FL 捕获、保持与 P06 前送

|指标/同任务窗口|本轮 B|deterministic|stochastic|
|---|---:|---:|---:|
|FL 越沿→捕获 tick|2468→2677|2728→未捕获|2726→2790|
|FL 越沿→捕获用时 s|1.7417|28.0917后终止，未完成|0.5333|
|同窗 rate RMS rad/s|0.016297|0.005444（不可排名）|0.042260|
|同窗峰值倾角 rad|0.054812|0.037430（未完成）|0.017343|
|同窗 body collider最低z mm|117.557|131.448（未完成）|121.180|
|捕获后→P07 的 FL TOP/verified-bearing端点|311/311|未到达，null|37/240|
|同保持窗 FL AIR端点|0/311|未到达，null|203/240|
|同保持窗 FL 最大gap mm|0.014|未到达，null|27.736|
|P06→P07 用时 s|20.6667|未到达，null|15.9333|
|P06 rate RMS rad/s|0.010536|未到达，null|0.047068|
|P06 峰值倾角 rad|0.052611|未到达，null|0.089218|
|P06 body collider最低z mm|125.122|未到达，null|118.546|
|P06 body/障碍保守分离下界 mm|81.027|未到达，null|68.546|
|P06 body 实测净前送 mm|307.597|未到达，null|295.279|

deterministic 的低角速率对应 FL 未落脚停滞：越沿后3372个物理观测全部AIR，最终gap8.078 mm。它在50.825 s由semantic `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`结束；独立物理 `success=false, termination_reason=null`，不能改成成功或机身碰撞失败。

stochastic FL捕获是真实事件，但保持显著不足：捕获后至P07的1915个120 Hz接触观测中，AIR1643、障碍接触272；B同窗2484个物理观测均有障碍接触。表中TOP-bearing另取已保存的evaluator端点，不把稀疏样本数伪称连续承载时长。B历史placed与当前支撑一致；stochastic不能因历史placed而给AIR的FL记载荷。

stochastic 较早进入P07，但这两条P06起点构型/净位移不同，不等于同距离驱动速度改善；并且RMS/倾角更高、body几何空间更小。轮端位移与角速度不能单独证明轮驱是主动力。

## RR：B完成，stochastic窗口止于真实失败

|P07准备→RR放置/实际终止|本轮 B|deterministic|stochastic|
|---|---:|---:|---:|
|窗口 tick|5160→6155|未到达，null|4704→5046碰撞|
|窗口状态|完成|未到达|未完成/物理失败，不可排名|
|观察时长 s|8.2917|null|2.8500|
|rate RMS rad/s|0.052065|null|0.251545|
|峰值倾角 rad|0.210474|null|0.429490|
|body collider最低z mm|86.544|null|49.934|
|body/障碍保守分离下界 mm|36.544|null|0.000|
|RR 轮底相对顶面 gap 最大值 mm|85.442|null|−40.514|
|RR qualified /cross /placed|5434 /6155 /6155|null /null /null|5001 /null /null|

stochastic 的5001是曾发生的qualified事件，不是保留到终点的有效抬起。5008 AIR/current_lift_valid=true；5016仍AIR但当前有效性已false；5018真实GROUND撤销资格。最终history active_lift.RR=false，旧event tick5001仍保留，不能当作越沿或放置。5046由真实base_link/obstacle pair持续触发 `TASK_FAILURE_BODY_COLLISION`；最后6个物理tick属于未返回决策，由独立manifest保留。精确的当前有效性丢失tick未记录，只知(5008,5016]，不补造。

## 数据口径与历史候选

RMS统一来自完整120 Hz观测的wrapped相邻roll/pitch导数：`sqrt(∫(roll_rate²+pitch_rate²)/2 dt /T)`，不再除T。时长/位移另列。body collider z不是base z；AABB分离是保守下界，不是精确mesh净空。初始轮子低于顶面时gap为负是原始几何值；零分离本身不替代真实碰撞判定。未到达/缺失保留null，未完成窗口不参与完成任务的排名。

只用已保存旧B分析JSON核对，未重扫旧raw：本轮B四腿 lift/cross/placed主要事件、阶段入口与最终8857tick/73.808333s均与旧成功B一致。**没有进行完整逐点轨迹比较，不能宣称两条全轨迹完全相同。** 本轮配对以新B为准，旧成功候选继续保留。

机器可读结果：`CP187904_current_B_comparison.json`；本轮B完整窗口：`currentB_event_windows.json`。全部工作仅CPU离线读取和outputs写入，没有启动Isaac或修改生产/当前续训。
