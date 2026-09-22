# CP189952 stochastic：封存物理事件窗口

run `b66847e1dd544fdba49da0c6bb528346` 在5087 tick /42.391667 s、P09发生真实 `TASK_FAILURE_BODY_COLLISION`。FR placed1684、FL placed2988；**RR从未取得qualified，未越沿/放置**。训练式conditional Gaussian、policy seed4101、物理seed4001、评估optimizer updates=0；不是混合训练轨迹或独立−3干预。

以下对照同版本当前B、CP187904 stochastic及本次CP189952 stochastic，每种只有一条轨迹，不是成功率统计。JSON由既有事件工具提取，含全部窗口/原始单位；只读取新C已封存raw，B与旧C复用既有结果。

## FL真实捕获不等于持续承载；P06有真实推进

|物理事件窗口/指标|当前B|CP187904 stoch|CP189952 stoch|
|---|---:|---:|---:|
|FR P01→捕获时长s|12.5167|13.4250|14.0333|
|FR rate RMS rad/s|.117889|.105738|.113457|
|FR collider最低z mm|91.272|89.727|88.051|
|FL越沿→捕获 tick|2468→2677|2726→2790|2488→2988|
|同捕获窗时长s|1.7417|.5333|4.1667|
|同捕获窗rate RMS rad/s|.016297|.042260|.038443|
|FL捕获→P07：TOP且verified-bearing端点/样本|311/311|37/240|7/192|
|同保持窗AIR端点/样本|0/311|203/240|185/192|
|同保持窗120Hz障碍接触/AIR观测数|2484/0|272/1643|54/1479|
|同保持窗FL最大gap mm|.014|27.736|43.322|
|P06入口→P07 tick|2680→5160|2792→4704|2992→4520|
|P06时长s|20.6667|15.9333|12.7333|
|P06 body净前送mm|307.597|295.279|243.544|
|P06 rate RMS rad/s|.010536|.047068|.046794|
|P06峰值合成倾角rad|.052611|.089218|.108270|
|P06 collider最低z mm|125.122|118.546|122.201|
|P06 body/障碍AABB分离下界最小mm|81.027|68.546|72.201|

本次FL捕获是真实事件，但3042 tick已再次AIR；捕获后至P07的1533个120Hz观测有1479个AIR，TOP-bearing端点也只有7/192。不能用历史placed代替当前支撑。表中稀疏evaluator端点不转换成连续承载秒数，raw障碍接触也不冒充逐tick TOP承载判定。保持窗口标记COMPLETE只表示观察到了P07这个窗口终点，不表示FL成功保持。

本次P06比旧C短3.2 s，同时净前送少51.735 mm；起点/构型不同，不能称同距离更快。P06 rate RMS仅略低，峰值倾角反而增加，虽最小几何空间较旧C多3.655 mm，仍低于B。FR也比旧C慢.6083 s、RMS更高、最低collider低1.675 mm；不能据局部角速率宣传任务或稳定性提升。轮轴旋转、轮端位移和body净前送是不同量，本分析不识别主牵引来源。

## RR未qualified与实际碰撞时序

|P07准备→RR放置/实际终止|当前B|CP187904 stoch|CP189952 stoch|
|---|---:|---:|---:|
|窗口tick|5160→6155|4704→5046|4520→5087|
|结果|完成RR放置|未完成/碰撞|未完成/碰撞|
|观测时长s|8.2917|2.8500|4.7250|
|rate RMS rad/s|.052065|.251545|.213251|
|峰值倾角rad|.210474|.429490|.502318|
|collider最低z mm|86.544|49.934|49.985|
|AABB分离下界最小mm|36.544|0|0|
|RR最大轮底相对顶面gap mm|85.442|−40.514|−49.790|
|RR qualified/cross/placed|5434/6155/6155|5001（后撤销）/null/null|null/null/null|

本次RR history无initial-clearance/qualified事件；RR当前有效资格在全部72个保存端点均false。6个非终态AIR端点（4600、4784、4792、4800、4808、4816）也全未qualified，不把轻微离地称为有效抬起。终点另有真实AIR观测，最近接触参考tick5085、连续AIR2样本、当前升高仅.206 mm，current_lift_valid=false、body_control_evidence=false；FL在整个568个RR准备窗raw观测均AIR，不能声称FL已承载或接收CoM载荷。

机身collider最低点在P07入口4520为124.976 mm；其后首次低于120/100/80/60/50 mm的tick分别4583/4815/4832/4851/5086。最后5087确认base_link/Obstacle真实pair、连续接触2tick、当拍法向力9.059 N并终止；不是由AABB零距离单独宣判碰撞。最后决策5080→5087只步进7tick且environment_step_returned=false；既有工具使用独立manifest保存的准确终态，没有丢掉这7tick或用5080替代。

新旧C的RR窗口都未完成，不以本次较低RMS排稳定性优胜；本次峰值倾角更大且连qualified也未出现。B完整成功仍是73.808333 s，新C不能因42.39 s失败而称更快完成。

口径：RMS为完整120Hz wrapped roll/pitch相邻导数的时间均方根(rad/s)，没有再除时长；collider z不是base z，AABB是保守分离下界而非精确mesh净空。缺失/未到达为null。全部为只读CPU分析及outputs交付，未启动Isaac、未改生产或当前诊断logger。
