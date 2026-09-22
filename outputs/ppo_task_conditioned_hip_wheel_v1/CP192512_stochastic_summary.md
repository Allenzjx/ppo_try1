# CP192512 stochastic：真实机身碰撞，RR资格曾建立后撤销

封存run `57f370bd94e9499e8ab7a10b18115e10`在**5775 tick/48.125 s、P09**发生独立物理判定`TASK_FAILURE_BODY_COLLISION`，不是时间预算或单纯控制器停滞。最后部分动作的7个物理tick保留，以manifest精确终态为准。固定CP随机评估（物理seed4001、policy seed4101、training_style_conditional_gaussian），optimizer updates=0。

FR qualified21/cross1647/placed1661；FL qualified1738/cross2503/**placed2986**，捕获是真实事件。FL越沿→捕获耗时4.025 s；但捕获至P07入口4584的1599个120Hz观测中，FL障碍接触105、AIR1494。当前TOP+verified bearing+support仅13/200保存端点（6.5%），其余187端点AIR，最大gap51.352 mm；不能用历史placed说持续承载。窗口被标COMPLETE仅表示观察到P07，不代表保持成功。没有再扫描raw去推补“首次AIR精确tick”，该值留null。

P06为2992→4584，13.2667 s，**实测base净前送258.109 mm**。rate RMS .046197 rad/s、峰值倾角.122201 rad；body collider最低世界z123.479 mm，保守AABB分离下界73.479 mm。轮端位置/转速不能替代这个base位移，也不能从位移证明轮驱牵引占比。该窗口内FL原始AIR1494/1593，非全程有效接触。

RR实际顺序为：4854初始free-air上升 → **4861 qualified**（40.5083 s）→ **4936回地并撤销资格**（41.1333 s，before-cross）→ 5775机身碰撞。RR从未cross/placed；准备窗4584→5775共9.925 s未完成，其中仅9/150保存端点current_lift_valid=true。终点RR虽又AIR两帧，但当前free-air参考5773、升高仅.254 mm、current_lift_valid=false、active_attempt=false，不能把这次近地AIR或旧4861时间戳当有效后腿完成。

工具自动派生的第一未完成字段`RR:active_lift`取自**被撤销后的当前历史布尔**，不表示RR从未合格。实际第一未完成任务是维持RR有效抬起并完成越沿/受控放置。整个RR准备窗1192个raw观测中FL全部AIR，不能虚构CoM运动已由FL承载。该窗body collider最低49.996 mm、AABB下界到0，rate RMS .159362 rad/s、峰值倾角.498626 rad；终止依据是真实物理碰撞判定，不是单独拿AABB或高度阈值判失败。不将这条未完成窗口排为稳定性优胜，也不以48.125 s失败称更快完成。

完整指标为`CP192512_stochastic_event_windows.json`，精简事件/当前资格摘要为`CP192512_stochastic_summary.json`。没有重扫旧run或额外raw扫描、没有改生产/评分，没有运行新物理或优化。
