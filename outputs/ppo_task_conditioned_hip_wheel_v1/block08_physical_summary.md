# Block08：两个碰撞回合与一个预算尾

源 run：`20260921T0916266524903Z_gee5a9651591d_87559dc9adb340abb997ffa03fc0da89`。本块1536 decisions、12次更新（1454–1465）、240 optimizer steps，保存CP192000。下列均为边采样边更新的训练轨迹，不是固定checkpoint评估或稳定性排名；首回合复核与此前有界结果一致。

|日志回合|decisions／时长|FR qualified→cross→placed|FL qualified→cross→placed|RR qualified→回地撤销→碰撞|
|---|---|---|---|---|
|0|612／40.7500 s|29→1710→1722|1760→2550→2917|4755→4770→4890|
|1|644／42.9167 s|26→1889→1906|1946→2790→3110|4843→4882→5150|
|2，预算尾|280／18.6667 s|29→1847→1864|1939→null→null|未到达／null|

时间均为各回合physics tick（120 Hz）。前两回合独立physical evaluator均valid=true、`TASK_FAILURE_BODY_COLLISION`，首个未完成任务为RR保持有效抬起并越沿/放置；两者RR均从未cross/placed。保存的RR current_lift_valid端点分别只有4760、4768，以及4848–4880每8tick五点；回地后资格撤销，终态的近地AIR回弹仍invalid，不能拿旧qualified事件认成功。

## P06：实际前送不等于FL保持，也不证明轮驱牵引占比

|回合|P06进入→P07 tick／时长|实际base净前送|FL有效TOP支撑／AIR端点|首个已存AIR／最大gap|body collider最低z／AABB分离下界|
|---|---|---|---|---|---|
|0|2920→4456／12.8 s|+268.023 mm|31/193（16.06%）／162|2984／40.266 mm|121.260／71.260 mm|
|1|3112→4552／12.0 s|+265.998 mm|26/181（14.36%）／155|3224／33.974 mm|120.043／70.043 mm|

支撑需当前TOP、verified bearing及support，不用历史placed替代；这些是约15 Hz端点占比，不是连续支撑时长。base进展使用实测base相对固定障碍前沿，不用FL轮端距离或平均轮速推算。几何最低值来自120 Hz已写奖励审计；collider z不是base z，AABB下界不是精确mesh净空。

第二回合RR qualified4843后，body collider首次低于100/80/60/50 mm分别在4858/4876/4895/5149；4882先回地撤销，5150才碰撞。第一回合对应4758/4775/4793/4889、4770撤销、4890碰撞。这里的高度仅为离线描述标线，不是新增硬限，也不单独证明某轮/关节是因果责任。

## 实际质量扣分与覆盖

|回合|front真实样本／成本|geometry有效eligible样本／成本|terminal事件奖励|
|---|---|---|---|
|0|1695／0.0172914|4850／0.0170686|−40|
|1|1879／0.0192233|5110／0.0389287|−40|
|2|1831／0.0188723|2200／0|0|

成本为已实际加入奖励的累计扣分绝对值，不是未加权诊断量。本块front=0.0553870、geometry=0.0559974，合计body_stability贡献−0.1113844；geometry扣分全出现在P09，其他eligible阶段实测分离满足20 mm margin，所以是已测零成本而非补缺失为0。12160个eligible geometry样本均valid，另120个P03/P04样本不eligible，无terminal测量省略。其余质量families本块实际贡献0。

实际policy request覆盖：P01–P09依次6、670、12、3、343、372、2、2、126；P10–P13均0。后腿未放置，不能把前腿/几何样本充作后腿成功样本。

首个−40已进入update1458（global191076，terminal总reward−42.3888056，禁止bootstrap）；第二个−40在global191720，terminal总reward−42.4461997，同样禁止bootstrap。第三回合末端tick2240/P05，physical valid=true、termination=null、success=false、time_outs=false、bootstrap=true：仅预算截断。FL此时AIR、gap24.974 mm、尚未越沿/放置，P06及RR窗口为null，不追加未观察到的终止罚分。

仅离线读取本run及receipt，生成本报告与JSON；没有改生产、诊断脚本或启动Isaac。精确连续接触保持、完整COM/髋安装点数据本训练日志不足，未补造。
