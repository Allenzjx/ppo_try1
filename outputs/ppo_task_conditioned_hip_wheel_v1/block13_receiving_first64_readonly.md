# Block13：P12 最早64个学习动作，只读冻结窗口

来源：`train/20260921T1314224958148Z_g97ecd305afb5_728292550a8b4d38a844358c483ac74c`。只读 learner audit 第1–64完整行（tick6184–6688，global196609–196672）；7,256,361字节前缀SHA256记录于同名JSON，不是仍增长文件的全文件hash。仅CPU，无生产修改或物理运行。

真实 successful-N 前缀为772动作/6176ticks，在P12交接；0 PPO credit，前缀未进rollout。RR lift5434、cross/place6155均属于前缀，不能归给PPO。交接RL尚未qualified：首个学习端点6184历史lift=false，且没有此前合格事件；6176的直接接触快照未在已读prefix摘要中，保留null。

## 接续事实

|端点|RR当前状态 / bearing|RR前缘距离|RL当前进展|
|---|---|---:|---|
|6184|TOP / 12.394 N|+14.134 mm|GROUND；未qualified；gap −50.578 mm|
|6232|TOP / 15.529 N|+51.662 mm|短暂AIR，但未qualified|
|6240|AIR / 0 N|+54.433 mm|已回GROUND；仍未qualified|
|6536|TOP / 13.481 N|+20.555 mm|6533真实qualified；AIR，gap −34.234 mm|
|6688|TOP / 3.939 N|+4.751 mm|有顶部反力13.518 N，但front −9.646 mm、withinXY=false，未cross/place|

RR首次失去当前TOP承载的15Hz可定位区间是 **(6232,6240]**，此时为AIR、仍withinXY，不是落地。全64端点RR TOP36、AIR28、GROUND0；期间反复重新TOP，不能把首次AIR描述成永久掉落。历史placed始终true，和当前承载分别报告。FL当前TOP为30/64，不能把其历史placed当持续支撑。

RL初始离地事件6234未形成合格抬升；第二次6531→6533合格。RL最大净空59.685 mm（6568），终点gap −1.628 mm但尚未有效越沿/放置。成功N现成事件摘要的RLqualified6251、cross6658/place6727仅作为事件参考；这里没有跑完，不排名稳定性或宣称完整PPO成功。

6184→6688实际body前送净 **+22.426 mm**，RR轮端x净 **−9.383 mm**，RL轮端x净 **+131.710 mm**；body先前进再回撤（6240→6688约−54.927 mm）。轮端位移不能替代body位移，也不能凭轮速证明牵引来源。

## 同拍四轮通路（顺序FL / FR / RL / RR，canonical rad/s）

|tick|N = mapped N|REQUEST = effective|最终target|实测canonical角速度|
|---|---|---|---|---|
|6184|−1.070 / 0 / 0 / 0|+.120 / −.084 / −.014 / −.015|−.950 / −.084 / −.014 / −.015|−1.042 / −.083 / +.124 / +.077|
|6232|−1.070 / 0 / 0 / 0|−.276 / −.115 / +.142 / −.078|−1.346 / −.115 / +.142 / −.078|−1.363 / −.112 / +.144 / +.084|
|6240|−.300 / −.300 / −.300 / −.300|−.271 / −.104 / +.135 / −.064|−.571 / −.404 / −.165 / −.364|−.513 / −.582 / +.028 / −.364|
|6536|0 / 0 / 0 / 0|−.475 / −.091 / +.320 / −.144|−.475 / −.091 / +.320 / −.144|−.466 / −.091 / +.320 / −.145|
|6688|0 / 0 / 0 / 0|−.239 / −.033 / +.070 / −.159|−.239 / −.033 / +.070 / −.159|−.239 / −.172 / +.198 / −.192|

N本身的变化：6184 `[−1.07,0,0,0]`；6240四轮−.3；6504四轮0；6552四轮+.3；6648四轮0。因此6240的负向目标同时含源N与负向residual，不能全部称为policy倒转。RR学习REQUEST64/64为负（均值−.11716 rad/s）；N为0时，负RR目标确实来自同拍residual，不是输出丢失。RL的正residual也会部分抵消负N。这些是作用链事实，不是对body回撤的唯一因果证明。

四轮最终target逐端点等于mapped N+effective（最大误差0）；12维mask全1，全部120Hz派发核验通过。最后写入读回源为 `robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim`。final_stop_owner全窗未取得所有权；完整逐通道source stop/owner账本和原生实测joint qd未在此日志提供，记null；JSON保留实际native target轴符号，不能与canonical速度混用。

冻结终点6688：P12、terminal=false、物理evaluator valid=true/termination=null。此报告不涉及后续结果，不把短窗资格或顶部反力称为后腿越障成功。
