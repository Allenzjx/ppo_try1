# Block07：RR真实放置后失去顶部接触，截至7208

只读run `12b59026033c4ad08c274c684926dadc` 已写完的端点，不推断后续结果。这是N真实前缀到5456、随后PPO的课程训练回合，不是自然P01固定checkpoint成功。窗口含actor update1450（global190080 /tick6480），不能按一个冻结策略作排名。

|事件/保存端点|RR真实当前证据|RL状态|
|---|---|---|
|6311 /6403|历史真实越沿 /首次placed；准确tick来自本回合事件日志|尚未越沿/放置|
|6408|TOP、verified bearing .331 N、support=true、within_top_xy=true；gap−.952 mm|GROUND|
|6428|RR尚在TOP|RL取得qualified；6497接地撤销|
|6568 /6626|RR继续TOP接触|RL再次qualified /再次接地撤销|
|6773|RR已接近前缘、仍有TOP接触|RL第三次qualified；6821接地撤销|
|6784|最后保存的TOP端点，bearing9.384 N；已outside top XY，gap−13.701 mm|AIR，尚未越沿|
|6792–6800|OBSTACLE_AMBIGUOUS、bearing_verified=false、support=false；不是AIR也不是TOP|AIR|
|6808|精确`current_lift_revoked_ground`事件；GROUND、bearing77.589 N、gap−50.009 mm|随后6821接地撤销资格|
|7208|GROUND、support=true但TOP=false、current_lift_valid=false；gap−50.573 mm、front−68.368 mm、bearing4.755 N|GROUND+FRONT_WALL，active_attempt=false，未cross/placed|

仅有15Hz evaluator端点，**退出TOP的精确物理tick未知，限定(6784,6792]**；6808接地撤销则有准确事件。6408–7208共101端点：RR TOP48、ambiguous2、GROUND51。RR current_lift_valid在6424已因body_control_evidence=false暂时失效、之后又恢复，最后true端点6704；它不是“是否当前仍在TOP接触”的同义字段。

RL曾有三次qualified，不代表目前保持抬升；准确撤销6497/6626/6821均保存，末history.active_lift.RL=false，旧event tick6428仍保留。窗口RL最高gap23.164 mm，但没有cross或placed，不能称carry完成。

## 机身与轮端不是同一位移

准确6403时的base端点未保存，故使用**首个放置后保存端点6408→7208**：

|实测量|净世界前向位移mm|
|---|---:|
|base_x−固定obstacle_front的差|−133.687|
|RR轮心front_distance差|−113.511|
|RL轮心front_distance差|+52.873|
|FR轮心front_distance差|−127.221|
|FL轮心front_distance差|−94.403|

body值直接由evaluator的实测base_position[0]−front计算，**未用FL变化推算body**。RL轮端向前并不阻止机身向后，二者反映关节构型与机身运动叠加。

RR wheel最终canonical target为负98/101端点，实测轴角速度为负93/101；所有检查端点native派发/映射verified。例6408 target/actual=−.0651/−.0349 rad/s，6480=−.1801/−.5435，6800=−.0923/+2.2853，7208=−.1382/−.1674。存在实际跟踪/载荷变化，不能仅凭转速符号宣称轮驱是后退的唯一原因；这里只证明机身后退、RR后退/滑回及失去接触是同窗真实响应。

## 判定语义核查

这批证据首先支持**放置后的接触保持丢失**，没有发现将末段GROUND错误判为TOP或成功的迹象。源码`placed_on_top`字段直接引用历史`history.placed`，因此它在7208仍true不表示当前顶部承载；当前TOP=false、ground=true、current_lift_valid=false及6808撤销事件是相容的。历史placed不因回落而抹去；末physical success=false、final_region_valid=false、final_support_available=false、traversal_task_complete=false，物理termination=null。截至此窗口没有新任务成功或碰撞结论。

训练流没有该段独立120Hz原始接触/四髋geometry文件，未补造缺失tick、安装点或滑移因果。未修改评分/生产/运行，不阻挡正式训练；小JSON保留边界、关键证据与口径。
