# Block11 P04-origin 首回合：FL真实放置，RR仅初始离地，P06高度FALL

仅一次读取同run `train/20260910T1326149079057Z_g7db0d17f398d_ff94228e94e648f5b9fe4b59228b03f1` 的首条completed及截至首terminal的244条audit；不读下一前缀、physical大流或checkpoint。固定global148353–148596，原始18,219,679 bytes，SHA256 `5a2d55e084ac67ab80a914855ef18d62c278c33ce0d0819399446c64c541d8d6`。

实际credit start为P04 tick1696/14.133333 s、teacher_initialized_suffix、from_P01_current_policy=false、prefix_teacher_data_in_ppo_storage=false。策略执行1947 ticks/16.225 s，终态P06 tick3643/30.358333 s，FALL、task_success=false。**新策略样本P04=1/P05=147/P06=96，共244；P07–P13=0。** 30.358333 s包含教师前缀，不是全部策略执行时间。

## 任务事件与承载变化

FL在P05实际产生I1753/Q1757，2068回地撤销；随后I2104/Q2110，**C2544（21.2 s）/P2876（23.966667 s）**。tick2880/24 s由placed_FL=1进入P06。新FL I2/Q2/C1/P1；history首次Q时间1757不替代第二尝试Q2110。FR Q23/C1665/P1695均早于1696信用起点，属于教师历史，不能计新policy成绩。

| 真实状态 | FL | FR | RL | RR |
|---|---|---|---|---|
| P06入口tick2880 | TOP，0.924260 N，load0.032255 | TOP，11.514398 N | GROUND，12.889976 N | GROUND，3.326562 N |
| FALL末态tick3643 | TOP，11.387127 N，load0.299654 | TOP，12.836428 N | GROUND，10.684062 N | GROUND，3.093318 N |

表中接触、bearing和load均验证有效，四腿在这两个端点support=true；FL入口确有顶面接触，但承载很轻，不能写成主承载。**P06的96个决策末态中FL只有27个TOP/support，69个AIR**，载荷未知数为0。终态重新承载不能反推它一直承载；此前悬空时不能用placed history虚构支撑。终态FL距前缘+359.931 mm、台面净空-0.315 mm，真实TOP与历史P一致，但支撑腿数量不保证身体高度安全，也不能据此推造CoM迁移方向。

RR仅有新 **I3044/25.366667 s/P06/global148521**：excursion3.167898 mm，全身实测关节运动15.606008°，命令运动29.451881°。无Q、C、P，current_lift_valid决策末态数为0；不能把这一初始响应称为已建立可控抬升或carry。末态RR已GROUND，距前缘-339.264 mm、台面净空-50.768 mm，initial/established/current valid均false。首个未完成阶段任务仍为P06后腿靠近/准备（rear_approach=0.494191），没有到达或完成RR越沿放置。

## FALL与更新边界

终态原记录source=PHYSICAL_SAFETY、physical VERIFIED/valid=true、reason=`fall or physics explosion`。policy/critic均372维且相同、finite_fallback=false；按既有schema固定scale/offset，relative bottom=-0.014894172549 m，locked bottom=0，回解 **base_z=0.014894173 m<0.015 m**。gravity_z=-0.971702635，世界线速度[0.490082,0.040396,-0.181629] m/s、角速度[-0.957829,-0.821210,0.093472] rad/s，均未达到5/20模长阈值；8个实测关节位置在各自硬限内。此为保存的实测float32观测回解，不声称读取了原始双精度位置。低高度分支支持原FALL；即使末态四腿接触，也不能改成成功或放宽高度阈值。

1947/1947 native-effect tick verified、own-phase1945，2个普通交接hold，禁止的episode内state write=0，未发现新的下发/普通阶段重置缺陷。按主控本次派发快照，首128条已更新并保存1125/global148480；另116条虽已真实采集、终态已持久化，但尚未进入该checkpoint，不能提前计入已优化/已保存信用。该轨迹跨一次更新，不是固定策略评估，更不是自然P01全程成功。

仅新增本报告；无生产、配置、主报告、checkpoint或进程更改。
