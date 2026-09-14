# Block25 episode0：自然P01创造 RR 抬升入口，连续接管后仍未越沿

固定范围：`train/20260911T0053577123642Z_gd4e46006b382_6f25b311ef674a58984702fd575ad5d4`，audit前349行，global **162689–163037**，逻辑字节 `[0,26182239)`；2026-09-11 01:03:39.191–01:03:39.360 UTC 单次解析，另读completed首行。未读后续回合、大物理流或CP，未修改生产。主控已核实普通恢复、无迁移/教师；本次首策略P01的0→8、初始事件历史为空，整349条均为自然轨迹策略信用，不是教师后缀。

## 实际阶段与新前腿事件

来源阶段样本：**P01=2、P02=136、P03=9、P04=2、P05=39、P06=96、P07=1、P08=5、P09=59**。实际边界tick依次16、1104、1176、1192、1504、2272、2280、2320，普通切换均非done。348×8+末6=**2790 physics ticks**，全部native verified/actual effect，own-phase=2782；349条无episode内状态写入、Full12 mask全部开放、setter/native mapping全部通过。

- FR：I13(P01)、Q28(P02)、C1105/P1172(P03)。
- FL：I1227/Q1234/C1425/P1502，全部P05；均是本回合新策略事件，无教师信用。
- 1504 P05→P06时FL确实TOP、verified承载5.712152N，FR TOP7.038527N、RL GROUND7.648547N；1536已出现FL AIR端点。历史P不保证后续持续承载。

## RR：先由前驱创造入口，再连续学习，未把回地写成继承

| 尝试 | 实际事件 | current有效决策端点 |
|---|---|---|
|初始探查|I2223(P06)，2240端点回地，未Q|0；不是Q撤销。|
|第一Q|I2257/Q2265(P06) → 回地撤销2285(P08)|2272(P06)、2280(P07)，共2条。|
|第二Q|I2312/Q2318(P08) → 回地撤销2544(P09)|2320–2536每8tick一条，共28条。|
|第三Q|I2586/Q2593(P09) → 回地撤销2616(P09)|2600/2608，共2条。|

直接计数 **I=4、Q=3、真实回地撤销=3、current有效端点=32、C/P=0**。每个回地的两个资格/current标签只计一个物理事件；`event_ticks.active_lift.RR=2265`保留首次Q，不代表后两次Q未发生。末active_lift=false不能抹除这些真实事件。

关键连续链：Q2265在2272 P06→P07和2280 P07→P08仍current=true，2285确有地面接触才撤销；不是切阶段清历史。重新Q2318后，2320 P08→P09仍AIR/current=true/地面相对净空11.103717mm，2328接管后为16.485022mm，同一次尝试保持到2536端点。并没有要求RR先回地再抬。

这些事件包含全身实测响应：I2223的全身关节运动32.462945°，I2312为30.048022°；第一/第二Q事件自身近期关节运动为10.012233°/6.471666°。P06及P08的RR nominal仍`[0,0]`，实际residual/final已有非零变化；不能把入口归为源P09已经替策略完成抬升，也不能据此推定单一腿动作因果。

## 净空、前送和真实支撑

最高记录端点 **2400**：RR AIR/current=true、地面相对净空 **234.883637mm**、台面净空+184.889983mm，但front仍 **−265.512254mm**，没有越沿。之后2536仍current=true但净空降至28.664207mm、front−164.908316mm，2544实际回地。合格抬升期间最靠前是第三次的2600，front−137.875524mm、净空11.689242mm；全回合最靠前2776则已经GROUND/front−91.023323mm，不能冒充空中carry成功。所有front都未过零，C/P一直未完成。

32个current有效端点均AIR且body_control_evidence=true，独立FR TOP与RL GROUND支撑均verified。FL在这些端点中12次TOP、19次AIR、1次OBSTACLE_AMBIGUOUS/承载未知；不是固定FL承载组合。最高净空2400，FL TOP3.830380N、FR TOP11.928166N、RL GROUND14.129181N均verified；P08→P09的2320则FL AIR/0N，FR/RL分别12.493589N/13.686137N。短窗body-control证据与大净空都不是完整稳定性保证，不能用后来的碰撞追溯抹Q，也不能凭抬得高宣布稳定改善。

## 动作继承与wheel变化

八个普通交接记录均`previous_projected_residual_full12 == carried_projected_residual_full12`，没有forbidden/phase-scale丢弃。P05→P06前后1496/1504/1512四轮nominal均+.3，最终wheel仍独立非零。

2320端点的nominal `[+.3,−.63,+.3,+.3]`不是无条件清零：已有P07源0.333333s明确请求FR wheel−.63，其未结束owner继续运行；2328 P09在当前有效AIR且远离前缘的条件下建议四轮+.3。2536/2544的FR nominal为0，与P07源1.333333s停轮及当前近前缘、低于台面时“不增加盲目wall push”的条件一致；final FR wheel仍约−.708/−.711rad/s，residual未被清除。这里只核对已记录端点和明确源请求，不杜撰逐物理tick动作曲线。

RR final target也持续改变：2320 `[1.037460,5.452781]`、2328 `[7.387460,3.498740]`、2400 `[54.470617,1.044635]`、2536 `[20.867971,−48.794652]`°。实际净空/接触是运动事实，不能把负膝角直接命名为“时钟强制下降”或单因归责。

## 原始失败保留

349条末决策执行6tick至**2790/23.25s，P09 BODY_COLLISION / BODY_CONTACT**；物理评价`TASK_FAILURE_BODY_COLLISION`、`VALID / VERIFIED`，reason为central body/obstacle collision。没有成功，非deadline；terminal_bootstrap=false。聚合证据没有具体body collider/pair、冲量或signed BODY distance，不由姿态补造。

终态RR AIR但仅0.403366mm地面相对净空、front−91.972854mm、Q/current/C/P=false；这是三次回地后的末端状态，不是新合格抬升。FL AIR/0N，FR TOP10.395915N、RL GROUND14.258751N仍verified；有腿支撑并不排除机身碰障。首未完成任务是RR再次取得可用抬升、保持净空完成前送越沿，随后受控放置。

结论：这是自然P01 stochastic训练中真实创造并继承后腿入口的证据，同时仍是完整任务失败；不是确定性全程成功或稳定性改善保证。未发现新的明确dispatch/继承缺陷，不设置新训练门禁，不改安全标准。
