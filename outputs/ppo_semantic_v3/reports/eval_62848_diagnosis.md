# C62848 固定 P01 评估：RR 真放置后退回，P11 的 RL 接近/减载未完成

2026-09-06。只读 PowerShell 核验已结束评估；未运行 Python、Isaac、optimizer、测试或修改生产。仅新增本报告。

## 结果与证据身份

评估运行：`runs/ppo_semantic_v3/validation/20260906T1440514266953Z_g68631e932c7d_945a0a5ecd3243c9a82d56baf14c968b`。

固定 checkpoint `checkpoint_step_000062848.pt`，manifest记录 checkpoint SHA256 `e5c91c4f443c4fcc05671eb4d5ac517d33b6b65449c08b2be9d55bd9cef0a68b`；本次读取该记录，不冒充重新hash/tensor load。runtime `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`，seed2001，deterministic policy，从真实P01起点执行，**optimizer_updates_during_evaluation=0**。

根代理报告进程exit0；本次独立读取run manifest lifecycle=`SUCCEEDED`，含义是执行正常完成。实际任务结果为 **993 decisions / 7,944 ticks / 66.2s / P11 `INCOMPLETE_CONTROLLER_BLOCKED`**，task_success=false、window_ended_before_task_terminal=false。physical evaluator valid=true、physical failure/termination reason=null。不是任务成功，也不是软件异常或碰撞中止。

读取：evaluation/run manifests、全部993条decision audit、stage transition evidence，并从120Hz physical observations/native audit抽取限定事件邻域。几何/接触表使用当前真实raw字段，不重新运行历史控制器。

## 实际阶段与第一个未完成任务

| 请求阶段 | 实际决策数 | 转出tick / task time |
|---|---:|---|
| P01 | 1 | 8 / .066667s |
| P02 | 174 | 1,400 / 11.666667s |
| P03 | 3 | 1,424 / 11.866667s |
| P04 | 1 | 1,432 / 11.933333s |
| P05 | 150 | 2,632 / 21.933333s |
| P06 | 246 | 4,600 / 38.333333s |
| P07 | 1 | 4,608 / 38.4s |
| P08 | 1 | 4,616 / 38.466667s |
| P09 | 114 | 5,528 / 46.066667s |
| P10 | 2 | 5,544 / 46.2s |
| P11 | 300 | 没有转出；7,944 / 66.2s终止 |
| P12/P13 | 0/0 | 未进入 |

FR Q/C/P=48/1,411/1,424；FL=1,509/2,487/2,632；RR=**5,276/5,521/5,528**；RL无qualified/cross/placed。它们是本次固定P01策略实际执行中的历史，不是teacher suffix输入。

P11入口在46.2s，终止stage_age恰20s。当前配置P11需同时 `workspace_RL` 和 `load_ready_RL`，不是复现旧关节入口或固定支撑组。真实转入时entry.valid=true，physical_valid、FR/FL/RR placed均满足；此后没有旧入口拒绝或body collision。

## RR 合法过程与其后的真实退回

raw轮字段为 `rear_right_ankle`，XY使用当前center，净空使用当前bottom减实测top；contact分别读其body的ground/obstacle exact pair。下表的force为既有pair normal_force_n标量，不宣称竖直支撑分量或接触点法向。

| 实际120Hz tick / time | RR front mm | RR clearance mm | 当前接触与事件 |
|---|---:|---:|---|
| 5,276 / 43.966667s | −27.941 | +.01827 | ground/obstacle均false，实测AIR；qualified事件 |
| 5,521 / 46.008333s | +.530 | +12.456 | 仍AIR、两pair force0；真实front crossing |
| 5,527 / 46.058333s | +1.650 | −.341 | obstacle active，force19.1555N；ground false |
| 5,528 / 46.066667s | +2.060 | −.432 | obstacle active，force14.0502N；合法placed事件并转P10 |
| 5,536 / 46.133333s | −.663 | −.023 | 首次退回center front plane后方；obstacle仍active，force3.3031N |
| 5,544 / 46.2s | −6.583 | −2.035 | obstacle active，force4.2842N；转P11 |
| 5,711 / 47.591667s | −97.065 | −50.156 | 放置后首次GROUND active，force1.1212N；obstacle false |
| 5,735 / 47.791667s | −97.360 | −50.129 | ground/obstacle均inactive，重新AIR；不是above-top lift |
| 7,944 / 66.2s | −255.710 | −45.453 | AIR、两pair force0、load0；当前已不在平台区域 |

为避免用15Hz摘要猜“第一次”，本次连续检查raw ticks5,529–5,711（183条），确认首次GROUND为5,711；检查5,529–5,544确认首次退回front为5,536；在回地邻域逐tick确认再AIR为5,735。后续存在接触变化，不把5,735→7,944称作一整段连续AIR；末帧自己的consecutive_air_samples=370。

退回后的 `top_contact=false` 与 obstacle pair仍active可以并存：当前TOP/loaded还要求几何/front条件，body-pair接触本身不保证平台放置。不能把5,536的“不是当前TOP”倒推成5,528从未真实placed。

## 当前capture retention确实下降，历史没有被清空

本版本仅对已placed腿把原1改为 `.8+.2*retention`；`retention=min(clip(1−outside/.25), .008/(.008+max(0,−.015−clearance)))`。outside来自现有5mm扩张平台矩形的真实欧氏外距。它不是新hard gate，不要求AIR腿必须接触平台。

| 决策末tick | RR历史placed | 当前retention | 说明 |
|---|---|---:|---|
| 5,528 | true | 1 | 在平台XY/净空范围内 |
| 5,536 | true | 1 | center略退到front后，但仍在既有5mm测量容差内 |
| 5,544 | true | .993669543 | XY外距1.582614mm |
| 5,712 | true | .185398719 | 已回地，主要由当前低净空限制 |
| 5,736 | true | .185491953 | 虽AIR但仍低且在前方区域外，不能因AIR恢复满信用 |
| 7,944 | true | 0 | outside=.250709675m，已越过原.25m衰减范围 |

末帧RR仍保留Q/C/P历史，current capture份额已经耗尽；隔离RR这一项相对retention=1的global phi差为 **−.85/4*.2=−.0425**。当前总phi=.614693255，另受RL workspace/unload等影响，不能把整个轨迹phi下降或物理退回全归因于一个项。该修订确实在记录中感知退回，但“有负向进度信号”不等于已学会保持。

## P10→P11 入口与动作继承

P10的两个决策末：

| tick | RL front mm | workspace_RL | 当前其它支撑 / RL load |
|---|---:|---:|---|
| 5,528（进入P10） | −227.263 | .970947880 | support_RL=1；RL load .284357 |
| 5,536（P10首决策） | −223.311 | .986757893 | support_RL=1；RL load .344919 |
| 5,544（P10第二决策，进入P11） | −219.762 | 1 | support_RL=1；RL load .316350 |

所以P10确实短时把RL带到workspace，但P11此时 `load_ready_RL=.854562134<1`。同时RR已退到front−6.583mm，历史placed满足下阶段顺序，不代表当前RR仍有效TOP承载。没有用旧入口强制恢复现场状态。

真实native tick记录：5,544(P10)→5,545(P11首tick)的nominal全12维保持相同，projected residual仅浮点约1e−15差；没有forbidden channel或phase cap裁剪。bridge记录的最大logical servo/wheel action jump分别3.55e−15deg /2.78e−17rad/s。

- 这两个tick的native wheel target完全相同：`[−.188020989,−.170617312,+.286132455,+.009437785]` rad/s，负号属于冻结native轴映射。
- native servo仍可沿已有mapper/final slew继续：例如RR hip在这两个tick从 .313893944 到 .292077333rad，约1.25°。logical交接保持不等于把所有物理执行器冻结或reset mapper。
- 下一tick5,546的P11 raw请求已改变残差，native wheels变为 `[−.185471743,−.171826884,+.287432611,+.012269540]`。第一P11决策8ticks全部native verified/effect，其中own-phase request effect=7，只有交接首tick排除，**不是整个决策丢raw**。
- 名义继承仍在执行：RR knee建议在5,528/5,536/5,544为−37.8/−34.6/−29.3deg，5,545保持−29.3，5,546→5,552继续−28.05→−27.2；FR hip在P11依次0→1.25→3.7deg，并于5,560为5.8deg。这是有限层/既有slew继续，没有恢复旧固定入口；不能仅凭joint sign把它解释成实际末端下降。

全评估native verified **7,944/7,944**，own-phase request effect **7,934**（10次普通交接首tick排除）；993行no-in-episode-state-writes验证均true。以上不能独立重建所有隐藏filter/mapper缓存，但实际tick连续、桥接和native目标均无reset/全通道清零证据。

## P11最终缺口：workspace与减载，不是碰撞

P11的300个决策末中，workspace_RL达1的有8行，load_ready_RL达1的为0，两者同时达标为0。RL最近的决策末front为−212.814mm，说明曾在workspace内；不是“P11从来没有接近过”。统计为15Hz边界，不冒充每个subtick从未瞬时满足。

终止当前状态：

- RL：GROUND，front **−397.662770mm**，横向有效，clearance−50.443178mm，load **.490085993**，无Q/C/P。
- RR：AIR、front **−255.709675mm**、clearance **−45.452739mm**、load0，Q/C/P只是已完成历史。
- FR/FL当前TOP，载荷约 .501039555 / .008874451；不能把四腿历史和当前承载混成“全部仍在平台”。
- `workspace_RL=1−(.397662770−.22)/.25=.289348919`。
- 当前其它支撑条件允许计算减载，但 `load_ready_RL=(1−.490085993)/(1−.20)=.637392508<1`。
- stage_age=20s、physical valid=true、physical failure=null，因此按既有P11期限真实incomplete。没有进入P12的RL lift/cross，更未进入P13 stop。

末帧nominal四轮为零，实际canonical轮target仍 `[+.161140690,−.157424704,−.255746415,+.032695180]` rad/s；native为对应冻结映射 `[−.161140695,−.157424703,+.255746424,+.032695182]`。策略仍在动作，不能把不足归因为PPO被关闭；这些target也不是实际车体位移或因果结论。

## 结论边界

本次确有RR真实qualified→cross→placed，随后当前放置区域/承载丢失；RL在P11未能同时维持workspace与减载。当前capture shaping已区分历史完成和当前退回，但尚不能据此声称已产生稳定行为改善。

到达P11比某些旧评估阶段更后，只是本次轨迹覆盖事实。没有paired稳定性指标/重复成功证据，不能仅凭phase编号称优于C54656或FSM，更不能将正常exit0写成完整任务成功。保持下一既定训练，不修改配置/参数，不新增入口或成功门禁。

只新增本报告；写完停止。
