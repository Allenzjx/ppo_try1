# Block24 episode2：末决策内部建立 RR Q，随后低机身高度 FALL

固定范围：`train/20260911T0005451949265Z_gd4e46006b382_08d59fe9ad444f60b1098f90dbe1fca0`，audit **409–629**，global **162073–162293**，逻辑字节 `[31824117,48923592)`；2026-09-11 00:40:53.864–00:40:54.024 UTC 单次解析。前408行只流式定位、不解析；另读 completed 第3行，未读629之后的活动样本、大物理流或checkpoint。

## 信用和交接

真实教师P01→P06信用起点3584，prefix attempt=2，教师数据不进PPO；此前FR Q23/C1665/P1695、FL Q2417/C3115/P3583全部是继承事实。新增221样本：P06=184、P07=1、P08=2、P09=34；**1768 native ticks全部 verified/actual effect，own-phase=1765，221条无episode内状态写入**。

实际边界5056 P06→P07、5064 P07→P08、5080 P08→P09均done=false；后继交接记录5064/5072/5088的Full12 residual原值完整继承，forbidden/phase-scale clip列表均空，所有mask开放、setter/native mapping校验通过。三个边界RR均GROUND、I/Q/current=false，P09不是已合格抬升入口。

四轮nominal从5048的+.3到5056的0，不能写成全程非零：同tick真实前缘RR=−.213968m、RL=−.209549m，`p06_rolling_retirement.measured_fraction=1 / wheel_gain=0 / bounded_current_geometry_recoverable`，属于已声明的几何退役，不是普通切换清零residual。5056最终wheel仍为 `[−.222842,−.768207,−.410031,+.445533]`rad/s；5080为 `[+.107158,−.438207,−.740031,+.578572]`，连续独立探索保留。

## 新 RR 抬升与未完成任务

| tick | 实际证据 |
|---|---|
|5328|GROUND，front−47.549542mm，台面净空−50.294482mm；I/Q=false。|
|**5335**|新I，P09；事件upward excursion=3.414028mm，全身关节响应12.105640°，command motion=20.580949°。|
|5336|FRONT_WALL且ground=false，地面相对净空3.916796mm，front−46.835568mm；body_control_evidence=true。|
|5344|仍FRONT_WALL/非地面，地面相对净空7.997192mm，台面净空−42.297290mm；I=true/Q=false/body=true。|
|**5345**|新Q，P09；事件upward excursion=8.517693mm、台面净空−41.936669mm，RR自身近期关节运动3.435442°。|
|**5352**|FALL；FRONT_WALL/ground=false，历史established=true，current=false/body=false，reason=`physical_safety_abort`。front−45.294341mm、台面净空−39.165109mm、地面相对净空11.129373mm。|

**I=1、Q=1、回地撤销=0、C=0、P=0。** Q不是早先教师或P06事件，而是最后一个8tick决策5345–5352内部新成立；221个决策端点current有效数为0，不能因此删掉Q。末决策逐tick审计只保存native有效性，不保存每个tick的完整物理/current快照，因此不能声称精确的5345–5351连续有效区间或Q当刻完整支撑力/速度；仅可确定Q时点、末端失效和没有回地撤销。相邻5336/5344/5352是FRONT_WALL；不能将通用事件标签当作Q当刻AIR证明。

生产`semantic_supervisor.py:812–826`要求Q建立时无既有failure、有短窗其他支撑证据及实测净空/动作响应。故5345 Q本身确证这些当时的功能条件，不可用5352失败追溯抹除；它也不是静态稳定或未来carry成功证书。5352是本范围最靠前、净空最高的记录端点，仍未越沿。首个未完成子任务是**继续保持可用净空并越过前缘，继而放置RR**；不是已经成功carry，也不是Q后回地重试。

FL实际情况与历史P分开：5056/5064为TOP、verified bearing分别12.870013N/9.680481N；5080已AIR/0N。Q附近5336/5344/5352全部FL AIR/0N，独立当前支撑是FR TOP与RL GROUND，5344分别10.234786N/14.198482N，5352分别10.684116N/14.048728N，均bearing_verified=true。RR墙接触bearing/load_fraction_valid=false，不能把缺失载荷解释为已承载；其他腿已验证接触仍是独立事实。本回合FL共93个TOP有效承载端点、128个AIR端点。

5336→5344→5352，RR nominal均 `[−6.9,−37.8]`°，projected residual从 `[+.119393,−3.910733]` 经 `[−3.366527,−3.289417]` 到 `[+.633473,−6.763785]`，final target分别 `[−8.030607,−44.210733]`、`[−11.516527,−42.339417]`、`[−7.516527,−45.813785]`。全身和轮子仍在下发；不能从角度正负推导轮端运动或单一失败原因。

## 真实 FALL 与证据限制

completed episode2=221决策、t5352/**44.6s**，`FALL / PHYSICAL_SAFETY / SAFETY_ABORT`，`VALID / CONTACT_BEARING_UNVERIFIED`，full_task_success=false、terminal_bootstrap=false。并非碰障终态或阶段deadline。

终态372维policy观测全部有限、finite_fallback=false。按当前schema尺度与锁定障碍底面z=0回解：base_z≈**12.907103mm < 15mm**；gravity=`[.078928538,.092870235,−.992544889]`，线速度=`[.104540043,−.029121762,−.236497536]`m/s，模长.260207m/s；角速度模长.747609rad/s。符合`isaac_fsm_backend.py:6520`的低高度FALL分支，不是gravity_z>−.30或速度爆炸阈值。高度为锁定平面+float32相对几何推导，不冒充原始双精度base坐标；Q当刻的base高度/速度未独立保存，不能量化其距安全边界余量。终态实测RR约 `[−8.796345,−43.777087]`°，与final target不同，八个实测servo均未越硬限。

保留“确有一次功能Q、随后真实低高度失败”的两件事实。未发现新的可确认执行/交接缺陷，不增加门禁、不修改生产或安全阈值；后续训练与checkpoint累计由主控另行记录。
