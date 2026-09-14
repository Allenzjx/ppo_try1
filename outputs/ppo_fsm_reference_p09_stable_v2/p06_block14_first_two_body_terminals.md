# P06 block14 前两个 BODY 终态：固定247条诊断

## 范围与采样账本

只扫描 run `20260910T1553525221025Z_g7db0d17f398d_3568eb4dc65742e6902a900026543883` 已落盘的前两个完整回合：**210＋37＝247 policy decisions**，global151937–152183。用 ReadWrite/Delete 共享 FileStream 先固定初始字节长度，再一次解析；没有追读后续教师前缀或策略。

- audit：2026-09-10 16:19:37.4571558Z，18,796,321B，恰247完整行，SHA256 `d0c004374cf5aa43f6fd2461dec71af4e23a623acaa5940b1e66f566f0659af4`。
- completed_episodes：16:19:37.4868729Z，164,246B，恰2完整行，SHA256 `f187add554f517316272ee3bc1e36a2b46cedc49d8da6d28de2598fec8ae2665`。
- 两份文件的每个 terminal_info 逐字段序列化完全一致；终态已耐久落盘，不等于本次 rollout 已更新或 checkpoint 已获核验。这里的 SHA 只绑定固定日志字节，未加载或哈希 checkpoint。

主控提供的固定更新证据仅到 **PPO update1153 /本块1次完整更新＝前128条P06**。另 **119条已采样但尚未进入已完成更新**：ep0后82条＋ep1全部37条，即P06=112/P07=1/P08=1/P09=5。还需要**新增9次策略采样**才凑齐下一128条；这9条不是已经取得的信用。第二终态回调可能仍在后续 reset 返回之前，不能仅凭 audit 存在断言该拍已进入 `alg.process_env_step`。本报告不确认新 checkpoint 字节、后续更新、停止完成或最终保存；13个完成块主账不改。

## 两回合真实结果

| 回合 | 策略样本与阶段 | 策略起始tick | 终态tick / 物理时间 | 策略物理ticks |
|---|---|---:|---:|---:|
| ep0 | 210；全P06 | 3584 | 5261 /43.841667s，BODY_COLLISION | 1677 |
| ep1 | 37；P06=30/P07=1/P08=1/P09=5 | 3584 | 3873 /32.275s，BODY_COLLISION | 289 |

两者均 seed1001，教师前缀时间和历史不计策略信用；29.866667s之前不是当前策略从P01执行。合计P06=240/P07=1/P08=1/P09=5，**1966 policy-native ticks**（16.383333s策略物理时间）；不是将两回合带前缀的物理时钟相加作为策略时长。FR P1695、FL P3583属于真实教师前缀历史，不是这247条新赚取的前腿放置。

### ep0：P06 BODY，未形成 RR I/Q

210条中 RR 事件列表没有I/Q/C/P，current-valid端点0；未进入P07。FL当前接触随动作变化：P06端点 **TOP146 /AIR60 /其他4**，不能由历史P推断始终承载。

终态FL真实TOP，有效承载1.898449N/载荷0.167044；RR GROUND，有效9.466539N/载荷0.832956。FR与RL当时均AIR、有效载荷0。RR前缘距离−249.018606mm、台面净空−50.093267mm、ground-relative lift0。其 continuation reason=`physical_safety_abort`、current-valid=false；这是 BODY 任务失败使继续运动失效，不是把 BODY 改分类为独立安全中止。

全1677 ticks逐一连续且 verified/effect=true、own_phase1677；最后一个决策实际执行5ticks即物理终止。没有普通后腿阶段交接，不应虚构P06→P09成功。

### ep1：P06产生Q，连续进入P09，随后BODY

| tick / 秒 | 事件或阶段接管 | 当前 RR / 当前 FL |
|---|---|---|
| 3764 /31.366667 | P06新I，excursion3.077929mm | 初始事件，未成为Q |
| 3821 /31.841667 | P06再有I，excursion3.679190mm | 不能计为第二个Q |
| 3824 /31.866667 | **P06产生Q**，同tick P06→P07 | RR AIR/current-valid；excursion9.076247mm、台面净空−41.623328mm。FL TOP，有效1.701579N/load0.055504 |
| 3832 /31.933333 | P07→P08，同次Q接管 | RR current-valid，FL TOP，有效0.935909N/load0.031254 |
| 3840 /32.000000 | P08→P09，同次Q接管 | RR current-valid，FL TOP，有效5.159326N/load0.165984 |
| 3873 /32.275000 | P09 BODY_CONTACT | RR仍AIR，但 current-valid/continuation=false，reason=physical_safety_abort；FL与FR AIR、有效载荷0 |

本回合 **I2/Q1/已成立Q的回地撤销0/C0/P0**，7个决策末态 current-valid。两个I的全身实测关节运动20.147534°/11.000814°；Q时RR自身运动3.953451°。首次I到第二次I之间没有Q，不能编造一个“Q回地撤销”。Q3824发生在实际发出P06动作期间，不按returned P07或终态P09倒推其发生阶段。之后没有记录新的Q或ground撤销；物理中止使已建立历史与当前可继续状态分离。

接管仍不代表靠近前缘任务已按固定姿态完成：P06→P07的rear_approach仅0.230917；P07→P08 role_prepared_RR=1；P08→P09 transfer_ready_RR=1且edge_proximity_RR约0.331685。监督器明确记录 `qualified downstream motion already active; continuous takeover`，允许当前真实运动接管，没有等待静止姿态。

终态 RR 前缘仍 **−379.771438mm**，台面净空+163.593894mm、ground-relative lift214.293469mm；较高AIR不等于越沿，C/P均false。FL当时AIR/0N，即使其P历史保留也不能称承载。RL GROUND5.687593N/load1为当时唯一被验证的腿支撑，FR同样AIR/0N。全回合FL端点分布：P06 TOP19/AIR11，P07 TOP1、P08 TOP1，P09 TOP4/AIR1。保持当前接触与历史放置的区别。

## 动作交接、终止和原生执行证据

三次学习阶段桥接 P06→P07→P08→P09 的 **previous residual与carried residual全12维严格相等**，被禁通道/scale clipped列表均空，handoff_hold=true。首拍 residual步差最大仅1.42e−14（浮点量级），wheel桥接差最大1.11e−16；没有整组清零。

交接端点的RR residual hip/knee从[12.907235,−2.532503]到[16.407235,−6.032503]再到[19.907235,−9.532503]deg；实际RR target分别[11.557332,−1.578153]、[15.057332,−5.078153]、[18.557332,−8.578153]deg。实际四轮持续非零，并非阶段done/reset后的重新抬升。

**不能把 residual连续夸大为所有动作完全无跳变。** 桥接器的 applied-action请求指标仍有逻辑servo变化：P06→P07最大14.8°、P07→P08为7.4°、P08→P09最大8.5°（其中FL hip8.5°/RL hip4.2°/RR hip1.6°）。这些是 projector请求/nominal边界指标，不是独立重建的真实native actuator单tick跳变；不据此断言mapper重置或把它归因为碰撞。

每回合首次记录的P05→P06桥接，previous/carried residual全零，是零信用教师前缀向P06策略的初始入口；首个策略动作已非零。它不是三个学习中普通阶段桥接的“残差被清零”证据。

三次普通阶段转换的 terminal=false/time_outs=false/bootstrap=true。两次真正BODY终态均 terminal=true/time_outs=false/bootstrap=false；终态policy/critic各372维、完全一致、有限且finite_fallback=false。两回合最后决策只执行5tick和1tick，真实终止不必等8tick动作块结束。

全 **1966/1966 native ticks连续、verified且actual_effect=true**，own1963，差额3对应普通handoff hold；四类禁止episode状态写入最大值全0。原生动作执行有效与任务失败可以同时成立；**本次BODY不能称为执行链损坏**。没有把后腿AIR或局部Q当完整任务成功。

## BODY判定已确认的层次与未保存的细项

两份终态证据都为 physical evaluator **valid=true/run_validity=VALID/physical_evidence_status=VERIFIED**，`TASK_FAILURE_BODY_COLLISION`、source=`BODY_CONTACT`、reason=`central body/obstacle collision`。这与视频6的CONTACT_BEARING_UNVERIFIED不同，不应混用。

当前 [监督器](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:470)消费 authoritative `observation.body_collision.detected`；[sensor reader](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/sensing/sensor_reader.py:285)接入 [BodyCollisionDetector](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/sensing/body_collision_detector.py:46)。现有检测规则要求已验证exact base_link–/World/Obstacle活跃接触对，并满足≥2tick持续或≥1mm live collider穿透佐证；腿与wheel接触不是这个BODY条件。

但训练audit不保存原始BodyCollisionStatus的real_pair_active/persistent/geometry_penetration或原始BODY接触对细项。因此这里只确认运行链消费了权威判定，**不能独立指定此次分别由持续接触还是穿透分支触发，也不能编造穿透量**。ep0 body AABB zmin49.639549mm、ep1 zmin50.010606mm不能当signed penetration，不能据它们改判成功或传感器错误；完整AABB保存在附JSON。

两次均是应保留的BODY任务失败，后腿越沿/放置尚未完成。既不能证明P05 capture-wheel overlay小缺口是失败原因，也不把对该已确认小问题的后续修复或本次BODY当成新的optimizer成功门禁。

仅新增本MD及[固定小JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p06_block14_first_two_body_terminals.json)。未读后续prefix/策略、巨型physical/native流、checkpoint/pointer/tensor，未操作进程、生产、四主报告或CSV。原始文件和失败记录保留。

