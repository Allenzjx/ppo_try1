# RR 已封存轨迹诊断：CP177152 与保留 zero

## 实施状态（更新）

**最新完整评估：CP178432 从P01运行到49.225 s，第一未完成任务为P05的FL放置，未到RR。** 原终止 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，P05 age37.225 s；物理记录有效、没有collision/fall终止。详见[正式P01失败诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/formal_P01_CP178432_failure.md)及[JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/formal_P01_CP178432_failure.json)。2500–5907全部3408物理步FL无真实接触；不能用后腿后缀训练覆盖这次完整评估的前段阻塞。

RR 修复已在 `6c2121b68654eaaa2afa7c19e9d02ae770493d2f` 实现并通过定向测试；现已有新真实P06后缀训练证据，但**未证明持续carry或完整任务成功**。启用标识为 `functional_free_air_lift_v3` 与 `current_free_lift_before_pending_knee_and_roll_v1`；旧 v2 保留。原 CP177152 的封存 `SUCCESS` 标签、视频和物理记录未修改，本报告不将其改标，也不把组件回放冒充新 PPO 成功。

最新分离报告：[CP178432训练RR交接诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/training_RR_carry_178432.md) / [JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/training_RR_carry_178432.json)。512真实决策、4更新；RR新AIR资格4001/4640/5108/5434，ground撤销4376/4668/5259；5726cross/5731place后6053再落地。最终P12、RL未place，训练执行SUCCEEDED但`full_task_success=false`。新pending guard7个保存事件均满足许可，没有wait；不能将曾自由抬升当持续carry证明。正式自然P01评估由主任务另行报告。

- 新增测试：`tests/unit/test_p09_free_air_lift_v3.py`、`tests/unit/test_p09_free_air_carry_source_v1.py`。与旧 lift/source-order/late-reconfiguration/TOP 延续用例合跑 **155 passed**；与 backend/all-stage/transfer-role/functional-reward 用例合跑 **176 passed**；最后事件诊断补充后新旧 lift 子集 **32 passed**。这些套件有重叠，不能相加为独立测试数。
- 真实旧输入的 RR 组件回放 **2202 帧**：CP4292–5432 共1141帧，新资格只保留4633，删除旧4940接触上升资格；zero5100–6160 共1061帧，5434资格和6155越沿保留，逐帧 `current_lift_valid` 与旧组件一致。回执：`rr_v3_evaluator_replay_receipt.json`。未播种前腿历史，未重算完整任务成功，未运行物理。
- 生产范围仅 `semantic_supervisor.py`、`semantic_backend.py`、`semantic_transfer_roles.py` 的 RR 路径。没有修改 geometry、reward、sigma、动作上限、12通道残差许可或原始观测维度。终端 home 修订由独立工作处理，不属于这份 RR 因果结论。

## 结论与边界

CP177152 原封存 `SUCCESS` 标签未修改；但其第二次 RR 抬升资格与“先有效卸载/自由抬升，再越沿”的目标不一致，不能用该标签证明新版要求已经满足。日志证实**带动力的障碍接触上升**，尚不足以唯一归因为“纯 wheel 驱动”：记录提供接触合力与平均接触点，未分离轮驱动力、全身关节作用、摩擦和约束冲量。

它不是从来没有抬腿：第一次确有自由抬升，随后落地撤销；判定缺口发生在第二 attempt。CoM 也不是完全没有向 FL 移动；但移动不等于 FL 承载，资格时 FL 悬空。

原始只读诊断范围：CP ticks4292–5464，zero ticks5100–6184；仅提取对应决策快照和原始物理/控制记录。该分析无 policy forward、无 GAE 重算、无仿真；后续生产实现状态单列于上方。角度为 canonical deg，wheel 为 canonical rad/s；时间为 episode physics tick/120。接触判别沿用原 `physical_contact_surface`，未借新规则重算整场成功。

## 关键物理证据

| 事件 | CP177152 | 保留 zero |
|---|---|---|
| P09 任务标签开始 | 4393 | 5177 |
| 实际 P09 源动作起点 | 4600 | 5408 |
| 首次有效自由抬升 | 4633；AIR 连续段4614–4730 | 5434；AIR 连续段5413–5939 |
| 此自由段最高轮底净空（相对障碍顶面） | +19.073 mm，tick4682 | +85.442 mm，tick5505 |
| 重新落地 | 4731，原资格正确撤销 | 未发生 |
| 接触前缘 | 4833：FRONT_WALL 且仍有 ground 接触 | 5940：TOP，此前已充分自由抬升 |
| 第二次 initial / qualified | 4900 FRONT_WALL / 4940 AMBIGUOUS | 无需重获资格 |
| 第二资格→越沿/放置 | 4940→5428 | 5434→6155 |
| 此窗口连续接触上升 | 4940–5176：1.975 s，轮底上升32.258 mm | 5434–5939：4.217 s 真 AIR；之后才 TOP |
| 此窗口最高 AIR 净空 | −0.838 mm | +85.442 mm |

CP4940–5176 始终 `OBSTACLE_AMBIGUOUS`，轮底从顶面下42.764 mm升到下10.507 mm；RR 实测角速度 +0.250～+0.487 rad/s，最终 target +0.2810～+0.2816 rad/s。接触合力由4940的 `[-11.710, -0.006, +1.133] N` 到5176的 `[-8.871, +0.398, +8.846] N`。不能将分类器返回的未验证 bearing=0 当作真实不承载：对应 load fraction 在本报告明确为 `null`。角点平均接触点落入顶面5 mm边带，但水平力仍大于竖向力，因此既非已验证 TOP，也不满足现有 FRONT_WALL 的严格点位条件。

两条轨迹都允许合法 TOP 接触后完成轮中心越沿；不能要求所有接触发生前轮中心已经越沿，也不能把所有角点接触直接判失败。

## CoM、支撑与逐通道动作

在CP4944，过去0.5 s CoM沿窗口起点“CoM→FL”方向的投影为 **+4.557 mm**；zero5440为 **+23.398 mm**。这是世界坐标下实际移动，不是固定姿态相似度。二者当时 FL 均 AIR / bearing=0；真实其他支撑为 FR、RL。CP4940 FR=12.115 N、RL=14.031 N，RR有未知类别障碍接触。zero5434 FR=13.067 N、RL=16.547 N，RR AIR且 load fraction=0。故不能虚构 FL 承载，也不能仅凭正向 CoM 投影宣称转移已充分完成。

CP4944四轮如下；该窗口12维 mask 均1，实际派发 audit 均通过，未发现 wheel mask 或 nominal 被清零。

| 四轮顺序 FL / FR / RL / RR | 数值 rad/s |
|---|---|
| 源/N nominal | +0.300 / +0.300 / +0.300 / +0.300 |
| 投影后残差 | +0.13794 / −0.00260 / +0.11808 / −0.01839 |
| 最终 target | +0.43794 / +0.29740 / +0.41808 / +0.28161 |

RR 不是被 policy 正向加强：其残差实际降低了 nominal wheel 速度。此时 RR knee target −52.754°、actual −52.222°；不能把几何层的额外调整全部算成 PPO 当拍残差。Q→cross最大 raw 绝对值仅0.14270，不是饱和探索样本。

## 为什么第一次 AIR 丢失：可确认到哪一层

源 P09 hip/knee 序列按 CP+808ticks 与 zero 对齐，例如 CP4680 / zero5488 的 RR nominal 都为 `[55.6°, 0°]`。该窗口没有阶段切换、动作历史 reset 或第二次执行器覆盖。

1. CP4632→4731 roll 从−8.52°转到+7.85°，pitch从−4.19°到−13.42°；zero对应5440→5540约保持roll−10～−11.2°、pitch−4.2～−3.73°。CP的全身旋转先于贴墙；base origin反而从约51.8 mm上升到83.4 mm，所以不是简单“base_z下降造成脚落地”。
2. CP4682达到仅19.073 mm净空，4688源 knee 已走到−4.9°，4704轮底降到顶面下1.20 mm，4731重新落地。局部几何投影在4696记录固定base两DOF预测 `Δz=+5.109 mm`，但实测轮底持续下降；该投影从来不保证全身物理运动。
3. 源四轮在整个初次lift至4832仍为0，但CP当时残差使FL约+0.15、FR约−0.016、RL约+0.115、RR约−0.019 rad/s。FR/RL承担支撑，FL仍AIR；这是与zero四轮target=0的实际差异，不能认定所有轮真静止或FL提供牵引。
4. 4833源四轮才切到+0.3，同tick首次FRONT_WALL接触。因而顺序为“全身趋势/关节过程丢失净空→落地→继续原源前送→贴墙”，不是RR先加速把首次lift拉掉。其余腿残差、轮差速、入口状态与几何层共同耦合，单轨迹不能指定某一通道为唯一机械原因。

生产线索：`_continuous_advisory` 的有限 source wheel owner 保留作者的脉冲与停止，不受额外 AIR carry suggestion gate 覆盖；ground撤销任务资格后，有限 joint/wheel源仍可继续。`semantic_nominal_geometry` 是固定base、RR两DOF的建议，远离前缘仅维护ground+既有8 mm量级，并未包含刚观测到的全身姿态变化。在新版中应有依据地保护实际已形成的连续carry，并处理ground revoke后的旧前送段接续；不能仅收紧成功判定、盲目等固定时间或恢复历史关节入口。

另用已保存的相邻两个 dispatch context 中实际 physical q 差值与该拍 Jacobian 做一阶数值核对（不是重放）：

| CP tick | 真实轮底单步Δz | RR实际关节Δq的一阶JΔq | 差额（含base运动和线性化误差） |
|---|---:|---:|---:|
| 4682（峰值附近） | +0.017 mm | +1.179 mm | −1.162 mm |
| 4688 | −0.539 mm | +0.710 mm | −1.250 mm |
| 4696 | −1.202 mm | +0.202 mm | −1.404 mm |
| 4720 | −1.737 mm | +0.240 mm | −1.977 mm |

这比“knee往负就是下降原因”更准确：RR自身实际关节在局部固定base模型里仍有向上分量，整体轮底却下落；全身项不可忽略。差额不是严格因果分解，也不能凭此将单一轮残差定为根因。

## 原 v2 判定缺口与已实施的窄修复

旧 v2 的 `TaskEvaluator._observe_functional_rr` 将 non-ground（包括wall/ambiguous接触）计入连续样本；whole-body response、相对最后ground增高≥8 mm和两个其他支撑即可新建资格。旧模式不要求当前 AIR 或已验证非承载。ground撤销是正确的，但第二资格可把接触上升作为新的抬升；新版通过显式模式选择，不静默改写旧模式。

约662–675行的 crossing 首先接受 historical active_lift；wheel-only兜底只累计FRONT_WALL，而且要求 attempt 内没有 joint demand、没有 active initial。AMBIGUOUS 会清掉这段wall记录；历史任何全身joint demand也会阻止该兜底。因此现有旧 `SUCCESS` 不能排除这条实际反例。后端主要信任该任务结果，legacy wheel-only只作诊断。

已实施最小资格组件：沿用既有2个样本和8 mm尺度，**新资格需要当前真实AIR + 连续AIR段内的自由增高**，基点采用该AIR段之前最近一次真实接触的轮底高度；不能把ground之后已沿障碍抬高的高度加进去。initial 的3 mm也使用同一自由增高定义。保留whole-body response、当前至少两个已验证其他支撑与安全条件，不要求RR自身关节先变化，不要求FL接触。ground重新接触清除attempt信用。资格已建立后可保留合规角点/TOP连续接管，未知接触不自行生成新资格；从边沿脱离后如果确有新的自由增高仍可恢复，不永久锁死。缺失合法相邻实际样本的自由增高/轮底速度诊断返回 `null`，不填造数据。

数据反例足以检验“晚AIR洗白”：CP4731以后各AIR段相对上一接触的最大自由增高，ground微跳最多 **1.136 mm**，后续TOP/AMBIGUOUS脱离段最多 **0.516 mm**，均未到8 mm；zero5434刚好首次达到8 mm，继续自由上升。CP第一次4633仍应获得资格，4731后必须撤销，不能全程一刀切判“没抬腿”。

这只是有证据的门控及源时序修复，不是新的全任务成功证明或纯wheel因果结论。carry 的窄源保护也已实现，但其闭环效果仍需真实训练和完整P01评估；不以把旧轨迹改标签替代实现。正反例夹具仅验证组件，不能宣称新物理复现。

### 实现范围与未实施的扩展

- `semantic_supervisor.py:TaskEvaluator._observe_functional_rr`：上述free-rise状态；旧ground撤销与普通phase连续性保留。测试zero5434获资格、CP4633获资格但4731清除、4940接触增高不能获新资格、后续短AIR不能用接触累计高度洗白；合法资格后的轻触不能自动判wheel-only失败。
- 同文件 `NominalMotionProvider._sequence_permission` / `_continuous_advisory`：给“已获得lift后又ground revoke”的P09后续正向wheel原子组增加当前carry恢复条件，不能把已有旧资格当推进依据；显式stop仍生效，未修改的持续通道仍继承，全12 residual和其他准备/恢复动作仍可执行。限定为丢失已形成carry后的接续，不把所有初始全身抬升前轮动作都禁掉。
- 未实施 geometry 扩展：`semantic_nominal_geometry.py:capture_nominal_geometry_context` / `_bounded_rr_correction` 保持不变。实际轮底下降由新 evaluator 的相邻世界坐标 collider-bottom 有限差分记录，用于窄源时序许可；没有引入新的固定base预测保证。

源时序实现从 contract 的 changed/atomic channels 和 source 值推导待执行 knee waypoint 与首次正向四轮组，不以本次六条 waypoint 或固定 tick 作为验收常量。pending knee 要求当前有效自由抬升；低于既有15 mm且实测下降/缺失合法速度时暂不消费 source 时钟，合法 TOP/drop 区域才可例外，`within_top_xy` 单独不能绕过。首次四轮组允许有效 AIR（即使 top-gap 为负）或合法当前 TOP 延续；显式 stop 不被吞掉。无待执行事件时清空该就绪诊断，跨阶段实际 history 与全12 residual 保留。此实现不等于已验证的闭环成功，仍待新物理证据。

### 窄源时序建议的原输入回放

`rr_carry_predicate_replay.json` 检查原source的六个RR knee waypoint（source .6667～1.0 s）和第一次四轮前送原子组（1.9333 s）。只计算“原轨迹输入下是否会暂缓”的predicate，不推进另一个controller、不声称暂缓后一定保住净空。

原输入 predicate：下一条确为该carry膝部源waypoint、当前已建立有效AIR、未在platform、真实轮底正在下降且净空低于既有15 mm（生产比较严格 `< .015`）。它不是“膝角为负就下降”，也不覆盖整个P09的自然前送/落轮过程。

| 原输入检查 | CP177152 | zero |
|---|---|---|
| 第一次would-hold dispatch | 4697（输入4696） | 六个waypoint均不触发 |
| 该输入front distance / gap / 实际vz | −228.135 mm / +9.892 mm / −0.14419 m/s | 对应5504：−246.657 mm / +85.352 mm / +0.01153 m/s |
| 六个knee waypoint检查 | 原输入后四个会触发 | 最低gap仍+74.177 mm |
| 后续四轮正向组 | 4833：Q=false、ground，不应盲接 | 5641：Q=true、AIR，可前送 |

关键边界：zero5640 gap已−12.008 mm，但仍有有效连续自由抬升历史、AIR且距前缘−105.103 mm。因此**rolling许可不能套用15 mm门槛**。zero5936近corner时gap−7.219 mm、距前缘−24.367 mm，但knee源已恒定，并无pending膝部waypoint；上述窄规则不触发。若实现扩为持续carry guard，必须用真实轮/障碍几何区分正常corner landing，不能把全程下降/低净空都暂停，也不能凭固定历史距离或膝角判物理失败。

## 文件与溯源

- 完整逐tick、同步决策、力/接触/CoM/12通道JSON：`outputs/diagnostics_v1/residual_RR_diagnosis.json`。
- 连续AIR自由增高正反例夹具：`outputs/diagnostics_v1/rr_lift_qualification_replay_fixture.json`。
- 窄源时序原输入predicate回放：`outputs/diagnostics_v1/rr_carry_predicate_replay.json`。
- 已实现组件的2202帧回执：`outputs/diagnostics_v1/rr_v3_evaluator_replay_receipt.json`。
- 封存后分析脚本：`outputs/diagnostics_v1/analyze_sealed_rr_v3.py --source <run>/source --start-tick <起点> --end-tick <终点> --output <新JSON>`。先检查终态 manifest，再读取显式有限窗口；保存原 RR 自由增高/资格/再接触/source-wait、真实 CoM/contact/qd 和原终止结果。决策诊断不插值成每物理步；最后 action 被固定观察终点中断时保留真实 endpoint 与较早的最后返回 step 快照区别。旧输入 smoke 仅11物理帧/2决策，新 RR 物理运行尚未用此脚本分析。
- CP正式run：`20260916T0550249947408Z_g00050a2b1452_086805789a6e491392ef4656c7708c2c`。
- zero保留run：`20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6`。
- 接触类别来源：`semantic_physical_sensing.physical_contact_surface`；力为exact pair合力、点为平均点，不是逐接触法向分解。
