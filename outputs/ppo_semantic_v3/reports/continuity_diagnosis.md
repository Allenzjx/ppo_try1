# P07–P09 连续任务诊断：历史 A 视频与 C10112

日期：2026-09-05。性质：只读历史证据分析，不是新验收门、重分类、配对优越性结论或新的物理实验。生产代码随后有并行 v3 修改；本文控制逻辑结论固定到各 run 自己的版本，而不是当前工作树。

## 结论先行

1. **A 视频不是碰撞/纯轮爬升/传感器/编码失败。** 原冻结控制器在 P10 `WAIT_ENTRY` 被历史 RRK 入口位置与回弹速度条件阻断，65.3667 s 未完成全任务。独立物理评价仍 valid、无 physical failure；RR 已真实越沿、放置，RL 未完成。视频编码校验通过，不能因文件存在而发布为成功 baseline。
2. **C10112 是 P09 超时未完成，不是记录中的 physical failure。** 有真实 RR 抬升，但在前沿前回地、撤销当前尝试资格，后来未实际越过前沿。30 s 阶段期限到达；保留失败训练证据，不抹掉 10,112 个真实训练决策、44 updates/880 optimizer steps。
3. 最有力的连续性证据是 **进入 P09 的动态条件不同、P07/P08 的有限建议被提前切换，以及 P09 固定时序回收与当前位置不匹配**。不是已证明的 mapper bug，也不是这条 deterministic 策略撞到 ±4°/.12 residual 上限。

## 源证据与复现

- A：`runs/ppo_semantic_v2/video_eval/baseline_A/20260905T0902190128552Z_g00b94fbb276a_e0ddaa746bc84bddb8e4f80506e09fac/source`。
  `semantic_video_source_manifest.json` SHA256 `bb333d4868afb951ec2066a3e9e40e45be00c71c5c07d708421a9f91c3ef4ab9`。
- C：`runs/ppo_semantic_v2/validation/20260905T0847270847096Z_gd19c655713bf_bb1e5013136b4b8787f63444760db2d8`。
  `evaluation_manifest.json` SHA256 `28ba0d1a86a406c531fe940ee46a1c797b12ef47fdda968f7f156c539421926c`。
  实际加载 `outputs/ppo_semantic_v2/checkpoints/history/checkpoint_step_000010112.pt`，SHA256 `7397162192242fde4d7532b3ab529b49bea7b41689f766f3ee8c7b27ae3940fb`，seed2001。
- 分析脚本：`tools/diagnose_semantic_continuity.py`。只读 JSONL/JSON，未 import Isaac、策略、训练或现行 TaskEvaluator；不会把新规则套回旧 manifest。
- `continuity_diagnosis.csv`：A/C 从 40 s 起到各自末帧的 7,302 条 120 Hz 记录，包含实际 nominal、native mapper 输入、combined bias、最终 canonical drive、float32 PhysX targets、关节位置/速度、四轮几何/接触/载荷、CoM 和历史事件。
- `continuity_diagnosis_facts.json`：摘要、manifest 哈希、关键帧、历史事件和 native 审计计数。重新生成须给脚本一个全新 `--output-dir`，现有输出不覆盖。

时间与 phase：CSV `phase_source` 是该物理 tick 使用的前一帧控制阶段；阶段事件记录在该 tick 的观测之后。因此 transition tick 仍显示旧 source，下一 tick 才使用新 phase。这是正常因果次序，不是错帧。

字段限制：`mapped_*` 是 ACK 的 `native_drive_target_full12`（最终 bounded feedback 前），不是实测关节；`drive_*` 是该 tick ACK 最终下发的 canonical targets；`native_target_*` 是真实 float32 dispatch buffer，伺服用物理 rad、轮用 rad/s。伺服 `drive - mapped - bias` 非零可由冻结最终 slew 产生，不能直接称漏加 residual。`*_relative_body_*` 是 world 轴位置差，不是旋转后的 chassis 坐标。roll/pitch 由完整原始 quaternion 算出。CoM→FL 速度是 world CoM 速度投影到“当前 base→FL 的水平单位向量”，包含前向分量；不是载荷、不代表 CoM 必须到 FL 中心，也不是因果控制贡献。

## A 的具体结束原因

| 项目 | 真实结果 |
|---|---|
| 视频生命周期 | `DIAGNOSTIC_FAILURE`，错误 `SemanticVideoError: episode did not meet common physical task` |
| 原控制器 | P10 / WAIT_ENTRY / `CONTROLLER_BLOCKED` |
| 阻断 guard | `reference_entry_compatible`，`controller.live_servo_vs_v010_actual_start` |
| RRK 位置 guard 样本 | actual −45.8583059°，reference −50.3975984°，error +4.5392925°，容差2° |
| RRK 速度 guard 样本 | actual +2.6106870°/s，reference +23.5853331°/s，error −20.9746461°/s，容差3.53779996°/s；要求正向回弹 |
| 独立任务评价 | valid=true，success=false，termination_reason=null；FR/FL/RR placed=true，RL=false |
| 真实长度 | 65.3666667 s，7844 physics ticks，981 次已发决策；末次只推进4 ticks |
| 视频技术校验 | valid=true，989帧，error为空 |

A RR 的同一条真实过程：57.825 s/tick6939 获得抬升资格；59.208333 s/tick7105 越前沿；63.158333 s/tick7579 放置。不是仅根据原 FSM P09 标签推断。

A 全7844执行 ticks 原生审计通过，raw residual 与真实 residual effect 全零，禁止状态写入计数全零。7845 原始观测（含 tick0）finite，body collision 全零。所分析40 s后窗口四轮 ground/obstacle pair 均 verified，CoM 无缺失。A 视频有64 ticks真实零输入 pre-roll；这会改变录制起点物理状态，不能只凭此次与旧成功录像不同就证明它是 P10 阻断原因。应保持 A 控制冻结，单独核对自然起点和录制流程，不把历史入口约束迁入 B/C。

**确切初始化差异：** 源码原自然 reset 先执行 `SETTLE_TICKS=180`（1.5 s，原生command tick0–179）。本A source记录 `reset_count=1`，随后额外64个 `pre_action`，实际原生tick180–243（0.533333 s），每tick真实零输入、root writes=0、task_credit=false。其第一/末个任务命令native tick为244/8087，对应任务episode tick1/7844。因为未共同物理成功，实际post-success ticks=0。它不是从原180 settle的最后64ticks取图，而是又推进64ticks；旧普通baseline流程没有这个video pre-action调用。此项是已识别、可配对测量的初始化差异，不是已证明的因果解释。后续录制可从既有settle捕获以避免额外warmup，但本次诊断未修改录像逻辑，也不承诺旧A历史成功可以直接发布为新版共同评价基线。

## P07–P09 的实际动态差异

| 时刻/条件 | A | C10112 |
|---|---:|---:|
| P07 进入 | 55.4000 s | 44.8667 s |
| P07 持续 | 1.7333 s /208 ticks | 0.5333 s /64 ticks |
| P08 进入 | 57.1333 s | 45.4000 s |
| P08 持续 | 0.4667 s /56 ticks | 0.0667 s /8 ticks |
| P09 进入 RR 中心距前沿 | −61.081 mm | −205.553 mm |
| P09 进入 RR bottom 相对台面 | −41.720 mm，AIR | −51.553 mm，GROUND |
| P09 进入 FL | AIR、0实际载荷 | AIR、0实际载荷 |
| P09 首次 FL 接触 | 57.7083 s，进入后0.1083 s | 49.6167 s，进入后4.1500 s |
| RR 第一次峰值净空 | +127.798 mm @58.3417 s | +58.360 mm @46.1667 s |
| RR 首次 knee 名义弯曲时距前沿 | −139.993 mm，净空+124.965 mm | −244.425 mm，净空+57.285 mm |
| RR 首次 hip 名义下降时距前沿 | −62.641 mm，净空+95.018 mm | −169.746 mm，净空−9.771 mm |
| RR 首次回地撤资格 | 无 | 46.7583 s，距沿−146.120 mm |

C：45.875 s 的 RR lift 资格是实测 above-top AIR（净空+0.590 mm，近期向上行程52.515 mm，关节运动27.747°），不是未抬腿。46.7583 s 回地有 verified ground pair、RR分担69.92%当时四轮有效法向载荷；FL仍 AIR，净空+138.387 mm，载荷0。之后 RR 最接近前沿在 tick6104/50.8667 s：−1.477 mm、AIR、净空+12.530 mm，但仍未真正越过中心前沿。末帧 RR 距沿−49.893 mm、ground=true。旧 evaluator 的 above-top资格不能表达所有 whole-body 初始离地进度，这是设计覆盖缺口；不应把这次末成任务改写为成功。

## 九项假设：证据、未知与最小修复方向

| 假设 | 判定与直接证据 | 最小方向（不是新增门禁） |
|---|---|---|
| H1 P08瞬间低载荷即结束 | **支持。** C P08只有8 ticks；completion `workspace_RR=1, load_ready_RR=1`，末帧RR仍ground、load fraction=.11897。v2 `load_ready` 仅当前其他支撑数和载荷比例；没有持续动态准备表述。A比它晚0.4 s切P09，且RR已离地。不同控制器/轨迹，不能把该时长差本身当最短时间标准。 | 继续现有全身动作/工作空间目标，暴露实时卸载和初始AIR进度；不要引入历史时间或固定末姿等待。 |
| H2 P09按钟下放而未到前沿 | **强支持。** 历史 NominalMotionProvider 每次phase调用 `MotionExecutor.start_phase`，只重启建议时钟；除P02外无当前RR位置反馈。C在距沿244 mm时已开始knee弯曲，随后距沿170 mm且低于台面时hip下降，回地后建议尾静止，直到阶段期限。 | 名义建议目标按 lift/carry/place/retry 实时几何连续选择；还没到前沿时不能因为建议时钟到尾就唯一剩下下放姿态，residual仍有主导权。 |
| H3 P08→P09清零或丢动作 | **清零说法被反证；语义提前终止被支持。** tick5456→5457，12维 nominal及residual全部保留；bridge dropped/clipped全零，handoff_hold=true，最大servo residual数值误差5.6e−17°。但之后P08时钟被P09替换。C P07在FR wheel仍−.6rad/s时结束，P08随后slew到0；P08的RL hip只到14.3°即换阶段，而A P08结束已31.2°。 | 保留现有mapper/bridge；修复有限建议的跨阶段任务连续性，不能复制大范围reset新架构。 |
| H4 想更大但被小caps截死 | **本条deterministic C不支持直接饱和因果。** P09最大raw=.13505；RR hip residual最大abs .16945°，knee .53694°；四轮最大abs .007603rad/s。远小于4°/.12。 | 按新规范审计/扩大整体探索workspace，但如实区分“空间先验很小”与“这次策略已要求而被截断”；新范围尚需真实短probe。 |
| H5 指令到达但映射/身体响应不同 | **到达获强证据，mapper bug未获证据。** C9056/9056、A7844/7844 native audit verified，实际重建与float32 setter/PhysX一致。C回地时nominal RRH/RRK38.05/−37.8°，measured46.834/−37.684°，相位/接触动态并非目标数值本身。 | 保留成熟mapper；将native目标、实测关节、轮几何/接触联合诊断，不把command误当姿态，不用未经控制实验的差分声称控制增益。 |
| H6 有效动作被物理分类错判 | **无本次成功被否决证据；初始progress覆盖不足。** C确实先获lift再在verified ground撤资格，所有旧lift事件仍保留；末端无RR frontcross。没有TASK_FAILURE_WHEEL_ONLY/碰撞。后来最近−1.477 mm还未跨plane，不能只因接近就记成功。 | 分开 active whole-body initial lift、越沿资格和真实placement；保留GROUND撤当前尝试及合法retry，不把初始离地当完整跨越。 |
| H7 FL准备/CoM趋势断开后孤立RR抬腿 | **支持动态差异，不宣称单一因果。** C P09前497 ticks FL均AIR，而A仅13 ticks。A P08 CoM朝当前FL水平射线速度始终正（.00347到.19662m/s）；C仅一决策、范围−.01421到+.02512。C RR抬升/回地阶段FL持续AIR。 | 保留全身姿态、CoM动量趋势、其他实际支撑和RL协同；禁止把FL AIR算FL承载，也不要求CoM到FL中心。 |
| H8 P08已有离地被handoff重置 | **A初始离地事实支持；C“重置致回地”被反证。** A P08末RR bottom约高于地8.28mm、AIR、RR nominal hip/knee仍0；C P08八ticks RR全部ground，P07末短AIR但bottom仍低于地约1.43mm，不能等同可靠几何净空。C全9056ticks禁止root/velocity/force/gravity写入为0，handoff连续。 | 保留历史和实时几何，区别接触阈值短AIR与真正初始离地；不凭A现象虚构C已被reset清掉的状态。 |
| H9 RR自己关节没动就拒绝whole-body初始lift | **结构缺口支持；不单独解释C未完成。** v2 lift依赖自身关节运动≥2°、AIR、8mm上升且bottom到台面；A P08自身RR nominal均0已经离地，证明初始抬升可以来自whole-body协同。C在P09曾得到完整资格，故其首次回地不能归因于完全不认lift。 | 让初始抬升用已记录真实全身控制/几何/卸载表述，独立于RR自身角度阈值；完整成功仍需对应过程越沿、落顶、最终受控通过。 |

## 排除与保留的不确定性

- C9057条原始观测全finite、无body collision；40 s后核对四轮exact pair/CoM均无缺失。所有9056执行ticks都有真实非零float32 effect、无禁写。**这排除了证据缺失/零效应冒充，但不能证明所有物理响应都是理想或bug-free。**
- 本报告没有重新训练、没有新B/C配对、没有直接从旧A成功summary推定统一评价成功。A video 与C validation还有64 ticks pre-roll及控制版本差异，比较是定位相关证据，不是控制变量实验。
- 旧库中的P07建议动作含FR wheel −.63rad/s约1 s和FL姿态回收；其历史时序在C的早切换下不会全部执行。应修复任务上下文连续性，而非强制复播1 s或historicalendpoint。
- 接触点可能为多个触点均值或None；本分析只保留实际字段，不伪造top/side标签。FL的实际pair载荷与CoM运动明确分列。
- 不增加前置B成功门、不改冻结A、不改历史manifest。本诊断仅为最小连续任务修复和下一轮真实PPO/物理probe提供事实。

## 本轮 v3 残差 workspace 派生审计

`tools/audit_semantic_workspace.py` 生成 `outputs/ppo_semantic_v3/metrics/residual_workspace_audit.csv` 和相邻 manifest。312行=2角色×13阶段×12通道，其中72行有P07–P09实际窗口；未到达阶段标 `phase_not_reached_no_live_data`，实际已访问但不在本次窗口标 `outside_P07_P09_focus`，均不伪造测量统计为0。输入v3 execution profile SHA256 `39839d163b8f661da627182ea6feee0464f51806788f9d3191931d5969ca6470`，原v2 cap来自固定d19c655源版本。

保留四种不同量：source nominal余量、postmapper无当前residual的硬余量、projector原2°保留带、最终执行器硬限。CSV的 `*_intersection_*` 列显式取 requested cap、nominal保留带与postmapper硬余量交集；原 `*_after_hard` 列仍是硬限单层诊断，不拿它冒充完整投影空间。

- 原硬限：hip[-135,135]°，knee[-60,210]°，wheel ±2.0943951024rad/s；原projector安全区：hip[-133,133]°、knee[-58,208]°，轮不额外留2°。
- 新P07–P09请求范围：hip±24°、knee±36°、wheel±.6rad/s。它们并不保证任意姿态下均可双向用满。
- C P09 RR knee最小postmapper向负硬余量20.95°；nominal=-37.8°时原projector向负只允许20.2°。最终三者交集的最小向负cap=20.2°，向正=36°，双向总范围相对请求72°最低保留78.0556%。不是“36°全部可用”。
- C P09 RR hip及各轮在所观察窗口的名义/projector/执行器静态范围充足，不等于对未来接触几何仍安全、可达或必成功。
- residual slew仍60°/s与1.8rad/s²：120Hz最多每tick改变.5°/.015rad/s；15Hz八tick改变4°/.12rad/s。冻结最终servo target slew仍150°/s，即1.25°/tick、10°/decision。扩大cap不会自动扩大每次决策的瞬时可发变化。
- `*_held_native_1tick/8tick_*` 是把该tick真实native和controller bias冻结，代数推进residual slew/原2°projector保留带，并直接复用冻结 `bounded_drive_feedback_step` 算出的目标空间包络。没有调用mapper.advance、adapter.apply、Isaac step；8tick不是未来真实rollout，关节/身体实际到达还受动力学、接触和下一tick反馈影响。
- 原C observed eligible projection差异在上述关键RR/FR-wheel窗口为0，进一步说明此deterministic轨迹并非在要求远大动作后被投影剪掉。新的探索范围应在真实短probe中验证，而非从旧轨迹小信号差分杜撰FK因果增益。

工具第一次生成的本轮派生审计经授权补充了安全保留带交集；更新前校验原派生CSV和输入CSV哈希。A/C原始日志、manifest、checkpoint、冻结动作及物理文件未改。最后一次Python分析正常退出；后续live barrier期间不再启动Python或Isaac。
