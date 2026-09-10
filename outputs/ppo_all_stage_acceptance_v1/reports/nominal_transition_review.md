# P01–P13 nominal 所有权、动作依赖与 12 处交接只读审查

审查基线：`285307603f43bfc2846dada5f3ede73b0dc5f536`，2026-09-08。已完整阅读本轮附件 `0610eac7-775f-41a0-a772-223fe0147ef9/pasted-text.txt`。本报告只审查源码、当前配置、compact Recording 合同及已有小报告；没有启动 Python/Torch/测试/Isaac，没有重新扫描训练 raw、加载 checkpoint tensor 或复算大文件 hash。生产文件未改。工作树已有一个无关 untracked 文件，未触动。

所有 13 阶段均为 **CODE_REVIEWED / PHYSICALLY_UNVERIFIED_IN_THIS_REVIEW**。下文 COMMAND_OBSERVED 是命令/代码事实；历史 MEASURED_RESPONSE 只按原报告范围引用，不重授当前版成功；HYPOTHESIS 不是控制效果或失败因果证明。短阶段、非零继承和有限动作本身不自动判错。

## 1. 优先反馈：确认了什么，尚不能确认什么

### N1：P03 两个 wheel 通道存在明确的静态取消范围缺口

合同 P03 从相对时间 0 到 1.2 s 建议 FL wheel `−0.79`、RL wheel `+0.61` rad/s。当前 P03 两通道 residual cap 都是 `±0.6`。在自然 B/C 的 controller wheel bias=0、建议达到该值的状态下，忽略 slew 的宽松闭区间分别为：

| 通道 | nominal | residual cap | nominal + residual 宽松区间 | 0/反向 |
|---|---:|---:|---:|---|
| FL wheel，index 8 | −0.79 | ±0.6 | [−1.39, −0.19] | 不可达 |
| RL wheel，index 10 | +0.61 | ±0.6 | [+0.01, +1.21] | 不可达 |

有限 tanh 不能达到闭端点，所以不会比此表更宽。上述区间没有触及原 `±2.0943951023931953` wheel hard limit；hard clamp 不能补足取消范围。nominal 的 3 rad/s² 与 residual 的 1.8 rad/s² 又限制瞬时抵消。该结论不依赖当前 raw 是否碰 cap，也不证明这导致任何一次失败。

这是**局部可表达性缺口**，不是所有阶段都要扩 cap：P01/P04/P05 的 +0.3 已能用 ±0.6 取消；P07 FR −0.63、P09 FL −1.07、P13 FL +1.09/FR −0.72 已被后段 front cap 1.2 覆盖，P12 ±0.3 与 P13 rear −0.43/−0.39 被 rear cap .6 覆盖。修订若选择，应仅针对有必要取消/反向的通道并保原硬限，不用改噪声或频率。

### N2：捕获成功不会释放尚未结束的旧 nominal servo 层

`NominalMotionProvider._continuous_advisory` 对每个已创建 layer 每 tick 无条件 `motion.tick()`；除 P06 wheel gain 外，`placed/current region/current load` 不参与旧层终止或保持。旧 owner 持续写它曾 touched 的通道，直到后来的 owner 覆盖。

具体可复现代码反例：P05 在其源动作末尾之前已合法 FL placed 并进入 P06；P06 只拥有 wheels，因此 P05 FL hip/knee 仍会继续既定建议（hip 峰值 48.2，末 22.8；knee −36.7，末 −13.4），而不是绑定本次实际捕获状态。P02 的 FR knee 有同类提前捕获后继续摆动的问题；P12 的 RL 动作可能延续进入只先开轮的 P13。它们不是“新阶段把全部 start_full12 重置回旧 anchor”，而是**旧层仍有动作所有权**。

这是当前代码缺少“已先进合法进展后的 owner 适用性”处理的确证；旧动作是否破坏某次支撑，仍需该次 q/geometry/contact。不能据此固定历史姿态、冻结全 phase 或一概取消未完成前驱动作。正例必须保留：未完成的支撑调整仍可在下一 phase 继续；反例应针对已完成目标后不再合适的旧建议。

### N3：P06 有限轮建议已延续，但退役峰值不能因回退重新激活

当前 source 的 25.5333 s 零 endpoint 不再自动结束 P06 wheel 能力：满足 live 条件时复用原 +.3，并乘 `1−peak`。但 peak 用两后轮最小 front distance，在既有 `−.22..−.215 m`、lateral 有效范围单调累积。一度到 −.215 m 后 gain=0，随后退回不会恢复；即使短暂进区发生在两个 15 Hz phase 判定之间、最终仍留 P06，也是如此。

因此这是**nominal 恢复能力的明确局限**，不是动作整体被关闭（四 wheel residual 始终开放），也不是当前新版“25.533 s 到时永久断轮”。原意是避免过期 P06 持续滚动，不能不分后 owner/安全情形删除退役。

### U1：计时关节建议不等于物理下降；保护范围只覆盖后腿两个阶段

所有层按 source 时间推进，没有按 knee 正负暂停。当前 post-mapper geometry 仅 `P09→RR(6,7)`、`P12→RL(4,5)`；在当前未到 place XY、非 ground、lateral 有效时，用同 tick q/J 和原 15 mm margin 限制名义一阶向下分量。它不读当前 residual；不可行时明确 degraded bypass，不是实际净空保证。

P03/P05 没有该几何调整，P02→P03 也没有同类局部前腿约束。代码可证明仍采用计时源目标，**不能仅凭角度降低证明正在 world-down**。需对实际 current q/J 和 raw 净空核验后，才可把某窗口定为“未到放置区域而被迫下降”。不恢复曾删除的 knee-sign pause，也不把跨线本身设成开启创造跨线动作的前提。

### U2：一般速率差仍在，不能把 history 连续误写成任意旧目标可保持

除 P06–P13 FL hip 已降至 60°/s 外，其余 nominal servo 上限仍 150°/s；全部 residual servo 为 60°/s。每 physics tick 分别 1.25° / .5°。这使某些当 tick 无法完全抵消 nominal，但非所有 source 段都达到该上限。后段其他通道是否需要同样修订，应由真实保持/反向需求决定，不全局扩 cap/降 rate。

已有 block1/block3 窄窗报告确证过部分 RR/RL hip 旧目标不在当时 nominal+cap 静态区间，且有少数 tick 加 residual slew 后不能保持紧邻 target；并未证明负 knee nominal 必然排除旧膝姿态或导致 FALL。该历史来自旧 FL timing 版，不能称本 HEAD 的配对物理验证。

## 2. 共同执行不变量及 12 通道检查

通道顺序固定：`0 FL hip, 1 FL knee, 2 FR hip, 3 FR knee, 4 RL hip, 5 RL knee, 6 RR hip, 7 RR knee, 8 FL wheel, 9 FR wheel, 10 RL wheel, 11 RR wheel`。

`build_semantic_projector` 实际构造每 phase mask=`111111111111`；不是仅配置注释。scale 由 cap / 原 physical span 得到，输出请求为 `cap*tanh(raw)`，再受原速率、真实 headroom 和 hard safety 限制。所有 stage 的 allowed_action_channels 也都是该 12 通道；没有因 source 未动某个支撑腿就禁用该 residual。

| 范围代号 | 前 8 cap，单位 ° | 后 4 cap，rad/s | nominal servo rate | residual servo/wheel rate |
|---|---|---|---|---|
| E：P01–P05 | [18,24,18,24,12,18,12,18] | [.6,.6,.6,.6] | 8 通道均 150°/s | 全部 60°/s、1.8 rad/s² |
| L：P06–P13 | [32,36,24,112,24,36,24,36] | [1.2,1.2,.6,.6] | index0=60，其余150°/s | 同上 |

两组 nominal wheel slew 均 3 rad/s²，即 .025/tick；residual wheel slew .015/tick。caps 在 E→L 增大，其余 handoff 不变，构造器显式拒绝沿阶段缩 cap。FR knee 112 是动作范围，不是必须达到 −40°的姿态门；原 knee hard [−60,210]、2° reserve [−58,208] 不变。当前 REQUEST 历史编码前 8 scale 为 [4,4,4,6,4,4,4,4]、wheel 为 .12，raw history scale=1；这些是观测归一化，不是 cap。

执行链证据：

1. `MotionExecutor` 按时间量化选择完整 absolute waypoint；未变化通道不是新的 delta。每个 layer 对比自身 `phase.start_full12/last` 确定 touched；`proposed[i]=source_value`，不是 `+=`。
2. 当前新 layer 首次创建不回灌完整 anchor；handoff 的一次 `evaluate` 保留原 `nominal_full12/tracking`。所有 layer 时钟仍前进；另外 `_source_motion` 是逻辑源 bookkeeping，不是第二次物理 mapper。
3. `PhaseTransitionBridge` 保留可行 previous REQUEST，首交接 tick 用保持该请求的 raw 反解；只按硬 mask/safety 裁剪。后续 tick 再向本次真实 raw 的目标 slew，不全局归零。
4. `SemanticEpisodeEnv.step` 在同一次 8 tick 决策内继续新 phase；更新 HISTORY 而非 reset。只有真实 termination reason 才 done；phase label 不 reset mapper/观测builder/接触历史/Phi/GAE。
5. `apply_semantic_residual` 使用唯一 mapper advance；geometry 后再按真实 native+controller 的 2°余量裁有效 residual，最后原 hard 和 servo 1.25° final slew。REQUEST 与 effective/final 分账；当前 raw=0 但上一 REQUEST≠0 不误走零历史 fastpath。原子 8+4 只一次 write_data_to_sim。
6. WAIT_ENTRY 只是 lifecycle 标签：`SemanticControllerAdapter.step` 仍先 evaluate nominal，不用 entry_valid 禁止创造入口的动作。真实安全/任务终止才收尾 wheel=0；普通阶段没有停稳门或 mapper reset。

## 3. 全部 13 阶段：source owner、动作依赖与审查结论

下表源时长只是 compact 合同有限建议长度，**不是物理动作完成时刻或任务通过证明**。owner 列为本层会 touched 的通道；其他通道仍由此前 owner 保持/继续。所有行 mask 都为 12/12，E/L 引用上一表的完整 12 个 scale。

| Phase | 物理目的；本层 nominal owner / 源时长 | 当前动作依赖与期限 | 结论及一组正/反例建议 |
|---|---|---|---|
| P01 | FR 卸载准备；RL hip 0→37.6，四轮+.3后stop；13.267s；E | physical_valid 即可开始；30s task deadline；RL正向与 Recording 一致 | 保留目标腿仍加载时全身动作开放。正：FR仍承载可执行RL调整；反：低载但无有效支撑/响应不能仅用nominal时钟通过。 |
| P02 | FR 抬升/接近；FR knee0→45.9；.467s；E | Q+clear≥15mm+front[−.005,.10]；15s。当前Q/净空/other supports满足且尚远时+.3 assist | 动作开放；较先进FR已合法placed但clear不足/超过上界仍可被阻挡，交给共同验收修订。正：远处保持抬升并前送；反：不能用计时直接交P03或要求先进位置退回。 |
| P03 | FR 跨越/捕获；FL knee0→−22.9、RL hip37.6→17.5、FL−.79/RL+.61轮；1.2s；E | entry lifted_FR，不要求placed才开此层wheel；20s。没有FR专属pending-capture续轮或前腿geometry保护 | N1确证取消范围缺口；有限源结束后的捕获能力待验证。正：首次未placed仍有wheel协作；反：残差任取有限值也不能取消上述两个nominal轮值。 |
| P04 | 利用FR，为FL准备；四轮+.3/4.4s停、RL hip17.5→6.9；4.8s；E | placed_FR历史入口；edge_FL+role_prepared_FL；20s。RR动作不是本层显式owner，但其residual开放 | 不强制FL先AIR；FR当前支撑与history分开。正：FL已合法AIR连续接管；反：不能凭FR旧placed称它当前承载，不能把RL回向误作精确home门。 |
| P05 | FL摆动/捕获；FL hip0→48.2→22.8、knee−22.9→−36.7→−13.4、四轮+.3；9.733s；E | 30s；pending_capture可在未placed时续+.3，并要求实时support/几何/安全 | 原“placed之后才开捕获轮”的环已局部修复；首TOP(1/2)不会因air=false自动撤销。N2捕获后旧servo层继续；U1前腿降向未验证。正：1/2TOP仍续助；反：ground/非法/过前界不能盲续。 |
| P06 | 前部推进/后腿接近；仅四轮+.3；25.533s；L | 前腿历史placed入口，rear_approach要求两后轮edge；40s；finite-tail复用+.3，P06 gain受测量峰值退役 | P07/P08空间动作没有phase mask禁用；不存在25.533s必断轮的旧缺口；N3回退不复活。正：仍远且live时尾建议续；反：后owner轮值不应被P06再次相加或改写。 |
| P07 | RR接收侧/桥接准备；FL hip22.8→49.2→38.6、FR knee45.9→31.1、FR wheel−.63；1.667s；L | 两rear edge+role_prepared_RR；15s；FL hip限速60，其余150。receiver FL可AIR | 保留后续层下的有限继续、没有RR必须AIR入口。正：FR/RL实际支撑时FL AIR仍可准备；反：edge值不能独自称完整workspace/支撑已建立。 |
| P08 | RR减载/初始摆动；RL hip6.9→31.2；.4s；L | edge_RR+transfer_ready_RR；20s；继续P07，RR自身与全部wheel residual仍可用 | 不要求静止、固定receiver承载或历史膝速度。正：全身已造成RR有效AIR直接接续；反：单帧跌落低载/无有效响应不能通过。 |
| P09 | RR净空/跨越/捕获；RR hip0→55.6→−6.9、knee0→−37.8；后续+.3及FL/RL协同+FL−1.07；7.2s；L | 30s；当前Q且clear≥15mm且front<−.005可override四轮+.3；后腿nominal geometry可局部限降 | N2旧层/捕获适用性，U1非物理保证。正：延续P07/P08与RR前送；反：ground重试不冻结旧AIR姿态、不沿用被撤销Q。 |
| P10 | 利用RR，为RL准备；RR knee−37.8→−27.2；.133s；L | placed_RR历史入口，edge_RL+role_prepared_RL；20s；不含旧膝角/回弹速度门 | P10直接FR开空间owner尚在P11，靠前驱协同/全身响应或PPO可能形成准备；不是已证明循环依赖。正：小回弹但真实RR放置允许动作；反：历史RRplaced不等于当前RR支撑。 |
| P11 | RL减载/初始摆动；FR hip0→12.2→3.7；.467s；L | edge_RL+transfer_ready_RL；20s；P09协作层与P10 knee可继续 | 全12开放，FR receiver AIR不是动作拒绝项。正：RL已由全身作用离地保留；反：不借用旧window或接收腿名制造承载。 |
| P12 | RL跨越/捕获；RL hip15.4→.5→31.2→−10.1、knee19.4→35.3→−18.7、四轮−.3；4.667s；L | 30s；同RR的Q/clear/near-front正向assist与RL geometry；完成需placed_RL | 没有强制重新做标准单腿抬升；U1/N2仍适用。正：已有合法AIR连续carry；反：不能以P13标签补placed或掩盖RR回地。 |
| P13 | 整体恢复/停轮；四轮+.3到16.733s，16.8s恢复servo+轮[1.09,−.72,−.43,−.39]，17.8sstop；L | 源endpoint后向8servo zero+4wheel zero slew；60s、总200s。home不是hard唯一姿态；前段不是立即停轮 | 保留finite建议与residual；基本任务/严格质量由共同验收审查。正：不同关节姿态但真实受控完成；反：旧placed/P13标签不能掩盖回退/FALL，源endpoint不能自动成功。 |

13个局部期限实际均在 `observe_and_update` 比较 age 后产生 `INCOMPLETE_CONTROLLER_BLOCKED`；不是仅日志/nominal时长。所有 nominal evaluate 都仍运行到实际终止，不存在“MotionExecutor endpoint让PPO停止更新”的分支。本报告不裁定各期限是否物理合理；保留为 root 的统一期限审查输入。

## 4. 全部 12 处交接：保留证据、问题与正反测试建议

每一行都继承第2节共同不变量。以下是**应复用/补齐的测试建议，未在本次执行**；正例不是假造物理成功。

| 交接 | 所有权/依赖事实与分类 | 正例 | 反例 |
|---|---|---|---|
| P01→P02 | 保留：新P02仅FR knee，P01 RL hip和未结束wheel继续；不灌P02 anchor的RL37.6到尚未到位的当前值 | P01只跑短窗口即有效交接，RL目标继续slew，wheel保持既有事件时钟 | 未执行完RL动作不能在handoff被phase.start直接跳到37.6；P01 wheel stop事件不能永远丢弃 |
| P02→P03 | 保留：P02 FR knee仍owner，P03只接FL knee/RL hip/FL+RL轮；问题：P02窄域/先进状态拒绝及P03 N1 | 符合几何的FR跨阶段保留Q/当前动作，P03未placed即可协作 | 远处不得仅时钟切换下降；已经合法越沿/placed不能因旧上界必须退回；P03两个轮取消能力单独负例 |
| P03→P04 | 保留：P04没有新FR knee owner，不整向量恢复anchor；U：旧P02若未完仍可能动FR knee（N2） | 捕获后handoff首拍target/REQUEST继承，P04轮行开放 | 历史FR placed但当前AIR不应被日志称support；不得让待完成旧swing无条件拆除已得捕获 |
| P04→P05 | 保留：P04 RL动作可继续，P05先FL knee，FL hip到.2667s才新owner，wheel到.8667s才新owner | 已有效FL离地时不强制先落地/复做姿态；旧RL动作继续一次 | 不允许P05 start_full12把RL提前回6.9；源未touched轮通道不能清掉前层动作 |
| P05→P06 | 保留：P06只接四轮，FL servo和动作历史保留；问题：N2旧P05后续姿态仍会执行 | 未placed的1/2TOP仍在P05续捕获；真实placed后首拍不清空FL请求 | 不能先要求placed才给全部捕获协作；早placed后的源后续旧目标必须有明确适用性处理而非默认继续 |
| P06→P07 | 保留：P06 rolling仅自身gain，P07先FL、稍后FR knee/FR轮；问题N3峰值退役无恢复 | P07 FR轮−.63接手后，P06不能再加+.3；其余轮按各owner | 短暂rear进区后回退且未完成时，不把不可恢复gain=0说成当前距离已完成；不能重建旧前缀P06队列 |
| P07→P08 | 保留：FL/FR/P07轮继续，P08只接RL hip；短P07不表示动作已结束 | FL AIR但其他支撑有效、RR进展合法时连续交接且FL名义rate≤.5°/tick | 不要求FL永久承载或强行50/50；P08新anchor不能把FL立刻拉到38.6 |
| P08→P09 | 保留：P08 RL hip继续；P09先RR hip，再knee/轮/协同；U2旧hip保持域可能不足 | 前面形成的AIR/速度/Q和REQUEST保留，RR自身q变化小不重置 | source负knee不能被自动解释world-down；旧参考膝目标不应成为重新开始门；无证据AIR不能升级placed |
| P09→P10 | 保留：新owner仅RR knee，旧RR hip及FL/RL协作仍按P09；无历史回弹入口门 | 已真实RR Q/C/P且低回弹可执行P10，首拍q/target不重置 | 不能凭history_RRplaced冒称当前TOP；P09未完协同对已placed RR的影响需明确，不以旧负膝角准入 |
| P10→P11 | 保留：P10 RR knee继续，P11新FR hip；直接FR开空间从本阶段开始 | 不同RR当前姿态与FR AIR仍保持动作开放及同一mapper | P10很短不是bug；但只有RR历史没有真实当前支撑，不能填成“已利用RR支撑”；不得双加RR膝delta |
| P11→P12 | 保留：P11 FR hip继续，P12 RL hip/knee及随后轮接手；Q/尝试历史不按phase清 | RL已自然AIR/Q，继续源动作与名义局部geometry，不要求标准重抬 | RL low-load无持续响应不能凭label认转移；ground回退撤销本次Q但不抹其他腿历史 |
| P12→P13 | 保留：首拍不home、不停轮；P13先轮+.3、随后才恢复。问题：N2早捕获后P12剩余RL源仍owner | RL真placed后首拍保留REQUEST/final历史，既有捕获姿态不瞬跳统一home | 四腿history不代表当前support；不能让末源/P13恢复破坏捕获却截掉尾段，也不能以17.8s endpoint自动成功 |

## 5. prefix 与自然 P01 的队列差异必须保留在证据中

- 自然 P01 B/C：同一 `NominalMotionProvider` 从P01累积真实已创建层。checkpoint_policy prefix 复用同一个 current core，仅前缀 actor 是独立冻结副本；前缀不进PPO，接管不重建队列。
- frozen_fsm：shadow supervisor观测老师真实轨迹；handoff用老师上一真实 nominal/tracking/mapper receipt 创建 `from_handoff` provider，只创建目标阶段及以后实际运行层。不能宣称已恢复新版P01起的全层队列，也不能人为补全旧P06 layer/retirement peak。
- 两种prefix都不是历史传感器快照注入；Q/C/P、AIR/current support必须读真实handoff。offset前目标离开/terminal会按真实失败fallback，不能把请求阶段当实际信用起点。
- 本轮不用prefix差异建立新训练gate；它只是解释动作所有权证据来源的必要限制。

## 6. Recording 与单位依据（一次有限复用）

已读 compact 合同 P01–P13 的全部 start/end、源时长、active/touched 通道和 waypoint 命令；未重跑9版本。复用既有 `recording_mechanism_evidence.md` 的五个重叠窗口与 `recording_evidence.json`：成熟 `recording_fast_plan.py::fast_plan_rows` 直接调用 `playback.plan_from_steps(profile='fast')`，不是本次重写parser。原稀疏命令保持未出现通道，明确stop才停；跨step绝对servo状态不是再次相加的delta。Fast计划时长、人工录制时长、合同实际120Hz时间不混用。

关键旧证据：v010 P05末FL在台面上方但AIR/0N，P06轮行后才TOP/5.510N；支持捕获与轮行重叠，不支持AIR直接placed。RR准备时FL AIR、FR/RL承担实际支撑；RL准备时FR AIR、FL/RR承担支撑；不是永久名单或固定载荷比例。

FR knee“约−40°”：既有9版本检索未发现FR绝对−40命令；v010后腿准备是45.9→31.1（delta−14.8），实测q46.269→31.205，adapter target31.225，三个字段不同。UI/JSON是相对standing的canonical absolute；native物理目标 `rad(standing+sign*canonical_final)`，真实q另测。FR sign=+1，rear sign反向，不能用同角度符号推world-z。−40只是合法范围内可探索候选，不是本报告成功/入口要求。

## 7. 可复用测试位置与完成边界

建议优先复用而非新建13套框架：

- `test_semantic_continuous_v3.py`：whole-body AIR、尝试撤销、未完P08跨P09层、bridge/history。其显式bridge参数列表只有9对；补齐 P03→P04、P04→P05、P06→P07 的统一12对表，不能把已有9对称全覆盖。
- `test_semantic_transfer_roles.py`：真实当前support、receiving AIR、pending_capture与时间window；扩首TOP1/2仍续辅助与后续真实placed owner场景。
- `test_semantic_p06_wheel_tail.py` / `test_semantic_p06_rolling_retirement.py`：后owner优先、finite endpoint、terminal不赚新peak、回退不复活反例。
- `test_semantic_fl_hip_temporal_authority.py` / `test_semantic_front_wheel_authority.py`：已有实际mapper/bridge seams，补P03 FL−.79、RL+.61无法取消的定向反例，不引入新硬限。
- `test_semantic_nominal_geometry_*`：真实q/J、native同一mapper结果与无current-r归因；front阶段当前没有接线，不能靠rear tests声称已保护四腿。

本次只交付此审查报告；没有新增或执行测试，没有实施上述候选，没有更改旧报告或成功率。应先修可定位共享/局部问题，再由 root 按最小受影响范围验证并继续真实学习；上述待验证控制假说不应变成新的全阶段预先成功门禁。

## 源文件定位

代码引用均相对以下仓库根，报告固定对应上述 HEAD：`C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`。

- `configs/ppo_semantic_v3/stage_task_spec.yaml`：nominal配置约80–114，13阶段约120–236；`execution_profile.yaml`：caps31–44、rate/mode约47–68。
- `src/wlr50_clean/ppo/semantic_supervisor.py`：predicate599、stage推进811、NominalMotionProvider897、retirement977、from_handoff1010、continuous1025、evaluate1108、controller.step1206。
- `src/wlr50_clean/fsm/motion_executor.py`：start_phase187、tick226；`configs/recording_motion_contract.json`：各phase完整source。
- `src/wlr50_clean/ppo/semantic_backend.py`：build_semantic_projector65、dispatch路径约246；`semantic_residual_adapter.py`：apply23；`semantic_nominal_geometry.py`：ACTIVE23、capture55、correct198。
- `src/wlr50_clean/ppo/phase_action_masks_v2.py`：PhaseTransitionBridge256；`action_projection.py`：ActionProjector386；`semantic_env.py`：reset110、step136。
- `src/wlr50_clean/ppo/semantic_transfer_roles.py`：角色窗口与pending_capture约135–228；`semantic_prefix.py`：老师handoff约155–203；`semantic_checkpoint_prefix.py`：reset153、_roll_in约220。
- 既有输出：`outputs/ppo_transfer_roles_v1/recording_mechanism_evidence.md`、`recording_evidence.json`、`rear_handoff_authority_diagnosis.md`；此报告未改写它们。
- 成熟parser：`C:/robotics_sim/wlr_robot/height_based_obstacle_replay/fsm_50mm_recording_derived_v3/recording_fast_plan.py`，`fast_plan_rows`183。

## 实施补记：定向 nominal 修订及 CPU 边界

上述章节保留为修改前的只读结论。本节记录随后获准实施的独立范围，不把代码测试当真实物理覆盖。

- 仅 `NominalMotionProvider` 在 `physical_acceptance_version=all_stage_v1` 时启用：首次观察到真实 placed 历史，撤销当时已存在层对该腿两个 servo 的旧所有权，保持当时连续 nominal，而非恢复入口角、actual q 或最终执行目标。源时钟仍继续，其他腿与所有 residual 保持开放。
- 另一个独立边界是新建 P02/FR、P05/FL、P09/RR、P12/RL swing 层：只有当前 placed、within_top_xy、非 GROUND，且当前 TOP 或台面上方 AIR（clearance≥0）时，才抑制新层目标两 servo；历史 placed 但已回地、区域外或台面以下 AIR 不跳过。既有协作通道逐值保持，未来 P07/P11 接收侧及 P13 恢复层可以重新取得所有权，不永久锁腿。当前证据缺失或非法时在源时钟变化前拒绝。诊断单列 `suppressed_new_swing_layers`。
- P06 wheel gain 在新模式下使用 `1−当前真实工作区进度`，而非 `1−历史峰值`；0..1 有界，可在退回后恢复原有限幅度建议，保留原 wheel slew、后 owner 优先级及 terminal 不激活规则。旧峰值仍为诊断字段。局部恢复预算与整体200秒由 root 的监督器改动负责，不在此专属测试中宣称已完成真实 deadline 验证。
- 新 `execution_profile.yaml` 只改 revision 与已批准定向 cap：P03–P05 的 FL/RL wheel 各 1.0；P06–P13 的 RL wheel 延续 1.0，原 FL wheel 1.2 不动；P01/P02 和全部其他通道不动。最小后续延续是现有 phase-cap 单调约束所需，未放宽真实 hard limit 或 slew。P03 nominal FL−0.79/RL+0.61 因而可用有限 latent 表达净命令 −0.02/0/+0.02；这是命令可达性，不是物理响应证明。

最终专属 CPU 测试 `tests/unit/test_all_stage_nominal_transitions.py`：**71 passed、0 failed、0 error、0 skipped，4.988s**，receipt `nominal_transitions_trial02.xml`。覆盖13阶段无capture旧路径一致、全部12交接的 nominal/request/history 保持、capture旧owner退役、新swing正反例、后续恢复owner、P06恢复/后owner/terminal、非法证据先拒绝，以及cap定向范围与有限latent反向。此前 `trial01.xml` 的50项是同一套测试较早子集，不能与71相加。

此实施没有运行 Isaac，没有检验实际净空、承载、碰撞或完整任务成功。当前 nominal hold 不等于真实姿态保持，也不自动重放已退役的摆动；回退重试仍由开放 residual、当前任务调度和后续合法 owner 完成。一般150°/s nominal 与60°/s residual 的其余通道差异、以及前腿未接线 rear-only 几何投影，仍维持前述有界诊断，不因本次定向修订宣称全部物理控制风险消失。
