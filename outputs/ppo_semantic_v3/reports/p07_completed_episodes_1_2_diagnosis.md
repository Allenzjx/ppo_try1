# P07已完成episodes 1–2：连续后腿、控制与奖励信号

## 固定范围与结论

仅使用 run `20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5` 截至第三条真实终止的已完成数据。ep0已有详报，本报告只作对照，不重复checkpoint/hash/master累计账本。

| 项目 | ep0（对照） | ep1 | ep2 |
|---|---:|---:|---:|
| 固定global范围 | 73089–73166 | **73167–73618** | **73619–74070** |
| 已完成PPO决策 | 78 | 452 | 452 |
| P07/P08/P09决策数 | 1/1/76 | 1/1/450 | 1/1/450 |
| 信用起点 | tick5952 /49.6s /P07 | 同左 | 同左 |
| 终止 | BODY_COLLISION | P09 INCOMPLETE_CONTROLLER_BLOCKED | 同ep1 |
| 终止physics tick /全物理时间 | 6574 /54.783333s | 9568 /79.733333s | 同ep1 |
| 信用段物理时长 | 5.183333s | 30.133333s | 30.133333s |
| RR新initial / Q事件数 | 2 /1 | 16 /3 | 9 /3 |
| RR新C / P事件数 | 0 /0 | 0 /0 | 0 /0 |

两回合不是简单“没能抬腿”：均多次真Q，接近前沿后退回；ep1末已GROUND且Q撤销，ep2末仍Q+AIR却离前缘180mm。相较ep0，本两回合**没有记录物理安全失败**并运行到P09的30s任务deadline，但这不足以称稳定性改善，更不是成功。first unfinished始终RR crossing/placement，未进入P10/RL任务。

数据支持“有执行、能真实抬升、未完成横向推进/保持”的有限结论。看不到纯传感报错、整阶段PPO请求被禁用、PBRS公式断裂或nominal/residual双重smooth处罚的证据；同时无法凭单批随机轨迹排除接触模型、牵引条件、critic学习速度或其他控制耦合问题。未选择生产修复。

## 数据与时间绑定

- JSON仅解析前982行（78+452+452），终止global严格为73166、73618、74070；未读后续非terminal tail作失败。
- 一次CPU脚本执行读取 `rollout_000537.pt` 至 `rollout_000544.pt`，只解码ep1/2的904个真实324维观测。八个文件runtime一致；schema及实际编码源小文件hash相符，policy/critic观察相等、raw action/old_value/done与对应audit行精确匹配。无clip±20触及、无非有限值；未构造模型、未优化。
- saved observations是**predecision**，例如最后global73618/74070的观测是tick9560，动作之后的terminal current state是9568。下文姿态/关节表注明pre；几何、loads和reward累计使用每决策末current state。
- 编码定义同首回合报告：q/qd为实际canonical逻辑deg/deg/s，wheel速度为canonical rad/s；body线/角速度组是body轴，CoM速度为world轴。CoM为13刚体质量加权，不是base-only COM。
- 只有决策末物理snapshot、事件真实tick和逐tick native/hold标志；没有完整120Hz raw geometry/contact点/力向量，不把边界统计乘8当持续接触时间或宣称取得每个子tick峰值。
- CPU线程/interop均1、map_location=cpu、无CUDA/Isaac，唯一执行exit0/4.490862s。产物同目录 `p07_completed_episodes_1_2_decode.py/.json`；无真实权重或任何训练计数写入。

## 真实后腿事件与最近几何

### Episode 1：三次Q均在越沿前回GROUND撤销

- initial真实ticks：6022、6263、6757、6869、6944、7622、7719、7772、7889、7931、8184、8506、8560、8753、9383、9439。
- Q6395 → GROUND撤销6736；Q6774 → 撤销6832；Q6976 → 撤销7518。
- 452个末边界中Q=true117次，但C/P始终false。后续initial不是已完成的cross，更不能把initial重复计为placement。
- 最近前缘：**tick6576 front−9.416934mm /clear+32.601075mm /AIR/load0**。距离现有tolerance-expanded top XY仍有4.416934mm outside；中心未越front plane。
- 末边界净空峰值：tick6584 clear+39.165066mm，同时front−14.909383mm。最高与最近不是同一时刻。
- 终止tick9568：RR **GROUND**，front−405.837450mm、clear−47.308757mm、load.060309、Q/C/P均false；较最近前沿明显退回。

### Episode 2：前两次Q撤销，第三次保持到deadline但仍未cross

- initial真实ticks：6048、6144、6338、6780、6934、7020、7062、7099、7131。
- Q6386 → GROUND撤销6754；Q6800 → 撤销6908；Q7184 → 保持至9568。452个末边界Q=true359次，C/P始终false。
- 最近前缘：**tick6544 front−17.734047mm /clear+30.088053mm /AIR/load0**，outside XY12.734047mm。
- 末边界净空峰值：tick8008 clear+56.173966mm，但front已−142.199096mm。不能把更高解释成更接近放置。
- 终止：RR **AIR**，front−180.413839mm、clear+32.287807mm、Q=true、C/P=false、load0、outside XY175.413839mm；当前连续AIR count2445，表示当前AIR段并非末端短暂离地。

这里hard Q会在实测GROUND且未cross时撤销，与记录的回地相符；没有“只因阶段label变化清Q”的证据。亦不能要求RR自己必须新动2°：事件记录包含whole-body响应，已有合法initial有own motion<2°。

## FL历史placement与当前承载完全分开

FR/FL的Q/C/P在tick5952教师后缀信用起点前已存在，不能记为本课程新增PPO硬事件。两回合均从同一5952真实读回起点开始（选定q、qd、姿态、速度等保存组相同），首PPO动作后当前接触分化；不能沿用教师placed bit当持续FL承载。

| 452个决策末的当前类别 | ep1 | ep2 |
|---|---|---|
| FL | AIR415、TOP37 | AIR233、TOP219 |
| FR | TOP451、AIR1 | TOP452 |
| RL | GROUND451、AIR1 | GROUND452 |
| RR | AIR242、GROUND208、obstacle非TOP2 | AIR401、GROUND44、obstacle非TOP7 |

负载是当前exact pair force标量份额，不是独立的竖直支撑力分解。

| 终止腿状态 | ep1：front / gap（mm），载荷 | ep2：front / gap（mm），载荷 |
|---|---|---|
| FL | AIR；−142.329 /+77.602，0 | TOP；+96.371 /+.755，.080770 |
| FR | TOP；+62.825 /+2.307，.471635 | TOP；+263.990 /−1.903，.473484 |
| RL | GROUND；−589.307 /−51.672，.468056 | GROUND；−384.949 /−48.841，.445747 |
| RR | GROUND；−405.837 /−47.309，.060309 | AIR；−180.414 /+32.288，0 |

ep1不止RR退回：FL也已移到front后方且AIR。因global potential同时含各腿当前物理进度/retention，不能把所有phi降低孤立解释为RR一条腿。ep2终止FL/FR有TOP载荷而RR仍悬空；同样不能反过来把FL AIR规定成错误或强制某个支撑构型。

## 姿态、速度与“较安全”措辞边界

| 决策末统计 | ep0 | ep1 | ep2 |
|---|---:|---:|---:|
| body线速均值 /最大（m/s） | .078001 /.241587 | .081967 /.282004 | .075163 /.274011 |
| body角速norm均值 /最大（rad/s） | .286687 /.916013 | .288098 /1.316335 | .313745 /1.271995 |
| 终止body线速 /角速norm | .007641 /.071983 | .105663 /.165856 | .081509 /.202082 |

ep1/2虽然没有BODY_COLLISION/FALL等已记录物理失败，但上述速度统计并非一致下降，也非静止保持，不能称“已学会稳定”。

| 最后动作前 | RPY（deg） | base线速度 body轴（m/s） | 全机器人CoM速度 world轴（m/s） | 实测四轮canonical速度（rad/s） |
|---|---|---|---|---|
| ep1，pre tick9560 | [10.937, -14.581, -1.576] | [0.0741, -0.0150, 0.0393] | [0.0427, -0.0234, 0.0183] | [0.1731, -0.4161, -0.2499, -0.5154] |
| ep2，pre tick9560 | [-7.283, -7.382, -1.296] | [-0.1531, 0.0083, 0.0082] | [-0.1249, 0.0128, -0.0008] | [0.8612, -0.1747, 0.0721, 0.1509] |

ep1 pre roll范围−11.617°至+15.794°，pitch−19.121°至−.023°；ep2 roll−11.376°至+13.830°，pitch−19.143°至−.922°。这是重力固定参考下实际姿态，不是与A历史姿态的误差。

顺序FL hip/knee、FR hip/knee、RL hip/knee、RR hip/knee：

| 最后动作前 | 实测q（逻辑deg） | 实测qd（逻辑deg/s） |
|---|---|---|
| ep1，pre tick9560 | [-30.84, -39.67, -3.38, 16.62, 7.51, 16.10, -10.21, -47.15] | [1.86, 17.37, -26.48, -3.32, 6.59, 25.09, -6.79, -2.38] |
| ep2，pre tick9560 | [-31.06, -40.67, -9.40, 12.57, 13.06, 25.26, -12.80, -47.16] | [-9.72, -14.85, 44.61, 11.20, -19.25, -29.24, -8.45, 0.55] |

这些是pre9560，不是末9568；例如ep2 pre body Vx=−.153084m/s，而末body speed norm=.081509m/s。不能将两时刻速度互换以解释瞬时接触。姿态和全身响应均在变化，不是整段执行冻结。

## nominal/residual相抵：两回合后段机制不同

四轮顺序FL/FR/RL/RR，均使用canonical向前符号；PhysX FL/RL轴符号相反，在此已不混用。末150个决策为各自固定g73469–73618、g73921–74070，覆盖最后10秒动作；该窗口的统计为：

| 回合 | nominal均值 | residual均值 | 逻辑actual drive均值 | measured wheel速度均值 |
|---|---|---|---|---|
| ep1 | [0.0000, 0.0000, 0.0000, 0.0000] | [0.1285, -0.3407, -0.3868, -0.0438] | [0.1285, -0.3407, -0.3868, -0.0438] | [0.1281, -0.4343, -0.2745, -0.0927] |
| ep2 | [0.2658, 0.2658, 0.2658, 0.2658] | [0.1241, -0.3458, -0.3694, -0.0167] | [0.3899, -0.0799, -0.1036, 0.2492] | [0.4037, -0.1344, 0.0345, 0.2498] |

- **ep1后段不是长期“残差抵消非零nominal”**：最后一个非零nominal末边界是7472（四轮+.075）；7480–9568的262个末边界四轮nominal均0。其末150决策的drive等于残差，FR/RL平均−.340711/−.386807，实测轮速平均−.434308/−.274483；这是真实反向请求与运动，不是nominal清零顺带把PPO关掉。相同晚段RR末边界front继续退157.735mm。
- ep1 terminal四轮nominal0，residual=drive=[+.117556,−.225964,−.328525,−.052364]。
- **ep2确有长期相抵/反转**：晚段nominal平均每轮+.265833，而FR/RL residual均值−.345766/−.369438；actual均值变为−.079933/−.103605。150个末边界中FR/RL有102/112次actual与当时非零nominal方向相反；RR大多数仍向前。同期RR front退32.174mm。
- ep2 terminal nominal四轮+.3，residual=[+.086958,−.407018,−.407546,−.033607]，actual=[+.386958,−.107018,−.107546,+.266393]。
- ep2 RL measured均值+.034485与负actual均值不同，提示接触/外力耦合下读回不是理想速度指令的逐点复制；没有轮地接触点/牵引力矩或完整tick数据，不能单凭该差异判断执行无权限或传感错误。

两回合各3616 physics ticks中native审计全verified，own-phase-request effect均3614（两个交接首tick排除），所有in-episode state-write检查true。P07/P08普通交接不terminal，其后仍有真实请求效应；不支持“整段raw无执行权”的假设。它也不意味着任意原始12维请求都能无约束实现：合法mapper、servo限位、slew、接触动力学仍在，不能把native effect布尔当作无限执行能力。

## 奖励五family的实际有符号累计

以下仅各已完成信用段的环境奖励，不含教师前缀；不是entropy、PPO loss或优势。ep0时长不同，不能直接用累计绝对值比较稳定性。

| family（已乘生产权重） | ep0：78决策 | ep1：452决策 | ep2：452决策 |
|---|---:|---:|---:|
| task_progress | -43.566181033 | -48.672390489 | -49.232344648 |
| body_stability | -0.080034326 | -0.581900280 | -0.407343240 |
| contact_motion_quality | -0.000074438 | -0.001130540 | -0.001053926 |
| control_smoothness | -0.245143813 | -1.307663151 | -1.246582420 |
| control_regularization | 0.000000000 | 0.000000000 | 0.000000000 |
| **总计** | -43.891433609 | -50.563084460 | -50.887324235 |

task_progress进一步分解：

| 分量 | ep0 | ep1 | ep2 |
|---|---:|---:|---:|
| potential shaping累计 | −3.462514366 | −8.069723823 | −8.629677981 |
| 真实终止event | −40 | −40 | −40 |
| elapsed-time成本 | −.103666667 | −.602666667 | −.602666667 |
| 正shaping决策数 | 11 | 44 | 35 |
| 正shaping累计 | +.671531489 | +2.967953980 | +2.153052540 |
| 负shaping累计（含吸收态） | −4.134045855 | −11.037677803 | −10.782730521 |

两长回合smooth成本约1.31/1.25、body约.58/.41，contact约.0011，远小于−40真终止，但不能据累计量直接认定critic已经正确学到相应目标或小成本对局部动作无影响。

### 是否重复惩罚或“主动AIR自动罚rebound”

生产 `smoothness_components=applied_only`：只用actual first/second difference平均并乘.1，不把记录的nominal/residual diagnostic再加一次。

- ep1 diagnostic actual first=19.259710028、second=6.893552995；`−.1×(first+second)/2 = −1.307663151`，精确对应family。
- ep2 actual first=18.773533298、second=6.158115105，同式=−1.246582420。
- nominal/residual first difference虽有记录（ep1 .715752/17.632745，ep2 .582418/17.759609），**不是额外加罚**。
- 两回合 confirmed_post_touchdown_rebound累计均0，尽管RR/FL频繁AIR；没有本样本中“主动AIR自动被rebound罚”的证据。
- touchdown diagnostic分别91、122；contact chatter分别1.683333、2.291667是时间积分诊断，不把它们直接当已乘权重的contact奖励。regularization开关false，family为0。

### 局部进展与其他成本确有tradeoff，不等于已证明冲突主因

真实正/反例（均非terminal）：

| 回合/global /末tick | RR front前进 | gap前→后 | shaping | body /smooth成本 | total |
|---|---:|---:|---:|---:|---:|
| ep1 g73173 /6008 | +3.344mm | −50.262→−50.064mm | +.002070026 | −.000433793 /−.003079580 | **−.002776681** |
| ep1 g73175 /6024 | +2.015mm | −50.036→−45.709mm | +.083085431 | −.000575636 /−.003112564 | **+.078063898** |
| ep2 g73642 /6144 | +7.651mm | −49.225→−46.117mm | +.053855469 | −.001119861 /−.003190222 | **+.048212052** |

ep1/ep2分别6/7个决策是shaping>0但total<0，证明局部小进展可被已有运动成本盖过；同时有明显正total反例，不能说环境把所有主动抬升/前进都惩罚了。global phi还包括FL/FR/RL当前状态，表中不是控制其他腿不变的因果隔离实验；不能由一条负total或负advantage推断奖励设计必然与任务冲突。

## PBRS连续性与真终止

对固定全部982行（包含ep0比较）的逐决策实际double日志检查：

- `F = 5×(.995×phi_after − phi_before)` 最大误差 **0**。
- 每回合内 `next_row.phi_before == previous_row.phi_after` 最大误差 **0**，包括普通P07→P08→P09交接。
- 非terminal的reward phi_after与同row物理task potential一致；terminal强制吸收态phi_after=0，误差0。跨fresh-prefix reset不要求前回合吸收态和新初态phi连续。
- 两回合初态phi同为.4874797925854255；ep1末实际物理phi=.4283578455003407、ep2=.5485562348257462，但reward末phi均为0，正确区别physical snapshot与吸收态。
- terminal前phi分别.4267047511714009/.5501538882296245，吸收态shaping分别−2.133523755857/−2.750769441148；另−40终止event，不是把同一个family重复计两次。
- 三个失败段的折扣shaping总均约 **−2.43739896292713 = −5×初始phi**（误差约1e−15），符合有限真终止的PBRS telescoping。ep2较高潜力轨迹的未折扣shaping累计更负，不是符号bug；不能用整回合未折扣sum替代局部学习信号或成功排名。

所以当前实现有真实的中间正/负task信号，普通相切不清零potential，但同起点失败轨迹的折扣shaping最终回到同一边界项，这是设计数学性质，不是本诊断建议打破的吸收态规则。保留原主规范：不加阶段/入口/AIR事件奖金，不保留true-terminal phi。

## 安全/任务截止与可证伪的结论边界

ep1/ep2每条末snapshot均physical evaluator valid=true，termination_reason=null；task supervisor在P09 age30.0触发INCOMPLETE_CONTROLLER_BLOCKED，episode全物理时间79.733333<200s。不是BODY_COLLISION、轮式越障非法判定、NaN、FALL、旧入口姿态检查或200s全任务超时。RL尚未开始任务，真实first unfinished是RR crossing/placement。

目前可支持/不能支持的区分：

- **可支持**：两回合避免了ep0那次已记录body collision，继续执行到任务deadline；RR有真实Q、有几次接近前缘，但最终未cross并退回。不能升级为普遍“更安全、更稳”。
- **未见纯传感错误的正证**：904真实obs有限且未clip，exact contact/current geometry/history彼此能解释当前GROUND撤Q、AIR保持等记录。没有完整raw就不能排除任何传感/几何标定误差，更不能因没成功臆断sensor bug。
- **不支持整段执行权限不足**：几乎全部tick有own native request效应、实际关节/轮速持续变化；合法限位/slew与接触约束仍可能限制某些请求的物理效果，不能用布尔audit全排除。
- **可见策略/建议相抵与后退相关**：ep2 FR/RL目标常反向，ep1晚段nominal0后残差仍反向；这值得后续固定策略/目标日志检验，但不足以单独解释姿态、牵引与前缘净空的耦合。
- **有局部成本tradeoff，但未证明致命reward冲突或缩放bug**：PBRS和applied-only成本均数值闭合，主动rebound成本0，有正total进展样本。没有额外实验可把“优化不足”与“目标/牵引/信号分配不足”唯一分开。

当前运行不因报告新增门禁或参数变化。只提交这份固定已完成窗口的事实与未知；后续非terminal tail不计失败，不轮询新训练。

