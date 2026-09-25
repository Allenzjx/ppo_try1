# 两次学生 RR 捕获后的首次掉载：892385 窄只读分析

**结论：两次最早 AIR 都不是 late-owner 覆盖造成；它们均短暂重接，之后才发生持续掉载。共同的后续风险是 FR knee 持续负向 REQUEST，把正向 nominal 抵消到 −58°命令边界，并最终实测越过 −60°硬限。wheel 的 source stop／反向组也在相关窗口真实发生，不能全归给单个关节或 policy。**

源为已封存 P07 512 块 `20260924T1527536734451Z_g892385cba8a7_f00e6fb48d074b26a2a98898505632e6`，仅提取 episode 1（225867–225979）及 episode 4（226065–226200）的 JSON 审计；同时核对所指定 P12 首442决策、CP225792 DET 报告和当前源实现。没有 Torch、模型、Isaac 或生产修改。数字与完整向量见同名 JSON。

## 1. 先区分首次短暂 AIR 与持续丢失

关节顺序 FL/FR/RL/RR，每腿 hip/knee；角度为 canonical °。表中实际角度来自同拍 tracking receipt 的**端点前1 tick**原始传感器，按 `nominal_deg−current_actual_canonical_error_deg` 精确重建；接触/力/gap 为端点本身。不是把 target 当 actual。

| episode / tick | RR当前接触；力N；gap mm | RR FINAL hip/knee° | FR knee FINAL / 实测前1tick° | 已提交 owner active（下一输入对齐端点） |
| --- | --- | --- | --- | --- |
| 1 / 5696 | TOP；12.543；−.711 | −10.317 / −58.000 | −41.899 / −34.566 | 无 |
| 1 / 5704 | AIR；0；−.424 | −14.317 / −58.000 | −45.899 / −38.403 | 无 |
| 1 / 5792 | AIR；0；−.532 | +5.183 / −44.880 | −58.000 / −58.240 | RL hip/knee |
| 4 / 6048 | TOP；5.114；−.965 | +13.821 / −48.002 | −29.703 / −23.864 | 无 |
| 4 / 6056 | AIR；0；−1.048 | +13.463 / −44.017 | −33.203 / −26.745 | 无 |
| 4 / 6112 | AIR；0；+.006 | −12.674 / −45.305 | −58.000 / −53.939 | FL、RL hip/knee |

- episode 1：学生 `placed=5697`。5704 的连续 AIR 计数为1、free-air参考5703，确认该拍刚离接触；5720 又有合法 TOP 5.956N。随后5792的计数3定位当前持续 AIR 起点5790，之后直到6061终止未恢复；本次是**合法XY内失去接触**，不是退出落脚区。
- episode 4：学生 `placed=6033`；6032已测TOP 16.393N。6056计数1，6064又测TOP 1.208N；6112计数5定位当前持续AIR起点6108，直到6246终止未恢复。两次首次掉载时XY均合法。
- 仅有决策端点接触、当前连续计数和事件 tick，**没有完整120Hz接触序列**。可定位上述 AIR 起点，但不能排除相邻端点间更早低力TOP或短暂重接；不把5704→5720、6056→6064臆写为整段连续AIR。
- 首次掉载时其余腿仍实测承载：5704 FL/FR/RL = 2.367/14.962/11.237N；6056 = 1.171/13.126/10.444N。AIR RR未计入支撑。

## 2. 第一处控制差异与责任边界

**episode 1：首次掉载前后 source 与 mapper N 8关节均保持。** RR N hip/knee=−6.9/−37.8，mapped=−9.4/−39.05；5696→5704 RR hip REQUEST −.917→−4.917，FINAL −10.317→−14.317，实测前1tick −4.399→−8.207。RR knee REQUEST −20.327→−20.093，但都被headroom裁成同一个−58° FINAL；FR knee REQUEST同时−71.749→−75.749，FL/RL请求也变化。当前P09 sourceclock334，未到late648门，owner公开17全0，**12维FINAL与同拍headroom candidate差为0**。所以不是新source接管、nominal跳变或owner把RR目标抬走；也不能由共变日志证明“负RR hip导致掉载”。

**episode 4：P10必要接续与首次掉载重叠，不能粗暴全部推迟。** 6040→6056 RR knee source −37.8→−27.2、mapped −39.05→−27.2；REQUEST −17.744→−16.817，FINAL −56.794→−44.017，实测前1tick −56.429→−50.082。RR hip FINAL约14.513→13.463，未大幅负收。FR knee FINAL −26.203→−33.203，同时轮速 source stop 生效。6056的P09 clock648正在等RR bearing，late组尚未发，owner active全0，**FINAL仍等于headroom candidate**。这包含真实source knee展开与policy全身变化，不能只称policy退回旧目标。反例：episode 1 的相同P10 knee展开发生于5712→5720，并伴随恢复TOP接触；现有证据不支持全局禁掉该捕获/重构动作。

全身跟踪也不可省略：episode 4 / 6048 的FR hip FINAL +7.940°，实测前1tick +26.201°（约18.26°差），当时FR仍TOP 11.662N。不是仅凭nominal或目标角就能断定实际接收构型已经形成；这份日志不能单独区分负载耦合、动力学与驱动跟踪的因果份额。

`capture_owner_hold`明确保持的是**捕获时 nominal**，不是实际关节或已捕获FINAL。RR receipt在5697/6033都记 held N=−6.9/−37.8；后续policy仍可变，P10也可合法接管。没有“placed自动锁定捕获FINAL”的机制，不能把receipt名称误读为此保证。

**owner后续贡献有限且可区分：** 该层仅管FL/RL四个servo，永不直接改FR/RR或wheel。episode 1 到5792才有RL owner suspension；episode 4 到6064已经发late全身组，公开active四个，6072再次释放，6112掉载后再次suspend。下一输入439尾部给出上次已提交anchors/active，能排除首次掉载时owner覆盖；但原生owner逐tick receipt未存，后续 `FINAL−headroom candidate` 是owner与最后slew等的**合计**，不可都算policy或都算owner。

## 3. 四轮变化确有两种来源，不是mask

以下全部已是 canonical forward-positive，FL/FR/RL/RR，rad/s；物理 evaluator 的 wheel readback也是canonical，不能再次乘native轴符号。

- 5704：N=+.3四轮；REQUEST=[−.731,−.142,−.104,−.213]，FINAL=[−.431,+.158,+.196,+.087]，实测=[−.393,+.165,+.230,+.087]。FL反向是policy抵消，非late−1.07脉冲。
- episode 1 5784→5792（持续掉载起点所在区间）：**P12 source四轮+.3→−.3**，对应原contract 0.4667s的有限反向组；5792 FINAL=[−1.399,−.482,−.322,−.319]，实测=[−1.434,−.320,−.397,−.320]。这是source与policy共同造成的反转。该脉冲后续stop正常，6061 N四轮均0，不能声称stop丢失。
- episode 4 6048→6056：P09 source明确四轮stop，N+.3→0；6056 FINAL=REQUEST=[−.327,−.161,+.070,−.167]，实测=[−.336,−.193,+.186,−.177]。stop没有屏蔽合法残差，因此FR/RR此时反转来自policy，并非nominal遗漏。
- episode 4 late组在6056以后才发：6064 N FLwheel=−1.07，policy另加−.447，FINAL−1.517；6112进一步−1.821、实测−1.870。它不能解释最早6056 AIR，但参与后续全身运动，不能只看stage=P11而忽略仍在执行的P09 late源动作。

## 4. FR knee 饱和先于安全中止

| episode | 首个FRk FINAL−58端点 | 该FINAL端点数 | 终止tick / 实测FRk° | N / mapped N / 终止REQUEST° |
| --- | ---: | ---: | --- | --- |
| 1 | 5736 | 42 | 6061 / −60.018756 | +31.1 / +29.85 / −111.866 |
| 4 | 6112 | 15 | 6246 / −60.016454 | +31.1 / +32.35 / −97.552 |

终止实测来自已封存439终端观测，因此含float32量化；两次FINAL均仍−58，硬限−60未放宽。FRk不受后腿owner投影控制。相同−58执行结果来自多个不同raw/REQUEST，后续继续向负侧扩大请求没有增加有效目标空间。不是“只是一次随机极端sample”：例如episode 1首次掉载5704，raw FRk−1.143、条件mean−1.130、sigma .024；episode 4持续掉载6112，raw−1.949、mean−1.996、sigma .446。均值/HISTORY演化也在负向请求里，但不能把一次轨迹据此推广成全部策略分布。

## 5. 对照含义与最小下一实验

P12首块的RR placement由nominal前缀产生；其中学生RL新资格6437后，RR在6496退出合法XY时仍有11.235N TOP力，与本次两个学生捕获后的“XY内AIR”不是同一种第一损失。该块说明“预先有RR接触”也不保证后续RL保持/前送，不能用它证明本次捕获已经可持续。CP225792自然DET则在8312–9432一直AIR，RR knee全141拍裁−58，FR knee均值约−20；这是未捕获状态，不能套用上述学生已捕获后归因。

**最小可证伪候选（仅建议，未准备/执行新控制）：** 等当前自然评估封存后，若需独立物理对照，复用已有单通道方向探针机制，只在新鲜RR合法TOP承载后、FRk尚未饱和的窗口，测试一次FRk REQUEST相对触发时固定入口值向正4°的连续1s ramp/1s hold/1s release；保持其他11通道为同一当前学生及原有mapper/HISTORY/物理限位，保留source knee/wheel/stop，不同时改owner或reward。任务是检验“取消持续负漂是否改善RR接触保持、FR跟踪和RL准备”，不是把FR自动设+30°或正式修复；原sample/logp与实际干预分别记录，零PPO/AUX信用。如该回合没有合法触发，就明确未触发，不假造已放置入口。若担心3s过长，可先限定同机制更短预算，但不要重复叠加+4°。

**实际续训仍用兼容最新模型和原前段KL保护**；只增加步数不等于消除这个执行分布。先在已有后腿连续采样中跟踪“捕获后的FRk REQUEST/FINAL饱和、RR当前接触保持、source反向组时刻”，不以重训前腿、放宽硬限、新rear姿态脚本或扩大sigma替代诊断。未完成单通道反事实前，不声称FRk是首次或持续掉载的唯一原因，也不提前移除P10展开或整个late组。
