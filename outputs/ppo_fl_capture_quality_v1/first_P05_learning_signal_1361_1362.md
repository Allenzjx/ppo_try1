# 首批真实 P05 学习信号：updates 1361–1362

只读分析，未修改生产、未启动仿真、未新建策略分支。源 run：`runs/ppo_fl_capture_quality_v1/train/20260918T0412242524179Z_gf2e552406ea7_541c9a6d6c8e436d811fa7bcd64a885d`。仅取前 384 条完整决策记录和已完成更新的 `rollout_001361.pt`、`rollout_001362.pt`、对应真实 optimizer likelihood 日志；不计未优化尾部。每批 optimizer 回执均为 20 步、观察到有限非零梯度。

结论：新训练真实取得 FL 捕获并进入 P06，但随后的机身碰撞仍是失败。捕获奖励和后续失败都进入了实际 PPO 更新；没有证据支持把固定负 hip 指令改为训练目标。128 步 rollout 的边界与旧 critic 估值使相邻两批的优势方向明显不同，不能把归一化优势正号直接解释为奖励偏好悬空。

## 物理事件与输入状态

| 事件 | tick / 秒 | 真实证据 |
| --- | --- | --- |
| FL qualified lift | 1535 / 12.7917 | 保存的物理历史事件 |
| FL cross | 1711 / 14.2583 | 保存的物理历史事件 |
| update 1361 收集截止 | 2048 / 17.0667 | FL AIR，gap +64.064 mm，无 FL placed；其他三腿承载 |
| FL placed | 2157 / 17.9750 | evaluator 物理事件；不是用 gap 代替接触 |
| P05→P06 | 2160 / 18.0000 | FL TOP、14.649 N、gap −0.344 mm，任务条件满足 |
| BODY_COLLISION | 2249 / 18.7417 | `BODY_CONTACT`，`central body/obstacle collision`，物理证据 VERIFIED；不是语义超时 |

2157 是保存的 placement 事件时间；本报告没有把它冒称为首次接触力采样时间。决策级首个 TOP/承载读数在 2160。P06 的 12 个决策末端 FL 均实际承载，末端仍为 TOP/11.427 N，因此不是“FL 历史 placed，但当前悬空”造成的这次失败。

全身状态不应忽略：2160 FR 暂时失去承载，2168 恢复；2184 起所查决策末端 RL 无承载。P06 实测机身角速度峰值 0.81049 rad/s。FR knee 的 nominal 保持 48.15984°，已保存的 post-mapper residual 从 2152 的 +17.41970°增至 2249 的 +64.06664°，最终下发 112.22648°；属于现有 P06 112°请求范围和 slew 路径内的大偏移，不是 wheel/servo mask 丢失。没有对多通道闭环耦合做单因果归责。

## 实际奖励、保存的 GAE 与优势

`raw GAE = saved returns − saved values`，没有重算 GAE。保存 advantage 是官方整批标准化结果；gamma=0.9985、lambda=0.99。两批的 actions/old μ/old σ/rewards 与同步逐决策 audit 均逐项零差异。

| 已完成批次/阶段 | 样本 | 平均即时 reward | 平均保存 raw GAE | 平均保存 advantage |
| --- | ---: | ---: | ---: | ---: |
| 1361 / P05 | 71 | +0.005242 | +0.072968 | +0.745710 |
| 1362 / P05 | 14 | +0.013337 | −17.230973 | −1.845348 |
| 1362 / P06 | 12 | −3.525260 | −19.176756 | −2.107520 |

1361 中，34 个“已 cross、top XY 内、仍 AIR”样本平均 reward −0.002812，但标准化 advantage 全为正，raw GAE 30/34 为正。这批截止时尚未看到后面的碰撞，只能利用当时 critic bootstrap；不是后来 terminal 被删去，也不能据此宣称悬空获得正的即时任务奖励。

1362 中，FL gap 在 tick 2128/2136/2144/2152 依次为 23.354/11.747/4.448/1.185 mm，真实总 reward 为 +0.001379/+0.008973/+0.015013/+0.017664；2160 真实捕获为 +0.118939。但这 14 个 P05 样本和随后 12 个 P06 样本 raw GAE 全负，捕获当拍 advantage −2.010372。2249 terminal reward −42.317661，其中任务失败 −40、potential 归零项 −2.317496、单物理步时间成本 −0.000167，`done=true`、bootstrap=false。P05→P06 本身没有截断回报。

以上是保存的全任务 potential/shaping，不把所有变化都归给独立 FL 项。新 FL 25 mm/3 mm 双尺度只改合法 top 区域中的软接近份额；P05 此处没有额外 body quality 成本。

实际 optimizer 内日志进一步确认：每个样本使用 5 次。1362 的 P05 70 次使用和 P06 60 次使用均为负 advantage；各样本最后一次被使用前的 likelihood ratio 均值分别 0.78708/0.61157，13/14 与 12/12 低于 1。它们是实际优化过程中的读数，不是最终 checkpoint 重新 forward，也不保证每一维均朝理想方向改变。

## 探索、HISTORY 和 nominal 的区别

rho=0.9、temperature=0.25，来自当时保存的 policy request。1361 的 FL hip base mean 平均 −0.0320，但 conditional mean +0.1984、sampled raw +0.2370；HISTORY 足以改变实际采样中心，不能用 base mean 代替动作。1362 的 14 个 P05 hip base mean 平均 +0.0205，而 conditional mean −0.1940、raw −0.3424。

关键 tick 2152：hip base mean +0.12610，conditional mean −0.77106，effective σ 0.66689，实际 raw −2.70659（innovation −1.93552，约 −2.90σ）；投影/slew 后 residual −16.63159°，最终目标 32.81841°。下一决策 conditional mean −2.41757，体现了真实 raw HISTORY 延续，而非新学到固定的负 base mean。1361 hip 无 `|tanh(raw)|≥.95`，1362 的 P05 为 1/14；不能把这一段泛化为持续全面饱和。

本次捕获时 FL nominal 是 [49.45°, −37.95°]，controller bias=0，并非此前独立 ±hip probe 的 late endpoint [22.8°, −13.4°]。因此不能把这次捕获解释为“学习复现了 −2° probe”。下降与接触确有有用探索，但伴随其他关节、轮速、载荷和身体运动，最终未安全延续。

建议保持已安排的 P04/P05→捕获→P06 连续课程，增加真实安全后续覆盖。当前仅 85 个 P05 请求样本、一次真实捕获、12 个 P06 请求样本纳入这里的两个完成更新；不足以证明已经学会稳定捕获，也不支持新固定姿态奖励、再改 N 或扩大探索范围。完整 P01 与前段 quality 的评估结论另由正式重载评估提供。

## 补充：P05→P06 的尺度、raw HISTORY 与物理连续性

独立有界核验仅读取同一封存 run 前 282 条决策、`rollout_001362.pt` 已保存观测，以及当前未改动的生产实现。未运行模型 forward、优化器或新仿真。细表与可复算脚本：`P05_P06_scale_history_audit.json`、`audit_P05_P06_scale_history.py`。原始前 282 行 SHA256 为 `b59765e1020a0c065d11ac7b1e1c5e088c142350ec90f99530cf5fdcf42ba785`。

结论是**有明确的物理尺度放大机制，但没有瞬时目标跳变、mask 丢失或重复加 residual 的证据**。P06 扩大范围是已声明的设计，用于保留全身替代动作能力；当前代码实现了这一范围。问题在于 rho=.9 继承的是 raw latent，占新范围的比例，而不是上一阶段相同的物理修正量。范围合法、slew 合法不等于该探索对当前承载安全。

| FR knee 决策末 tick / 请求阶段 | base μ | conditional μ | effective σ | sampled raw | cap·tanh(raw)，限速前 ° | 已滤波请求 ° | 最终下发 canonical ° |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2152 / P05 | −.167756 | 1.252378 | .314805 | .919838 | 17.419702 | 17.419702 | 65.579544 |
| 2160 / P05→P06 | −.160394 | .811815 | .425363 | 1.208104 | 20.066637 | 20.066637 | 68.226479 |
| 2168 / 首个 P06 | −.123431 | 1.074951 | .337380 | 1.072714 | 88.533921 | 23.566637 | 71.726479 |
| 2249 / P06 terminal | −.088922 | 1.063797 | .236765 | 1.424428 | 99.738140 | 64.066637 | 112.226479 |

首个 P06 的 previous raw 与最后 P05 sample 逐项相同。FR knee 同一个 raw=1.208104 在旧 cap24°下要求 20.066637°，在新 cap112°下要求 93.644307°。尺度单独贡献 +73.577670°；真实新 raw 反而减小到 1.072714，按新尺度贡献 −5.110387°，合计成为 88.533921°。首个新 sample 仅比 conditional mean 低 .00663σ，并不是突然采到一个大正噪声。base μ 只从 −.160394 改为 −.123431；它在此 P06 全窗保持负数，正的条件中心主要由继承的 raw 产生。交接位于同一批 1362 收集内部，不在 optimizer 更新边界。

bridge 已有保护：首个 P06 物理步 2161 把上一拍 20.066637°原样保持，servo 最大误差 3.6e−15°、所有通道无丢弃/无缩 cap 裁切。2162 起重新追踪 `112·tanh(raw)`。残差限速为 60°/s，即 .5°/120Hz tick；2162–2249 共 88 拍恰好增加 44°。不是一次性执行了 93.6°修正。所有 P06 决策末端 native final 也保持 +.5°最后一拍变化，低于独立 final-drive 1.25°/tick 上限；该窗 FR knee 无 headroom 裁切、无 final-slew 额外裁切，mapper tracking 未使用新的反馈修正，mapped nominal+controller 恒为 48.159842°。dispatch 前实测 native knee 从 tick2160 的 1.126544rad 到 tick2249 的 1.852748rad，说明不是仅记录了不动的目标；这是末物理步前读数，不冒称该步后的 actual。训练未存全量逐物理步 native 数值，因而“每拍”的数值结论限于 bridge 公式、light tick 验证及决策端点相符，不虚构未保存传感数据。

同 raw 的扩大也作用于其他通道。下表是**保持最后 P05 raw 不变的代数比较，不是另一次运行**；实际首 P06 sample 还会改变目标。完整 12 通道表见 JSON。

| 通道 | cap P05→P06 | 同 raw 限速前请求：旧→新 |
| --- | ---: | ---: |
| FL hip / knee | 18→32 / 24→36° | −15.1147→−26.8706 / .5192→.7787° |
| FR hip / knee | 18→24 / 24→112° | 4.0637→5.4182 / 20.0666→93.6443° |
| RL hip / knee | 12→24 / 18→36° | .0644→.1288 / 1.9855→3.9709° |
| RR hip / knee | 12→24 / 18→36° | 5.1438→10.2876 / −5.0503→−10.1005° |
| FL / FR wheel | 1→1.2 / .6→1.2rad/s | .7895→.9474 / −.2623→−.5245rad/s |
| RL / RR wheel | 1→1 / .6→.6rad/s | 不由 cap 放大 |

因此不能说当前完全没有 physical history：bridge 保持一拍，残差和最终执行器各有限速，372 观测也保存上一物理请求及最终目标。但是**固定 rho 的 actor 条件中心仍使用 raw HISTORY**，这些连续性保护不保证持续的物理修正目标不被新范围放大。本段仍不能隔离证明 FR knee 是身体碰撞唯一原因；FL 已承载、其他关节/轮子/载荷也同时变化，必须以真实闭环评估判定。

### 仅设计的最小候选，未实施

当前 372 维已经足够，无需私有缓存或增加观测才能表达“cap 增大后的首个决策，以先前物理 REQUEST 作历史中心”。零基半开区间及固定逆归一化为：

| 观测位置 | 含义 / 解码 |
| --- | --- |
| `[0:13]` | current stage one-hot，选择版本绑定的 cap 表 |
| `[20]` | stage elapsed ×200 秒；本次首 P06 恰为 0 |
| `[158:171]` | completed stages，核对线性前驱完成 |
| `[195:207]` | previous raw，scale1；现有 rho 来源 |
| `[207:219]` | previous **filtered REQUEST**，乘 `[4,4,4,6,4,4,4,4,.12,.12,.12,.12]` |
| `[219:231]` | previous-previous filtered REQUEST，同尺度 |
| `[231:243]` / `[183:195]` | previous final drive / previous mapped nominal；不可直接混同于已实现 residual |

当前 supervisor 只在 `physics_tick % 8 == 0` 正常换阶段，stage age 同时重置；固定 8 tick/decision 下，首拍 age=0，下一拍 .0666667s。已保存 rollout1362 的 index14/global178703 实测 stage=P06、age0、previous raw1.208104134；由观测解码的 FR previous REQUEST=20.066637039°，与高精度 audit 差 2.74e−7°，仅 float32 表达误差。当前各 cap/观测固定尺度都在 clip20 内，正常请求没有此处裁切信息损失。此充分性依赖**当前线性、对齐的阶段与真实 prefix/reset 合约**，不能推广到将来任意跳阶段/任意控制周期。

最小候选可仅在 cap 增大的首拍、仅对 cap 改变通道，令 `h = atanh(previous_filtered_request / current_cap)`，用 `.1·base_mu + .9·h` 替换该拍的 raw-history 中心；其他阶段/通道保持旧规则。当前 FR 数值为 h=.181121314，条件均值 .150666125，对应均值处物理请求16.748069°，而非旧规则88.627742°。这只是给定已记录状态的代数，不是闭环预测。若希望保留旧“未滤波意图”而不是已滤波 REQUEST，也可由已知前驱 cap 和 previous raw 做精确 tanh→物理→atanh 变换；两者在一般 slew 跟不上的状态不同，必须先明确选择，不能统称“实际执行历史”。简单将 raw 乘旧/新 cap 比例并非精确 tanh 逆变换。

该候选不改112°搜索范围、不改 N、不清零 residual、不改rho/.25σ/限速/奖励；Gaussian 仍保留整个 latent 支持。但它改变了**条件分布和 deterministic 输出**，绝不能作为无版本 helper 热补丁。实施前须独立 policy/kernel 版本与迁移：保留兼容 learned tensors、Adam、RNG、Identity、LR；清空旧未完成 rollout，以新 kernel 重采 old μ/σ/logp，minibatch 重算 new logp 同用保存的372观测、相同首拍判据与 cap 表。禁止把旧 logp配新中心或把 sample 后另做变换却仍声称旧 Gaussian likelihood。prefix 冻结 actor 的精确 class/version guard、正式 deterministic/stochastic video eval、更新审计均须接受同一新内核；新内核上线前完成的旧数据不混入其首批更新。只从观测复算中心而不改 raw sample 的坐标时，不引入额外 Jacobian；若改为物理空间采样/事后改 raw，需另定义正确分布，非本最小候选。

目前**不实施**。根任务先完成当前冻结 runtime 的已计划训练与同 checkpoint 真实视频，再考虑单因素新版本；本补充不增加训练/发布门禁，也不把代数候选称为已修复安全性。

### 输出目录内 CPU 原型核验（不是生产实施）

随后仅依根任务明确授权，在本 outputs 目录新增 `cap_transition_history_prototype.py` 和 `test_cap_transition_history_prototype.py`，没有修改生产 actor、注册表、迁移、配置或正在训练的 runtime。当前方案严格使用 `age==0 AND predecessor_completed AND cap_increased`，只重表达变化通道的历史，拒绝非法 one-hot/age/归一化与超前驱 cap REQUEST，不通过静默 clipping 掩盖输入异常。类无新增可变 cache、parameter 或 buffer；仍使用官方 heteroscedastic Normal，rho=.9、温度.25、学习到的 sigma 和12通道搜索范围不变。

15/15 CPU 单线程测试通过，XML 为 `cap_transition_prototype_tests.xml`。覆盖真实 rollout1362 的交接、非交接/前驱未完成/相同 cap/P01 的 deterministic 与随机 sample/RNG 逐位旧内核一致、零历史等价、P02→P03 仅 FL/RL wheel 变化、前驱请求恰在 cap 边界、非法输入、真实 state-dict 和 Adam 的严格加载，以及官方 `rsl_rl.PPO` 的 32 个合成转移、5×4 minibatch 更新、内存保存/新实例严格加载、Identity/LR1e−5/空 rollout 保持。后者用保存观测与构造 reward、新原型采样，仅检验 optimizer 接线；**新增真实 policy decisions=0、真实 PPO updates=0，绝不作物理训练或成功证据**。

真实权重来源是当时采集此批前的 CP178688（1361 updates / 27220 optimizer steps），SHA256 `ad9086fe22618cd733054a636d5da00605e8ccbdc5f14ce753d80b5d43679578`；没有使用事后训练好的网络反推当时行为。在 global178703 已保存观测上，CPU FR knee μ 从旧 1.074950695 变为原型 .150666118，σ 两者逐位相同 .337379485（variance .113824919）。同一个原始实际 Full12 action 的 logp：旧 CPU 1.093932271，原型 .192038178；原 GPU 记录1.093930840，CPU/GPU差1.43e−6。原型似然确实不同，印证必须新 kernel 版本和新 rollout，不能把它假作旧行为完全兼容。其他时刻 σ 本来仍可能较宽，此原型没有解决/更改这一独立问题。

这是待根任务审阅的可执行提案，而非已注册的新策略；不得把该输出模块指定给当前真实运行。后续正式迁移仍需统一 training、frozen prefix、deterministic/stochastic eval 及 likelihood 版本，并在安全边界决定是否采用。
