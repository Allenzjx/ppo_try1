# 两次正式课程首失败 episode：P07→P09 有界只读对照

仅读取 block02 的前 401 条、block03 的前 305 条 `residual_and_projection_audit.jsonl`，分别截止真实 BODY_COLLISION：global186769/t4710/39.2500s、global187185/t5116/42.6333s。未读取其后新 episode；未启动仿真、采样、forward 或 optimizer；新增训练量 0，未改生产文件。

源 run：

- block02：`runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0509408527373Z_gee5a9651591d_35cb60975d944752a3227f41558c2fca`。首 episode 有效样本：P04=1、P05=148、P06=209、P07=1、P08=1、P09=41，其余阶段 0。
- block03：`runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0525533703252Z_gee5a9651591d_91ae17ea1dff4aab8f836d9b3ca6ebf4`。首 episode：P06=229、P07=1、P08=1、P09=74，其余阶段 0。前缀不计入这些数。

## 最早的关键区别在 P08 以前，且不是仅由随机尾巴造成

block02 在 P05/t1584/13.2s 已出现 RL hip REQUEST −3.095°、RR hip +3.723°；RR knee 在 t1528 已为 −3.022°。整个 P06 的 REQUEST 范围分别为 RL hip [−10.955,−5.838]°、RR hip [+6.530,+14.233]°、RR knee [−11.614,−5.883]°。

block03 从合法 N 前缀 P06 接管，首拍 HISTORY 为 0。仅第 2/3/4 个学习决策便分别出现 RR hip +4.192°（t2696）、RR knee −3.223°（t2704）、RL hip −3.221°（t2712）。P06 全段范围分别为 [−10.816,−0.927]°、[+2.381,+14.608]°、[−10.520,−1.587]°。这些 3°界线只是便于展示首次实值，不是任务门槛或“过大”的判据。

两次都把已经改变的后腿构型及 HISTORY 带入 P07/P08；不能将失败全部归给 P09 新动作。另一方面，下表 network mean 自身也分别支持 RL−、RR hip+、RR knee−，所以也不能将现象只归因于 ρ=0.9 反向响应过慢。减小新采样 σ 不会自动撤销已学均值和既有历史。本读数不建立单关节因果关系。

## P08 首个真实 policy request（唯一小表）

数值来自同一条采样记录，不是重算 N 的差分。`μ=.1 network+.9 H`；network/H/μ/σ/raw 无量纲；R、N 单位 °。R 是日志 `requested_policy_residual_full12`，且逐值等于该拍 headroom `effective_policy_residual_full12`。N 是 `baseline_native_plus_controller_full12`，是 mapped/controller 后基线，**不是源 Recording 角度**。R 相等不把后续 final-slew、native 写入或物理跟踪混成同一层。

| Run / channel | network | H | μ | σ | raw sample | R=request=effective ° | mapped N ° |
|---|---:|---:|---:|---:|---:|---:|---:|
| b02 FL hip | +.032713 | +.056014 | +.053684 | .037342 | −.018623 | −.595877 | 38.85 |
| b02 RL hip | −.387879 | −.491069 | −.480750 | .047126 | −.465193 | −10.423372 | 6.90 |
| b02 RR hip | +.569795 | +.641302 | +.634151 | .043755 | +.693520 | +14.405722 | 0 |
| b02 RR knee | −.314821 | −.294978 | −.296962 | .009654 | −.293319 | −10.266709 | 0 |
| b03 FL hip | +.023641 | −.013331 | −.009634 | .051623 | −.035178 | −1.125221 | 37.60 |
| b03 RL hip | −.444537 | −.262994 | −.281149 | .046020 | −.262299 | −6.154661 | 6.90 |
| b03 RR hip | +.621925 | +.488309 | +.501670 | .052091 | +.509068 | +11.261256 | 0 |
| b03 RR knee | −.317440 | −.285994 | −.289138 | .011746 | −.271755 | −9.549272 | 0 |

b02 是 global186728、episode t4376→4384（36.4667→36.5333s）；b03 是 global187111、t4520→4528（37.6667→37.7333s）。分布取动作前状态，下面几何/接触取步进后的端点。native audit 的末次派发 tick 分别4563/4707；两 run 所有选定 episode 的 native tick 都等于 episode endpoint tick+179（含180tick bootstrap、末派发在端点之前），不能直接混用时钟。

P08 后两次都是 **FL AIR/0N，FR TOP support，RL GROUND support，RR GROUND support**，不是 FL 承载。RR bearing 分别3.283/2.295N，RR top-plane gap −51.334/−50.158mm；body collider minimum world z=123.565/127.718mm。两次 P07/P08 各仅一个决策：P07 完成值为 edge proximity RR/RL 与 role_prepared_RR，P08 完成值为 edge_proximity_RR 与 transfer_ready_RR；阶段名进入 P09 不等于 qualified lift，P09 的实际抬升任务尚未完成。

从 P07 到终止，43/76 个端点全为 `RR_preparation=1`、`RR_current_valid_lift=0`，12维 σ 均正（最小 .007061/.007667）；从 P06 到终止，phase mask 均为全1，12维 REQUEST/effective 每拍相等。没有发现本窗口的 mask 或 headroom 丢弃请求；不据此保证实际轨迹正确。FL 在全部43/76个端点都 AIR；P06 中 FL support 仅6/209、27/229，说明前驱“放置历史”没有变成可靠持续支撑。

## Collider 下降与后腿动作的先后

- **b02**：P07 端点 t4376 z=123.771mm；t4440/37.0s 首次比该值低10mm（z109.563），当拍 RL/RR hip/RR knee mapped N 仍为6.9/0/0°，R为−11.103/+14.097/−10.348°。这次下降早于 nominal 的 RR hip 上行；但随后 z 回升，P07→RR nominal 启动前范围107.283–131.084mm，不能误称持续单调塌落。nominal RL hip 开始增长见 t4576；nominal RR hip 首个 >1°端点为 t4624（N2.85°，R+11.122°，z120.807）。t4680 z78.778；最后 t4710 z50.003，原终止源 BODY_CONTACT。没有任何 RR 初始/qualified lift 事件；端点短暂 AIR 5次不能替代 qualification。
- **b03**：P07/t4520 z127.937；t4584/38.2s 首次比该值低10mm（z116.525），N仍6.9/0/0°，R为−6.447/+13.070/−10.902°；之后同样回升，RR nominal 启动前范围115.418–132.774mm。nominal RL hip 增长见 t4720；RR hip 首个 >1°端点 t4768（N2.85°，R+10.368°，z111.359）。RR 只有一次真正初始 free-rise 事件 t4793/39.9417s（3.097mm），从未 qualified；t4832 z75.515，终止 t5116 z50.009/BODY_CONTACT。共有6个短 AIR端点，不是持续有效抬升。

这说明后腿残差偏移先于身体严重下降，且进入准备时早已存在；随后的身体变化同时涉及 FL/FR/RL 源准备、RR source ramp、残差和接触重分配。没有配对干预，不能把时间先后或相同符号升级为“RL负向单独导致碰撞”，也不能把 RR 自身关节动作大小当成抬升事件判定。

## 与既有 RR-A/B 的有限对齐，以及下一入口要分辨什么

已复用 `RR_prepare_RL_minus3_N_evidence.md`、`RR_lift_FL_RL_minus3_N_evidence.md`；只额外读取 RR-A 的单条 `probe_entry.json`，未重扫参考轨迹。成功 B/诊断A 的共同 P08 入口 t5168：collider z131.456mm，FL AIR，FR/RL/RR真实支撑，RR bearing1.684N、gap−50.925mm、RR/RL front distance −.218189/−.224047m；12维 residual anchor=0。正式两入口接近同一 rear-distance 窗口，但不是同一全身状态或匹配前缀：P08动作前 z分别123.771/127.937mm，且已有大幅后腿残差历史。

RR-A 在这个匹配入口逐步加 RL−3°，取得 qualified lift 后又在 t5504 回地，未完成 carry；RR-B 则先保留 B 的已取得 qualification，再有限施加 FL/RL−3°并成功放置/释放。两者都没有 PPO credit。方向相同不代表时机与入口可互换；当前正式策略的持续 RL−6至−10°、RR约+11至+14°/−10°也不是这些小幅诊断动作的复现。

下一次合法未AIR P08真实前缀课程可分开两条证据：①进入学习前的真实 body/contact、nominal、RR qualification=false、完整 HISTORY；②接管后分布与 REQUEST 是否重新快速形成上述偏移，以及其后的真实 lift/collider轨迹。若合法前缀仍快速复现偏移，才增加“当前阶段 policy 分布自身有问题”的证据；若显著改善，只能说明前驱状态/历史很重要，不能据一次后缀成功宣称自然 P01已解决。本报告不要求修改ρ/LR/reward，不将这项观察设成训练门禁。

## 补记：新 P08 合法入口后的固定前15拍（非终态结论）

新 run `train/20260921T0546278314342Z_gee5a9651591d_ca2f2a9132e740f2a8165a412622dbd2`；严格只读前15条 global187393–187407，不读取之后样本、不 forward、不等待更新。前缀首记录为 `source=successful_nominal / policy_credit=false / target_phase=P08 / entry_is_exact_historical_snapshot=false`；实际课程入口 t5168/43.0667s，`from_P01_current_policy=false`，P08仍在active时开始计credit。首拍372布局对应 request 日志的 previous_raw、history_center、previous_filtered_REQUEST 全12维均为0；stage index7、age0、predecessor completed=true、cap transition gates全部false，current RR qualification=false。这里检查的是实际 request 所记录的输入切片，未冒称读取尚未封存的完整 rollout372张量。

下列三元组顺序固定 **RL hip / RR hip / RR knee**；前五组为无量纲 raw 分布，R为°：

- 第1拍 global187393，t5168→5176，P08→P09：network `[-.556905,+.603794,-.364414]`；H `[0,0,0]`；μ `[-.055691,+.060379,-.036441]`；σ `[.048150,.070338,.017247]`；sample `[+.032428,+.025361,-.011096]`；R `[+.778002,+.608534,-.399428]`。
- 第3拍 global187395/t5192，R已到 `[-2.351915,+4.609726,-3.409304]`；第8拍 global187400/t5232为 `[-2.705296,+7.373451,-6.566236]`。
- 第15拍 global187407/t5288：network `[-.405237,+.519945,-.315205]`；H `[-.121406,+.471312,-.224817]`；μ `[-.149789,+.476175,-.233856]`；σ `[.041987,.047359,.011819]`；sample `[-.111780,+.463221,-.241075]`；R `[-2.671601,+10.384941,-8.514397]`。

这15拍 network 的三通道分别始终负/正/负，范围 `[-.556905,-.318623] / [+.498475,+.641371] / [-.373830,-.292330]`；历史从真实零入口逐渐形成，而非继承前两失败episode的旧负历史。其 R 范围 `[-3.441,+.778] / [+.609,+11.122] / [-8.514,-.399]`。RR hip/knee 已快速重建近似旧偏移；RL hip目前仅约−3°，**尚未重建旧例−6至−10°幅度**。这增加“当前 P08/P09 分布自身会产生这些后腿偏移”的证据，不证明这些方向各自造成终态失败，也不排除先前前驱入口的额外影响。

15拍 mapped N三通道均为 `[6.9,0,0]°`，12维 mask全1、σ全正（全窗口最小 .009953）、headroom REQUEST=effective逐值相等。每拍 RR均GROUND、current qualification=false、FL均AIR；无terminal。collider z端点范围123.013–131.922mm，第15拍131.733mm，因此不能把短暂下探解读成已持续塌落或即将必然碰撞。窗口仅1秒，尚不足检验完整准备/抬升/放置，更不是正式自然P01成功。

## 补记：同一 P08 课程前两次已结束 episode 的动作—回报链

固定读取前111条（global187393–187503），只覆盖 episode0的61拍及 episode1的50拍。分别在 t5653/47.1083s、t5565/46.3750s 由 BODY_COLLISION结束，真实 learner窗口4.0417/3.3083s；其余时间为真实前缀，不计PPO样本。两次 RR均未 qualified/crossed/placed；episode0只有t5459的一次3.087mm初始free-rise，episode1连初始事件也没有。两次前腿历史placed成立，不代表当前FL支撑或后腿任务完成。

**同权重，从合法零历史再形成偏移。** 两次首拍 network/H/μ/σ逐值相同，H全0；但随机sample不同。episode1第15拍（global187468/t5288）已经是 RL/RR hip/RR knee REQUEST `[-12.911789,+8.671885,-9.052653]°`，相应 network `[-.326684,+.623863,-.320809]`、H `[-.553500,+.352399,-.262977]`、μ `[-.530818,+.379546,-.268760]`、σ `[.047339,.044849,.010436]`。相比episode0同位置的 `[-2.671601,+10.384941,-8.514397]°`，RL历史路径尤其不同，RR双关节同向偏移则重复形成。全部111拍的三通道 network 符号始终为RL−/RR hip+/RR knee−；不同时间的network值差异是不同输入下的输出，不能当成两episode间发生了参数更新。

**成本实际有效，但它是次级、有界质量项。** 日志每120Hz样本的实际几何系数β=.03/s、分离下界裕度20mm。body reward逐拍恰好等于负的 `task_space_weighted_geometry_cost`（最大算术差<5e−19）；绝非“只写配置未进入reward”。episode0/1累计几何成本分别 .029376/.011198，对应21/10个决策非零。首个非零都落在第41拍/t5496（首个物理非零时间45.7417/45.7583s，分离下界19.919/19.475mm），明显晚于RR偏移形成；当裕度足够时成本依法为0，不惩罚所有后腿动作。终止body最低z为50.012/49.999mm。

这里的 `.029 vs 40` **不是权重bug或应自动增大成本的证据**：几何项只负责有界质量，任务失败才是主项。两次实际 terminal event均−40；terminal potential置0、bootstrap=false，终拍总reward −42.309795/−42.362984（含PBRS−2.307732/−2.360943、实际dt时间成本及几何成本）。整段learner reward和为−42.713471/−42.642552；终拍之前的累计reward也为−.403676/−.279568。未qualified时确有少量局部正进展reward，不等于被记为抬升成功：例如episode1倒数第二拍 t5560 的PBRS对应Φ .456246→.472189，reward+.073233，但紧接真实碰障，终态Φ清零。

**GAE边界与只读重构。** 首次审查时 `optimizer_updates.jsonl` 与 `advantage_audit.jsonl` 均0字节，`rollouts/`为空：本节不声称这些111拍已被优化，不提供伪造的官方normalized advantage或head gradient。为核对终态任务信号是否能沿这两个完整episode传播，仅使用已记录float32 reward/old V，按当前显式return profile γ=.9985、λ=.99、真实terminal零bootstrap，以普通双精度算术独立重构 `δ=r+γ Vnext−V; GAE=δ+γλ GAE_next`；未调用actor、采样或optimizer，且没有跨episode连接。

重构raw GAE：episode0全61个为负、范围[−13.299507,−6.698239]，首拍−6.979613、均值−9.434169；episode1全50个为负、范围[−13.354944,−7.701993]，首拍−8.089708、均值−10.311471。终态向第一拍的γλ传播因子仍为 .500030/.567780，不存在普通P08→P09切done造成的断链。在保持所有记录状态/old V不变的纯算术分解中，几何成本对首拍raw GAE贡献仅−.016387/−.006637；这不是另一条物理轨迹，也不代表应加大质量项。上述重构不是官方保存returns；整个128拍的标准化中心/尺度仍须实际storage写出后核对，**raw GAE全负不保证normalized A全负，更不能预言每个hip均值更新方向**。

当前可支持的结论只有：合法N入口并未阻止该分布重新建立偏移；主任务碰障惩罚和有界几何成本都真实写进reward，有限episode回报链没有丢失。尚不能声称这次PPO已吸收或拒绝这些失败。后续如检查同一1430完整更新，只需对这批样本的实际标准化A、相关mean/logσ head导数与同checkpoint闭环证据作有限核对；不把联合logp变化替代RR/hip均值学习，也不据这里的比值修改ρ、LR或reward。审查不构成训练/视频门禁。

## 1430 正规封存后的实际 advantage / terminal / head-gradient 核对

只读取该 run 已保存的 `rollouts/rollout_001430.pt`、`update_001430_likelihood.json`、对应 update/advantage audit及前128条逐拍日志；无模型forward、采样、优化器调用或仿真。实际更新1430：128新decisions、20 optimizer steps，终点187520；学习率1e−5、KL=.0164593、clip_fraction=.225，actor hash确已变化。实际覆盖P08=3、P09=125，前缀没有进入buffer。

**数据及终态链。** 保存观测为128×372，`obs[:,195:207]` 与逐拍 previous_raw逐值相等，索引0/61/111三个真实episode入口均全12零history。actions、old logp、old V、reward、done与逐拍日志逐值相等。done仅在60/110；两个5-physics-tick碰障终拍都不bootstrap，returns恰好等于终拍reward。三个普通P08→P09都不done、允许连续回报；最后第128拍是第三episode的非terminal rollout边界，采用既有official last-value bootstrap，并非任务成功或失败。

**实际A与之前重构的区别。** 官方前两episode raw GAE与此前双精度回算最大误差1.706e−5/7.615e−6，符合float32递推精度；失败传播链得到实证。整128拍 raw GAE均值−8.532077、样本std3.741060，按 `(GAE−mean)/(sample_std+1e−8)` 得到的A与storage逐值相同。分组为：

- episode0：raw GAE全负，实际A却有22/61正、39负，范围[−1.274353,+.490196]。
- episode1：raw GAE全负，实际A有12/50正、38负，范围[−1.289171,+.221887]。
- 第三episode未完成的17拍：raw GAE范围[−.579288,+.129377]，8正；实际A全部17正，范围[2.125812,2.315241]。这来自合法尾部bootstrap与整批标准化，不能把它们标签化为成功，也不能因前两episode失败而人为改A符号。

例如第一个episode t5288/global187407虽然后来碰障，raw GAE−7.545790但实际A+.263639；第二个episode同t5288/global187468为raw−8.934534、A−.107578。episode1倒数第二拍t5560即刻reward+.073233，实际A仍为−1.250720；终拍A分别−1.274353/−1.289171。立即reward、raw GAE及正式normalized A不能互相代替。

**真实head导数，不以joint logp替代。** 128个样本各在实际optimizer前hook出现5次。所有记录的conditional mean/σ、mean-head及logσ-head导数有限。实际mean-head导数满足未strict-clip时 `−.1*A*ratio/32*(raw−μ)/σ²`、strict-clip时0，最大误差1.532e−7；.1为ρ=.9的真实链式因子，未另乘10。前111个失败episode样本共555次使用，58次strict-clip；其余497次RL hip、RR hip、RR knee的mean-head导数都非零。总loss导数是在parameter clip之前，不能解读为该单样本独立Adam更新；logσ也可能含entropy导数。

按每个样本**最后一次实际hook（各自minibatch optimizer之前）**对其collection输出比较，前111样本 network均值Δ平均为 `[RLhip +.014256, RRhip −.009223, RRknee +.006424]`；对应conditional均值Δ仅其.1倍。RL111/111向较不负移动、RR hip111/111向较不正移动、RR knee107/111向较不负移动。σ倍率平均 `[.999643,1.016979,1.011567]`。因此这一次更新不能描述成“相关均值完全没梯度”或“只有σ在吸收”。在固定oldσ先换μ、再换σ的**有次序**逐通道logp分解中，RL mean部分明显超过σ部分；RR hip两者相近；RR knee mean部分更大，这也不是“所有通道都是同一种吸收机制”。

这些last-hook对应不同minibatch的权重时点，**不是CP187520最终整网forward，也不是自然确定性闭环入口**；变化幅度小且均值仍保留原符号，不能据此宣布后腿任务已修复。σ变化与全身共享参数同样不能由个别样本梯度直接归因。接下来以同CP真正重载后的det/stoch完整尝试判断任务能力即可，本分析不改ρ、reward、LR或增加训练门禁。
