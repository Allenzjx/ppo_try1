# 任务状态探索：372 维现状与最小实现建议（只读审查）

审查基线 HEAD `73c128d0f43c628a0d0c3bdb8e60c5f3414a6233`；相对 e735 的 src/scripts/configs/tests 无差异。完整读取本轮 07768750 附件。本文没有改生产、运行 actor、优化器或 Isaac，新增训练量为零。下面数值是待物理诊断校准的单一候选表，不是已部署配置或有效性结论。

## 1. 结论与表达

可以保留当前 372 维模型、全部权重、均值、ρ=.9、REQUEST-HISTORY 入口修复、tanh 容量和实际投影，只增加一个**纯当前观测的 sigma 规则**。不需要换动作表示、添加滤波器或额外随机过程。

令 C_i(s) 为现有阶段请求 cap，B_i(s)>0 为下面表中的物理等效创新尺度：

`effective_log_sigma_i = learned_log_sigma_i + log(.25) + log(B_i(s)/C_i(s))`。

仍在官方 raw Gaussian 内采样、计算 old/new logp、entropy/KL；不在采样后挑动作或加负偏置。这里 B **不是实际 sigma、动作容量、角度目标或每拍 delta**。局部物理标准差代理是 `.25 * B * exp(learned_log_sigma) * sech²(conditional_mean)`，只描述 tanh 附近的未滤波一阶创新，不等于 REQUEST 增量/执行器位移/实测波动。举例 learned_sigma=.4 且 μ≈0 时，B=12° 对应约 1.2°、B=24° 对应约 2.4°；B=.6 rad/s 对应约 .06 rad/s。若学出的 sigma 很大，这不保证物理小扰动，仍需报告实际 sigma/创新与已有投影。

旧 e735 在 P01–P05 的 B=C；P06–P13 仅 FR knee 的 B=24°，其他 B=C。新表可作为对该旧物理尺度规则的替代，不能再叠乘旧 FR 24/112 因子两次。

对称 sigma 不能“偏向负号”，只能让经实测有用的通道有更多尝试。FL/RL 负向候选必须依靠真实诊断、任务状态回报和后续 PPO 信用吸收；不能把用户文字变成 policy 外负 bias。

## 2. 实际观察索引（0-based，切片右端不含）

来源是 `configs/ppo_fl_capture_quality_v1/observation_schema.json`、`semantic_observation.py` 和 `semantic_transfer_roles.py`，不是按报告猜测。

| 语义 | 索引 / 原始单位恢复 |
|---|---|
| P01–P13 one-hot | 0:13；task phase progress 16，全局 potential 17 |
| task/stage 时间 | 18:21 ×200 s；stage age=20 |
| FL / FR / RL / RR gap、center-front x、load fraction | 21:24 / 24:27 / 27:30 / 30:33；gap/x 单位 m |
| support count / body-front x / body speed / angular speed / max wheel speed | 33 / 34 m / 35 m/s / 36 rad/s / 37 rad/s |
| actual servo position / velocity | 38:46 ×90° / 46:54 ×180°/s |
| actual wheels | 54:58 ×2.1 rad/s；顺序 FL,FR,RL,RR，不是所有轴相同符号等于前进 |
| chassis roll,pitch,yaw / roll,pitch derivatives | 69:72 ×π rad / 72:74 ×5 rad/s；74 为 derivative valid |
| body linear / angular velocity | 75:78 m/s / 78:81 ×5 rad/s |
| CoM relative base / velocity world | 87:90 ×.5 m / 90:93 m/s |
| obstacle planes relative base | 93:99：front,back,left,right,bottom,top；98=top_z−base_z，不是 body 最低净空 |
| wheel bottom x−front、x−back、z−top | FL99:102, FR102:105, RL105:108, RR108:111，m |
| wheel world linear velocity | FL111:114, FR114:117, RL117:120, RR120:123，m/s |
| exact-pair normal forces | FL123:125, FR125:127, RL127:129, RR129:131；每腿 ground/obstacle，×100 N |
| exact-pair active | FL131:133, FR133:135, RL135:137, RR137:139；每腿 ground/obstacle |
| load fraction / support diagnostics | 139:143 FL/FR/RL/RR；143 count，144 margin valid，145 margin m |
| lift / crossed / placed | 146:150 / 150:154 / 154:158；腿顺序 FL,FR,RL,RR |
| completed stages / N / previous-dispatch mapped N | 158:171 / 171:183 / 183:195；N servo ×90°、wheel ×2.1 rad/s |
| previous raw / previous filtered REQUEST | 195:207 / 207:219；REQUEST scales=[4,4,4,6,4,4,4,4,.12,.12,.12,.12] |
| earlier REQUEST / final histories | 219:231 / 231:243 / 243:255；并非可互换的历史 |
| transfer-role blocks FL/FR/RL/RR | 324:336 / 336:348 / 348:360 / 360:372 |

每个 role block 的 offset 0..11 是 valid、workspace_progress、preparation_progress、transfer_progress、motion_fraction、preparation_ready、transfer_ready、fixed direction x/y、short_support_continuity_fraction、continued_response_fraction、window_evidence_fraction。均 scale=1。例如 RR valid=360、preparation=362、transfer=363、ready=365/366、support continuity=369、transfer maturity=370/371。`valid` 不等于 normalized-load-valid，也不是静态稳定证明。

**特别重要：当前 functional_free_air_lift_v3 在 `semantic_supervisor.py` 的 snapshot 中，将索引149所对应 RR lift bit 改为实时 `current_lift_valid`。** 真正历史 Q/C/P 仍独立存在 evaluator/history。RR 回地后该观测 bit 会撤销，可以直接用来区分准备与当前有效卸载，不能再把它叫陈旧 AIR 标志。FR 索引147仍是 lift history，需与当前 AIR、gap 和支撑信息合用。

12 维输出/servo顺序严格为：FL hip,knee；FR hip,knee；RL hip,knee；RR hip,knee；FL,FR,RL,RR wheel。`command_batch.py` 的 rear servo native sign 为−1，wheel forward sign 为[−1,+1,−1,+1]。sigma 表在 canonical 单位，无需按 native 符号翻转 sigma。

## 3. 一个可实施候选表

以下 B 的前8项为度，后4项为 rad/s；所有未选状态回退**原 e735 规则**，不统一降低全部阶段。FR preparation 尚未形成可用抬升时也保持原规则。表中“主要方向”只说明试验/学习关注，不添加动作偏置。

| 观测任务状态 | 前8维 B=[FLh,FLk,FRh,FRk,RLh,RLk,RRh,RRk] | 四轮 B=[FL,FR,RL,RR] | 目的 |
|---|---|---|---|
| FR 可用 AIR approach，P01/P02 且 FR未 placed | [9,12,12,18,12,12,6,9] | [.6,.6,.6,.6] | RL hip 保留当前尝试幅度；RR旁路约减半，FR摆腿仍可调，轮协作不缩 |
| P05、FL crossed 且未 placed、当前 AIR/gap>0 | [24,24,9,12,6,9,6,9] | [1,.6,1,.6] | FL hip 等效创新比当前18增至24；knee保留；减少无关支撑腿乱动，wheel保留 |
| P06 rolling，未进入 rear prepare blend | [12,12,12,12,12,12,12,12] | [1.2,1.2,1,.6] | 腿部仍可修姿态但减少大幅循环；不削已有四轮探索能力 |
| P06 前腿历史placed、RR未placed，但当前双前支撑代理不成立 | [24,24,12,18,12,12,12,18] | [1.2,1.2,1,.6] | FL/knee保留恢复机会，其他非主要关节不恢复旧大创新；不是永久双前接触要求 |
| P07–P09 RR preparation，当前 RR lift=false | [24,24,12,18,24,24,12,18] | [1.2,1.2,1,.6] | 保留FL/RL全身重构；RR直接大摆随机频率降低，不是锁死 |
| P06–P09，RR lift 当前有效且未 placed | [24,24,12,18,24,24,24,36] | [1.2,1.2,1,.6] | 保住FL/RL高度调整能力，恢复RR路径探索；不强制RR先动关节才能入此状态 |

这不是必须采用的唯一参数。只需一次有限诊断后选定明确版本，不应同时扫 rho、mean bias、新表示和所有 sigma。P06示例的 FR knee B12 是相对 e735 的 B24再减半，必须明确披露；其他关节不是在112等大cap上沿用同raw噪声。非主要通道最小B仍6°/9°，且Gaussian全支持，不是永久mask。

建议按确定优先级计算状态：RR-current-valid > RR-preparation > P06 rolling/recovery > FL-pending > FR-approach > fallback，适用阶段集合本身互斥后仍留显式优先级。RR placed后退出RR和P06规则，P10–P13回原e735规则，避免rolling偏好污染后腿或停车。

FR gate 可用 `phase∈{P01,P02} & !placed_FR & lift_FR & no_current_FR_pairs`，再以当前FR gap从0到现有15mm净空尺度连续插值，而不是只有已完全达到15mm才允许任何修正。此门只调整探索幅度，不作成功/承载判定。无FR lift时不减掉必要RL准备。

P05 gate 可用 `phase=P05 & crossed_FL & !placed_FL & no_current_FL_pairs & gap_FL>0`。建议gap从0到3mm细尺度连续退出该AIR探索表，接触拍退回默认，不继续强推下降。AIR↔contact无需历史缓存，不影响阶段done或REQUEST。

P06使用当前 `max(RL_front_x,RR_front_x)`，即任一后腿较早接近，在[-.27,-.22]m的 .05m 物理带连续 blend 到 RR preparation，而非固定秒数。早期原型曾取min及[-.275,-.225]，根审查后已修正，旧写法未部署。该 .05m 是待有限实测选定的探索调度尺度，不修改原进入/验收阈值。P06中若当前RR lift=true，立即使用有效卸载表，不能等待阶段名到P09。支撑退化时使用新增recovery表保留FL恢复机会，不恢复所有旧大cap创新；rolling/recovery均按相同rear距离退出，允许以后必要的FL AIR。

## 4. 372 中缺什么，不能造什么

现有观测没有逐腿 `within_lateral_span/within_top_xy`、当前 `TOP/WALL/EDGE` 分类、bearing validity、FL当前 `consecutive_top_samples`、真实机身碰撞几何最低净空、各髋世界高度。wheel_pair_active 只说明 ground/obstacle exact pair，**obstacle active不等于TOP**。X范围及历史 crossed 也不能证明当前Y仍在合法ROI。

因此最小方案中的状态 gate 必须命名为“观测调度代理”，不包装成新的物理验收。当前P06已有 placed历史但后来FL可能AIR；不能称这等于前腿现在都承载。sigma调度可以用两前轮当前contact/force/gap作为保守代理，真实评价仍完全调用同一物理evaluator。若根决定必须由“当前真实TOP/有效支撑”精准驱动分布，需要**显式版本化追加少量观测位、保留旧输入权重并初始化新列**；不能从不可见 `info`/隐藏timer修改Gaussian而仍用旧372似然重算。

机身空间质量可以由当前evaluator在reward侧使用真实几何（critic已有state可能不完全可观测，但reward本身不要求由actor观测重建）；不要把 `-observation[98]` 的base对顶高度当最低机身净空。诊断应从原始body bounds/髋实体读取，不能用这张表填造。

## 5. HISTORY风险：目前保持ρ，不默认修它

真实公式是 `.1*base_mean + .9*history_center`，除扩大cap首拍外 history_center=previous raw；扩大cap首拍为前一filtered REQUEST/current cap的atanh。sample再进入现有tanh/限速，故raw、REQUEST、effective/final不是同一历史层。

固定base、无观测反馈的简化AR(1)中，ρ=.9的扰动半衰期约6.58 decisions=.439s，1/e约9.49 decisions=.633s，消退95%约28.43 decisions=1.896s。这只是固定输入参考；真实base读到history/接触且mapper有自己动态，实际持续时间可以不同。不能拿此计算宣称当前偏移一定两秒消失。

固定观测时，单通道 `d logp/d base_mean = (1-rho)*(sample-mu)/sigma²`，均值通路有.1因子；但 sigma、Adam归一化、共享MLP、优势符号和clipping会改变实际更新，不能把梯度直接乘10或宣称有效LR必然只剩十分之一。

已读 block12 first-update信用证据：FL捕获即时reward+.205475但normalized A−.348037；最后观察conditional FL hip只移−.000809，sigma−6.0285%，联合概率上升主要来自其他10通道。它支持“应该测量学习吸收/持续历史”的风险，不证明ρ单独有bug。先按状态sigma采样并记录base/history/conditional/currentσ，再查正负历史衰减和实际 mean/sigma更新贡献；若后续证据需要改ρ，必须独立版本，不能本轮与sigma变化混成无法归因的多变量包。

## 6. 最小生产触点和窄验证

1. `semantic_history_actor.py`：新增一份纯latent→state→B/logσ helper及actor版本；沿当前一MLP/一Gaussian调用。deterministic委托原REQUEST均值路径，cache/RNG不动。不要改旧e735类的历史语义。
2. `semantic_policy_distribution.py`：显式version/class/完整state表与单位合同，固定372、全12、当前cap表、rho与mean语义；识别/metadata严格校验更新。可把状态表置于选定配置中的版本化条目，但必须进入完整runtime/hash契约，不能全局可变mask。
3. `semantic_training.py`：runner接受版本，`audited_history_policy_request`使用同一helper并记录state/blend/B/currentcap/learnedσ/effectiveσ；真实storage/likelihood hooks不重建另一套sigma。`audited_ppo_update`/exact-type限制相应增加，不用isinstance偷偷接收未知版本。
4. `semantic_migration.py`、`semantic_cli.py`、`semantic_checkpoint_prefix_policy.py`：显式由真实最新兼容CP迁移；Adam所有状态、参数、Identity normalizer、LR实际scalar/param_group、RNG及counter全保留；旧未完rollout清空；prefix source/effective version分别记录，不能把无optimizer的迁移算更新。当前旧迁移builder要求所有六配置不变；若同步改reward/配置，应做一份清楚的联合新MDP迁移，不在旧sigma-only白名单中偷放新reward。
5. 新experiment还有明确路由触点：`run_semantic_ppo.ps1` ValidateSet、`semantic_cli.version_paths/parser/validate_request`及跨root来源、`semantic_migration.experiment_namespace`、训练request/likelihood audit的experiment枚举、`semantic_video.py`任务窗口/相机枚举、`semantic_video_cli.py`的action与stochastic许可/A共用reader。特别是video action旧else会固定 `stochastic_output=False`，不能漏扩新experiment后还声称随机评估。N+0无checkpoint路径不得使用学习sigma helper。

定向测试应覆盖：

- schema自动定位/已知slice合同、腿/通道顺序、canonical单位、RR149实时资格正反例（回地撤销）、FL历史placed与当前AIR分开；不把wall contact叫TOP。
- 同权重同观测 deterministic μ逐项相等、REQUEST首拍carry不变、全12 sigma严格正有限；选定状态改变比例正确，fallback逐位保留e735；invalid阶段/bits/NaN等拒绝。状态混合向量维度/广播，batch每行不同状态且不会串行污染。
- 一次MLP/一次Gaussian draw；sample、old stored μ/σ/logp、Normal重算、entropy/KL同规则；PPO未更新第一minibatch ratio≈1。使用raw Gaussian概率，不用物理σ代理替代。
- 官方PPO一次CPU合成更新→保存→load→冻结合法prefix wrapper→det/stoch loader；权重/Adam完整保留，非默认LR保持，fresh rollout、RNG恢复。合成计数始终不算真实训练。
- sigma不是residual mask：零residual保留N，单通道不覆盖其他通道；既有source端点/部分stop/nominal与REQUEST分离及phase non-done小回归足够，勿重新千项审计。
- 第一真实完整128 rollout检查实际state/B/σ是否出现预期状态及非零全12，关键P01/P02、P05近接触、P06推进是否实际入optimizer；按真实样本而非课程名称汇报。

本文没有提出或实现gSDE、CAPS、有色噪声、共同wheel新表示、固定负bias、actor reset或改变N/硬件参数。

## Output-only 原型结果（2026-09-20）

按根追加授权，在本目录 `candidate/task_state_sigma_prototype.py` 实现纯张量原型；原始25项CPU定向检查通过，随后新增P06 recovery/max-rear正反例，最新 `candidate/test_task_state_sigma_prototype.py` 为26项全通过，XML为 `candidate/task_state_sigma_prototype_tests.xml`。覆盖上述状态/正反例、部分连续blend、各batch行独立、全12正sigma、cap/mean/原始obs/RNG不变、REQUEST入口重表达、单Gaussian抽样与logp/entropy公式。人工候选B仍未采用、没有物理效果证明。

第一次测试仅因隔离输出目录没有src import路径而在收集时失败，无测试/训练动作；随后在测试文件中显式加入该项目src路径后25项通过。没有改全局PYTHONPATH或reset配置，没有装依赖。原型gate明确命名proxy；例如P06当前顶部附近障碍反力不被称为精确TOP，缺失Y/力向量有效性继续披露。

根随后授权 candidate/src 中3个生产文件的独立候选（不是共享生产）：semantic_history_actor、semantic_policy_distribution、semantic_training。新版本 `task_conditioned_hip_wheel_sigma_v1`、新类 `SemanticTaskConditionedHipWheelHistoryMLPModel`，共同sigma helper供actor/request audit，rho=.9/conditional mean/REQUEST/cap不改。联合factor `task_conditioned_hip_wheel_factor` 有独立loader，保留source full Adam/Identity/RNG/LR/权重/counters并清空旧rollout；archive-only layout factor独立支持但不改变kernel。新branch origin/counters随checkpoint保存。该3文件候选与migration/CLI/prefix候选的完整源→新联合验证由另一agent联测，本报告不预先宣称通过。

最新定向结果：`task_sigma_selected_regressions.xml` 中 **89 passed / 0 failed / 0 skipped**（18新candidate集成+71既有actor/REQUEST/quarter回归）。新集成实际调用官方RSL-RL在纯CPU合成core中采128行、20 minibatch、一次完整update并保存/重载，再接续第二CPU update；128行stored raw/μ/σ/logp与当拍audit逐位一致，首未更新minibatch ratio≈1，非默认Adam LR2.3e−5/betas/eps及normalizer/RNG保留。每种phase一MLP/一Gaussian、det/cached/RNG原路径一致，P06AIR recovery与任一后腿先近的正反例通过。request新增 `task_state_audit_topology=one_actual_N1_request_not_a_batched_N8_summary`，不把N1逐项标量audit称为N8支持。

新增3个测试最初仅因JSON float32的1.200000047与Python十进制1.2逐位比较而失败；改成相同float32期望值后通过，未放宽行为/概率/物理限制。上述CPU更新全部留在pytest临时目录，**零真实policy decisions/零真实PPO credit**；三production文件仍未部署，B值仍待真实probe后选定。
