# Request-history cap-transition：最小生产接线提案（未实施）

本次仅只读源码、写 outputs 文档；没有注册 actor、修改生产、重新训练或启动 Isaac。根任务先完成当前冻结版本 ≥2048 决策和正式视频，再决定是否实施。CPU 原型的 15 项测试不等于下面生产 loader/audit/prefix 已接通。

## 已确认的阻断与非阻断

1. **request audit 当前必然拒绝新类。** `semantic_training.py:1261 audited_history_policy_request` 的 `type(actor) is SemanticQuarterTemperedHistoryMLPModel` 精确 guard 会先抛错；即使只放宽 guard，1282 的旧 `history_conditioned_head(head, obs[195:207])` 仍会在 cap 首拍和实际新 μ 不同。原型不是绕过 MLP hook：仍只调用一次 `self.mlp(latent)`，原 `register_forward_hook` 能观察这次真实 head。必须分版本调用同一个纯 history-center 函数，不能额外 forward 或补一次采样。
2. **真实 minibatch likelihood 数学不需改。** `audited_ppo_update` 包装的 `actor.get_output_log_prob(stored_raw)` 和官方 PPO 调用是正确接点；其 `history_source="this_saved_observation_195_207_not_shuffled_neighbor"` 已不完整，须为新版本记录额外的 stage/age/completed/REQUEST 索引。不能修改存下的 raw 或以滤波后动作代替 likelihood 参数。
3. **训练/视频已有 actor factory，可复用。** 不是新建训练框架。新类需完整通过 policy contract、runner config、load、prefix 精确类检查；不能只改模型文件或只 monkeypatch `forward`。

## 必改的 6 个生产文件和职责

| 文件 / 现有接点 | 最小改动 | 必须保持 |
| --- | --- | --- |
| `semantic_history_actor.py` | 只追加新显式版本 class 与纯 `request_transition_history`；actor 和 audit 共用纯函数。新 forward 保留现有372/layout/Identity检查、先 unpad，再取 latent；单次 MLP、单次随机 draw。 | 旧 HISTORY/.5/.25 类及旧函数字节/行为不动；新 state_dict 名称/形状、无私有状态/新增 buffer；JIT/ONNX继续明确拒绝。 |
| `semantic_policy_distribution.py` | 新 POLICY/ACTOR_CLASS 常量；接 `policy_contract`、`configure_policy_distribution`、`supported_heteroscedastic_contract_version`、`policy_version_from_metadata`。完整绑定 gate、REQUEST片段/尺度、精确 cap 表和120/15对齐假设。 | 只接受 v3/372/Full12/Identity/.25。旧 contract 不改。新 mean 明确会变化，不沿用“same deterministic mean”的迁移声明。σ仍是旧 learned logσ+log(.25)，非 stationary σ。 |
| `semantic_training.py` | `semantic_runner_config` 支持新版本；新增窄 factor validator；`load_semantic_checkpoint` 接 source→target kernel pair、exclusive observation factor/runner差异；`audited_history_policy_request` 分版本重建中心并记录；`audited_ppo_update` 修新版本 provenance 字段。 | 官方 `PPO.act → storage → PPO.update/actor.forward → get_output_log_prob` 不换算法；不重置 mean/Adam/normalizer/RNG；只允许空 rollout/pending transition为空；实际 LR1e−5恢复。 |
| `semantic_migration.py` | 追加独立同372 kernel factor 的 builder/重建验证和精确 allowlist。绑定 source CP+manifest/hash、真实 source/target HEAD/runtime/6config hashes、目标 actor/runner contract、中心语义。 | 不借“temperature factor”伪装均值变化；不拓宽旧 factor。一次只改此 kernel，排除 reward/cap/mapper/HISTORYρ/σ温度/验收变更混入。 |
| `semantic_cli.py` | `_preflight_checkpoint` 在新 factor 验证后选择目标版本；使用该目标构造与解析 contract，允许且仅允许已审 source/target runner 差异；训练/普通 eval 复用该选择。 | 不能用旧 `--new-mdp-warm-start/--target-policy-version` 路径意外重置 optimizer；保留当前 `--resume-migration` 和同种子。无 migration 的旧CP→新runtime必须拒绝。 |
| `semantic_checkpoint_prefix_policy.py` | 精确 version→class 映射添加新 class，并保留 `.forward/get_latent` 精确函数、official Normal、Identity、参数hash独立copy检查。 | 不能扩大为任意 subclass。N1 frozen checkpoint prefix 继续 deterministic、无 optimizer/PPO样本信用、全段独立冻结。 |

以上是最小核心触点，不包含改名新 experiment 的额外路由。**若继续现有 `fl_capture_quality_v1` profile，仅用新 policy/runtime version 和独立 run/branch metadata，6个配置可逐字不变，PS/视频主循环无需机械修改。** 若根任务另选新的 experiment id，则必须显式追加 CLI 路径、PS ValidateSet、训练 audit/likelihood分支、video stochastic允许集合和camera/task-window集合；不能仅换目录名后默认继承。不要为本单因素候选同时更改任务 profile。

## request / likelihood 审计的准确字段

旧 `previous_raw_from_current_observation_full12` 必须继续记录真实 raw，不能悄悄改成物理中心。新 schema/version 追加：policy version、`history_center_full12`、`cap_transition_gate_full12`、stage、encoded stage age、predecessor completed、current/previous cap、decoded previous **filtered REQUEST**、history-source semantics。保留真实 base mean、conditional mean、learned/effective sigma、sample、tanh、logp。审计用 hook 已捕获 head 和当前372纯计算重建中心；校验 stochastic actual distribution params / deterministic actual returned mean 一致，额外forward/draw仍为0。

调用点：训练 `train_semantic` 当前1459分支包住真实 `runner.alg.act(obs)`；视频 `semantic_video_cli.checkpoint_loader.action` 当前56分支包住真实 actor；二者在保留 FL namespace时自动使用修改后的审计。minibatch audit 应用 actor 当前 official params 而非重新 forward；新 provenance必须说明中心来自该条保存观测，不能使用当前环境或 shuffle邻居。新收集数据 old μ/σ/logp 由新 kernel 产生；每次 update 中同一保存观测重建 gate，故“首拍”是观测谓词，不是只允许执行一次的 mutable flag。

## age=0 的合法来源和边界

实际索引：stage `[0:13]`；age `[20]×200秒`；completed `[158:171]`；raw `[195:207]`；previous filtered REQUEST `[207:219]×[4,4,4,6,4,4,4,4,.12,.12,.12,.12]`。严格 gate：`age == 0 AND phase> P01 AND immediate_predecessor_completed AND current_cap > predecessor_cap`，逐通道应用。

| age=0 情形 | 应有行为 |
| --- | --- |
| 自然 episode P01、失败后真实 reset/fallback P01 | P01无前驱，gate关；不把初始零历史误认 transition。 |
| 普通相邻阶段换段 | supervisor `observe_and_update` 仅 tick%8==0换段、同拍 `stage_started_s=now`；下一15Hz决策 age恰0。仅P03的FL/RLwheel与P06的前10通道 cap增大，其余旧中心。 |
| 同阶段重复查询、deterministic回放、PPO shuffle重复使用同个age0观测 | 每次纯函数给同一中心；不能“第一次调用消耗标志”。重复读取不是额外物理决策。 |
| 直接构造 `TaskStageSupervisor(initial_stage_id=P06)` 的合法离线/单测首观测 | 初始化 completed为空，因此gate关。不能只凭P06+age0认定前驱已执行。 |
| checkpoint-policy prefix 在目标初入、offset=0处开始信用 | 原 `SemanticResidualEnvironment` 连续step已填真实 raw/filtered REQUEST，prefix返回同一个core observation，不reset；若该阶段cap增长，首个被信用决策正常用同gate。P04 prefix以后自然到P06同理。 |
| successful-N prefix 的目标初入 | 同样连续真前缀；raw/filtered REQUEST为0，新中心也0，与旧kernel等价，但actor仍可有非零 `.1·base_mean`，不可误写成强制零动作。 |
| 前缀 teacher offset>0或本阶段第二拍以后 | stage age>0，不触发。阶段内source调度/substage切换/末段source-home不是stage age重置。 |
| legacy frozen-FSM teacher接管 | 不改旧路径。它的残差历史在其环境初始化时是0；不能借新kernel补造PPO历史。默认候选使用已验证自然P01/checkpoint-policy/N-prefix合约。 |

当前实现只在 stage初始化和实际阶段迁移赋 `stage_started_s`，没有因采样、局部恢复或优化器更新重置它。最小正物理间隔为1/120s，除以200后的float32远离underflow，不会把正常非零age舍入成0。`-0.0`按0处理无差别；负数/NaN/非onehot/错误completed编码应拒绝。前驱cap严格小于新cap使合法旧REQUEST在atanh域内，即使正好旧cap边界也无需裁切。仅容忍有依据的float32解码误差，不接受任意超界值。

372的充分性只覆盖当前单向、相邻、120/15对齐的阶段图与固定cap表。若将来允许跳阶段、时钟reset或不同周期，应换合约/增加明确prevphase信息；不能凭completed累计列表可靠推断任意上一控制状态。生产 preflight必须核对实际 stage `next_phase` 链和cap配置与新actor绑定一致，不能把输出原型的硬编码矩阵复制进去却不绑定runtime。

## previous filtered REQUEST 不是“已实现动作”

`semantic_env.py:163` 在每个真实physics step把 `projection.safe_projected_residual_full12` 保存为 `previous_residual_full12`；这是 tanh×cap、mask、残差slew/阶段限制后的请求。post-mapper geometry/headroom/final-drive slew在后续派发层还可能改变它。因此新中心维持的是“上次**请求**对应新cap的latent表示”，不是关节 actual、不是相对某条独立重算 nominal 的最终差值，也不是确保 final target 无变化。现有bridge单tick hold、60°/s、final1.25°/tick、nominal/controller所有权和物理硬限必须原样保留。不能以此候选删掉它们。

原型中首P06 FR的中心μ1.07495→.150666，σ依然.337379；其他通道的σ也未变。因此仍可能采到较大innovation；单因素候选不承诺免碰撞或全面稳定。`atanh(prev_request/current_cap)`用于Gaussian中心，最终 raw变量坐标不变，所以无需人为增添tanh Jacobian；若改成事后变换sample或物理空间分布，是另一方案，需要重新定义正确likelihood，不在本清单。

## prefix、det/stoch eval 与保存/迁移不能遗漏

- `semantic_video_cli.py:checkpoint_loader` 已通过 `_resolved_policy_version` 构造actor；新版本加入完整registry/loader后无需临时缩放。deterministic和显式`--stochastic-policy --policy-seed`同CP、同kernel；视频proof记录新contract和sampling mode。`semantic_cli.evaluate` 的无视频默认deterministic也必须走同class。保留FL experiment时这些入口本身不必改；必须测试而不是假定。
- 新 factor 同时进入 preflight 和实际 `load_semantic_checkpoint`，两次重验plan。新版本不满足当前仅temperature允许的runner/class差异，必须显式接入对应分支，不能通过放宽通用验证规避。factor加入exclusive observation-contract选择以及metadata精确检查；现有FL reward factor的历史ancestry保留，但不能把本次行为改动伪装成原reward factor。
- 保留 source learned actor/critic、完整Adam moments/steps、Identity、实际LR1e−5、所有RNG/总计数与已验证 ancestry；source weights hash可相同但**行为已变**。旧未完成rollout/pending action不复用。新factor记录migration本身+0样本/+0更新；若保存初始化CP须明确initialization，不能标新学习成果。
- `save_semantic_checkpoint` 已从runner输出新的runner_config+policy_contract并做实际save/load roundtrip；新runner marker不能漏。保存后的普通resume无需再次迁移，但仍必须完整contract匹配。采用独立branch计数或至少明确本次source counter origin，输出目录/immutable checkpoint不得覆盖旧版本；不必为此另建庞大发布体系。
- 同checkpoint prefix在**迁移当块**尤其要绑定有效目标kernel：tensor来自sourceCP，函数来自targetactor，provenance应同时清楚记录source checkpoint hash及迁移plan/target policy/runtime，而不能让人误以为磁盘sourceCP原先就是新kernel。最小做法显式补充现有prefix provenance的effective kernel/plan字段，或先保存+验证zero-credit新版本初始化CP再用它做prefix；二者择一，不能静默混称。
- 对照N物理/动作不变，旧zero媒体可保留，但新build比较不能声称重新进行了同build物理B。正式视频仍需该真实checkpoint自然P01全程，后缀不冒充完整成功。

## 最小验证清单（不是新增成功门禁）

1. 原型已有15项 + 将来生产类复跑真实1362观测；新旧非gate帧bitwise相同，old classes/state topology不变、σ不变；支持混合batch和padded/unpadded、shuffle的同观测logp。
2. 新request audit包住一次真实官方act；新/旧stochastic和deterministic均正确，开启/关闭审计权重/RNG/sample逐位一致；原始raw与实际center字段不混。
3. 官方CPU loader按真实plan恢复全部状态 → 新kernel采集小合成rollout → first ratio≈1 →真实PPOupdate → 官方save/load。旧rollout/nonempty storage、旧/newclass混淆、caps/age schema/tamper应拒绝。CPU合成数据零真实信用。
4. 同新CP冻结prefix exactclass/copy隔离 + offset0首拍age0与raw/REQUEST连续性；N-prefix零等价、不新增前缀PPO信用；video det/stoch两入口都确认新kernel且无临时缩放。

不要求额外零残差成功门禁，不改变现有运行。正式是否采用及真实验证顺序由根任务在当前训练、checkpoint与视频交付后决定。
