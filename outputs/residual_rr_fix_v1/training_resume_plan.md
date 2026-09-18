# Residual RR 修复续训方案与实际交付

状态：`TRAINING_640_SEALED_FORMAL_CP178432_INCOMPLETE_P05`。生产已提交并冻结；本次只更新outputs，未修改生产或启动Isaac。

## 1. 实际恢复、训练与待判结果

迁移源为CP177792 / 1354 updates / 27080 optimizer steps；正式plan `checkpoint177792_rr_physical_acceptance_migration.json` 已生成并重建验证，SHA256 `962d4a7a10cf8c44cc38bebd83ebc853adaa52453bca5915c059d321d4125a9c`。builder为同目录 `build_rr_physical_acceptance_migration.py`。

两块真实训练自然封存，实际新增 **640 decisions / 5 updates / 100 optimizer steps**：

| 真实run | 采样 | 实际新增 | 末态 |
| --- | --- | --- | --- |
| `20260917T0450450693297Z_g6c2121b68654_8996bc65765a494c877b12dae8792f81` | naturalP01/full_episode |128/1/20 | tick1024/P02非终态，无完整episode |
| `20260917T0454003203604Z_g6c2121b68654_7079dc3b6d4f471d8d3dfe5404d49924` | naturalP01 successful_nominal prefix→P06 suffix |512/4/80 | tick6776/P12非终态，无完整episode |

run均在 `runs/ppo_residual_rr_fix_v1/train/`；第二块于2026-09-17 05:10:23 UTC封存。生命周期 `SUCCEEDED` 只指训练操作完成。最新不可变history checkpoint为 `outputs/ppo_residual_rr_fix_v1/checkpoints/history/checkpoint_step_000178432.pt`，累计 **178432/1359/27180**。

- checkpoint SHA256：`19acb91f64a2b728e54f02d50dcaf187e11fd432ff68901ba91ff0af58f6a665`
- manifest SHA256：`78dc71b80f897fa843f365c0ae3ba7150ade970eaf5ff57325141e465a91453a`
- checkpoint_last指针、官方save/load round-trip与计数一致；此前每update CP178048/178176/178304保留。
- 分支样本：P01=2、P02=126、P03–P05=0、P06=166、P07=1、P08=1、P09=214、P10=1、P11=19、P12=110、P13=0。
- 共5120 PPO物理ticks；第二块335 teacher decisions/2680 ticks及其FR/FL完成历史独立排除。不是从RR已离地快照开始；前缀从自然P01合法重建。
- 640/640 full12许可/epsilon0，5120/5120 native dispatch验证与实际目标影响；7次普通阶段切换、0错误done。所有阶段继续可学习，未采用永久轮残差屏蔽。

末尾P12/RL未越沿或放置，RR虽历史越沿5726/放置5731，当前却GROUND、front=-.130729m、clearance=-.050347m、current_lift_valid=false、rr_placed_currently_usable=false。P12当前entry_invalid的理由为placed_RR。RR最初资格4001后来撤销，5434是此次越沿前的最后有效资格，6053当前抬升再撤销；不得把旧事件当作当前支撑或完整成功。

CP178432真实官方重载后的自然P01全程视频评估已封存 `DIAGNOSTIC_FAILURE`（run `20260917T0510347247858Z_g6c2121b68654_35383f247031424e9b890de765bc2b13`）。实际5907ticks/49.225秒、739全12决策、teacher0、无mask干预；checkpoint actor/critic/Adam/Identity精确匹配，评估optimizer updates=0、训练信用0。

首个未完成任务为P05的FL受控放置：FR Q23/C1406/P1425，FL Q1484/C2106、未placed；最终FL AIR、gap+.003470128m、bearing0。semantic原因 `INCOMPLETE_CONTROLLER_BLOCKED`，来源 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；37.225秒=基础30+实际进展延长7.225，有限任务终止、非外部截断、bootstrap=false。物理evaluator valid/VERIFIED，无安全终止；5908含tick0观测中body collision/nonfinite/hardlimit违规均0，最小关节余量21.47384°。wrapper exit1是共同任务未通过，不是Isaac崩溃。

完整评估并未进入RR，因此不能将P06课程的后缀事件称为正式后腿成功。新严格判据0/1完整成功，没有成功checkpoint指针。结果见 `evaluation_results.json` 及独立actual/safety receipts。下一步先补P05任务能力，然后处理进入RR后的carry和当前放置保持；quality epsilon=.05不启用。

实际receipt：`training_P01_177920_actual.json`、`training_P01_177920_signal.json`、`training_P06_178432_actual.json`、`training_P06_178432_signal.json`。独立总账：`recovery_branch_and_training_manifest.json`。未改旧task-first账本或best历史指针；epsilon.05继续延期。

## 2. 保护对象与实际版本边界

源CP177792原样保留：SHA256 `e4552bb18ef223fb51da31a616cf57cf3291ba1a108b08e7da40c07bee0db0ba`；manifest SHA256 `9f78b80a8128c3a44c4b13c3e6ea4fbf47e01b491eff8a62c73f0aae50dd1b0c`。源Git `fc14a68c037abd69e24f49b1524457fd6854543a`，runtime `bfa1e1471cc0e645ae33e873439a242142dde5adceb70a87bcc419152815c48c`。

目标Git `6c2121b68654eaaa2afa7c19e9d02ae770493d2f`，runtime `cfad9c3664f4e479fd16a9902178cc4e84327ab06aa03dd3db83e3d96a89aed1`。独立factor `rr_physical_acceptance_same372_factor`，schema `wlr50_clean.rr_physical_acceptance_same372_continuation.v1`；仅允许此次task-first→RR边界，不扩展旧reward/temperature/composition/final-stop授权。

receipt绑定精确源CP/manifest、source/target runtime/Git、六配置与实际代码diff。没有大型AST发布框架、全Recording扫描、A/B五连或人工非零探针全程成功门禁。

保护原73.808333秒成功zero及其所有源文件。新zero v2 `runs/ppo_non_residual_refine_v1/video_eval/prior_B/20260917T0424208857504Z_g6c2121b68654_13804a86a302480ab61ad2af0dcb8dcd` 亦已真实成功8857ticks/73.808333秒，是nominal-only，不是learned PPO。zero v1 `20260917T0343245466199Z_g3d231897c91c_0e58ddffca37441089dd17f34f4ed5d8` 最终轮速失败结果保留。CP177152与原evaluator成功标签不篡改，但当前 `RR_ACCEPTANCE_UNDER_REVIEW`，不能当新AIR/free-rise成功证明。保护引用见 `outputs/diagnostics_v1/checkpoint_and_protected_baseline_references.json`。

## 3. 实施语义与六配置精确差异

沿原工程使用 `configs/ppo_residual_rr_fix_v1/`、`runs/ppo_residual_rr_fix_v1/`、`outputs/ppo_residual_rr_fix_v1/`；用户说明集中于 `outputs/residual_rr_fix_v1/`，没有新建训练引擎。

五个非stage配置字节不变；stage parsed diff仅四处（详见 `planned_configuration_delta.json`，该文件是已实施配置说明，不是可执行迁移receipt）：

| stage路径 | 源 | 目标 |
| --- | --- | --- |
| revision | fsm_reference_p09_stable_v2 | residual_rr_fix_v1 |
| p09_lift_semantics | functional_lift_edge_v2 | functional_free_air_lift_v3 |
| nominal.rr_carry_source_semantics | 不存在 | current_free_lift_before_pending_knee_and_roll_v1 |
| nominal.final_stop_owner | current_physical_stop_nominal_owner_v1 | source_home_after_physical_stop_v2 |

新RR资格使用实际连续AIR中的free-rise证据，区分接触边沿上升/有效抬升/保持/越沿/放置。失去当前有效抬升时source/carry不因历史qualified提前接管；whole-body抬升允许、不能要求RR自身先有特定角度变化。terminal home v2使用0.5秒有界quintic nominal建议再exact source-home hold，保留原1秒物理观察窗口，不以画面复位授予成功。

372布局/字段顺序/Identity不变，但任务派生的历史、goal、role readiness及nominal值语义改变。明确 `task_acceptance_changed=true`、`nominal_control_changed=true`、`action_execution_changed=true`；不宣称完整物理MDP等价。物理场景、actuator能力、硬限、全12容量、几何v3、reward代码权重/gamma/lambda/epsilon0不变；task事件与potential输入语义改变单独声明。没有geometry v4或质量重调。camera仅影响取景与证据、不改变控制/传感器。

## 4. 状态续接与生产路由

完整policy contract保持quarter `SemanticQuarterTemperedHistoryMLPModel`、372/full12、innovation temperature=.25、rho=.9、原HISTORY/限幅/限速。新factor不是温度迁移，不改mean/std head。

保留actor/critic、完整Adam moments/steps/param groups、有效LR1e-5、Identity、训练RNG与累计counter/stage预算；runner配置默认LR3e-5不得覆盖实际加载的1e-5。真实update后actor/critic/Adam/RNG自然变化，最终状态不应仍等于源。新语义边界丢旧unfinished rollout和transition，合法reset后重新采集，普通阶段不reset mapper/HISTORY或done。第二块同runtime exact-resume无需第二迁移；通用summary里的null直接migration字段不是继承旧rollout。

生产已接通migration/loader/CLI、videoCLI/PS routes、新six configs、semantic_video task-window；core贯通supervisor/backend/transfer_roles v3 guard。观测布局、sensor物理阈值、geometry、actor和reward未顺带修改；video侧使用一致新判据。CLI continuation-only依靠现有v3 checkpoint必需规则，不能意外fresh-network启动。

loader写 `rr_task_branch` 与独立origin177792/1354/27080，保留祖先mean-head/quarter metadata；same-branch续训沿原exact-resume且审计每update likelihood。新root保存history，不伪增global step防旧文件撞名。

## 5. 已完成最小验证与实际审计范围

`outputs/diagnostics_v1/rr_migration_tests.xml` **33 passed**，`rr_migration_regression_tests.xml` **181 passed**，无failed/error/skipped。包括官方quarter CPU load→update→save/reload；非空Adam/有效LR/Identity/RNG/counts逐项保留；fresh storage；配置/namespace/temp/rho/容量/reward/物理/无关代码篡改与partial-rollout/无CP负例。CPU测试无Isaac信用，不代表物理成功。

两个封存真实块首update actor-before分别匹配CP177792/177920，官方loader强制状态校验与RNG恢复路径成功；没有捏造独立“加载后采样前”快照。官方save/load round-trip已记录。第二块4updates均有限非零梯度且actor变化，20steps/update、LR1e-5。末update KL=.02273239822、valueLoss=1.195866728；中间valueLoss7.451954也原样报告，不隐藏。

输出helper `outputs/ppo_task_first_recovery_v1/summarize_recovery_training.mjs --run <sealed_run> --branch residual_rr_fix_v1` 已支持新factor及已记录same-branch ancestry，未以旧namespace/hash断言挡训练。reward helper `summarize_task_reward_signal.mjs <sealed_run>` 复用已录GAE、无额外模型推理/反事实拼接。

P06后缀观察折扣回报−1.369575不是完整episode回报；P09势能合计+.056756与P12−.877219只用于说明当下信号。正GAE受critic bootstrap/rollout归一化影响，不能替代任务成功。所有新actual/signal/ledger在本独立目录，不给未来训练任何预付信用。

已录quality仅描述此49.225秒P01–P05失败轨迹：roll RMS=6.954382°、pitch RMS=5.749541°，roll/pitch rate P95=.018862/.017192rad/s。固定全程quality score=null且缺少P06–P13；长期P05驻留可压低速度指标，不能与73.808333秒成功zero比较优越性。没有质量、稳定性或泛化改善声明。
