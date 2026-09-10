# 全阶段进展保存快照：Block1 / checkpoint 139520

| 阶段 | 原目的 / 原条件简述 | 保留或已确认问题 | 本版实际修改 | 新完成 / 交接语义 | 代码测试类别 | a8 当前 C 覆盖：决策 / ticks | 该子任务是否完成 | 仍未知 |
|---|---|---|---|---|---|---|---|---|
| P01 | FR 减载准备；physical_valid→transfer_ready_FR | 保留全身协作；旧空间准备年龄可能借给首次减载，反作用范数不等于承载 | 准备/transfer 独立计时；未知载荷令角色证据无效 | 实测 transfer 成熟，或合法已 Q 的非地面过程接管；不等源计时结束 | R/E/N | 9 / 72 | 4 次记录交接；准备阶段完成，不等于 FR 已放置 | 实际载荷转移的充分性、稳定性 |
| P02 | FR 抬升、净空、接近；physical_valid→Q+clear+approach | 旧窄域无对称容差；已合法越沿/捕获可能被要求退回 | ±5 mm 对称测量容差；允许当前合法 advanced 过程；主动 early TOP 路径 | Q/净空/接近条件，或规则允许的当前 advanced 状态；不靠时间补 crossing | E/S/N | 450 / 3600 | 4 次记录交接；非整条任务完成 | 前腿未接后腿专用 world-down 投影；动作与物理响应不同 |
| P03 | FR 越沿并捕获；physical_valid+FR Q→FR P | 原 FL−0.79、RL+0.61 轮建议超 ±0.6 取消域；旧 swing 可在捕获后续动 | FL/RL wheel cap 1.0，并最低限度不缩延续；捕获退役旧 FR pair | 合法 Q/C 与连续 2 tick verified TOP 产生 P；历史与当前承载分开 | E/N | 22 / 176 | FR Q/C/P 各 4 条；4 次记录交接 | 名义保持不等于实际 q 保持，持续支撑未由历史证明 |
| P04 | 为首次 FL swing 开空间；physical_valid+FR P→FL edge/prepared | FR 历史 P 非当前支撑；空间收缩非真实 transfer | 共用承载有效性、独立准备计时及 advanced 接管；继承轮取消范围 | edge+prepared，或合法已 Q 连续过程；不是固定角度入口 | R/E/N | 10 / 80 | 4 次记录交接；准备完成不等于 FL 捕获 | 接收空间/实际动态支撑充分性 |
| P05 | FL 首次 swing 与 pending capture；physical_valid+FR P→FL P | 保留未捕获时轮 +0.3 协作；旧 FL 源可能拆掉早捕获 | 捕获退役旧 FL pair；当前合法捕获时抑制重复 swing；允许后续接收侧新 owner | 合法 FL Q/C/2 TOP→P；真正地面回退可撤销未跨沿的 Q，再重新取得 | E/N | 286 / 2281 | 至少一次完成：FL P 共 3 次；另 ep0 P05 FALL 未完成 | 前腿接触后稳定性；不能由 P 推当前承载 |
| P06 | 两后腿接近，重叠 FL 接收准备；前腿 P→rear_approach | 原 wheel gain 历史峰值不可回退恢复；并非源到时必断轮 | 仅已有 P06 轮层用当前几何恢复有界 gain；峰值只诊断 | 两后轮当前 edge 满足，或合法 RR Q 非地面过程接管；不伪造 edge 测量 | E/N | 247 / 1968 | 未完成：2 次 P06 FALL，另 P06 非终止尾；末 rear_approach=0 | 当前建议能否形成有效前送/后腿入口 |
| P07 | FL 接收空间与 RR 准备；前腿 P→两 rear edge+RR prepared | 准备 proxy 非 transfer；保留已修 FL hip 时间/范围 | 保留 FL hip 60°/s、cap32；共用成熟度/承载有效性；合法 Q 接续 | 可短阶段接管，但前层未完动作仍继续；不等同源动作做完 | R/E/N | 0 / 0 | 未访问，不能判完成或失败 | 本快照无 C 物理覆盖 |
| P08 | 为 RR 抬升转移载荷；前腿 P→RR edge+transfer | 旧空间年龄可借给首次新 drop；未知低载不应冒充卸载 | transfer 独立成熟 ≥1/15 s；未知载荷无效；不按 phase 清历史 | edge+真实 transfer，或合法已 Q 下游过程；不按标签补 Q/C/P | R/E/N | 0 / 0 | 未访问 | RR 卸载/响应链与前送是否足够 |
| P09 | RR 首先抬升、越沿、放置；前腿 P→RR P | 缺旧 Q 不等于纯轮爬；接触面和早捕获 owner 有旧缺口 | 主动 AIR/early TOP；正证 powered wall ascent；当前几何/承载分层；捕获退役旧 RR pair | Q+当前越沿几何→C，真实 2 TOP→P；RR_FIRST 不变 | E/S/N | 0 / 0 | 未访问；本块 RR 仅尾段 I，无硬 Q/C/P | 名义 q/J 限降非硬净空保证；无本块 C 下游证据 |
| P10 | 检查 RR 当前支持并为 RL 开 FR 接收空间；前腿/RR P→RL edge/prepared | RR 历史 P 可能对应当前 AIR/GROUND；旧回弹角度不是 B/C 入口 | 共用角色/承载有效性；RR 旧 pair 退役但新 knee owner 可动 | edge+prepared 或合法 RL Q 接管；不要求旧 A 回弹姿态 | R/E/N | 0 / 0 | 未访问 | RR 当前支撑与 RL 准备的实际协作 |
| P11 | 为 RL 抬升转移载荷；前腿/RR P→RL edge+transfer | 与 P08 同计时借龄；FR AIR 不自动失败，也不等于承载 | 独立 transfer 成熟及未知载荷处理；接收空间不冒充质心转移 | 连续实测 transfer 或合法 advanced 过程；不固定三腿承载名单 | R/E/N | 0 / 0 | 未访问 | RL 新卸载/响应与其他腿支持 |
| P12 | RL 第二个抬升、越沿、放置；physical_valid+RR P→RL P | 旧 active 模板/面分类；早捕获后源可能继续拆捕获 | 共用主动/early TOP 与 verified bearing；退役旧 RL pair，允许 P13 新恢复 owner | RL Q/C/2 TOP→P；进入 P13 标签不能补 P | E/S/N | 0 / 0 | 未访问 | RL 净空/前送/放置与当前四腿状态 |
| P13 | 全身通过并受控停止；四腿 P→whole_task_success | 原 body 原点、薄 z 层、近零指令与基本任务混合；先成功截尾会藏失稳 | 当前 body bounds/region；基本 event、controlled、strict quality 分层；固定 1 s 后观察 | 首次 controlled 后同回合观察 1 s；最终仍受控且无 region loss/安全失败才成功；严格质量另记 | E/S/N | 0 / 0 | 未访问；无完整任务成功 | 新后观察窗口、稳定性及成功视频均未实证 |

上表是**保存快照，不是当前实时总计**：只纳入已完成 Block1，运行 `20260909T0309011339161Z_ga8b148463115_75823f6efd5f416ab14add86a395a817`，runtime `a8b148463115`，信用范围 **138497–139520**。原条件简述以审计表的 source `2853076` 语义为基准，不把更早冻结 A 的 WAIT_ENTRY 角度模板移作 B/C 入口。I=初始离地过程，Q=硬有效主动抬升资格，C=越沿，P=放置；I 不等于 Q。表中“完成”只指至少一次记录中的阶段守卫/合法交接，绝不代表每回合都完成、当前仍有支撑或整任务成功。

测试类别：E=物理评价/接触合成 CPU；N=名义所有权、取消范围与交接 CPU；R=角色准备/转移成熟度 CPU；S=监督器规则/终止契约 CPU。它们是代码路径和有限构造状态的证据，不是物理通过证据。

## 12 个交接的保存快照

| 交接 | 物理依赖 / 本版语义 | 仍保留的动作与历史 | 代码测试正例 / 反例类别 | Block1 C 实际交接 | 下游完成边界与未知 |
|---|---|---|---|---|---|
| P01→P02 | FR transfer 成熟或合法主动过程；非源时间 | RL hip/四轮前驱层继续；不灌旧 anchor | R/N：短 P01 后继续 slew；未知低载不能借年龄放行 | 4 次，均非终止 | P02 后来完成；不证明整个 FR 链此刻已完成 |
| P02→P03 | Q+净空+接近；合法 advanced 当前状态可承接 | 未捕获 FR 源继续；捕获后旧 pair 退役 | E/N：对称容差/early TOP；区域外不能靠时间补完成 | 4 次，均非终止 | P03 后来 FR P 四次；实际长期支撑另验 |
| P03→P04 | FR 合法 C+2 TOP→P；当前承载另测 | 其他协作层继续；旧 FR pair 不再计时摆动 | E/N：首拍 nominal/request 连续；历史 P 不伪 current TOP | 4 次，均非终止 | P04 后来完成；无固定 home 入口 |
| P04→P05 | FL edge+prepared 或合法 Q 接管 | P04 RL 动作继续；仅目标 pair 的合法捕获可抑制新 swing | R/N：未完 RL 动作继续；不能恢复整 anchor/清轮历史 | 4 次，均非终止 | 3 次 FL P；ep0 在 P05 FALL，无 FL P |
| P05→P06 | 真 FL P；pending 时仍在 P05 允许轮协作 | 捕获旧 FL pair 保持连续 nominal；新接收 owner 可重开 | E/N：1/2 TOP 不算 P；GROUND/区域外 history 不能冒支撑 | 3 次，均非终止 | P06 都未完成：两终止、一尾段 |
| P06→P07 | 当前两 rear edge 或合法 RR Q；非历史峰值 | P06 自有轮层可恢复；后 owner 优先，无额外 +0.3 双加 | N：endpoint 前后回退恢复；后 P07 轮 owner 不被覆盖 | 0 次 | C 未观测；前送动态效果未知 |
| P07→P08 | rear edge+RR prepared，或合法已 Q 过程 | FL hip 等未完层继续，phase 不清动作/角色历史 | R/N：接收侧 AIR 可由真实其他支持协作；不固定 FL 永载 | 0 次 | C 未观测；prepared 不等于 transfer |
| P08→P09 | RR edge+独立成熟 transfer，或合法 Q | RL hip/前驱协作、尝试与速度历史保留 | R/E/N：新 drop 等成熟；负 knee 不等于 world-down，不能补 C/P | 0 次 | C 未观测；残余名义/残差速率差不构成必撞证明 |
| P09→P10 | RR Q/C/P；不要求旧回弹角速度 | RR 旧 pair 退役；P10 新 knee owner 可动 | E/N：合法不同姿态连续；history P 不冒 current TOP | 0 次 | C 未观测；RR 支撑保持未知 |
| P10→P11 | RL edge+prepared 或合法 Q；非短源结束 | RR knee、前驱层与新 FR 接收动作继续 | R/N：FR AIR 不自动拒绝；不能双加 RR delta/臆造承载 | 0 次 | C 未观测 |
| P11→P12 | RL edge+独立 transfer 或合法 Q | RL 自然 AIR/Q 和全身动作继承；几何建议不读当前 residual | R/E/N：持续成熟后接管；地面回退可撤销本次 Q，不抹别腿 | 0 次 | C 未观测；不保证实际净空 |
| P12→P13 | RL 真 Q/C/P；整体任务仍须另判 | 首拍不 home/全轮清零；旧捕获不永久锁新恢复 owner | E/N/S：固定后观察；event 后 FALL/退区不能被先验成功遮盖 | 0 次 | C 未观测；无 P13/完整成功证据 |

合计 **19 次非终止交接**，记录的 bootstrap 契约异常为 0。表中“0 次”是本块未观测，不是动态不可达结论；未独立从这些 compact 记录重建全部 mapper/filter 内部状态。

## Block1 实际账本与事件

- 新增 **1024 决策、8 次 PPO 更新、160 optimizer steps、8177 physics ticks**；阶段 ticks 直接来自 `coverage_table.json`，不是按 8×决策推算。8 次更新属于全阶段共享网络，不能解释为每阶段独立 8 次，也不能把表中重复注释相加。
- 终结回合：ep0 **170 决策/1353 ticks，P05 FALL**；ep1 **402/3209，P06 FALL**；ep2 **174/1391，P06 FALL**。监督器及共同评价器的来源均为 `PHYSICAL_SAFETY`，记录 `run_validity=VALID`、物理证据 `VERIFIED` 不代表任务成功。三个终末动作分别只执行 1/1/7 ticks，较 8192 少 15 ticks。
- ep3 为 **278 决策/2224 ticks 的 P06 非终止尾段**，已进入本块优化，但没有结束、没有成功；末 `rear_approach=0`。完整训练任务成功数为 0。
- FR 四回合 Q/C/P 各 4 条。FL Q 共 5、撤销 2、C/P 各 3：ep0 Q1105→撤销1329，无 C/P；ep1 Q1086→撤销1609→I1635/Q1638→C2057/P2195；ep2 Q918→C/P1168；尾段 Q1126→C/P1484。旧解析器读取完整 `lift_attempt_events`，按腿/事件/tick 去重，不仅取首个 Q，因此撤销和重新取得资格没有被压成单次。
- RR 只有尾段 I1727，无硬 Q/C/P；RL 有 9 次 I，无硬 Q/C/P。所有这些事件均归当前 PPO 信用，prefix 决策/ticks 为 0。历史 P 不替代当前 TOP、AIR 或承载证据。
- **8177 native-effect ticks，8158 own-phase-effect ticks**；native audit、四类禁止状态写入、prefix storage、bootstrap 异常行均为 0。这是保存的原生下发审计，不是所有命令都形成对应物理位移的证明。
- 8 次更新均记录 actor changed、finite nonzero gradient；相邻 actor hash 链连续。已有检查点收据为 **139520/1055 updates/21100 optimizer steps，save_load_round_trip=true**；本报告没有重新加载 PT 或重复哈希。源为 **138496/1047/20940**。

## B2 单独列账：不同 runtime 的非 PPO 诊断

B2 是 `2f942e824f82` 下零 residual 的 `semantic_prior_eval`，不是 Block1 的 C 策略训练，也不是完全同版本配对。它早于 a8 的 P13 finish/timer 最后修订；主线程确认 P01–P12 路径相同，但不能据此将整个运行当同版本比较。

B2 的 900 诊断决策/7200 ticks/60 s 不加入训练账本。阶段计数为 **2,182,4,1,146,330,1,1,233,0,0,0,0**；FR Q24/C1487/P1502，FL Q1556/C2586/P2677；RR 仅 I5909，无硬 Q/C/P，RL 仅 I7。记录 8 次非终止交接，止于 P09 外部诊断窗口，内部 `termination_reason=null`，不是完整任务终止或成功。末 RR 为 `OBSTACLE_AMBIGUOUS`、bearing 未证，负载比例无效；不能补成 current TOP 或失败因果。

## A 与旧结果保留，不增设 5/5 门槛

冻结 A 原结果仍是 **INCOMPLETE_CONTROLLER_BLOCKED，65.3667 s，P10 WAIT_ENTRY**：原入口 RR knee 差 4.5393°、回弹速度差 20.9746°/s；当时没有记录硬安全失败，但任务未完成。旧 A 使用 seed4001、180 settle+64 extra zero ticks，初始化与当前 C 不配对。原视频和原判词不改，新共同判据下缺当前几何/面证据，不能追溯宣布成功。

同样保留旧 B1 的 `TASK_FAILURE_WHEEL_ONLY_CLIMB`、源 C138496 的 P02 不完整原记录；新规则对“缺 Q 不足以证明纯轮爬”的纠正不是把旧失败改写为成功。本快照不要求 A 先 5/5 才继续训练，不把历史不同版本尝试拼成单策略重复成功率。

## 相对 source 的生产改动清单与测试界限

以下清单来自只读 `git diff --name-status 2853076 a8b148463115 -- src configs scripts`；不是本报告新修改：

| 范围 | 文件 | 作用 / 保留边界 |
|---|---|---|
| 新实验配置 | `configs/ppo_all_stage_acceptance_v1/{action_schema.json,execution_profile.yaml,observation_schema.json,quality_score.yaml,reward_config.yaml,stage_task_spec.yaml}` | 独立 namespace；全阶段物理/终止语义及定向取消范围；旧配置不覆盖 |
| 新测量适配 | `semantic_physical_sensing.py` | 当前姿态 collider bounds、接触面/竖直承载、机身证据；点集算术优化不等于已验证动力学优势 |
| 监督器/名义层 | `semantic_supervisor.py` | 主动过程与 advanced 准入、当前完成/严格质量分层、有限 deadline/post 观察、capture owner 与 P06 恢复 |
| 角色状态 | `semantic_transfer_roles.py` | 准备与实际 transfer 的独立成熟度；372 布局不改，不声称完整历史严格 Markov |
| 后端/共同 A 评价 | `semantic_backend.py`、`semantic_legacy_evaluation.py` | 新共同测量/终止结果接线；冻结 A 控制器/物理动作链不因新评价而重写 |
| 加载/训练/入口 | `semantic_migration.py`、`semantic_training.py`、`semantic_cli.py`、`scripts/run_semantic_ppo.ps1` | 显式新 MDP、独立路径和结果字段；保存网络/std/critic/normalizer/RNG/累计账本，验证源 Adam 后 fresh Adam、空 rollout；不把同输入权重一致称轨迹一致 |

上述生产相对路径以 `src/wlr50_clean/ppo/` 为模块目录。actor kernel 实现、372 的顺序/scale、全 12 通道、原 mapper/hard limit/slew、rho、gamma=.9985、lambda=.99、128 rollout 均不是本轮新架构，网络参数仍实际更新。reward 配置字节与五个奖励家族保留：1 个 task 家族、3 个启用的稠密质量家族及 1 个关闭的弱正则。physical Phi/角色观测语义改变，因此首次加载是显式新 MDP，不能叫无变化续训；之后同版本块则普通续训并保留 Adam。

记录的 **376、166、102** 为不同回归批次，覆盖有重叠，**不得相加**。E31、N71、R25 等是专属有限 CPU 范围，也不能再叠加到集成批次当独立总数。N 的早期 trial01 50 项是后续 71 项的旧子集。本报告未重跑任何测试；synthetic 条件成立不等于 Isaac 全阶段物理接受、真实传感器永远可靠或稳定性优于旧版。

## 交付与缺失证据

本快照**没有完整自然 P01 重载评估、没有本版成功视频、没有稳定性优越结论**。P01 开始的随机训练回合不等同固定 mean 评估；当前没有依据计算成功概率。Block2 或之后的 live/完成数据均不在这里，不能把保存快照当最新累计数。

依据文件（均为已经保存的证据，不重扫 raw）：

- [audit_tables.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/audit_tables.json)：完整 13 阶段/12 交接语义与 B2、Block1 索引；其中准备期说明保留历史，当前覆盖按两个分立证据字段解释。
- [Block1 summary](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/block1_summary/training_block_summary.json)、[coverage](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/block1_summary/coverage_table.json)、[固定累计账本](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/metrics/cumulative_after_block1.json)。
- [B2 completed](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/reports/B2_completed/prior_diagnostic_summary.md)、[历史原判词](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/manifests/historical_verdict_comparison.json)、[逻辑/reward 修订说明](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/reports/logic_and_reward_changes.md)。

此次仅新增本 MD；没有修改 audit 输入、生产、CSV、检查点或原结果。
