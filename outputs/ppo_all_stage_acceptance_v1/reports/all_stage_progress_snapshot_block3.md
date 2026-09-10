# 全阶段保存快照：Blocks1–3 / checkpoint 140032

| 阶段 | 原目的 / 原条件 | 保留或问题 | 本版实际修改 | 新完成 / 交接语义 | 代码测试类别 | a8 C 决策 / ticks | 子任务完成边界 | 未知 |
|---|---|---|---|---|---|---|---|---|
| P01 | FR 减载准备；transfer_ready_FR | 全身协作保留；空间年龄曾借给首次减载 | 准备/transfer 独立计时，未知承载无效 | 实测成熟或合法已 Q 非地面接续；不靠源结束 | R/E/N | 9 / 72 | 4 次记录推进；非 FR 整链完成 | 动态转移充分性 |
| P02 | FR 抬升；Q+clear+approach | 旧窄域可能拒绝合法先进状态 | ±5 mm 容差；主动 early TOP/advanced 准入 | 当前 Q/净空/接近或合法先进过程，不补越沿 | E/S/N | 450 / 3600 | 4 次记录推进 | 前腿净空保持、后续稳定性 |
| P03 | FR 越沿与捕获；FR P | 旧 FL/RL 轮取消域不足；旧 swing 可拆捕获 | 两轮 cap1.0、最小非缩延续；退役旧 FR pair | 合法 Q/C+2 tick verified TOP→P | E/N | 22 / 176 | PPO FR Q/C/P 各4；历史非当前支持 | 持续支撑 |
| P04 | 为 FL 开接收空间；edge+prepared | FR 历史 P 非当前支持 | 承载有效性、独立准备、advanced 接续 | 当前准备或合法后续过程，非固定角度 | R/E/N | 10 / 80 | 4 次记录推进 | 实际支持/空间协同 |
| P05 | 首次 FL swing/capture；FL P | 保留 pending 时轮+.3；旧源可拆早捕获 | 仅退役旧 FL pair；不永久锁后续 owner | 合法 FL Q/C+2 TOP→P；地面回退可撤销未跨沿 Q | E/N | 286 / 2281 | PPO FL P3；另 P05 FALL 未完成 | 捕获后持续承载 |
| P06 | 两后腿接近/桥接；rear_approach | 旧 wheel gain 峰值不可恢复 | 当前几何恢复有界 gain，峰值只诊断 | 双 rear 当前 edge，或合法 RR Q 接续 | E/N | 610 / 4872 | 2 次进入 P07；不证明原双 rear 目标同时满足 | 前驱准备是否充分 |
| P07 | FL 接收空间/RR 准备；两 rear edge+prepared | 保留 FL hip 时间/范围；准备非 transfer | FL hip60°/s、cap32；角色有效性/合法 Q 接续 | 短阶段可接续，前层动作不因此完成 | R/E/N | 7 / 56 | 7 次进入 P08；其中5次始于教师 P07 接管 | 当前策略能否自行形成同入口 |
| P08 | RR 减载；edge+transfer | 空间年龄不能冒充新减载成熟 | transfer 独立≥1/15s，未知载荷无效 | 实测 transfer 或合法 Q 接续；不按标签补事件 | R/E/N | 33 / 264 | 7 次进入 P09；不代表 RR C/P | 减载/前送的实际充分性 |
| P09 | RR 首先抬升越沿放置；RR P | 缺 Q 不自动等于轮爬；面/owner 曾有缺口 | AIR/early TOP、正证违规、当前面/承载；捕获退役旧 pair | 合法 Q+当前几何→C，真实2 TOP→P | E/S/N | 109 / 851 | 未完成：7 个 P09 终止回合；无 RR C/P | Q 后保持/越沿；限降不是物理保证 |
| P10 | 用 RR 当前支持为 RL 开空间；edge+prepared | RR 历史 P 不代表当前 TOP | 承载/角色共用有效性；允许新 knee owner | 当前准备或合法 RL Q 接续，非旧回弹姿态 | R/E/N | 0 / 0 | 未访问，不能判成功或失败 | RL 准备与 RR 支持 |
| P11 | RL 减载；edge+transfer | FR AIR 不自动失败亦非承载 | 独立 transfer 成熟、未知载荷处理 | 当前成熟或合法先进过程；不固定支持名单 | R/E/N | 0 / 0 | 未访问 | RL 减载与接收侧响应 |
| P12 | RL 第二个抬升越沿放置；RL P | 早捕获后旧源可继续拆捕获 | 共用主动/面证据；退役旧 pair，允许 P13 新 owner | 合法 RL Q/C/2 TOP→P；P13 标签不补 P | E/S/N | 0 / 0 | 未访问；无 RL C/P | RL 物理链 |
| P13 | 全身通过并受控停止；whole_task_success | 原点/薄z层/严格小指令曾与基本任务混合 | 当前 bounds/region；event/controlled/quality 分账；固定1s观察 | 首次受控后1s，末仍受控且无区域丢失/安全失败 | E/S/N | 0 / 0 | 未访问；无全任务成功 | 后观察、稳定性、成功视频 |

## 12 个交接

| 交接 | 依赖 / 保留动作与历史 | 正例 / 反例测试类别 | C 实际非终止交接 | 完成边界 / 未知 |
|---|---|---|---|---|
| P01→P02 | transfer 或合法 Q；RL hip/前送层继续 | R/N：短阶段仍 slew / 未知低载不借年龄 | 4 | FR 后来捕获，不是交接瞬间已完成整链 |
| P02→P03 | Q/净空/接近或合法 advanced；未捕获源继续 | E/N：容差/early TOP / 远处不靠时间放行 | 4 | FR P4；持续承载另验 |
| P03→P04 | FR 真实 P；旧 pair 退役，其他协作继续 | E/N：首拍连续 / 历史 P 不伪 TOP | 4 | 无固定 home 入口 |
| P04→P05 | FL 准备或合法 Q；未完 RL 动作继续 | R/N：协作不断 / 不灌 anchor、清轮历史 | 4 | FL P3，另一次 P05 FALL |
| P05→P06 | FL P；旧目标退役，P06 新 owner 可重开 | E/N：捕获保持 / 不永久锁后续腿动作 | 3 | P06 仍可失败；历史非当前支撑 |
| P06→P07 | 双 rear 接近或合法 RR Q；wheel tail/桥接继续 | E/N：合法连续接续 / 不从远处补 edge | 2 | 均来自 Block2 PPO；教师到 P07 不计入此项 |
| P07→P08 | rear edge/准备或合法 Q；FL hip 未完继续 | R/N：首拍不跳 / 不把准备当 transfer | 7 | 2 个 P06 策略前驱+5 个教师 P07 接管 |
| P08→P09 | transfer 或合法 Q；全部 mapper/request/history 连续 | R/E/N：独立成熟 / 不借空间年龄 | 7 | 进入 P09 不等于 RR C/P |
| P09→P10 | RR 合法 C/2 TOP→P；旧 pair 退役、新 knee 可动 | E/N：新 owner 可动 / 不用墙接触补 P | 0 | 无 RR C/P，后续未验证 |
| P10→P11 | RL 准备；FR 接收动作继续 | R/N：协作不断 / RR 历史不伪当前承载 | 0 | 未访问 |
| P11→P12 | RL transfer 或合法 Q；FR hip/旧层继续 | R/E/N：真成熟 / 不固定三腿永载 | 0 | 未访问 |
| P12→P13 | RL 真 P；允许恢复新 owner，首拍不清动作 | E/S/N：保历史及后观察 / 不以标签或 milestone 截尾成功 | 0 | 未访问、无任务成功 |

## 已保存账本与边界

这是 **a8b148463115 的保存快照，不是实时总计**。仅完成 Blocks1–3：source **138496/1047/20940 → 140032/1059/21180**，新增 **1536决策/12共享PPO更新/240优化器步/12252实际ticks**。各阶段 ticks 来自三份完成 coverage 表，不能用8×决策补齐短终止。35次交接均记为非终止；阶段推进可能采用合法 advanced 接管，不能一概写成原所有目标同时满足。

Block3 原请求1152、实耗128、余1024；5个完成回合全部首未完成 P09：

| 回合 | 决策 | 实际ticks | 终止 |
|---|---:|---:|---|
| ep0 | 14 | 105 | FALL |
| ep1 | 14 | 109 | BODY_COLLISION |
| ep2 | 39 | 310 | BODY_COLLISION |
| ep3 | 50 | 395 | FALL |
| ep4 | 11 | 85 | BODY_COLLISION |

共1004ticks，较128×8短20。Block3没有已优化非终止尾；终止后第6次前缀成功不产生第6个已信用回合。累计10个完成回合为7 FALL/3 BODY_COLLISION；Blocks1–2另留278+89=367个已优化非终止尾决策，不算成功。

前缀累计 **5778决策/46224ticks、9次接受尝试**全部排除PPO。Block3单块为4434/35472、6次接受。事件按尝试计数，不是成功腿数：当前PPO FR Q/C/P各4；FL Q5、撤销2、C/P各3；RR I13、Q7、撤销2；RL I12、Q2、撤销2；**RR/RL C/P均未记录**。继承前缀 FR/FL Q/C/P各8另列，不算当前PPO新成就，也不证明当前支持。I=初始离地过程，Q=有效主动抬升资格，二者不等同。

保存审计记12252 native验证ticks；native、四类状态写、交接bootstrap和prefix入库异常均0。这是已有审计汇总，不是本报告重新执行物理或逐tick复算。前缀compact缺完整native effect/四write字段，保留UNAVAILABLE，不将聚合中的0误读为实测零。

CP140032记录round-trip=true。Block4 `20260909T0532461768313Z_ga8b148463115_50c0b9ea678a413c87c11ebbefe70f6e` 的 P06/1024 已启动普通resume、保留Adam，但本快照不计其数据。首4096分配：P01 1024+P06 384+P07 128已完成，P06 1024进行中，P10 1024/P05 512计划中；尚余2560计划量不是实际信用。

E=物理评价/接触CPU，N=nominal所有权/权限/交接CPU，R=角色成熟度CPU，S=监督器/终止CPU。376、166、102等回执有重叠，不相加，也不把synthetic当physical。B2另属2f942的900决策/7200ticks非PPO诊断，不与a8声称完全同版配对。冻结A原结果/原因未改，不设A5/5门；本快照仍**无正式自然P01评估、best-success checkpoint或成功视频，无稳定性优越结论**。

来源：[累计摘要](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/metrics/cumulative_after_block3.json)、[Block3完成摘要](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/block3_summary/training_block_summary.json)、[审计表](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/audit_tables.json)。原配置/迁移/代码清单与A边界见[保留的Block1快照](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_all_stage_acceptance_v1/reports/all_stage_progress_snapshot_block1.md)。本次只新增本报告，未修改元数据、CSV或生产，未读活跃轨迹、加载PT或重哈希。
