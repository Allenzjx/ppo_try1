# P09 offset=0 真实课程块：前缀、信用与优化样本

状态：最终只读对账。`run_manifest.json` 和 `training_manifest.json` 均记录 `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`。本报告只用 PowerShell 检查实际日志，不运行 Python/Isaac，不修改生产代码。课程占比调整不是任务成功或优化器准入门。

## 来源与计数边界

- Run：`runs/ppo_semantic_v3/train/20260906T0515507234089Z_g314f5a8d6bf6_fa38b1752ea547a899ae53e6b7435bc6`。
- 源 checkpoint global=16768；N1，P09，teacher offset=0。原计划 `planned_requested_policy_decisions=2048`；停止后 manifest 的 `requested_policy_decisions=actual_policy_decisions=1024`，`unconsumed_requested_policy_decisions=1024`，rounding overrun=0。不能把原 2048 请求写成全部完成。
- `optimizer_updates.jsonl` 最终已完成边界 global=17792、累计 PPO update=104。audit 恰好 1024 行，global 16769–17792 连续，无超过该优化边界的残留行；完成 8 个新 PPO update、160 个新 optimizer step。
- 1024 个已优化决策全部属于 **P09**，物理 tick 合计 **8192**。其他阶段的本 run 策略优化样本均为 0；前缀中经过 P01–P08 不能计作 C 在这些阶段的优化样本。
- 8 次更新均记录 actor 参数改变和有限非零梯度；这是实际更新，不是只采样或单测。两次真实终止和第三回合的未终结尾部均已经进入完成的 PPO 更新，见下节。

## 最终回合分账

| 当前策略回合 | 已优化 global 范围 | 策略决策 | 实际结果 |
|---|---|---:|---|
| 0 | 16769–17218 | 450 | P09 `INCOMPLETE_CONTROLLER_BLOCKED`，tick9688 / 80.733333s，stage age=30s |
| 1 | 17219–17668 | 450 | 同为 P09 30s 阻塞，tick9688 / 80.733333s |
| 2（尾部） | 17669–17792 | 124 | 未终结；最后 tick7080 / 59.0s，P09，termination_reason=null |

两次终止均 `task_success=false`、`full_task_success=false`、`time_outs=false`、`terminal_bootstrap_allowed=false`。第三回合没有实际 task terminal，不人为追加失败或成功。run 正常保存停止不等于任务完成。

第一次终止 RR front −69.327202 mm、clearance −50.403992 mm、AIR 单样本、load=0；FL 当时也无 TOP 接触/载荷。第二次终止 RR front −52.720275 mm、clearance −49.701935 mm，GROUND=true、load=0.467096；FL TOP=true、load=0.527724。二者都未取得 RR qualified lift/cross/placed。

尾部最后 RR front −60.900059 mm、clearance −46.959149 mm，AIR 连续15样本、initial_clearance=true、load=0；仍未 qualified/cross/placed。这是初始离地探索，不能报为越沿或完成落脚。

## 第一次真实接管

`prefix_evidence.jsonl` 的首个 `policy_credit_start.start`：

| 字段 | 实际值 |
|---|---|
| 模式 | teacher_initialized_suffix |
| requested / actual phase | P09 / P09 |
| handoff / credit tick | 6088 |
| 同一任务时钟 | 50.7333333333 s；剩余 149.2666666667 s |
| teacher offset | 0；target_first_observed_tick=6088 |
| from_P01_current_policy | false |
| requested_phase_still_active_at_credit | true |
| dispatch receipt | source_control_tick=6087；command_physics_tick=6267（含底层 180-tick settle 偏移） |

接管时 RR 尚在地面，并不是已抬起/已越沿的精确历史入口：front distance −188.352790 mm，bottom−top −50.517044 mm；GROUND=true、AIR=false、obstacle=false、TOP=false，support=true，load fraction=0.196458；initial clearance、qualified lift、crossed、placed 均 false。

FL 则真实在障碍顶面：front +387.551907 mm，bottom−top −0.284473 mm，TOP geometry/contact/support=true，obstacle pair=true，ground=false，load fraction=0.158861，连续 TOP 样本 2507。

入口实际 nominal full12 为 `[22.8,-13.4,0,45.9,6.9,0,0,0,0.3,0.3,0.3,0.3]`，controller bias 全零；RR 实测 hip/knee 为 `[-1.016944,0.846854]` deg。P09 指新语义 supervisor 的阶段，不应误写为旧 A FSM 已执行过其历史 P09 抬腿轨迹。

## 教师信用排除和连续性

- 共 **3 个实际 accepted prefix**、3 次 `policy_credit_start`，各 761 条 `reset_only_prefix_decision` / 6088 physics ticks，无 fallback；合计 **2283 教师决策 / 18264 ticks**。所有前缀 raw policy action 为零、`policy_credit=false`，native audit 均 verified。
- 第一条 PPO audit 是 global16769、P09、tick6096，恰好在接管之后 8 physics ticks；policy decision_count=1，而 physical_core_decision_count_including_prefix=762。
- 该条及全部 1024 个已优化行明确 `prefix_teacher_data_in_ppo_storage=false`、`task_result_scope=teacher_initialized_suffix`。教师物理时间进入每次同一 200s 时钟，但教师样本不进入 PPO global 决策计数。
- manifest 的 physical core（含前缀）为 3307 决策 / 26456 ticks，精确等于 `2283+1024` / `18264+8192`，不是单回合持续时间。前缀阶段分账：P01=3、P02=621、P03=12、P04=3、P05=705、P06=888、P07=3、P08=48；这些全部为教师信用排除数据。
- 全部 1024 行的 8-tick native 汇总均通过；无在回合写入状态的违规记录。raw action、旧 log probability/value、reward 与实际执行审计都有真实记录。

## 实际探索范围与证据限制

- RR 曾出现初始离地事件：tick6111（+3.081795 mm excursion）、6377（+3.508713 mm）、6431（+3.799364 mm）、6750（+4.465204 mm）；这是初始离地诊断，不等于合格越沿或落脚。
- 首 384 个 decision-end 样本中：AIR=20、GROUND=331、obstacle pair=104；以上接触数仅为该子段决策末采样，不冒充全部物理 tick 的接触统计。最终 1024 行及两次终止的 RR qualified lift/crossed/placed 仍均未出现。
- 第一个决策末 obstacle 样本为 global16801/tick6352：front −50.177481 mm、clearance −46.767371 mm。决策末最高 clearance 为 global16848/tick6728 的 −43.796363 mm，当时已有 obstacle contact；不能解释为自由越沿净空。
- 在 global17152/tick9160（76.333333 s）RR 仍 GROUND，front −48.230019 mm、clearance −50.139702 mm，未 qualified/crossed/placed；FL 此时 TOP 并加载 0.436478。
- 该边界 RR nominal 为 `[-6.9,-37.8]` deg，projected residual `[+0.330956,-2.161246]` deg。最终 1024 个决策末样本全 12 通道均未达到配置 cap；RR hip/knee 最大 cap 使用率分别约 **49.55% / 46.94%**。这是随机策略的实际动作，不说明整个允许动作域都不可达。

## 保存、重载与后续使用边界

- 实际保存节点：16896、17280、17792。最终文件为 `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000017792.pt`；对应 manifest 记录 SHA256 `9fc9826c695e6e3e63d81737e211836064d2527adf2a06bbed4ed5b54a51a68e`，`save_load_round_trip=true`。
- 最终 checkpoint 累计 global=17792 / PPO updates=104 / optimizer steps=2080；源 ancestry 为 16768 / 96 / 1920。新块差值恰为 1024 / 8 / 160。
- 本 run actor 参数摘要从 `30a1327a42338ea837db28edeb24eb3da749838ead64065759595f33a421737a` 变为 `b599ff4c14b6262ca0c8dbe241c34826a75afb96d7b2abe7553497fb9d1afa48`。最后一次真实 update 的 KL=0.0192466、clip fraction=0.21875、entropy=−5.666774、value loss=37.61758；仅为训练诊断，不是性能胜出指标。
- checkpoint 明确 `physical_env_state_saved=false`、`resume_physics=legal_reset_not_bitwise_continuation`，normalizer 为固定版本观测 schema / identity RSL normalizer；不能声称保存了物理回合状态或可原地连续恢复。
- manifest 记录本块 wall time=919.108s，其中 roll-in=836.051s、reset=19.430s。这些时间解释成本，不是新准入门。
- `success_count=0`（fresh P01 current-policy scope），teacher_initialized_task_success_count=0。此次没有 fresh P01 评估，不能更新最新 P01 成功率、宣称 suffix success、全任务完成或稳定性优于 A/B。

证据：该 run 的 `training_manifest.json`、`run_manifest.json`、`optimizer_updates.jsonl`、`completed_episodes.jsonl`、`prefix_evidence.jsonl`、`residual_and_projection_audit.jsonl`，以及最终 checkpoint sidecar。原始 run 和 manifest 未修改；唯一写入为本报告。
