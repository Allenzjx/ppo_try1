# P01 完整课程：已优化样本与真实事件

状态：最终只读对账。run/training manifest 均记录 `SUCCEEDED`，指本轮训练预算实际完成，**不是任务成功**。本轮4096决策实际优化完成：3个P09阻塞回合，另有949决策未终结尾部；第四回合曾真实合格抬升RR，但未越沿，随后落回地面、资格撤销。原始 run、生产代码和 checkpoint 均未修改。

## 来源与优化边界

- Run：`runs/ppo_semantic_v3/train/20260906T0545269489491Z_gcae1d6e7cfdd_1813c27a4d0e408aab8fde1f0c4010c6`。
- 源 HEAD：`cae1d6e7cfdd8383fec67e8f919cc823f9e65448`；v3，N1，seed1001，from_phase=P01，stage=full_episode，原请求4096。
- 从 checkpoint17792 显式 new-MDP warm start。记录的变更文件仅 `stage_task_spec.yaml` 与 `semantic_supervisor.py`；network 保留 actor（包括 std）/critic 参数，Adam moments 重置，初始学习率3e−5，324维观测/12维raw action，旧 rollout buffer/物理状态不继承。
- 最终 `planned_requested_policy_decisions=requested_policy_decisions=actual_policy_decisions=4096`，unconsumed=0，rounding overrun=0。
- `optimizer_updates.jsonl` 最终已完成边界为 **global21888**（累计 update136）。audit 实际恰好4096行，global17793–21888连续无缺号或越界，**32次新PPO update /640次新optimizer step**；32次更新均记录 actor 参数变化和有限非零梯度。

| 已优化阶段 | 策略决策 |
|---|---:|
| P01 | 4 |
| P02 | 702 |
| P03 | 16 |
| P04 | 4 |
| P05 | 581 |
| P06 | 1070 |
| P07 | 4 |
| P08 | 4 |
| P09 | 1711 |
| P10–P13 | 0 |

phase分布由实际audit重算，与manifest telemetry逐项相符；共32768个策略physics ticks。P07/P08各仅4个样本是实际一决策阶段推进，不假装在这些标签驻留过更长训练窗口。

## 回合与未终结尾部

| 回合 | 已优化global范围 | 决策 | 物理结果 |
|---|---|---:|---|
| 0 | 17793–18819 | 1027 | P09阻塞，68.466667s /tick8216 |
| 1 | 18820–19864 | 1045 | P09阻塞，69.666667s /tick8360 |
| 2 | 19865–20939 | 1075 | P09阻塞，71.666667s /tick8600 |
| 3（尾部） | 20940–21888 | 949 | 未终结；最后P09，63.266667s /tick7592 |

三个terminal均 `INCOMPLETE_CONTROLLER_BLOCKED`、P09 stage age约30s，task_success/full_task_success=false，time_outs=false，terminal_bootstrap_allowed=false。第四回合尾部不是实际task terminal，不额外计失败/成功或伪造停止事件。`1027+1045+1075+949=4096`。

第二回合terminal RR front=−88.938501mm、clearance=−49.544110mm，AIR、无GROUND/obstacle/TOP、load0；第三回合terminal RR front=−66.126552mm、clearance=−50.037718mm，GROUND=true、无obstacle/TOP，load=0.418993。这三个terminal的history均未取得RR qualified/cross/placed。

## 第四回合真实资格及撤销：没有越沿成功

- 第四回合首次RR initial_clearance事件tick4762 /39.683333s，upward excursion=3.617238mm。
- **tick5032 /41.933333s**（global21568）记录 `qualified_measured_upward_lift`：upward excursion=46.204817mm，joint motion=16.105357deg，实际clearance=+0.798599mm，front=−32.270369mm，AIR且无ground/obstacle接触。
- 该回合决策末最大clearance=**+9.100513mm**（tick5040/global21569），front仍−31.929102mm，AIR、无ground/obstacle。最靠近前沿的决策末样本tick5320/global21604，front=−23.165699mm、clearance=+3.860721mm，仍未越沿。
- global21568–21619共**52个决策末样本**的RR active_lift=true；这是真实合格抬升历史，不是仅AIR判定。
- **tick5446 /45.383333s**记录 `qualification_revoked_ground_before_cross`。下一决策末tick5448/global21620，GROUND=true、load=0.335233，clearance=−50.980097mm、front=−51.283952mm，active_lift=false。随后有初始离地重试，但尾部最后仍未重新取得有效qualified/cross/placed。
- 最后一行RR front=−58.714386mm、clearance=−43.314113mm，AIR连续50样本、initial_clearance=true、load0；task仍未完成。
- 全run：RR active_lift=true决策末样本52，crossed/placed样本均0。**有真实合格抬升进展，但仍未完成carry越沿与落脚，不能报告suffix或whole-task success。**这些是随机训练过程中采样的轨迹，不保证最终checkpoint确定性复现。

## 首个自然 P01 回合的实际 RR 链

- terminal global18819，decision_count=1027，tick8216 /68.466667s：P09 `INCOMPLETE_CONTROLLER_BLOCKED`（stage age=30s），不是全任务成功。
- 从 P01 自然推进到 P09：P06→P07 在tick4600，P07→P08 tick4608，P08→P09 tick4616。阶段推进是实际 supervisor 记录，不代表 RR 已完成真正抬升。
- RR 首次 `whole_body_initial_clearance` 在 P06 tick4031 /33.591667s，upward excursion=3.286164mm。后续有多次初始离地/重试诊断，但无 RR qualified lift、crossed 或 placed 事件。
- 第一个 P09 decision-end（tick4624）RR front=−212.903615mm，clearance=−50.134181mm；AIR，但 initial/qualified/crossed/placed 均 false。
- 该回合 **决策末采样** 的 RR 最大 clearance 为tick5040 /42s 的 **−20.460999mm**，front=−64.061787mm，当时 AIR、无GROUND/obstacle接触、initial=true。未达到顶面净空，不能报为成功抬升或越沿。这里不把15Hz采样峰值冒充120Hz连续峰值。
- 最接近前沿的决策末样本在tick6472：front=−41.698661mm、clearance=−26.698223mm，已有obstacle pair；该回合决策末没有真正越过前沿。
- terminal时 RR front=−55.349466mm、clearance=−39.250574mm，AIR、无GROUND/obstacle/TOP接触、load=0；history 只有 FR/FL 的 qualified/cross/place，RR 均未取得。
- 结论限于真实探索：已出现初始离地和更大的离地运动，**本回合首个未完成任务仍是 RR 合格抬升→越沿→放置**。不据此新增历史姿态门或认定算法/物理接口错误，也不与不同seed/确定性评估混作性能比较。

## 自然 P01 信用，而非教师前缀

- 四次实际策略起点 global17793、18820、19865、20940 都是 P01，decision_count=1、physics_tick=8、sim_time=0.066667s；不是先进行50秒教师 roll-in 后才开始计数。三个完成回合的记录seed均1001。
- 本 run 参数明确 from_phase=P01 / teacher_offset=0。当前执行入口对 P01 使用 `SemanticIsaacBackend` + `SemanticRslAdapter`，只有非P01分支才构造 `PrefixSemanticIsaacBackend` / `PrefixRslAdapter`；本 run 没有 `prefix_evidence.jsonl`。
- 普通 P01 行没有 suffix 专用的 `prefix_teacher_data_in_ppo_storage`、`curriculum_start` 字段；这里不把缺失字段伪写成显式 false 证明。无教师前缀结论来自实际启动参数、明确的执行分支和从第8物理tick开始的策略信用。
- 最终4096行 native 8-tick 汇总全部 verified，无在回合状态写入违规。raw policy action、old log probability/value、reward 与 applied native 审计一并记录。manifest core计数4096/32768与PPO计数完全一致，无额外教师决策或prefix物理时间。
- 4096个决策末样本全12通道没有达到配置cap；RR hip/knee最大使用率约50.72%/60.28%。这只是实际采样动作的范围，不等于把整个允许动作域都探索完。

## 最终checkpoint与后续使用边界

- 最终文件：`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000021888.pt`。sidecar记录 SHA256 `05bf403d742426582e8c3cada15ee215b8e0c9f4be5b1abe65eb2c8cf91873d8`，`save_load_round_trip=true`。
- 最终累计 global21888 /PPO updates136 /optimizer steps2720；ancestry源17792 /104 /2080，差值精确4096 /32 /640。
- actor参数摘要从 `b599ff4c14b6262ca0c8dbe241c34826a75afb96d7b2abe7553497fb9d1afa48` 变为 `238d76693f7f6533e84b3d862288dacbf30f88b27e52647ce96a90ffc1cb69df`。最终update的KL=0.0214036、clip fraction=0.26875、entropy=−5.702108、value loss=0.01415854；仅为训练诊断。
- sidecar明确 `physical_env_state_saved=false`、`resume_physics=legal_reset_not_bitwise_continuation`，normalizer为固定观测schema/identity RSL；不能声称保存了当前物理回合状态或可原地继续。
- 该run已消耗的full_episode阶段账本由512增至4608；phase_suffix保持7168。manifest wall time=1451.495548s，success_count=0，training_success_is_not_task_success=true。
- training manifest与run result的lifecycle、预算、global、update/optimizer计数、actor摘要字段一致。本报告不预填新的确定性P01评估结果，不声称完成任务、胜过A/B或达到交付稳定性要求。

证据：该run的 `training_manifest.json`、`run_manifest.json`、`optimizer_updates.jsonl`、`completed_episodes.jsonl`、`residual_and_projection_audit.jsonl`、`new_mdp_warm_start.json`，以及最终checkpoint sidecar。唯一写入为本报告。
