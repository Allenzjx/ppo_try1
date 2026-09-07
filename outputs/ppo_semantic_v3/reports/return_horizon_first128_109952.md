# #33 return-horizon 首个真实 128 决策窗口

固定范围：**global 109825–109952**；仅读取该 run 前 128 条 decision audit、首条 optimizer update、首保存 sidecar 及启动参数。未读取后续 tail/后续 checkpoint，未重复 initial 权重迁移审计，未执行 Python/PT/pytest/Isaac、额外 probe 或文件哈希。以下不计任何 CPU 测试样本。

Run：`20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8`。实际运行 HEAD 为 `f4bfe2560bfd228541f7829d830fa441054eab1d`。

## 1. 有界实际结果

- source 为 **109824 decisions / 823 PPO updates / 16460 optimizer steps**；该窗口实际增加 **128 / 1 / 20**，首保存为 **109952 / 824 / 16480**。
- source phase 计数：**P01=1，P02=127**。未访问 P03–P13，不能为其填入“质量为零”或推断后腿改善。
- 共 **1024 个物理 tick**，每条完整执行 8 ticks，最后 episode tick=1024、sim time=8.533333333333333 s；无短 terminal interval、无终止回合。
- 启动参数为 v3 / P01 / offset0 / N1 / seed1001 / full_episode / NewMdpWarmStart；首条信用从 episode tick8、decision_count1 开始，至 tick1024、decision_count128，无 reset。该自然 P01 信用窗口不含 teacher prefix；没有借教师动作或 CPU 样本填充这 128 条。

唯一阶段切换发生在 g109825、tick8：P01→P02。该条 terminal=false、time_outs=false、terminal_bootstrap_allowed=true。下一条 g109826 的 handoff 记录将原 12 维非零 residual 逐值携带，handoff_hold=true、hard_safety_modified=false。整个窗口 **128/128 nonterminal、128/128 bootstrap_allowed、0 time_outs、physical evaluator valid=128/128**。

FR 硬 qualified tick=48，首次包含该记录的是 g109830；末条仍无任何腿的 cross/placed，RR/RL 无硬 Q/C/P。这里只说明初始前腿采样，不构成任务成功。

## 2. 实际 PBRS 是新 gamma，一次 policy action 一次折扣

首保存 `runner_config` 明确为 `semantic_return_profile=v3_gamma_09985_lambda_099_v1`、gamma=.9985、lam=.99、num_steps_per_env=128。runtime selected configuration 指向 `configs/ppo_semantic_v3/reward_config.yaml`，其记录 SHA 为 `7874da47f19f1a619bc8d9dbc8d2f01d78bf4a7628cd277881ab19d764db338e`；本次没有重复计算 SHA。

用每条日志的 before/after Phi 独立重算：

`potential_shaping = 5 * (.9985 * Phi_after − Phi_before)`

128 条最大误差 **0**。相邻非terminal记录的 `Phi_before` 与上一条 `Phi_after` 最大差 **0**，包括 P01→P02 边界，没有 phase-local potential 重置。若仅用旧 `.995` 代入同一对 Phi，最大差为 **0.003207781219431438**，所以不是只根据配置启用就声称新分支生效。

| global | Phi before | Phi after | 日志 shaping＝重算 `.9985` | 同状态旧 `.995` 数学反事实 |
|---|---:|---:|---:|---:|
| 109825 | 0.06804292995522962 | 0.06870537065466077 | +0.0027969132172458305 | +0.0015945692307892112 |
| 109826 | 0.06870537065466077 | 0.06832804325876529 | −0.0023990973039181296 | −0.003594838060946501 |
| 109952 | 0.1825682636220928 | 0.18016895346429418 | −0.013347817939975276 | −0.016500774625600456 |

每条 `elapsed_physics_s=8/120=1/15 s`，与实际执行 ticks/120 的差为 0；独立校验 `task_progress = shaping + terminal_event − .02*elapsed_physics_s`，最大差为 0。gamma 仅乘一次 Phi_after，没有每个 120 Hz tick 乘一次，也没有额外以实际 dt 对 gamma 取幂。128 条 dt 累加的浮点值为 8.533333333333323 s，末端真实时钟为 8.533333333333333 s。

五族仍为 task_progress/body_stability/contact_motion_quality/control_smoothness/control_regularization，实际 weighted 与 unweighted 的 1/.4/.2/.1/0 映射最大误差为 0；double reward total 与固定顺序五族和最大误差为 0。顶层进入 PPO 的 reward 是该 total 的 float32：与 PowerShell float32 转换结果逐条相等；其对 double 和的最大差约 4.78759e−9，不是 PBRS 公式偏差。该 128 条顶层 reward 合计 −0.1822701696655713。

**terminal_event 全部为 0，没有 SUCCESS/failure terminal。** 因而本窗口未实测 ±40 事件、terminal Phi=0 或短终止 tick，不能将配置中保留 ±40 当作它们在本窗口已触发的证明。新旧 shaping 表仅是同一日志状态的数值对照，不是新旧策略轨迹的因果比较。

## 3. 原生下发与禁止状态写入

- native command ticks **180–1203** 与 episode ticks **1–1024** 连续对应，偏移始终为 179，未发现缺口。
- **1024/1024 tick verified、actual native policy effect=1024**。own-phase-request-effect=1023，剩余 1 tick 是有记录的阶段 handoff hold；不是遗漏下发。
- 128 条详细末 tick audit 均 `verified=true`、`actual_mapping_matches_dispatch=true`、`setter_dispatch_targets_equal=true`、target dtype float32；previous-ACK tracking reference 独立核验为 128/128。
- root pose、root velocity、force/impulse、gravity 四类 episode 状态写计数在 128 条中全部为 0，且 no-in-episode-state-writes verified 全真。
- raw、old distribution mean/std、old log probability/value/reward 均有限，std 全为正。

这里依靠真实 compact 每 tick 验证及详细末 tick receipt，未独立重放全部四个物理缓冲区/mapper 数学，也不把 canonical double drive vector 当 native float32 readback。

## 4. 官方首更新与实际保存

`optimizer_updates.jsonl` 首条准确绑定 PPO update824 / global109952：optimizer_steps=20、actor_parameters_changed=true、finite_nonzero_gradient_observed=true，记录的 gradient norm 范围 **1.024659016007307–1.414213350458366**；KL mean=0.02052039839327543、clip fraction=.303125、value loss=.026930099260061978、surrogate loss=−.03735619122162461，均有限。

记录的 actor before `db277144…768b70` 与首保存 ancestry 的 source actor 一致；actor after `23fd4ca7…17b70` 与首保存 actor 字段一致。此为已记录链条对照，不是重复 tensor/hash 检验。该更新边界/结束 LR 为 **1e−5**，未逐 minibatch 记录 LR，不能断言全部 20 个 minibatch 始终使用该值。

[首保存 sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000109952_manifest.json) 的 checkpoint 文件实际存在，记录 `save_load_round_trip=true`；last_update 与上述首更新一致，source_run 指向本 run。stage budget 为 full_episode=49536、phase_suffix=50304、smoke=0，origin=10112。后续结果未读、不预填。

本次未加载 rollout `.pt` 或重新运行 GAE：`.99` 的 algorithm 配置由已绑定 runner metadata 确认，真实一次 PPO 更新由 update/save receipt 确认；不声称在此报告中独立逐 tensor 重算了官方 advantages/returns。initial 的完整 actor/std/critic/normalizer/RNG/fresh-Adam 审计由另一份独立工作负责，此处不重复。

数据来源：[固定 run 的 decision audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8/residual_and_projection_audit.jsonl)、[首 update 所在 JSONL](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8/optimizer_updates.jsonl)、[启动记录](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8/run_manifest.started.json)。
