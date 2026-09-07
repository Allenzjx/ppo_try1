# Continuous-stop 新 MDP：初始迁移与首个完整更新（有界 67072）

运行仍在进行：`20260906T1611147430138Z_g68cd9f5fca6c_846e7b65ab224b86a53905c55fa11f85`。HEAD `68cd9f5fca6c9ba06c38666fb55024b9ba04f9b0`；P06 offset 0 / N1 / seed 1001 / NewMdpWarmStart，请求 4096、保存 cadence 4。

本报告仅读取已落盘的 initial checkpoint、前缀、首条 optimizer update 和 **global 66945–67072 的 128 条 audit**，截止实际 checkpoint **67072 / PPO 489 / optimizer 9780**。不计仍在采集的尾段，不预记计划终点，不把当前训练说成完成或任务成功。

## 1. 显式新 MDP，不是静默 exact-resume

源 checkpoint 为 **66944 / 488 / 9760**，来自 HEAD 68631e932c7d。实际 `new_mdp_warm_start.json` 仅声明两项 runtime 变更：

- `configs/ppo_semantic_v3/stage_task_spec.yaml`
- `src/wlr50_clean/ppo/semantic_supervisor.py`

新模式为 `final.stop_progress_semantics=per_wheel_four_type_threshold_ratio_v1`。action schema、execution profile、observation schema、reward configuration、quality score 的源/目标 hash 相同；进度语义仍明确作为新 MDP 处理，而非因为 reward YAML 未变就声称奖励未变。旧 rollout 和物理状态均标记不继承。

独立初始文件为 [initial sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000066944_s8d5896c5fb02_g68cd9f5fca6c_b7d80703ee87e23902156fa0b2a5a9016ba1b66d21fc9d04444508497079f494_manifest.json)，`stage=initial_v3_warm_start`、`save_load_round_trip=true`。与已核实源 sidecar 实际字段逐项比较：

| 项目 | initial 与 source 的结果 |
|---|---|
| actor 参数 hash | 完全相等；包含已学习的 state-dependent std 参数，不是重置 std |
| critic 参数 hash | 完全相等；后续才可随新奖励在线更新 |
| normalizer state hash | 完全相等；固定 324 维 schema + identity RSL normalizer |
| training RNG state | 完整序列化内容逐项相等 |
| policy contract | 完全相等：heteroscedastic_log_v1 / 324 observations / 12 raw actions / log std |
| 累计计数 | 仍为 66944 / 488 / 9760，initial 不增加优化信用 |
| 阶段账本 / origin | full_episode=25088、phase_suffix=31744、smoke=0、origin=10112，均保留 |

此处依据真实保存/重载产物的参数摘要与 RNG 内容；没有在运行中的 Isaac 旁重新启动 Python 载入 tensor，也没有重复散列主线程已经核实的源 checkpoint 大文件。

Adam 是本次明确重置的状态：source optimizer state hash `ac0a9ed0…` → initial `ebddf13d…`，initial 记录 LR **3e−5**。已执行路径 `_load_v3_warm_start` 先验证源 actor/critic/Adam/normalizer，再构造新的 Adam、恢复训练 RNG；它在载入前要求新的 storage 为 `(128,1,12)`、324 维观测、step=0、未挂起 transition action。实际 initial 的 roundtrip 和后续首更新说明这条受校验路径走通。旧 Adam moments 不用于本次新 MDP 的更新；累计历史 optimizer 计数保持 9760，并不表示新 Adam 已有 9760 个内部 step。

## 2. 真实 A roll-in → P06，后腿尚在地面

首次 prefix accepted：**448 教师决策 / 3584 ticks**，逐行 `policy_credit=false`，raw policy action 和 projected residual 均为零，3584/3584 native ticks verified，`no_in_episode_state_writes_verified=true`。这些决策不进入 PPO storage 或 global budget。

实际接管为 P06、episode tick **3584 / 29.866667 秒**，剩余总任务时间 **170.133333 秒**；`requested_phase_still_active_at_credit=true`、`from_P01_current_policy=false`。接管保留的历史只含教师 FR/FL：FR Q71/C1665/P1695，FL Q2461/C3115/P3583；**RR/RL 均无 Q/C/P**。

| 接管时腿 | 当前接触 | front 距离 | clearance | 实际 load fraction | initial lift evidence |
|---|---|---:|---:|---:|---|
| RR | GROUND，非 AIR/TOP | −494.884 mm | −50.950 mm | 0.228028 | false |
| RL | GROUND，非 AIR/TOP | −510.247 mm | −49.513 mm | 0.298524 | false |
| FL | TOP | +85.492 mm | +0.605 mm | 0.224897 | true |
| FR | TOP | +181.732 mm | −1.061 mm | 0.248551 | true |

因此本次确实让策略承担 P06 前驱准备，不是只从已经腾空的 RR 历史入口开始。handoff 原字段为 source_control_tick=3583、handoff_tick=3584、command_physics_tick=3763；它们保留各自控制/episode/dispatch 时钟含义，不能混为同一个计数。

初始同状态逻辑对照记录 tick3584、P06、324 维观测，旧/新 profile 的 scaled/masked/bounded/rate/safe residual 及 applied action 六组 12 维差值均为 0，`optimizer_updates=0`、`actual_environment_or_bridge_history_modified=false`。这只是同一当前观测/nominal 的逻辑投影比较，**不是 native 后续轨迹或新旧 MDP 等价的证明**。

## 3. 首个完整更新：实际新增 128 / 1 / 20

首个策略决策 global66945、episode tick3592，至 global67072/tick4608；**128 条全部为 P06**，1024/1024 native ticks verified，未终结、无任务成功，所有教师 storage 标记 false、四项 in-episode 状态写计数为 0。

首条 update 精确为 PPO489 / global67072 / 本次20 optimizer steps，已保存 [checkpoint67072 sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000067072_manifest.json)，`save_load_round_trip=true`。actor-before 与 source/initial actor 完全相等，actor-after 与该新 checkpoint 相等；`actor_parameters_changed=true`、`finite_nonzero_gradient_observed=true`，normalizer 仍等于 initial。

记录的梯度范围 1.000774–1.152967，KL=0.0220276、clip fraction=0.315625、entropy=−5.233888、value loss=0.00123149。**初始 LR 为 3e−5，首个已记录 update 结束 LR 为 1e−5**；未逐 minibatch 记录 LR，不能把结束值当作整个更新期间的常数。

128 行真实 old distribution mean/std 均为 12 维；std 范围 **0.095112–0.261296**。用 raw action/mean/std 独立重算 Gaussian old log-probability，最大绝对误差 **1.095551e−6**；未用 filtered/applied 动作冒充 PPO raw 样本。这支持新 MDP 下已学习 heteroscedastic 策略接口正常工作，不说明 std 导致了任何物理改进。

固定窗口末，RR 仍 GROUND/front −273.027 mm/clear −50.061 mm/load 0.101818；RL 仍 GROUND/front −362.781 mm/clear −50.450 mm/load 0.451562，阶段还是 P06。首窗口没有进入 P13，**尚未检验新的 continuous-stop reward 主动分支或最终停止能力**。当前证据只证明迁移、真实前驱起点、信用排除和首个优化更新，后续结果保持未定。
