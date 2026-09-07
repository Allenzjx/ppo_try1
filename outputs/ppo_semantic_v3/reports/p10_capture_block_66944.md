# P10 capture 完整训练块：62848 → 66944（已完成）

运行：`20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc`。生产 HEAD：`68631e932c7deb08a7a3f2a2787b79fa7eb569ef`。配置为 P10 offset 0 / N1 / seed 1001 / phase_suffix / heteroscedastic_log_v1。

`run_manifest.json` 正式 lifecycle 为 **SUCCEEDED**：requested/planned/actual 均 **4096**，32 次 PPO update、640 次 optimizer step，rounding_overrun=0、unconsumed=0，wall time **2805.8859927 秒**。累计从 **62848 / 456 / 9120** 到 **66944 / 488 / 9760**。这是训练执行成功，**不是物理任务成功**；本块没有后缀任务成功或自然 P01 全任务成功。

本报告逐行读取完成后的 4096 条策略 audit、32 条 optimizer update、4 条 completed episode，并与最终 manifest/sidecar 对账。此前的 [前 512 窗口](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_capture_initial_512.md) 与 [前三回合有界报告](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_capture_first_three_episodes.md) 保留不变；本报告不重复 checkpoint 文件哈希计算。

## 1. 完整样本、前缀和短 tick 分账

策略 global **62849–66944** 连续无重号/缺号，共 4096 条，全部属于已完成更新。实际阶段分布如下；教师阶段不能填入这张策略表。

| 阶段 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO 样本 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 706 | 3380 |

| 回合 | global 范围 | 决策 | 物理 ticks | P10/P11/P12/P13 | 最末 tick / 秒 | 最末状态 |
|---|---|---:|---:|---|---|---|
| 0 | 62849–63815 | 967 | 7736 | 1/1/65/900 | 15320 / 127.666667 | P13 incomplete |
| 1 | 63816–64779 | 964 | 7712 | 1/1/62/900 | 15296 / 127.466667 | P13 incomplete |
| 2 | 64780–65232 | 453 | 3617 | 1/1/451/0 | 11201 / 93.341667 | P12 incomplete |
| 3 | 65233–66198 | 966 | 7728 | 1/1/64/900 | 15312 / 127.600000 | P13 incomplete |
| 4 | 66199–66944 | 746 | 5968 | 1/1/64/680 | 13552 / 112.933333 | **非 terminal，P13 尾段** |
| 合计 | 62849–66944 | **4096** | **32761** | **5/5/706/3380** | — | **3350 terminal 回合样本 + 746 非 terminal 尾段** |

四次正式终止均为 `INCOMPLETE_CONTROLLER_BLOCKED`。唯一短决策仍为 episode 2 / global 65232：在 tick 11201 终止，只执行 **1** 个物理 tick；其余 4095 条均为 8 ticks。因此 `4096 × 8 − 7 = 32761`，不是丢失 7 个策略样本，也不是 rollout 尾段漏记。该 P12 终止 stage age 30.008333 秒，physical evaluator termination_reason=null，非碰撞、非 200 秒超时；`time_outs=false`、`terminal_bootstrap_allowed=false`。

五次真实 A-teacher prefix 均 accepted、每次 **948 决策 / 7584 ticks**，均在 P10、tick 7584 / 63.2 秒接管，没有 prefix miss/fallback。教师共 **4740 决策 / 37920 ticks**，`policy_credit=false`，未进入 PPO storage；所有 4096 条策略行的 `prefix_teacher_data_in_ppo_storage=false`。完整物理 core 为 **8836 决策 / 70681 ticks / 5 回合**，不能把这些教师开销加到优化 budget。每回合仍共享包含教师时间的原 200 秒任务时钟，不是接管后重新获得 200 秒。

## 2. RL 真实事件与当前接触

五回合中教师 RR 的 Q6938 / C7109 / P7579 都早于接管 tick 7584；本块不是重新学出这些教师 RR 事件。策略信用段内 RL 的实际事件为：

| 回合 | RL Q/C/P tick | 最末真实 RL | 最末真实 RR | 尚未完成 |
|---|---|---|---|---|
| 0 | 7959/8070/8113 | AIR，load 0 | TOP，load 0.467820 | P13 受控停止 |
| 1 | 7969/8059/8089 | TOP，load 0.028772 | TOP，load 0.502175 | P13 最终条件/受控停止 |
| 2 | Q7961；8054 落地撤销；无 C/P | GROUND，front −252.466 mm、clear −51.362 mm、load 0.548419 | GROUND，front −233.655 mm、clear −48.397 mm、load 0.086997 | RL 越沿/放置，不能称 P13 失败 |
| 3 | 7950/8066/8105 | TOP，front +203.245 mm、clear −0.639 mm、load 0.052129 | TOP，front +122.837 mm、clear −0.077 mm、load 0.519788 | P13 受控停止 |
| 4 尾段 | 7960/8072/8111 | **AIR**，front +198.923 mm、clear −0.162 mm、load 0 | TOP，front +199.911 mm、clear −0.518 mm、load 0.537170 | 尚在 P13；没有终局 |

尾段四腿虽然历史 placed=true，但当时实际承载为 FL TOP/load 0.462830 与 RR TOP/load 0.537170；FR、RL 均 AIR/load 0。历史事件和当前台面上方区域不能冒充当前 TOP 接触/载荷。

episode 3 在 P13 stage age 60 秒结束，region=true、support=true、controlled=false、stable_for_s=0；尾段 P13 age **45.333333 秒**，region/support=true、controlled=false、stable_for_s=0，任务仍余 87.066667 秒。两者最末四轮 nominal 都已经为 0。其实际 filtered canonical 最大轮命令分别 **0.421738 / 0.334312 rad/s**，最大实测轮速 **0.420027 / 0.332451 rad/s**。这说明这些最末状态仍未停稳，不代表整段确定性 mean 轨迹或未来成功概率。

上述逻辑命令来自 `actual_drive_target_full12` 的 canonical double 值，**不是 native 左轴符号变换后的 float32 target readback**；native 证据另由实际 actuator audit 核验。尾段 `terminal=false`，不能为了训练块结束把它改成物理失败；其非终止 bootstrap 语义也不能与前述真正终止混淆。

## 3. 真实更新、raw Gaussian 与 native 证据

- 32 条 update 精确覆盖 update **457–488**，global 每次递增 128；optimizer_steps 每次 20，合计 640。每条 actor-before 等于上一条 actor-after，首条精确等于 source62848 actor，最后一条精确等于 checkpoint66944 actor。32 条均 `actor_parameters_changed=true`、`finite_nonzero_gradient_observed=true`。
- update 记录的 gradient norm 区间 **1.000304–1.414214**；KL 均值区间 **0.015107–0.031896**；clip fraction **0.221875–0.387500**；entropy **−4.569629–−3.954201**；value loss **0.001582–105.540139**。32 个已记录 update 边界/结束 LR 均为 **1e−5**；未逐 minibatch 记录 LR，不作全程常数断言。这些是更新诊断，不是物理能力改进的因果证明。
- 所有 4096 行均含 12 维 raw action、old distribution mean/std。std 全通道范围 **0.087193–0.301982**。以日志 raw/mean/std 独立重算 12 维 Gaussian old log-probability，最大绝对误差 **1.367925e−6**；没有把 filtered/applied 动作代替 raw 样本。该复算核对日志内部数值，不声称重新运行了 optimizer。
- 所有 4096 行 `all_ticks_verified=true`，verified native tick 合计 **32761**，逐回合等于实际 ticks；没有未核验行。四项 in-episode root pose/root velocity/force-or-impulse/gravity 写计数全部为 0，`no_in_episode_state_writes_verified=true`。

## 4. 同 MDP exact-resume 与最终 checkpoint

源与目标 runtime content 相同，仍是 HEAD 68631e932c7d。启动 checkpoint 为 62848；`new_mdp_warm_start=false`、`resume_migration=null`、`policy_distribution_migration=false`，没有新 initial migration，也没有本块重置 Adam 的分支。源 actor/Adam/normalizer 与首次真实 update 的 exact-resume 依据已在前 512 报告核对；本次额外核对首尾 actor 连续性和 source→final normalizer hash 相等。

最终 sidecar 保存 global **66944**、PPO **488**、optimizer **9760**、`save_load_round_trip=true`；source ancestry 明确为 62848/456/9120。`physical_env_state_saved=false`、`resume_physics=legal_reset_not_bitwise_continuation`：网络/优化器状态连续不等于物理场景或旧 rollout storage 连续，当前 run 是真实重新 prefix 后的新样本。

预算 ledger 从 `full_episode=25088, phase_suffix=27648` 到 **`full_episode=25088, phase_suffix=31744`**，本块只增加 suffix 4096；既有 origin10112 不变。两阶段之和 56832 加 origin10112 正好为 66944。

最终文件为 [checkpoint_step_000066944.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000066944.pt) 及 [对应 sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000066944_manifest.json)。sidecar 记录 checkpoint SHA `8d5896c5fb02b55ab09b9fcaf387c7a43c112795928c7a721ad31126cd0002bb`；实际文件与 pointer/manifest 的 hash 匹配由主线程已核验，本报告不重复散列大文件。

完成结论：本块真实获得 4096 个已优化后缀样本、32 次连续参数更新和 32761 个已验证 native ticks。4 回合正式未完成，第 5 回合有已优化但未终结的 P13 尾段；没有把教师前缀、历史放置、训练执行成功或新启动的自然 P01 评估当作全任务成功。
