# 捕获—失载—碰撞与 rollout 信用：只读评估

范围：当前 natural-P01 run `20260921T0727365666248Z_gee5a9651591d_6c75964ec21a46e296fcc856b2ee2064` 已写出的前 640 个学习决策／update 1434–1438、首个已完成 episode，以及两段既有正式训练捕获窗口。未改生产、配置、当前 2048 采样或 optimizer；不是训练门禁，也未合成 PPO credit。

## 实际间隔

下表 decision 为对应 run 的 learner 序号；括号内为 global decision。旧 block02 的 nominal P01→P04 prefix 不计 learner。首次 AIR 指策略采样端点的当前接触记录，不等同历史 `placed` 清零。

| 正式训练窗口 | FL 捕获 | 首次失载 | 捕获→首次失载／rollout | 后续碰撞与边界 |
| --- | --- | --- | --- | --- |
| 旧 block01 | 350 (186206), tick 2800；qualified tick 2793 | 358 (186214), tick 2864 | 8 decisions；均 update 1420 | 此窗口无终态；不能由该片段声称捕获→碰撞距离 |
| 旧 block02／P04 prefix 后 | 149 (186517), tick 2696 | 150 (186518), tick 2704 | 1 decision；均 update 1423 | 401 (186769), tick 4710：P09 BODY_COLLISION；相隔 252 decisions，1423→1425 |
| 当前 natural P01 | 381 (188285), tick 3048；qualified tick 3042 | 383 (188287), tick 3064 | 2 decisions；分别为 update 1436 第 **125、127/128** 条 | 624 (188528), tick 4988：P09 BODY_COLLISION；相隔 **243 decisions**，1436→1438（终态为该批第 112 条） |

当前 FL tick 3056 仍 TOP、3064 已 AIR，故第一次失载在 qualified capture 后 0.125–0.1833 s 的观测区间；不是等到 rollout 结束才失载。decision 384、385 仍 AIR，386–388 短暂恢复 TOP，389 再 AIR：这次重复失载距捕获 8 decisions，已进入 update 1437。捕获 qualified tick 3042→碰撞 tick 4988 为 16.2167 s（采样端点 3048→4988 为 16.1667 s；末拍只有 4 physics ticks）。

## 128 步实际切断了什么，保留了什么

当前 update 1436 覆盖 decision 257–384；普通 P05→P06 的 `done=false`。捕获 reward +0.194716、首次失载 −0.139531，失载时 Phi 0.465808→0.438826，**即时支持丢失信号已经进入同批数据**。实际官方归一化 advantage：捕获 +0.075087、下一拍 TOP +3.888141、首次 AIR +2.436062、批尾 AIR +2.491763。P05 的 125 个 raw GAE 全负；whole-rollout normalization 可改变单条符号，因此不能仅凭失载那拍 normalized advantage 为正判定接线错误，也不能宣称 capture 获得了可靠长期正信用。旧 update 1420 同样已经包含捕获后的 27 个 AIR 样本，不支持“只有增大 horizon 才第一次看见失载”。

官方 RSL `PPO.compute_returns` 先计算当时 critic 的 `last_values`，随后使用：

`delta_t = r_t + (1-done_t) * gamma * V_next - V_t`

`A_t = delta_t + (1-done_t) * gamma * lambda * A_next`

当前 gamma=.9985、lambda=.99。update 1436/1437 批尾均非终态，有正常 value bootstrap；递归在 storage 外没有后续实测 TD 残差，**不是把未来 value 当成 0**。decision 624 的真实碰撞后来进入 1438，不能回填已经完成的 1436/1437 GAE；它可经 critic 学习影响未来相似样本。日志没有独立保存 1436 的数值 `last_values`（明确为 null）；不能拿更新后 decision 385 的 old-value 冒充它，也不能由现有证据量化 bootstrap 误差或认定 horizon 是主要原因。

首个真实终态：`time_outs=false`、`terminal_bootstrap_allowed=false`、Phi_after=0，reward −42.311534638（event −40、Phi shaping −2.309948670，另含实际 dt 与小质量成本）。终态截断 GAE；1438 剩余 16 条属于下一 episode，尾部 bootstrap 不会穿过该 done 回传到前一 episode。没有发现明确终态或普通阶段信用接线错误。

## 若之后评估 256／512：最小合法范围（仅建议）

- 不修改本次 run。先完成当前更新／保存边界；从**当时最新兼容 CP**另建可追溯 collection/return-estimator 版本，保留 actor/critic、全部 Adam 状态与有效 LR、Identity normalizer、RNG、累计计数，重新分配空 rollout，不接用旧未完成数据。保持奖励/Phi 的 gamma、物理 MDP、动作概率/控制与安全规则不变；仅 horizon 变化不应冒称新物理 MDP。
- 目前不是一个合法 CLI 开关：`semantic_return_profile.py:26,54–69` 将两个已知 profile 固定为 128；`semantic_training.py:106–177` 的 factory/一致性检查与 `:802–815` 的 loader 严格比较 profile。需显式新 profile／collection contract、runner factory、save/load 与 source/target migration 绑定；相关 CLI/manifest 路径接受该版本。保留历史 128 profile 原义，不全局替换 `128`，不放宽原 task-joint factor，不借用会改变优化器语义的旧 new-MDP warm start。
- 实际采样 loop／预算本已读取 `num_steps_per_env`（training `:1724–1753`）；须测试新 storage 形状、整批及非整批预算、episode 内普通切段不 done、碰撞 done 不 bootstrap、非终态批尾正常 bootstrap、raw Gaussian log-prob 对齐、保存重载与完整状态保持，且历史 128 回归不变。正式模拟只在资源空闲后安排，不能复用跨 policy update 的旧段拼成 on-policy 大 rollout。
- N=1 下 128/256/512 的满批跨度约 8.53/17.07/34.13 s。若仍为 5 epochs × 4 minibatches，则每 minibatch 32/64/128 条；2048 decisions 对应 16/8/4 updates、320/160/80 optimizer steps（样本 epoch 曝光数相同，优化步频不同）。如另改 minibatch 数以维持步数，那是额外明确配置变更，不能悄悄等同“只增长 horizon”。
- **边界对齐反例：**仅把当前已发生轨迹从相同起点按 256 或 512 分箱，capture 381 与 collision 624 仍跨 decision 512 边界。旧 block02 的 149→401 在 512 分箱中可同批，在 256 中不同批。这只是冻结时间轴分箱，不是新 GAE／新训练预测；真实更改更新时机也会改变后续策略和轨迹。长 rollout 可增加一批中的实际后续反馈，但不保证这条捕获吃到碰撞结果，更不保证学会保持支撑或越障。

证据：当前 run 的 `residual_and_projection_audit.jsonl`（有界窗口）、`advantage_audit.jsonl` 前 5 条、`rollouts/update_001436_likelihood.json`、`completed_episodes.jsonl` 首条；既有 `update1420_FL_capture_credit.md`、`block02_P04_512_receipt.json`、`block02_block03_RR_first_failure_readonly.md`，另仅抽取旧 block02 前 165 条中的 capture 邻域。算法依据为本机 `rsl_rl/algorithms/ppo.py:163–210` 与当前生产 `semantic_training.py:237–250,1773–1840`；没有把 frozen evaluation 当作 PPO 数据。
