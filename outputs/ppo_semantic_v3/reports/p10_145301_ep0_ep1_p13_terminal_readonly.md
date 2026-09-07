# P10 145301：ep0/ep1 已完成 P13 末段审计

范围固定为 `runs/ppo_semantic_v3/train/20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc` 的前两个完成回合：ep0 global 62849–63815（967 credited decisions），ep1 63816–64779（964）。只读取 `completed_episodes.jsonl` 前两条及 `residual_and_projection_audit.jsonl` 前 1931 条；已存在的完整 update 472 / global 64896 覆盖两回合末端。不使用其后活跃回合或计划预算。

两回合都从真实 teacher-initialized P10 suffix 开始计入 PPO，物理起始 tick 7584 / 63.2 s；并非当前策略从 P01 开始的固定 mean 评估。teacher prefix 不在上述 credited decisions 内。两者最终 task/full-task success 均 false，终止原因均 `INCOMPLETE_CONTROLLER_BLOCKED`，P13 age 恰为 60 s。evaluator valid=true、physical failure=null，不是碰撞/物理失效，也不是完整或后缀任务成功。

## 1. 最终一刻缺项

| 末端指标 | ep0：g63815 / tick15320 / 127.6667 s | ep1：g64779 / tick15296 / 127.4667 s | 原硬标准 |
| --- | --- | --- | --- |
| final region | true | **false** | 必须 true |
| current support available | true（FL、RR） | true（FL、RL、RR） | 至少 2 个当前 TOP 支撑 |
| final controlled | false | false | 以下四类全部满足 |
| body linear norm，m/s | **0.0992424444** | 0.00831352438 | ≤0.05 |
| body angular norm，rad/s | 0.176296459 | 0.127552469 | ≤0.30 |
| max measured wheel，rad/s | **0.359396517** | **0.421290576** | ≤0.25 |
| max applied command，rad/s | **0.330504122** | **0.461020879** | ≤0.02 |
| continuous eligible settle，s | **0** | **0** | ≥0.5 |
| nominal wheel full4 | [0,0,0,0] | [0,0,0,0] | 不是硬成功判据 |

ep1 的 region 失败不是后腿退回前沿：FL/FR/RL/RR front 分别 +0.794507/+0.619070/+0.321028/+0.246887 m，四腿均在平台 XY 内。明确缺项是 FR `top_geometry=false`：FR AIR、clearance +0.0283143361 m，超过现有 `top_gap_max_m=0.025`；其余三腿 top_geometry=true。没有据此调整阈值，也没有将普通 AIR 一概视为错误。ep0 四腿 top_geometry 均 true，FR/RL 在 AIR 不妨碍其当时 region/support 合格。

## 2. 固定末段窗口，而非只挑终止帧

下表全部是 15 Hz decision **末端快照**；不是完整 120 Hz 控制条件占空比。零稳定时长也来自真实 evaluator，不从末端采样自行伪造 0.5 s 连续结论。

| 回合 / 窗口 | region false | support false | controlled true | wheel >.25 | command >.02 | linear >.05 | angular >.30 | nominal 四轮全零 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ep0，P13 全900条，g62916–63815 | 299 | 0 | 0 | 874 | 900 | 595 | 152 | 628 |
| ep1，P13 全900条，g63880–64779 | 461 | 0 | 0 | 883 | 900 | 629 | 155 | 628 |
| ep0，最后150 decisions（10 s），g63666–63815 | 36 | 0 | 0 | 144 | 150 | 90 | 23 | 150 |
| ep1，最后150 decisions，g64630–64779 | 58 | 0 | 0 | 149 | 150 | 105 | 21 | 150 |
| ep0，最后15 decisions（1 s），g63801–63815 | 0 | 0 | 0 | 15 | 15 | 9 | 2 | 15 |
| ep1，最后15 decisions，g64765–64779 | 4 | 0 | 0 | 15 | 15 | 5 | 2 | 15 |

两回合全部 P13 快照的最大 `final_stable_for_s` 都为 0。全部 P13 的最小 max-command 仍分别为 0.210842618、0.182448795 rad/s，远高于原 0.02。最后一秒最小 max-measured-wheel 分别 0.281272531、0.299630582 rad/s：不能把这两段简化为“实际已经停稳，仅命令日志没归零”。

## 3. 四轮命令、native target、实测速度分辨率

以下顺序固定 FL、FR、RL、RR；单位 rad/s。

| 末端四轮量 | ep0 | ep1 |
| --- | --- | --- |
| ACK 应用 logical command | [+.179704215, −.254245734, −.330504122, −.142333176] | [+.165566253, −.284693196, −.461020879, −.211314864] |
| 真正 native float32 dispatch target | [−.179704219, −.254245728, +.330504119, −.142333180] | [−.165566251, −.284693211, +.461020887, −.211314857] |

native audit 两个末端均 verified=true。左侧轮 native 符号与 canonical logical 约定不同，不是反向映射错误。零残差反事实 wheel targets 为零，实际命令非零不能归给尚未归零的 nominal wheel 建议；这仍不证明单独去掉残差就会在新的真实轨迹中停稳。

**逐轮 measured velocity 的诚实限制：**当前紧凑 decision/task JSON 只保留 `maximum_wheel_speed_rad_s`，没有四轮 measured 向量。上面的四轮 native 数值是 actuator targets，绝不能标成测得轮速。已保存 rollout `.pt` 含真实 324 observation，但本次只读 PowerShell 审计没有加载/解码张量；且存储的 observation 通常是决策前状态，不能拿它冒充同一终止后快照。因而本报告能证明实际轮速的 max 超阈值，不能确定该 tick 每一轮的 measured 数值或最大轮身份。这也支持停稳提案里单独保存同 tick 已验证的 measured 四元组用于复核，而非新增传感器。

两末端记录的行为分布也不是零 mean 加随机扰动：

- ep0 wheel raw mean `[+.336853, −.344485, −.672489, −.226009]`，sample `[+.308978, −.452245, −.619587, −.241828]`，state-dependent sigma `[.120544,.197130,.203145,.235834]`。
- ep1 wheel raw mean `[+.355006, −.455141, −.783882, −.244272]`，sample `[+.283286, −.515847, −1.016332, −.389621]`，sigma `[.109971,.195024,.248133,.255600]`。

这些是各当时状态/行为策略的已记录 raw 分布，不是冻结同一 actor 的整段固定 mean roll-out，也不是离线 mean 控制结果。命令还经过真实 residual/projector/slew；不得将 `0.6*tanh(mean)` 当成已执行命令。这里不推独立序列成功概率，不建议重置 std/关闭 PPO。

## 4. 当前 capture retention 与历史完成分开

两回合末端四腿历史 placed 都 true；900 条 P13 快照中，按当前生产 `_current_capture_retention` 用真实 `top_xy_outside_distance_m` 和 clearance 重算，每条每腿的 retention 均为 1。这不是退回前沿/ground 的那类历史完成掩盖当前丢失：平台 XY 外距离均未造成衰减，轮底也未低于现有 −15 mm 允许带。

ep0 末端接触/负载：FL TOP .532180，FR AIR 0，RL AIR 0，RR TOP .467820。ep1：FL TOP .469053，FR AIR 0，RL TOP .0287721，RR TOP .502175。历史四个 placement 不等于当前四腿都承载。现有 capture retention 有意允许平台上方 AIR 获满信用；它与硬 final region 的 top-gap 上界是不同量，所以 ep1 FR 在 +28.314 mm 时 retention=1 但 final region=false 并不构成数值矛盾。此审计不提议把所有已 placed 腿 AIR 自动罚掉。

## 5. 对连续 stop progress 提案的适用边界

真实后段仍覆盖原有 measured-wheel/body-linear/body-angular `clip(1-rate/tolerance)` 的平区，尤其两最后一秒 measured-wheel 全部超阈值。例：ep0 g63813→63814，max measured wheel 0.462317735→0.432255208，原 measured-wheel progress 仍 0→0；body/command 等同时变化，不能据此断言总 reward 缺失或因果改善。四轮非最大命令的数学盲区也仍存在，但本次没有逐轮 measured 数据，不伪算新四轮 measured 均值。

因此，`p13_continuous_stop_progress_proposal.md` 的四类连续、逐轮平均方案仍是有真实数据覆盖的局部学习信号候选；它不等价于“只修一个标量就已能成功”。ep1 region 独立不合格、两回合实际速率和命令都经常不合格、行为 mean 本身也非零。该候选不能代替 region/support/0.5 s 硬任务条件，不能证明当前失败主因，不能把本训练变成固定 mean 评估。

只读审计结束。没有 Python/Isaac 调用、生产/config/test 修改或提交，没有实施 reward/entropy/std 改动；当前训练照原版本继续。
