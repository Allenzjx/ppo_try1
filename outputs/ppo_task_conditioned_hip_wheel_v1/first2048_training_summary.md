# 首 2048 决策真实续训摘要

新增 **2048 policy decisions / 16 PPO updates / 320 optimizer steps**，连续覆盖 185857–187904；累计 **187904 / 1433 / 28660**。来源只限五份封存 receipt、CP187904 manifest，以及获准补读的本次联合迁移记录；本摘要未运行仿真、训练或重算网络。

## 实际样本与结果

| 块：学习入口 | 新决策 / 更新 | 不计学习的 nominal 前缀决策 | P09 碰撞次数 | 块末未完成阶段 |
| --- | ---: | ---: | ---: | --- |
| 1: P01 | 512 / 4 | 0 | 0 | P06 |
| 2: P04 | 512 / 4 | 376 | 1 | P05 |
| 3: P06 | 512 / 4 | 670 | 1 | P06 |
| 4: P08 | 128 / 1 | 1938 | 2 | P09 |
| 5: P01 | 384 / 3 | 0 | 0 | P06 |

首、末块从 P01 自然开始，共 896 学习决策；三个 nominal 初始化的后缀课程共 1152 学习决策。另有 **2984 前缀决策 / 23872 physics ticks / 7 次接受入口，学习 credit 全为 0**，不混入样本或成功数。人工方向诊断、视频评估亦不计学习。

四次记录的任务失败均为后缀课程 **P09 BODY_COLLISION**，发生于全局决策 186769、187185、187453、187503。五块末尾均为非 terminal 的预算边界，不是任务成功，也不另计任务失败；第 4 块在已核验更新边界停止。22 次普通阶段变换的 terminal 数为 0。从 P01 完整成功数 **0**，后缀成功数 **0**。

| 实际 request 阶段 | 学习决策数 |
| --- | ---: |
| P01 | 4 |
| P02 | 406 |
| P03 | 8 |
| P04 | 4 |
| P05 | 549 |
| P06 | 830 |
| P07 | 2 |
| P08 | 5 |
| P09 | 240 |
| P10 | 0 |
| P11 | 0 |
| P12 | 0 |
| P13 | 0 |

P10–P13 尚无新学习样本；达到 P09 不表示 RR 已完成越沿和放置。

## 质量成本与训练配置

实际加权 front 成本 **0.0328888957**，已知 geometry 成本 **0.0832318289**，合计 signed body contribution **−0.1161207246**。front 成本出现在 P01/P02；本批已知 geometry 成本只在 P09，P05/P06 为 0。其他三类质量 family 的实际加权贡献均为 0。详细逐阶段积分及真实 physics 样本数见 JSON；缺失终态几何不是“测得零”，P10–P13 无样本亦不是质量通过。这些成本不是稳定性胜过 zero/FSM 的证据。

实际目标为 `task_conditioned_hip_wheel_quality_v1`，记录的 ε=0.06（2048 条）；372 维观测、12 维动作、N=1、120 Hz 物理 / 15 Hz 决策；rollout=128，γ=.9985，λ=.99，5 epochs × 4 minibatches。全部 16 次更新记录有限非零梯度；有效 Adam LR 始终 **1e−5**（runner 初始配置 3e−5 并非实际恢复后 LR）。KL 均值范围 0.0148651–0.0305052，value loss 0.05348–83.72475，不声称已经收敛。

## Checkpoint 与迁移

最新 [checkpoints/history/checkpoint_step_000187904.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000187904.pt)，SHA-256：
`e3a405312514c8c99dd55195a7f8648cff5c44af48e33c4a7e7429a2a2d2eb4c`。
[checkpoints/history/checkpoint_step_000187904_manifest.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000187904_manifest.json) 已记录 save/load round-trip=true；本摘要未重新加载二进制。物理状态未保存，续跑是合法 reset，不是逐位物理延续。

[checkpoint185856_joint_migration.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/checkpoint185856_joint_migration.json)（SHA `92465aaa3eae5c500cdadb09fd05148fcefe4da5b891cc96c206fafcfe8e0f43`）绑定源 CP185856/1417/28340，完整保留 actor/critic、Adam（两组及算法 LR）、Identity normalizer、RNG 和累计计数；迁移本身新增更新/决策均为 0，旧 rollout 丢弃。本次变更是状态 σ + soft reward/global potential 的联合语义版本，不称 same-MDP；nominal、执行链、动作范围、物理场景和硬验收不变。该记录的目标运行 hash 与 CP187904 一致；未在本摘要重复逐 tensor 审计。manifest 内更早的 `new_mdp_warm_start` 等记录只是历史，不拿它们解释本次 Adam 处理。

## 固定 CP187904 的 P01 完整评估

**Pending**：deterministic run `91d8d06aaced49ebb0ad9d5a5c067b31` 已由主任务启动；未读取活动评估尾、未取得结果。det/stochastic 的完整成功、首个未完成任务与视频均等待实际评估，不能由这些训练 receipt 推断。

机器可读汇总、每块结局、输入 SHA、精确配置绑定见 [first2048_training_summary.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/first2048_training_summary.json)。
