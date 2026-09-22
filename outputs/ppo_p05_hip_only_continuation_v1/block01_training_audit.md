# Block 01：真实 PPO 封存审计

只读检查 16 个已封存 rollout（001526–001541）、2048 条原始动作记录、16 次 optimizer 更新及最新 checkpoint；CPU 计算，无生产修改、无 GPU 或 Isaac 操作。

新增 **2048 policy decisions／16 PPO updates／320 optimizer steps**。CP201728 为 **201728／1541／30820**，实际 LR `1e-5`；文件 SHA256 `a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6`，哈希及已记录的真实保存重载通过。

| 阶段 | 有效 PPO 样本 |
| --- | ---: |
| P01 | 6 |
| P02 | 734 |
| P03 | 8 |
| P04 | 2 |
| P05 | 604 |
| P06 | 242 |
| P07 | 1 |
| P08 | 1 |
| P09 | 450 |
| P10–P13 | 各 0 |
| 合计 | 2048 |

对应 **16384 个有 PPO 信用的真实物理 tick**，全部有 native 验证；不把 reset／episode 前稳定步进算作学习样本。16 个 rollout 均使用新版本 `128×1×389` 观测和 `128×1×12` 原始 Gaussian 动作，没有旧 rollout 或 teacher-prefix 信用。2048 条 sample、old mean/std/logp 与 collector、policy_request、applied 原始动作逐条相符；CPU Gaussian logp 重算最大差 `5.7221e-6`。此前对 001526／001530 的独立 minibatch 核验均为 20 个 minibatch、每条原始样本使用 5 次。

辅助可见且激活的 policy 输入 **564** 条；实际决策末拍由辅助拥有 FL hip/knee **565** 条（P05 312、P06 242、P07 1、P08 1、P09 9）。两者采样时刻不同，首次介入可在决策内部发生；RELEASED 440 条虽然仍可观测，但不拥有执行器。所有辅助介入仅作用 canonical 通道 0／1，其他 10 个候选输出不被补全器改变；原始样本没有换成最终角度。

Pending 可见 **353** 条，但 `scheduler_advanced_pending` 始终为 **0**：本训练块不能作为“未接触先跨阶段”的物理实测证据。

两条 episode 已结束，但都不是任务成功：第一条 236 决策、15.7333 秒，P02 未完成；第二条 1095 决策、73 秒，已真实放置 FR／FL，到 P09 未完成。第三条只采集 717 决策、47.8 秒，仍在 P05、FR 已放置、FL 未放置；训练块在完整 update 边界结束，没有将其写成任务终止或成功。

16 次更新全部有限且有非零梯度，actor 连续更新链完整；actor／critic 各 6 个参数张量与迁移起点不同，12 个 Adam 状态全部前进 320 步。**新增 AUX 为 0**，继承的 7 accepted／8 attempted 辅助学习账本保持不变。旧迁移文件中陈旧的 P12 sampling 字段已被正常 PPO 保存更新为自然 P01、无 suffix。

训练块 `SUCCEEDED` 只表示这块更新完成，不表示完整越障。真实 FL 捕获依赖已声明的补全器，不能称纯 policy 已经学会。

完整机器可读计数见同目录 `block01_training_audit.json`；运行版本为 `0001c3138b0b38a7278edc288b8ddc7444815770`，训练目录为 `runs/ppo_p05_hip_only_continuation_v1/train/20260922T0415552437349Z_g0001c3138b0b_7b34b44b4b9d49ddb087585dec7acef6`。
