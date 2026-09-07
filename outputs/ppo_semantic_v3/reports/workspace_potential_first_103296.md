# Workspace reciprocal potential：首 128 个已优化决策审计

状态：固定已保存边界 **103296 / 772 PPO updates / 15440 optimizer steps**。这是仍在运行的 4096 计划的首窗口，不是整块完成或任务成功声明。只读范围严格为 global **103169–103296**，未统计其后的采集或更新。

运行：[20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c)。HEAD `f1a9bbf650b1b8f80d5169e09173fcfc68797d99`，v3、N1、seed 1001、natural P01 / full_episode、offset 0、NewMdpWarmStart，checkpoint cadence 8（首更新另行保存）。

## 1. Source → 独立 initial → 首保存

以下是实际 sidecar 与 receipt 的字段对照；没有加载 PT、重新计算大 checkpoint 哈希或再次执行 tensor 验证。

| 项目 | Source | Initial | 首保存 |
|---|---:|---:|---:|
| global policy decisions | 103168 | 103168 | 103296 |
| PPO updates | 771 | 771 | 772 |
| lifetime optimizer steps | 15420 | 15420 | 15440 |
| full_episode 已消费预算 | 45312 | 45312 | 45440 |
| phase_suffix 已消费预算 | 47744 | 47744 | 47744 |
| v3 原点 | 10112 | 10112 | 10112 |
| 保存/重载 round trip | true | true | true |
| 记录的 LR | 1e−5 | 3e−5 | 1e−5 |

Source 与 initial 的 actor 全参数 SHA `91794a7d…92c4fd`、critic `e3dcabfd…19f042`、normalizer `c230b0db…3f4552` 完全相等。policy contract 仍是 `heteroscedastic_log_v1` / `HeteroscedasticGaussianDistribution`，state-dependent log-std、324 observations、12 raw latent actions；相等的是整个 actor 参数哈希，包含 learned std 网络，不只是 mean 分支。三份 runner_config 的完整 JSON 相等。

Source 与 initial 的完整 `training_rng_state` JSON 相等（Python、NumPy、torch CPU/CUDA、seed/device-count 等已有字段）。首更新后 RNG JSON 不再相等，符合继续采样的实际记录。normalizer 首更新后仍相等；actor/critic 首更新后分别变为 `5ac1562e…83915e` / `4938a6f2…9593cd`。

这是明确的新 MDP warm start，receipt `exact_mdp_resume=false`，**不是保留旧 Adam 的普通 exact resume**。receipt 记录 `reset_all_moments`、Adam initial LR 3e−5；source optimizer hash `fca108e5…54f048` → initial `ebddf13d…716614` → 首更新 `3946da39…45c44e`。当前 loader 先实际恢复并核验 source 的 actor/critic/Adam/normalizer，然后重建 Adam，再恢复 RNG；旧 moments 不用于新 optimizer step。

`old_rollout_buffer_inherited=false`、`physical_state_inherited=false`、`physical_env_state_saved=false`，`resume_physics=legal_reset_not_bitwise_continuation`。loader 的实际前置检查要求 storage actions shape `(128,1,12)`、obs 324、storage.step=0、无 pending transition。这是 receipt 加已执行路径检查的证据，不是本报告另行探查 live storage 内存。Initial 保存不增加任何样本或 optimizer 计数；source 文件保持独立。

## 2. 新 runtime 与 soft-Phi 绑定

Initial、首保存、warm-start receipt 的完整 target runtime contract 相等。source runtime content SHA 为 `37cad9a1279dacd8091fde0ed41ad55270eae4adbae50c15f7613468d5f109be`，target 为 `784d7f4def7889f36fe1392b1cb12df21a0256cb916baad9a901ff1a3759bb8e`。

实际 `runtime_changed_files` 恰为三个生产文件：

- [stage_task_spec.yaml](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/stage_task_spec.yaml)
- [semantic_supervisor.py](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py)
- [semantic_workspace_potential.py](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_workspace_potential.py)

29 个 frozen-A 文件的 source/target 哈希映射完全相等。配置转移中 execution_profile、action/observation schema、reward_config、quality_score 的 source/target SHA 相等；仅 task spec SHA `50c18b6e…92feb` → `d35afdca…2faa`。任务规范实际包含 revision `continuous_whole_body_v3_dense_workspace_potential`、`workspace_potential_semantics=workspace_interval_reciprocal_potential_v1`。势函数的两处 workspace/preparation 调用使用新 helper；公共 predicate、entry/completion 路径不调用它。奖励配置哈希未变不意味着 MDP 未变：Phi 的内容已明确改变。

首条真实观测 g103169 / tick 8 中，两后腿都未 placed、仍在 lateral span 内，前驱未完成，因此只能获得已有 `.1` preparation 份额。按当前日志几何重算：

| 腿 | front distance (m) | 原 clipped workspace | 新 reciprocal workspace | 对全局 Phi 的 preparation 贡献 |
|---|---:|---:|---:|---:|
| RR | −0.8281345929400956 | 0 | 0.2913295910184242 | 0.0061907538091415144 |
| RL | −0.8297610248820817 | 0 | 0.2907784753726050 | 0.0061790426016678560 |

这里使用既有区间 `[−.22,.06]`、尺度 `.25` 和全局系数 `.85/4 × .1`。这是实际状态上的软分量公式响应，**不是后腿准备完成、动作因果改善或新旧物理轨迹对照**。

128 条实际日志的 `5*(.995*potential_after-potential_before)` 与 `potential_shaping` 的 PowerShell double 重算最大差为 0；相邻记录的 Phi 链最大差也为 0。首条 before/after 为 0.06804292995522962 / 0.0687850464100407；窗口末 after 为 0.17851278131478165。总 reward 为 −0.560568100706405，不把势函数单项当成全部 reward。

`new_mdp_initial_action_comparison.json` 只在当前 P01/t0 的同一 324 观测、同一 nominal、fresh zero-history projectors 上比较逻辑动作输出；六类 12 维差值均 0，optimizer_updates=0，实际环境/bridge history 未改。该文件自己明确 scope 不含 native dispatch 或 subsequent trajectory；它**没有比较新旧 Phi/完整 observation 或物理轨迹，不能称 MDP/trajectory equivalence**。

## 3. 首完整 rollout 与真实更新

128 条 global 连续且唯一；phase 按动作 source phase 计 **P01=1、P02=127**。P03–P13 未访问，不给未覆盖阶段填“零质量”。P01→P02 在 episode tick 8 / 0.0666667 s 发生。窗口末为 tick 1024 / 8.5333333 s / P02，128 条全部 nonterminal，没有完成回合、task success 或 timeout。

这是全 P01 策略信用：sampling 与 implemented_reset_sampling 均为 `P01_full_task_only_initial_version`，curriculum prefix_request=null，首动作 decision_count=1，从 P01/t0 开始；没有 prefix evidence 文件或 teacher roll-in。**teacher/prefix policy credit=0，teacher physics ticks=0**。启动参数的 `prefix_source=frozen_fsm` 在自然 P01 路由未被调用，不能据该参数误计教师样本。初始化 settle 不算 PPO 或 teacher roll-in。

首 update 的 before actor hash 精确接 initial/source，after hash 精确接首 sidecar；`actor_parameters_changed=true`、`finite_nonzero_gradient_observed=true`，gradient norm 1.0013587193–1.4142128610，20 optimizer steps，KL mean 0.02623684709，clip fraction 0.3609375，value loss 0.001368555901，surrogate loss −0.03785672002。记录的是该 update **结束 LR 1e−5**，不能推断 20 个 minibatch 全程 LR 常数。

128 条 raw/old mean/old std 都是有限 12 维向量，std 范围 0.1188376471–0.2266453356。独立 double Gaussian log-probability 重算与已记录 float32 RSL old log probability 的最大绝对差约 `8.3984e−7`；不是另行 tensor bitwise 证明。生产循环实际以 `torch.equal` 检查保存的 raw action 和 mean/std 是否等于相应 transition，再计算 returns/update。

## 4. 同物理时钟 native 与状态写入审计

总计 **1024 physics ticks = 1024 compact native tick receipts = 1024 verified ticks**，每条 decision 都是 8 ticks，无 terminal 短 tick 差额。episode ticks 连续 1–1024，native command ticks 连续 180–1203，恒定偏移 +179；不是把 native clock 与 episode clock 当成同一计数。

128 个 decision 的 `all_ticks_verified=true`；native effect 共 1024 ticks。128 份末 tick 详细 audit 的 `verified`、`setter_dispatch_targets_equal`、`actual_mapping_matches_dispatch`、`same_tick_counterfactual` 及 `tracking_reference_previous_ack_independently_verified` 都为 true。headroom mode 全为 `same_tick_post_mapper_servo_margin_v1`，此窗口没有 headroom clipping、没有后腿 nominal geometry projection evidence。原生记录声明 float32 targets；报告没有把 canonical double command 当成原生 buffer readback。

四类 in-episode root pose / root velocity / force-or-impulse / gravity 写入计数在全部 128 行均为 0，`no_in_episode_state_writes_verified` 全 true。本次复核使用运行自身逐 tick fail-closed receipts 和末 tick 完整重建标志，没有另做 1024 次全 buffer 数值重建。

窗口末 FR 初始 lift tick 21、hard qualified tick 52；FR 当时 AIR，clearance +0.07812964817 m、front −0.08699246900 m、load 0，尚无 cross/place。FL/RR/RL 没有 hard Q/C/P；RL 初始 clearance tick 6 不等于 hard qualification。其他阶段和后续 episode 不在本报告范围，也不据此判断 workspace 修订是否提高实际成功率。

## 5. 固定证据文件与停止边界

- [Source sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000103168_manifest.json)
- [Initial sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000103168_s005d8794fc4f_gf1a9bbf650b1_784d7f4def7889f36fe1392b1cb12df21a0256cb916baad9a901ff1a3759bb8e_manifest.json)
- [首保存 sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000103296_manifest.json)

Initial checkpoint SHA（sidecar 记录）为 `797fb4718803b39ac8b1a7ea785033dcfff155f8de0c795169d58750093a7bf9`；首保存为 `3e37fca3b32e14b3e0e1b93f8a16c6ece45efefac95bfa0239fd5f514b2b5626`。本报告未重新哈希大文件。相对本轮计划只计实际 128/4096，余下 3968 在该固定边界尚未消费；不预填后续 checkpoint。只写本报告，未修改生产/config/tests，未运行 Python/PT/GPU/Isaac；完成后停止读取与写入。
