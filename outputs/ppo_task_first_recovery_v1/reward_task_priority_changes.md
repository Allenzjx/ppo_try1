# 任务恢复目标：epsilon = 0，保留成功 N 与真实安全

本次是独立目标隔离，不是已实现严格约束 RL，也不保证冻结均值会自行改变。成功 zero 的控制、阶段任务、动作容量、物理环境、真实失败标准都不变。

## 实际改动

独立配置目录 `configs/ppo_task_first_recovery_v1/` 从成功 zero 使用的 `ppo_fsm_reference_p09_stable_v2` 保留 5 份非 reward 配置语义。reward 标记为 `task_first_recovery_epsilon_zero_v1`，新增 `objective_profile: task_first_recovery_v1`、`quality_epsilon: 0.0`。

| 项目 | 原配置 | 本次任务恢复 |
|---|---:|---:|
| task_progress | 1 | 1 |
| body_stability | 0.4 | 0 |
| contact_motion_quality | 0.2 | 0 |
| control_smoothness | 0.1 | 0 |
| control_regularization | 0 | 0 |
| success / failure 事件 | +40 / -40 | 保留 |
| 全局物理势能系数 | 5 | 保留 |
| 真实物理时间成本 | 0.02/s | 保留 |
| gamma / GAE lambda / rollout | .9985 / .99 / 128 | 保留 |

`semantic_reward.py` 只增加 profile/epsilon/实际 family 权重的一致性校验及输出元数据，并澄清 failure_avoidance_bound 的边界。原计算与所有质量原始诊断继续保留；没有重复修改 carry allowance。epsilon 是现有权重的显式版本声明，不是隐藏第二次乘法或按阶段开关。`semantic_return_profile.py` 不需要改动。

保留公式：`r = terminal_task_event + 5*(gamma*Phi_next - Phi_before) - .02*actual_dt`。普通阶段切换与 rollout 尾不置零 Phi；真实任务/安全终态 Phi_next=0、不 bootstrap。正反轮速、特定姿态、非零 residual、stage 编号和 FSM 相似度均不成为新奖励。

四个关闭的 family 都是本工程的次级质量偏好：姿态、角速度/角加速度、触地冲击/反弹/滑动、执行动作一/二阶差分及可选残差幅值。真实机身碰撞、wheel-only 攀爬、跌倒、数值异常、关节硬限仍由独立物理判定中止并计失败，绝不随 epsilon 关闭。没有提高执行器能力或放宽限速。

## 完整真实轨迹回报核算

详值及每条来源：[actual_trajectory_return_accounting.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/actual_trajectory_return_accounting.json)。只读取已存实际轨迹；不拼接不同回合，不加载新 critic，不生成伪 GAE/returns，不把旧轨迹重新当作 on-policy 数据。

统一 `G = sum(gamma**t * r_t)`，每次实际 policy decision 折扣一次，最后不足 8 个物理 tick 仍为一次 decision，时间成本使用实际执行 dt。

| 单条真实轨迹 | 时长 / decisions | 真实结果 | 原目标 G | epsilon=0 同轨迹重估 G |
|---|---:|---|---:|---:|
| 成功 zero | 73.808333s / 1108 | 四腿越障及受控停车 | [6.509013, 6.510120] | 6.646374 |
| 正式 C174592 | 6.766667s / 102 | P02 安全中止，RR knee 硬限 | [-34.789543, -34.769492] | -34.724137 |
| 随机回合 0 | 16.458333s / 247 | FR 捕获后 P05 FALL | -28.482726 | -28.150331 |
| 随机回合 1 | 40.533333s / 608 | FR 捕获后 P05 local deadline | -17.460526 | -16.839682 |
| 随机回合 2 | 41.233333s / 619 | FR 捕获后 P05 local deadline | -17.226401 | -16.581892 |
| 随机回合 3 | 13.341667s / 201 | FR 捕获后 P05 FALL | -30.368557 | -30.082421 |

视频的最后一条记录由物理 observer 结束，`environment_step_returned=false`，因此没有存下最后 1 tick（zero）或 4 ticks（C）的 reward。表中 task-only 终态贡献由同一条真实终态、前一存储 Phi 和实际 dt 精确重估；旧质量尾积分未知，以各 family 每秒有界的总权重 0.7 给严格区间，没有冒称精确存储回报。zero 上述终态奖励重估为 35.000458；C 为 -40.649415。训练四回合全部有完整存储 reward，不需要补终态。

73.808 秒后的 +40 成功事件折扣只剩 **+7.592287（18.9807%）**；时间成本 -0.720203；势能总和 -0.225710。当前这些真实轨迹中，旧目标和新目标都把完整成功排在失败之前，因此“水平度成本使这条成功 zero 比失败更差”被数据否定，不能继续把它当作已证根因。随机回合质量扣分主要来自 smoothness，完整折扣贡献为约 -0.249 至 -0.557；去掉质量可能改变局部信用竞争，但是否修好闭环要看新 PPO 更新和重载评估。

长期阻滞失败比早期失稳回报高：约 40 秒后的 -40 分别折扣至 -16.082 / -15.819，而 6.767 秒失败仍为 -34.373。较小时间成本没有抵消这种负终奖延后的影响。这两条是“FR 已完成后长期未能完成 FL/P05”的真实回合，不冒充整段完全原地不动。仍需用任务功能排序模型，不能以失败轨迹总奖提高声称恢复。

`failure_avoidance_bound` 在旧配置为 37，在 epsilon=0 为 **5.888889**。它是有界质量/时间未来成本加 potential 的保守上界，只阻止某类靠早结束省成本的取舍；它没有证明所有不同持续时间的失败或成功按用户里程碑排序，也不能消除负终奖折扣造成的延迟偏好。

## 连续势能、前段有效片段与归因限制

完整终态轨迹实际检查：相邻 Phi 连续误差 0，存储 task_progress 重算误差 0，折扣势能望远镜误差 <=2.6e-15。相同初态 Phi0=0.04514193655003739，故所有这些终态轨迹的 shaping 总和都是 `-5*Phi0=-0.22570968275018696`。前送/捕获的瞬时正 shaping 不意味着整段会改变原目标排序，也不是独立的成功奖励。

四个已完成随机回合的 FR 首次放置 tick 为 1254 / 1046 / 1234 / 1048；到首个包含该事件的决策端点，其旧折扣前缀奖励为 +0.386516 / +0.480234 / +0.385122 / +0.457240，新 task-only 为 +0.597615 / +0.662288 / +0.609940 / +0.653591。这些都是非终态前缀，不能当完整回报或直接当 advantage。第五条 373 decisions 的非终止尾保持 bootstrap 语义，不凭空补成功/失败，完整值未知。

各随机回合跨更新采样：回合 0 使用更新后版本范围 1313–1314，回合 1 为 1314–1319，回合 2 为 1319–1324，回合 3 为 1324–1326，第五尾为 1326–1328。不能把全部动作归到最后 checkpoint174592。原值、GAE、ratio 与 critic 的独立核查见本轮 learning-signal 报告，而不是用上述重估替换旧 PPO 张量。

任务历史能辅助 Phi，但并不证明现有 372 维 actor 观测严格 Markov；不能用 shaping 理论宣称已经解决所有隐藏调度/接触历史问题。

## 定向测试与边界

`tests/unit/test_semantic_task_first_reward.py` 及既有 observation/reward、functional-carry、transfer-role suites 共 **160 passed**。覆盖：新旧 N/spec/schema 同语义；epsilon=0 时诊断保留但质量贡献全零；正转/反转/零转本身不奖励；相同真实进展允许倾斜；物理进展回退不能靠变平获益；不同合法姿态可成功；真实安全/任务终态、阶段交接与非终态尾；短暂回退的完整 shaping 一致性；marker 与权重错配拒绝；failure bound 不冒充普遍回报排序证明。这些是接线/数学验证，不是 Isaac 成功。

新目标必须在完整 update 边界迁移、丢弃未完成旧 rollout、保留兼容 actor，使用新采样拟合 critic。旧经验的本文件重估只作诊断，不进入新 PPO buffer。mean-head 恢复、优化器部分重置与实际训练计数由独立迁移记录说明。

方法依据：[Ng 等 potential shaping 原论文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)用于一致 gamma 与 terminal Phi 的核验，其不变性不能挽救不合适的基础目标；[Gymnasium 官方 time-limit 说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)用于保留有限时域任务终止、可观测剩余时间和外部截断的语义区别。未因此改变物理失败标签或普通 rollout bootstrap。

## 新目标首块结果与后续边界

首个真实 epsilon=0 恢复块已完成：CP174592→176640，+2048 decisions/+16 updates/+320 optimizer steps；2048/2048记录均为task-only且full12许可，16374物理ticks均有核验后的实际学习目标差异。P02=613样本、P05=1388、P06–P13=0；三条FR捕获后P05 local-deadline失败，尾138决策为P02非终态。P02原GAE均值+.375838，而标准化后−.019314；关闭质量项没有使所有前段动作自动成为好动作。601个P02正nominal决策中FL最终反向566次、FR接近零440次、RR601次增强；只能说随机采样中旧偏置仍普遍存在，正式均值能力等待CP176640重载评估。详见 [实际训练receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_preserved_mean_176640_actual.json) 与 [新目标信号](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_preserved_mean_176640_signal.json)。本段不授予M1/M3。

后续CP176640正式重载已完成：817ticks/6.808333s，P02 RR knee硬限，FR只有Q23、没有C/P，M1–M3为0/1；817/817实际full12/native派发/学习修正，未隐藏zero或teacher。最终轮速目标[-.224238,+.002450,+.029583,+.797650]，说明首块任务目标隔离未修复坏均值，不能继续只责怪水平reward。按用户授权启动单一mean-head恢复分支；corrected初始化只归零mean前12行及相关Adam动量，sigma/特征层/其他状态保留，所有RNG及save/load已核验。初始化本身0学习信用，新128检查块运行状态见RECOVERY.md；其正式任务能力尚未评估。

## 用户指定八项方法在本轮的短对应

| 主来源 | 本轮采用与限制 |
|---|---|
| [PPO，Schulman 等](https://arxiv.org/abs/1707.06347) | 使用同一存储 observation/raw Gaussian latent 核查 likelihood ratio、clip 和正负 advantage 局部梯度方向；PPO clip 不是 FSM 动作相似度门。新目标只消费新 rollout。 |
| [GAE，Schulman 等](https://arxiv.org/abs/1506.02438) | 原 old value、raw GAE、标准化 advantage 分开报告；不跨真实 reset，普通 phase handoff 不切断 credit，128 decision 尾保留估计。没有用新 critic 冒充原 GAE。 |
| [What Matters In On-Policy RL，Andrychowicz 等](https://arxiv.org/abs/2006.05990) | 将动作容量、采样 sigma/温度、更新幅度和输出初始化分开诊断；若旧均值在任务优先续训后仍失败，才采用独立 mean-head 恢复。论文不保证本任务最优超参数。 |
| [Potential Shaping，Ng 等](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | 核验实际折扣和终态望远镜和；理论不代表任意质量项保持任务最优策略，也不能把一拍正 shaping 当作正确学习方向。 |
| [Jump-Start RL，Uchendu 等](https://proceedings.mlr.press/v202/uchendu23a.html) | 复用现有 N 前缀创造合法后段起点，teacher 部分不计 PPO 信用；正式 M1–M3 仍必须自然 P01、无 teacher。课程是训练手段，不是成功替代。 |
| [Residual Policy Learning，Silver 等](https://arxiv.org/abs/1812.06298) | 保留已成功 N，学习附加修正；全 12 通道和合法差速/反向仍可探索。基础控制器成功不保证任意 residual 保持成功。 |
| [Gymnasium Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) | 200s 是已有有限时域任务终止且剩余时间可见；外部预算/rollout 尾不同于任务失败。不为修回报而改失败标签。 |
| [Isaac Sim 5.1 joint drives](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup_tutorials/joint_tuning.html) | 对应本地 5.1.0.0，区分 target、实际 q/qd、PD 需求估计与独立扭矩证据。RR target 约 -4° 而 actual 约 -60° 不被改写成直接命令硬限；不提高驱动能力掩盖策略问题。 |
