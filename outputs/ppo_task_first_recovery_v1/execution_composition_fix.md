# 几何 nominal 历史与 PPO 残差重复叠加：诊断与修复

范围：已结束的 CP176768 正式全通道 C、保留的成功 zero，以及离线计算验证。未重扫 Recording，未更改 N、reward、sigma、物理硬限或成功判据。修复版本：`ad0c1328f1772c440755f3b6a6622c35e464396e`。

## 已确认的执行链问题

旧链路是：上一步 **包含 PPO 的最终目标** → nominal geometry 的保持/向源邻域恢复 → 再加本拍 PPO residual → 最终限速 → 保存包含 PPO 的最终目标。几何 helper 虽未读取本拍 residual，却通过历史间接重新使用了上拍 residual；在保持/恢复分支中，恒定偏置可以被反复积分。这不是 wheel mask 错误，也不是只看单拍 `final − nominal` 得出的结论。

真实 C 的 tick6311（52.5917 s，P09 机身碰撞）提供直接数值链，单位均为 canonical degree：

| RR 通道 | 上拍最终目标 | geometry 输出 | 本拍有效 residual | 本拍最终目标 |
|---|---:|---:|---:|---:|
| hip | 87.55456156 | 86.30456156 | +2.55892814 | 88.80456156 |
| knee | −34.90899409 | −33.65899409 | −1.78637107 | −35.44536516 |

当拍 mapped N 为 `[54.35, 0]`，源邻域为 hip `[44.35, 64.35]`、knee `[−10, 10]`。geometry 已按每步 1.25° 向内恢复，但随后叠加 residual，实际目标仍向外移动。状态为 `bounded_hold_or_inward_recovery_tracking_exceeds_linear_trust`。不能把完整历史偏移全部归因于当拍动作，也不能据此声称这是碰撞的唯一原因。

`c176768_rr_composition_recurrence.json` 保留旧链路的 6 项离线正反例（固定源、固定实际状态，非物理模拟）：零 residual 从 `[20,−20]` 回到 `[10,−10]`；恒定 hip +2.56° 从20°积到30°；恒定 knee −1.786°从−20°积到−24.288°；排除前次偏置的两个诊断代理分别收敛到12.56°、−11.786°；该代理的零 residual 结果与旧零路径相同。**代理的“减前次请求偏置”未投产**，因为它不能完整处理 headroom/slew 后实际生效量。

## 比 P09 更早的状态差异

两个自然 P01 episode 不是相同 P09 入口；小均值不等于无影响。FL 捕获时 C 保持的 nominal hip/knee 为 `[48.2,−36.7]`，成功 zero 为 `[24.9,−13.4]`。这是当前构型的持有语义，不是必须恢复旧角度的要求。

| 对齐事件 | CP176768 C | 保留的成功 zero |
|---|---|---|
| P06 入口 | tick2352；FL 前缘距离 +0.455 mm；当前承载份额0；机身底部至障碍顶面间隙50.141 mm | tick2680；+73.432 mm；承载31.26%；间隙75.122 mm |
| P07 入口 | tick5984；FL AIR，顶面间隙11.322 mm；FL承载0；RR承载7.30%；机身间隙58.386 mm | tick5160；FL 顶面接触，间隙−0.667 mm；FL承载22.52%；RR承载26.18%；机身间隙82.135 mm |
| P09 入口 | tick6000；FL新接触顶面；机身间隙60.046 mm | tick5176；FL AIR；机身间隙81.117 mm |

这些承载数值来自当拍 evaluator 的实际腿状态，不把 FL 悬空解释为承载。首次 P07 对齐 source N 差异在 offset1（C5985/B5161）已经出现：FL knee −36.7° 对 −13.4°，其余11通道相同。P08/P09 source 启动相对 P07 分别仍为 +200/+248 tick，未发现旧时序截断在此窗口重现。

C 的 RR qualified lift 在6272，碰撞在6311；最后完整返回6304仍无 RR 越沿/捕获。机身间隙从6272的40.357 mm降至6311的−0.00777 mm。初始抬升不是后腿完成。最后7个物理 tick 由安全终止打断，未返回完整 env step；不能补造该段 terminal reward/GAE。该次正式评估没有 optimizer 更新。

## 实施修复与验证

本次四份生产文件：

- `semantic_residual_adapter.py`：独立保存纯计算的 nominal command history，每次 dispatch 更新，包括零 residual 快路径；phase 切换和 residual 撤回不重置。只允许新 adapter/已完成原零指令 settle 初始化，缺失、陈旧或外来写入后的历史拒绝继续。
- `semantic_nominal_geometry.py`：三个 geometry mode 均显式使用 `nominal_previous_servo_deg`；不再把包含 PPO 的实际最终目标作为 nominal 历史。实际状态、唯一 mapper 输出和 controller correction 不变。
- `semantic_migration.py`：显式 execution-composition 版本迁移，绑定代码变化；不伪装成仅 reward 变化。
- `semantic_training.py`：持久记录迁移因子。保留兼容的模型/优化器等学习状态，在新执行语义下重新采集，不能沿用旧未完成 rollout。

真实 actuator 的最终 slew 仍使用 **实际最终命令历史**，物理投影、headroom、全部12个学习通道保留；每步仍只有一次 mapper advance 和一次 articulation write。没有第二条 zero 物理轨迹。

Core 226 项定向测试通过，覆盖恒定正负 residual 不再积分、三种 geometry 模式、P09/P12、headroom 截断、phase 交接、撤回、源 wheel stop、180 tick settle、缺失历史及外来写入拒绝；另有主线42项 backend/CLI/prefix 测试通过。

保留的成功 zero 共8857 tick、73.8083 s。其 geometry-active 实录 P09 717条、P12 431条，共 **1148条**计算回放：新旧 Full12 目标最大数值误差0，status一致，重建 float32 actuator target 与原记录全等。证据为 `retained_zero_geometry_replay_receipt.json`；这是原成功行为的执行计算等价证据，不是新增物理成功次数。

## 观测与剩余风险

新增8维 trace 是控制器内部的 **命令记忆**，不是物理状态、不是新 policy observation，也未静默改写既有372维特征。ACK记录前后 trace及 `nominal_command_history_added_to_policy_observation=false`。这不宣称 policy 已观测完整控制器状态；执行语义变化已通过显式迁移处理。

此前 P06 successful-N prefix 训练中的机身碰撞需与此 P09 重复叠加分开。保存的 P06 rollout 显示探索分布仍较宽：平均 sigma 例为 FL hip .364、FL knee .226、FR hip .362、FR knee .289、FL wheel .442；raw 样本绝对值最大例为 FR knee约3.49、FL wheel约5.57（latent raw 单位，不是 degree/rad/s）。这支持继续审查探索与任务信用分配，但不独立证明碰撞根因；本次执行修复没有顺带调整 sigma/reward。

主线已从 CP176896 显式迁移并完成128个真实 P01 policy decisions，得到 CP177024。写本报告时，其正式完整评估仍在运行，最近主线状态为 tick3944/P06、FR/FL已放置。**尚无本次 M3 或完整 PPO 成功声明。** 成功 zero 候选与旧失败日志继续保留。

相关短证据文件：`c176768_p09_failure_window.json`、`c176768_rr_composition_recurrence.json`、`retained_zero_geometry_replay_receipt.json`，均在本目录。
