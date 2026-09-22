# Block05 已完成回合与补充快照：只读核查

运行 `20260922T0926100383683Z_g5fd88852bf20_344b6773d9854b41948d1632e1f0fc3f`。**证据固定截止 global decision 208653**；仅读取此前 781 条 audit（6244 physical ticks）、已完成 episodes 0/1，以及 episode 2 到 7.533333 s。此后结果不在本报告内；没有轮询等待、热改、Isaac/GPU 启动。stdlib CPU helper 已退出（最后一次 0.936 s）。

## 直接结论

前两个回合是 **FR 已抬起并有充足净空，但前送不足**，不是“P02 只抬了 RL”。第三回合的补充快照则确实是 **FR 回到地面、RL 悬空**；两种失败形态不能混称。没有看到 FR 指令错发给 RL、四轮 nominal 丢失、capture-assist 接管或执行器派发遗漏。策略确实改变全身目标和轮速，但不能凭一条轨迹证明某个 residual 是唯一原因。

| Episode | 截止 decisions / global | 完整物理时间 s | FR gap / front mm | 状态 |
|---|---:|---:|---:|---|
| 0 | 337 / 208209 | 22.450000 | 83.014 / -19.408 | P02 incomplete |
| 1 | 331 / 208540 | 22.050000 | 77.002 / -39.875 | P02 incomplete |
| 2 | 113 / 208653 | 7.533333 | -50.057 / -77.571 | P02 采样中（本快照） |

## Episodes 0/1 的第一个未完成条件

两者 P02 entry 有效、entry reasons 空、physical evaluator 一直 valid/VERIFIED、无独立物理失败；终态 `stall_diagnostic=false`。真实触发的是 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，不是额外卡住的源动作 guard：

- Episode 0：FR qualification tick25；到首个 tick80 决策端点已满足 15 mm 净空，此后未记录接地撤销。结束 `lifted_FR=1, clear_FR=1, approach_FR=0.9623661559`。P02 elapsed **22.316667 s** ≥ 15+7.3130110347 s。FR 当前 AIR、gap **83.013528 mm**，前缘距离 **-19.408461 mm**。
- Episode 1：qualification tick26，首个 tick64 决策端点满足净空。结束 `1,1,0.8805013221`，elapsed **21.916667 s** ≥ 15+6.9144065554 s。FR AIR、gap **77.002337 mm**，front **-39.874669 mm**。
- 当前 P02 approach 下界为既有 -5 mm 加 5 mm 测量容差，即 **-10 mm**；因此结束分别差 **9.408461 / 29.874669 mm** 才进入 P03 所需的 approach 区域，不是欠缺抬升。采样端点最接近时分别是 tick2640 的 **-12.074050 mm** 与 tick2416 的 **-34.410145 mm**，仍未完成。没有 crossing/placement 被误判成成功。

四轮 N 在两回合均从第一个观察到的 tick48 端点起保持 **[+.3,+.3,+.3,+.3] rad/s**，直到终点；在 tick1592/1600 的 P01 有限源结束窗口也没有丢失。本例当前 FR qualification、净空和其他支撑满足 `_approach_assist_required`，所以 P02 的物理目标前送继续生效。P02 不受 P07–P09 源 partial-order readiness 层阻塞；源 FR knee 末值 **45.9°** 已于 tick80 端点实际出现。没有“源从未启动”或“源 stop 清零四轮造成本次等待”的证据。

与 [block01_episode0_P02_readonly.md](block01_episode0_P02_readonly.md) 相比：同属局部 P02 任务期限未完成，但**不是同一个物理子类型**。旧例是 FR 接地、qualification 撤销、净空失败，有限源 stop 后无法满足额外前送条件；这次 episodes0/1 是持续 AIR、四轮仍前送而 approach 尚不足。

## 第三个回合：FR/RL 名称和实际状态

截止 208653 的 episode2 不是尚未发生过 FR 抬升：FR tick25 曾 qualification，**tick138 / 1.15 s 因 ground-before-cross 撤销**。全部已采样 FR 最大台面 gap 仅 **13.093780 mm**，未达到 15 mm clear 目标；到 208653 为 `lifted_FR=0, clear_FR=0`。这是早段净空丢失子类型，和上述旧例更接近，不应把它描述成底层 mask 故障。

- **208576 / tick288 / 2.4 s**：FR gap -49.999793 mm、真实 ground、1.854115 N；RL gap **+83.787284 mm**、AIR、0 N。
- **208653 / tick904 / 7.533333 s**：FR gap -50.056855 mm、front -77.571180 mm、真实 ground、1.450946 N；RL gap **+54.758627 mm**、AIR、0 N。

这里使用 evaluator 的显式 `current_legs.FR/RL`、对应接触 pair 和 native canonical 关节名，不是按画面或排序猜腿名。当前 recording motion contract 的 P01 末值本来包含 **RL hip +37.6°**，P02 继承它并追加 **FR knee +45.9°**。因此 RL hip 准备动作不是被错标成 FR knee；但“源包含 RL 准备动作”不等于允许 FR 回地的结果已经完成 P02。

208653：RL hip N **37.6°**，mapper **38.85°**，policy 有效修正 **-7.103535°**，最终 **31.746465°**，实测 **31.887359°**。FR knee 的独立通路见下表。RL/FR 两套关节实际都有响应；同时 FR 在地、RL 悬空是这次全身响应事实，尚无单独干预证明究竟哪一目标/载荷耦合是主因。

## 同拍控制与跟踪证据

FR 数组均为 **[hip,knee]，度**。mapper baseline 本窗口没有额外 geometry/controller 修正；effective 使用原生同拍 headroom receipt，不用重算 N 与 final 的差冒充 PPO。actual 是 dispatch 前最后一次真实关节读回，经已验证 canonical 变换；接触和 wheel actual 是该次物理步后的 evaluator 读回，时刻区别保留。

| Global decision | 同拍 mapper baseline | 有效 policy 修正 | 最终目标 | FR actual（步前） |
|---|---|---|---|---|
| 208209 | 0.000000, 44.969776 | 4.345191, -4.704899 | 4.345191, 40.264877 | 3.683101, 42.629586 |
| 208540 | 0.000000, 45.181733 | 4.227720, -8.000430 | 4.227720, 37.181303 | 3.975765, 38.432486 |
| 208576 | 0.000000, 43.603248 | 4.356050, 0.587749 | 4.356050, 44.190997 | 4.563155, 46.937779 |
| 208653 | 0.000000, 47.378623 | 3.463360, -3.080333 | 3.463360, 44.298290 | 3.568694, 42.854880 |

四轮顺序 **FL, FR, RL, RR**，单位 rad/s；上述四拍 N/baseline 均 **[+.3,+.3,+.3,+.3]**。

| Global decision | 有效 wheel residual | 最终 wheel target | 步后实测 wheel velocity |
|---|---|---|---|
| 208209 | -0.192041, 0.068217, -0.043312, -0.076599 | 0.107959, 0.368217, 0.256688, 0.223401 | 0.131981, 0.368593, 0.663185, 0.157729 |
| 208540 | -0.066045, 0.053978, -0.019694, -0.078898 | 0.233955, 0.353978, 0.280306, 0.221102 | 0.324565, 0.353467, 0.323282, 0.134274 |
| 208576 | -0.273257, 0.007451, -0.069288, -0.096257 | 0.026743, 0.307451, 0.230712, 0.203743 | 0.036499, 0.288645, 0.230918, 0.281133 |
| 208653 | -0.297938, -0.029732, -0.049215, -0.066144 | 0.002062, 0.270268, 0.250785, 0.233856 | 0.129668, 0.307399, 0.249295, 0.103363 |

特别是 208653，FL nominal .3 被 policy 的 **-.2979384554** 抵消到 **+.0020615446**；这是**策略抵消**，不是 mask。其当拍实测仍 +.129668，说明目标与受载运动并不瞬时相等。FR/RL/RR 最终目标及实测也都非零；不能推断“只有 RR 电机在转”，更不能把 FR 或 RL 的轮端高度当作电机角速度。FR 空中转轮不代表提供地面牵引，RL 空中也不虚构承载。

## 范围与处置

781/781 native endpoints mapping/dispatch verified，6244/6244 物理 tick 审计通过；12 个许可全开、capture-assist owner 空、无 headroom 裁剪、无状态写入。此结论同时检查了 source N、同拍 mapper、有效 residual、final、实际关节/轮速和接触，**不是仅以全1 mask 作结论**。仍有常规跟踪误差和载荷耦合，ACK 不能证明任务可达或因果唯一。

本窗口 RR 没有合格越沿，新的 RR post-cross preparation-retirement 条件未启用；不把 P02 的变化冒称该新项直接修复/破坏了动作。没有发现需中断当前块的执行链缺陷；报告的是当前真实任务不足。当前训练由 root 继续，未提出新门禁，也未改 reward、HISTORY、sigma、动作范围或控制器。
