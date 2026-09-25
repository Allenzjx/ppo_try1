# 已封存 RR 方向事实：只作下一次诊断依据

证据范围：复用 `diagnostic_actualmetrics_sealed.json`，没有重扫物理日志、运行仿真或拟合。其真实独立诊断来自 `diagnostic_hipminus25_kneeplus20_ecf205e/source`：自然 P01、CP225280 local0、旧 ecf205e / 447 维 v1；固定入口 FINAL 的 hip −25° / knee +20° 双通道干预，PPO/AUX credit 均为 0。这不是两个单轴独立实验。

## 实际执行与下降

RR 顺序均为 `[hip, knee]`，角度为度，gap 为轮底至台面毫米。相对入口的 FINAL 基准为 `[7.485144, −58.000000]`，actual 基准为 `[7.223447, −57.760433]`。

| tick / 仿真秒 | FINAL | actual | gap | TOP / hold |
| --- | --- | --- | --- | --- |
| 7984 / 66.5333，入口 | [7.485144, −58.000000] | [7.223447, −57.760433] | 58.988124 | false / 0 |
| 8104 / 67.5333 | [−8.070412, −45.555556] | [−7.065976, −46.287529] | 28.411804 | false / 0 |
| 8224 / 68.5333，最小 sampled gap | [−17.514856, −38.000000] | [−17.701442, −37.754707] | 4.059061 | false / 0 |
| 11220 / 93.5000，终止 | [−17.514856, −38.000000] | [−17.723827, −37.744195] | 7.429624 | false / 0 |

已核验的 0 / 0.5 / 1 / 1.5 / 2 秒稀疏决策端 gap 为 **58.988 → 46.113 → 28.412 → 12.454 → 4.059 mm**，支持一段连续执行中的明确下降趋势；本说明没有重新证明每个 120 Hz tick 单调下降。之后 gap 回升至约 6–7 mm，最终仍 AIR / 0 N，没有 TOP、可用承载或 hold，结果为 `INCOMPLETE_CONTROLLER_BLOCKED`，不是捕获成功。

最小 gap 时 FINAL 入口增量准确为 `[−25,+20]°`，actual 增量为 `[−24.924889,+20.005727]°`，说明这组有限方向确实传到实际关节，不只是 raw 意图。该点合法顶部 XY、距前缘 +47.736 mm；mapped N 为 `[−8.15,−39.05]°`，generic RR bias `[0,0]`，requested/effective 同为 `[−9.364856,+1.05]°`，没有该点残差 headroom 裁剪。原 policy selected raw `[−0.277023,−0.038440]` 与实际 issued override `[−0.412039,+0.029175]` 已分别记录。由于双关节、身体和其他支撑同时响应，不能从此断言单独 hip 或 knee 的因果效果。

## 与当前 v2 的边界

- 当前 448 / 0ff03ea 有本次有效抬升谱系字段、训练后的 local head 和有限 AUX64 谱系；旧实验是 447 / local0。已封存 CP228864 DET 的 998 决策前缀、gate 7979 和入口 FINAL 虽匹配，并不证明激活后的分布、HISTORY 和控制历史相同。
- 当前采用 `rr_local_defer_p09_late_and_new_p12_until_terminal_v2`，并修复为 `pending_source_tracking_inheritance_v1`。五通道 late pending、P12 pending-start、独立 stop clock 与旧 v1 不同。旧探针没有触地，不能据此推断两版全部 owner / tracking / stop 路径等价；最小 gap 处相同 mapped N 也只是局部数值相同。
- 因此此段可保留为“负向 hip + 正向 knee 可实际减小 gap”的方向/跟踪证据，**不能直接作为当前 v2 成功 AUX 标签、on-policy 样本或干预动作的旧 likelihood**。它本身没有完成接触/保持。

`OPTIONAL_KNEE25_DIRECTION_DIAGNOSTIC.md` 与 `INDEPENDENT_KNEE25_DRIVER.md` 中的 hip −25° / knee +25°（以及讨论的 knee +30°）只是候选/已准备驱动，**并非已运行的物理结果**。不把它们报成已验证进一步下降或触地。

另有旧 v1 真正下降→TOP→0.5 s hold 的短训练段，其可用性已经记录在 `AUX_success_short_window_availability_v2_control_review.md`：首次 TOP 9436 同时释放旧 P09 late 五通道，而当前 v2 延迟该组。因此旧成功也不能不经新控制实测就直接蒸馏；它与本独立方向探针是不同证据类别。本次不生成新数据标签，不改模型或生产实现。
