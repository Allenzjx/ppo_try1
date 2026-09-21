# CP180480 确定性 C：P05 未捕获诊断

结论：此次是 **FL 越沿后未放置，耗尽 P05 有界任务时限**，不是已发现的 FL 指令丢失，也不是进入 P06 后的 cap 扩大问题。记录显示策略修正真实生效，却与全身闭环一起长期维持悬空姿态；不能据此把某个关节正号单独判为失败原因。

## 封存结果与终止因果

- source：`runs/ppo_fl_capture_quality_v1/video_eval/validation/20260918T0524353774569Z_gf2e552406ea7_0d8dd902152a4ad68a84118544c00b3c/source`。
- CP180480，`deterministic_conditional_mean`，726 次决策；本次诊断无模型前向、无优化器更新、无仿真或生产改动。权重与源 manifest 哈希见同名 JSON。
- FR placed tick1603；P05 从 tick1616 开始；FL crossed tick2439，但直到 tick5804 / 48.366667s 都未 placed。RR 阶段未进入。
- 精确终止：`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。P05 age=34.9s；基础 30s + `min(10,30×0.5)×0.7²=4.9s`。终态 `placed_FL` 进度 0.7，不等于已放置。
- `stall_diagnostic=true` 是 6s 进度窗口诊断；配置 `terminates:false`，源码 `semantic_supervisor.py` 的时限分支先设置终止，随后才计算 stall。它不是单独的“停滞保护杀停”。软 capture potential 可变化，但不刷新该时钟。
- 终态物理 evaluator 有效、无物理失败终止；本次归类任务未完成，不能称碰撞失败或成功。FL gap=+25.728854mm，front=+150.444869mm，AIR，真实承载力 0N。

## 同拍 FL 目标链

下表关节均为 canonical 度，顺序 hip/knee；N 为该真实状态下源 nominal，mapped N 包括当时 mapper 状态，不是独立重跑 zero。raw/base/conditional 的完整精确数值及 native rad/pre-dispatch readback 见 JSON；所有 actual 为对应 physics tick 步进后读回。

| tick（秒） | base μ → conditional/raw（hip,knee） | filtered/effective residual | N → mapped N | final → actual | FL gap mm |
|---|---|---|---|---|---:|
| 2439（20.325）越沿 | .18630,.10213 → .18577,.10260 | +3.306,+2.454 | 48.2,−36.7 → 49.45,−37.95 | 52.756,−35.496 → 52.714,−35.501 | 76.630 |
| 2777（23.142）源末端前 | .18700,.09506 → .19034,.10008 | +3.385,+2.394 | 24.9,−13.4 → 23.65,−12.15 | 27.042,−9.756 → 37.814,−10.304 | 62.571 |
| 2785（23.208）源末端 | .18487,.09811 → .18979,.09988 | +3.376,+2.389 | 22.8,−13.4 → 22.8,−12.15 | 26.176,−9.761 → 32.155,−10.044 | 44.953 |
| 2905（24.208）末端后1s | .19293,.09526 → .19003,.09749 | +3.380,+2.332 | 22.8,−13.4 → 22.243,−12.15 | 25.623,−9.818 → 26.220,−9.811 | 26.813 |
| 5804（48.367）终止 | .18899,.09336 → .18870,.09348 | +3.357,+2.237 | 22.8,−13.4 → 22.224,−12.15 | 25.581,−9.913 → 26.045,−9.918 | 25.729 |

源末端 tick2785 是代码与记录推导：P05 首次源下发 tick1617、MotionExecutor 从 local tick0 开始、源 active duration 9.733333s=1168ticks，P05 无 sequence wait；此后记录的 nominal servo8 与源 endpoint 精确相同。P05 endpoint flag 未独立写入日志，故不是对未记录字段的“实测确认”。nominal wheel 在此后仍可按真实 pending-capture 条件协作。

## 执行链与末段全身状态

- 源末端至终止 3020 个 physics samples：FL 全部 AIR、障碍力全部 0N；gap 最小仍为 24.716mm。并非一次端点采样恰巧漏接触。
- 同窗口所有12通道 residual mask=1，实际写入/映射校验均通过；**FL requested residual 与 post-headroom effective residual 完全相同**，没有 headroom 截掉 FL。末6s hip 有30拍受已有最终 slew 修正，不能声称每拍完全无执行限制；终止拍 slew 差为0。
- 末6s（tick5085–5804）FL hip/knee target−actual RMS=0.601058°/0.006079°，最大绝对误差0.732864°/0.015044°。源动作下降段存在明显暂态跟踪滞后（例如 tick2777 hip 10.772°），但之后悬空又持续约25s，因此仅用这段滞后解释最终未捕获不充分。
- 末6s FL hip mapped N 在22.223870/23.473870°之间变化，actual 范围26.033433–26.379801°；这是已有闭环 mapper 摆动，非本诊断修复/消除。FL knee mapped N 固定−12.15°，加策略约+2.24°后 actual 约−9.915°。不能把 final−独立重算N 全部归为当拍 PPO。
- 末6s gap25.019–28.482mm，front139.896–150.735mm。FR 始终 OBSTACLE、RL/RR 始终 GROUND；其 pair normal force 均值12.319/14.191/2.167N。终态 evaluator 独立验证支持 FR/RL/RR，FL 不承载。末拍载荷比例约42.70%/49.64%/7.66%，不可虚构 FL 承载。
- 末拍全身 residual servo8 为 `[+3.357,+2.237,+1.080,−7.429,−1.442,+3.048,+3.485,−3.843]°`，不仅 FL 两关节改变。body z=0.098616m；线速度模0.024943m/s、角速度模0.061246rad/s。全身支撑及车体几何作用不能从一个 hip 符号分离。
- 四轮顺序 FL/FR/RL/RR。tick2785 N 为 `[.3,.3,.3,.3]rad/s`；tick2905 front>0.1m 后 N 为全0，符合 pending-capture 的 `front_distance < approach_max_m` 建议范围，而非 mask 清零。末拍 N=0，final=`[+.132521,+.003933,+.148504,−.035842]`，实测=`[+.132405,−.104779,+.090278,+.053915]rad/s`。承载轮实际与 target 可因负载耦合不同；FL 空转不等于提供牵引。

## 归因边界

可排除所查窗口内“FL residual 被许可 mask/余量投影吞掉、最终指令没下发”的解释；实测关节响应支持这一判断，不仅是 ACK。当前策略的均值输出趋稳，FL 修正和其它支撑腿/轮修正共同维持了未捕获状态，未创造接触入口。这是任务能力不足的实证，而不是单关节正/负方向的隔离因果实验。

此前 −2° probe 是其它闭环状态的独立干预，不应直接套到此状态判定“hip+必错”；旧 CP178432 终态 gap3.47mm 与本次25.73mm只是描述性对比。P05→P06 REQUEST-history 候选可处理另一个交接风险，但本次根本没有进入P06，不能借候选宣称此 FL 失败已修复。本报告不建议扩大时限、放松放置判据或改生产参数。
