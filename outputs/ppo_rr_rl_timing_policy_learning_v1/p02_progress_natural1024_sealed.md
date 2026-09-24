# P02-progress natural 1024：已封存真实学习审计

Run：`20260923T2322177593993Z_gd7e97ee7b7e4_1409145fe06a443cba28dbe61f3b4222`；runtime `d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd`。仅按已完成 optimizer journal 统计 global decisions **221697–222720**、PPO **1698–1705**，未读任何 `.pt`，未运行模型或物理。

## 实际信用与结束性质

- 新增 **1024 decisions / 8 PPO / 160 Adam / 0 AUX**；8 次 actor 均实际更新，记录 LR 均 `1e-5`。
- 分支 `ancestor220544_recapture_v2` 最终 **222720 / 1705 / 34100**，checkpoint 为该分支 `checkpoints/history/checkpoint_step_000222720.pt`；不是其他 namespace 或主线的同计数文件。
- 自然 P01 reset **1**；无 teacher/prefix 学习信用；completed episodes **0**、terminal samples **0**、reset/termination reasons **无**。
- 最后 tick **8192 / 68.266667 s / P09**，非终态、未成功；是完整 update 后的预算边界，使用官方非终态 value bootstrap。`SUCCEEDED` 只代表训练块完成，不能称完整越障或任务失败。
- 未消费请求 **0**；没有把 pre-update/pending tail 算成新增学习。

| 实际请求阶段 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10–P13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| optimizer-completed samples | 2 | 422 | 4 | 1 | 163 | 269 | 1 | 1 | 161 | 各0 |

## 真实物理证据

以下计数是 **1024 个已优化决策的物理末端状态**，不是逐物理 tick 或 pre-action 输入的计数。历史事件 tick 来自同一 evaluator 的实际记录。

- FR 实际 placed event **3420 / 28.5 s**。
- FL first observed TOP/bearing decision endpoint **4688 / 39.066667 s**；实际 FL placed event **4733 / 39.441667 s**；决策 endpoint **4736** 从 P05 接到 P06，下个 P06 请求 endpoint **4744 / 39.533333 s**。
- FL placed-history 433 个端点；当前真实 TOP 且 verified bearing **154** 个，不能把所有历史 placed 都叫当前承载。最后 FL 当前 TOP、bearing **2.152092 N**。
- FL capture assist 在 **282** 个决策末端持有 FL 通道；因此 FL 通过不能称纯 policy 单独捕获。RR assist owner **0**，rear task assist 全部 OFF。
- RR 实际本次 qualified-lift event **7150 / 59.583333 s**；首次在决策端点观察到 **7152 / 59.6 s**。
- RR 实际 cross event **7825 / 65.208333 s**；首次在决策端点观察到 **7832 / 65.266667 s**。
- RR 当前合格且非地面 **131** 个端点；历史 cross **46** 个；合格且顶面 XY 内 **48** 个。当前 TOP **0**、当前 verified TOP bearing **0**、placed-history **0**。没有用 AIR 的 `bearing_verified=true` 误计真实承载；承载判定同时要求真实 TOP/obstacle pair、非 AIR/非 ground、support 与 finite force≥冻结 spec 的 **0.2 N**。
- 上述48个合格合法XY端点的最小 gap：**28.038652 mm**，global **222696**，tick **8000 / 66.666667 s**，front **61.845425 mm**，仍 AIR、未 TOP/承载。此值不是整段逐 tick 最小值。
- 最后 RR Q/cross=true，AIR、0 N、未 placed；gap **62.192203 mm**，front **25.145832 mm**。当前未完成任务是 RR 捕获/承载；尚未获得 P10–P13 的学习覆盖。

## 原始 PPO 与执行证据

全部1024行 actual raw、conditional mean、effective sigma、old logp 与原 request 一致；均为1次真实采样、无额外 forward/draw。标准库重算 raw Gaussian old logp 最大误差 **2.7602e-6**。全部 native dispatch verified、12 residual mask=1、RR assist14为WAIT零值；这只是审计已保存的采样与派发，不重新 forward 模型、不把 transformed targets 当 Gaussian actions。

来源：同 run 的封存 `run_manifest.json` / `training_manifest.json`、8条 `optimizer_updates.jsonl`、8条 `advantage_audit.jsonl`、1024条 `residual_and_projection_audit.jsonl`、空 completed-episode ledger，以及实际分支 checkpoint sidecar。运行结束后单次有限读取；无生产改动。
