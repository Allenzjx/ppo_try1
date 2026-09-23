# c53119a v5：RR 接触前有限预算耗尽，未完成

仅分析已封存单回合；未新增物理步、优化器更新或模型前向。源 `20260923T0718224682359Z_gc53119ab332f_886689c9a357416293bc8242856d0be6/source`，实际官方载入 CP SHA `308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895`。完整输出和逐流哈希见同目录 `v5_capture_follow_fixed_COM.json`。1825 次真实评估 decision，14595 physics ticks，121.625 s，task=false；P10/RL 未到达。外层 `DIAGNOSTIC_FAILURE` 是真实任务未完成，分析没有改写结果。

## 直接结论

RR 在最后 5 s 仍实际下降，目标跟踪正常；最后一拍耗尽的是累计 **52° 行程**，不是 44 s 暴露上限（实用 42 s）。助控从 `DESCEND_PROGRESS` 转 `BLOCKED/finite_search_travel_or_margin`，同拍 `rr_capture_recovery_allowed=false/local_warning_only=false`。此前 current Q/cross、合法 XY 和三个其他支撑仍成立。没有证据支持把它归为关节限位、跟踪停滞或接触传感器丢失。

**27 微米不是接触。** RR 越沿 tick9555 后共 5041 拍，GROUND、obstacle_pair_active、TOP、support 全为 0，最大 bearing force 为 0 N；全回合无 RR TOP/placed。实际 collider-bottom 正净空的最小值恰在终态，为 **0.02695755 mm**。终态原始 ground/obstacle force_matrix、三拍力历史均为零，pair_verified=true。obstacle 的 contact_point 字段虽非空，但 active=false/force=0，不能把该坐标当作有效接触。尚不能排除未测的接触建模细节，但日志没有传感器错误证据，也无理由放宽 TOP/placed 判定。

## 实际末段（不是独立关节导数）

| 物理 tick / 时间 | RR gap mm | RR front mm | RR 实际 knee ° | knee 最终目标 ° |
|---|---:|---:|---:|---:|
| 13995 / 116.625 s | 4.483599 | 33.612314 | -18.283857 | -18.361176 |
| 14115 / 117.625 s | 3.381172 | 31.001214 | -17.286069 | -17.361176 |
| 14235 / 118.625 s | 2.449206 | 28.350178 | -16.288330 | -16.361176 |
| 14355 / 119.625 s | 1.685488 | 25.676433 | -15.290556 | -15.361176 |
| 14475 / 120.625 s | 0.743625 | 22.974495 | -14.292967 | -14.361176 |
| 14595 / 121.625 s | 0.026958 | 20.260695 | -13.296181 | -13.361176 |

最后 2 s 净下降 1.658531 mm，同时向前缘退回 5.415738 mm；最后 5 s 净下降 4.456642 mm、退回 13.351619 mm。终态 RR hip 实际/目标为 -2.122473/-1.785913°，knee 为 -13.296181/-13.361176°；目标减实际误差 +0.336560/-0.064996°。RR command bounds 余量远未耗尽，但这不是笛卡尔下降可行性证明。该连续动作同时有整机/轮控制，不能把上述变化归因于独立 knee 导数或 PPO 学到了该方向。

末段 base z 从 77.204019 到 77.191025 mm，roll/pitch 从 [-9.321055,2.473694] 到 [-9.329490,2.540616]°。真实 USD hip mount 在终态同拍的世界 z 为 **232.538027 mm**。5 s 起点的最近 height 样本是 tick13992（比表中13995早25 ms），z=232.326736 mm，单列而未冒充同拍。RR wheel 相对 base 坐标从 [-250.997,-321.392,-37.102] 到 [-263.577,-321.392,-42.403] mm；该坐标原点是 base，**不是 CoM**。

终态当前支撑：FL TOP 3.195532 N，FR TOP 13.586161 N，RL GROUND 11.971486 N，RR AIR 0 N。RR 历史 Q/cross 不被当成承载。

## 四轮及 owner

源 P09 finite wheel stop 已真实发出并有后续 ACK 证据；末拍 nominal 四轮 [0,0,0,0]，fresh wheel owner 为空，不是遗失轮指令。支持轮局部包络首次激活 tick9773/81.441667 s，FL/FR/RL 当拍各以 1.8 rad/s² 改变最终目标，RR 未被该包络改变。

末拍顺序 FL/FR/RL/RR（rad/s）：

| 层 | FL | FR | RL | RR |
|---|---:|---:|---:|---:|
| policy/mapper 后候选 | -0.937280 | 0.062743 | 0.000052 | -0.088528 |
| projection 后实际 canonical target | 0.000474 | 0.062743 | 0.000474 | -0.088528 |
| 写入 native target | -0.000474 | 0.062743 | -0.000474 | -0.088528 |
| 实测 canonical qdot | 0.021872 | 0.041256 | 0.240566 | -0.088218 |
| 实测 native qdot | -0.021872 | 0.041256 | -0.240566 | -0.088218 |

native target 来自已写入 simulator target buffers；native joint IDs 本流未记录，保留 null。正向 wheel 包络已改正 FL 的 -0.937 候选；不能由目标正值或 all-1 mask 推断牵引。RL 实测转速明显大于目标，未把它解释成电机屏蔽。RR hip/knee owner 仍为公开助控，末拍源和请求增量均 [0,0]；未进入 captured-follow/release/P10。

## 固定方向与有限下一步建议

P07 首入 tick8280 的真实 mass CoM→FL wheel 固定 XY 单位方向为 [0.7357788103,0.6772219299]，终态 CoM 投影 +188.056594 mm，接收脚自身投影位移 +105.459665 mm 单列未计作 CoM 信用；世界 CoM 增量 [221.500588,37.035355] mm，yaw 改变 -1.883175°。包含前送且并非纯横向载荷证明。P10 未到达，RL→FR 固定方向 unavailable。

数据支持下一版做**一次、不充值、很小的接触起始续行**，而非再次扩大无条件搜索：仅在现有 progress-reserve 已赚取、当前合法 Q/cross/AIR/XY/其他支撑/跟踪全部有效、当前近 TOP 且最近真实下降仍成立时，给至多额外 +1° knee / 1 s 的公开有界额度；保持 hip、原 wheel/源 owner/物理硬限，真实接触立即停止下降并进入既有有限接触确认。退出时机同时看 XY/支撑/跟踪，不凭预计剩余时间保证触地。必须版本化公开行程/时间语义，不能隐式充值、伪造 TOP 或把此控制续行称为 PPO 学习。末段退回前缘的事实要求继续观察 front depth；本报告不保证再 +1° 一定捕获。现有真实 HOLD/确认窗口应保留，不能在首个真实接触后一拍因局部 deadline 抢先终止。

分析代码只修正输出脚本对已完成 `DIAGNOSTIC_FAILURE` 生命周期的接受，读取所有相关流一次并核对封存哈希。未改生产、config、checkpoint 或原视频；所有分析 Python 已退出。
