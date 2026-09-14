# Video3：FL 已越沿，真实放置仍未完成

## 范围与结论

只读已完成 run `video_eval/validation/20260910T1138465345417Z_g7db0d17f398d_9bd793c700bd4040b4c517a0cd625ebb`。单次固定扫描 `source/video_policy_decisions.jsonl` 全部 758 行（49,997,728 bytes）；物理流只提取明确事件 tick 1872、1919、3466、5024、6060。policy 文件 manifest SHA256 为 `e979ad1611ba024a28319e65cc9f5b7e0270f116362f2866d4d6be911aa263c8`（引用保存的 hash，未另作重复哈希）。未读活动训练、未改生产或进程。

本次从自然 P01 开始，seed 4001，官方 `checkpoint_step_000145920.pt` 的 load proof 为 true；评估新增 PPO update 为 0。实际终态 P05 / 6060 tick / 50.5 s，`INCOMPLETE_CONTROLLER_BLOCKED`，来源 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。P05 age 34.9 s 达到当前 30 + 4.9 s 有界恢复期限。物理状态为 VERIFIED，无安全或机身碰障终态；**任务未完成，不是成功**。发布生命周期 `DIAGNOSTIC_FAILURE` 与正常物理终止并不矛盾。

首个未完成任务已是 **FL 顶面真实放置**，不是 FL 越沿：FL Q3466、C5024 有实际证据，P 始终 false。越沿后至末端全部 131 个决策末态仍 AIR、无顶面接触，距台面 29.539–33.017 mm，不能把过前缘或无碰撞当作落地。

## 阶段及事件链

| 发出动作的阶段 | 实际决策数 | 出口物理 tick / 时间 |
|---|---:|---|
| P01 | 2 | 16 / 0.133333 s |
| P02 | 218 | 1760 / 14.666667 s |
| P03 | 7 | 1816 / 15.133333 s |
| P04 | 7 | 1872 / 15.6 s |
| P05 | 524 | 6060 / 50.5 s，任务未完成 |
| P06–P13 | 0 | 未到达 |

FL 的初始抬升 I1919（15.991667 s）只有 3.125 mm 向上位移，没有 Q；不可把第一次 I 计作已经成立的抬升。之后新的 I3451（28.758333 s），3.067 mm 向上位移；Q3466（28.883333 s）累计 excursion 8.343 mm、FL 自身实测关节运动 2.924°，当 tick 为 AIR、离台面仍 -41.781 mm。这是经测量成立的初始功能抬升，不要求先越台面或固定悬停。初始 I 的记录同时有其他关节/全身运动证据；不能把它单因归于 FL 自身关节，也不能从 AIR 推断承载。

C5024（41.866667 s）：FL 轮心刚越前缘 0.033 mm、轮底高于台面 29.630 mm，当前 AIR 且 Q 历史有效，满足真实越沿几何。其后没有 Q 的 ground-revocation 事件，也没有 P。终态 `consecutive_top_samples=0`；FR 已有 Q27/C1772/P1811，不能把 FR 的放置或 FL 的历史 C 替代 FL 当前放置。

P05 共 524 个决策末态中，GROUND 176、AIR 331、TOP 0；其余不完整/歧义接触不计入 AIR 或 TOP。188 个末态载荷未知，不能填成零或承载。越沿后的 131 个末态 AIR 131、GROUND/TOP 均 0、当前 XY 与 lateral 判据均 true。

## 精确物理证据：通过落区，不等于落在台面

以下来自 `physical_observations.jsonl` 的精确 tick；行号分别为 tick+1。台面 z=0.05 m、前缘 x=0.5213121737735307 m、左右界 y=±1 m。

| 物理状态 | FL 轮心 x/y (m) | FL 轮底 z (m) | 实测接触与放置 |
|---|---|---:|---|
| P05 入口 tick1872 | 0.454267 / 0.098663 | -0.000746 | GROUND，14.061 N；未 Q/C/P |
| Q tick3466 | 0.472708 / 0.100133 | 0.008219 | 当前 AIR，地面/障碍对均 inactive |
| C tick5024 | 0.521345 / 0.102863 | 0.079630 | AIR，两个接触对均 inactive、法向力 0、pair verified |
| 终态 tick6060 | 0.581839 / 0.105608 | 0.081603 | AIR，两个接触对均 inactive、法向力 0、pair verified，P=false |

越沿后 `within_top_xy=true`、`within_lateral_span=true`、outside-distance=0，未发现横向落区判据阻断；它们是现有轮心/区域判据，**不等于已测得完整接触 footprint，更不等于有承载接触面**。终态载荷为有效测量的 0，support=false。当前 FL 仍在台面上方 31.603 mm，`top_geometry=false`、`top_contact=false`，没有可以接受为 placement 的物理证据。

实测 base x（不是 CoM）：P05 入口 0.226270 → Q 时 0.134413 → C 时 0.184546 → 终态 0.245029 m。先退 91.857 mm，再前进 50.133 mm、60.483 mm；P05 总体净前进仅 18.759 mm，不能描述为单调前进。C 至终态 FL 轮底反而增高 1.974 mm。终态 base 世界线速度 [0.014387, -0.000758, 0.022921] m/s；FL 刚体 z 速度 +0.024832 m/s，此瞬间仍不是向下落地。该趋势不证明由某一腿单独造成，也不把 base 位置当 CoM。

## 控制链、nominal 与实际响应

全回合 6060 个物理/native-effect tick 均验证；6056 为本阶段 request，4 为普通边界继承 hold tick。758 次环境 step 全部返回；无未验证 decision、无禁止的 episode 内 state write。未发现跨阶段 residual/filter/mapper 被清零或 native dispatch 丢失的新证据；普通 source owner 请求跃迁不能误判为 reset。

FL nominal [hip,knee] 最后一次变化在决策 381 / tick3048 / 25.4 s，此后一直为 **[22.8, -13.4]°**，直至终态；C 发生在这一最后变化之后 16.467 s。FR 的 capture-owner hold 记录存在，FL 无 capture-owner hold、无新 swing 被抑制。P05 全部决策 FL mask 均开、无 FL headroom clip。下表为真实请求/目标/实测，不把指令当成动作成功：

| 时刻 | nominal FL (°) | residual FL (°) | final target FL (°) | actual FL (°) |
|---|---|---|---|---|
| Q 后 tick3472 | [22.800,-13.400] | [7.141,20.550] | [30.677,8.400] | [30.021,8.345] |
| C tick5024 | [22.800,-13.400] | [7.468,20.548] | [30.953,8.398] | [30.345,8.385] |
| 终态 tick6060 | [22.800,-13.400] | [7.708,20.612] | [29.966,8.462] | [30.406,8.447] |

精确终态 raw physics 的 hip/knee 命令误差为 -0.441° / +0.014°；关节速度为 -5.837 / -0.288°/s。目标确已下发，尤其 knee 并非“命令变了但完全没动”。单凭 hip/knee 的正负号不能推出笛卡尔下降，但当前实际轮底趋势清楚：未落到台面。

C 与末端 nominal wheel 都为 [+0.3,+0.3,+0.3,+0.3] rad/s；末端 final 为 [-0.320,+0.125,+0.089,+0.803]，actual 为 [-0.321,+0.059,+0.167,+0.976] rad/s，策略仍在改变各轮方向/配合，并非无轮动作。末端较小的 actual-target 误差或 wheel 响应偏差不能单独解释整个未放置结果。

静态核对 `semantic_supervisor.py:1777` 的 P05 pending-capture 支援：该分支只续接 wheel prior，保留已有 servo 层；没有依据当前 FL 台面净空生成新 servo 下降目标的分支。本次观察窗口内确实**没有剩余的新 source nominal 下降请求**，PPO residual 也未形成真实放置；这是可用于后续学习/诊断的目标与运动结果，不是证据足够的通信/状态交接软件缺陷。未修改 reward 或增设放置捷径。

验收代码 `semantic_supervisor.py:632–638,687` 把 XY/净空几何、当前 TOP 接触与历史 C 分开；P 需要 C 后两个实际 TOP 样本（既有抗噪采样，不是新增静止停留）。本次 TOP 全为 0，判未放置与原始接触一致，不是只因轮心未过沿或横向条件过严。

## 质量与视频完整性

全局质量样本覆盖真实 6060 ticks / 50.5 s：roll RMS 0.085629 rad、peak 0.192340 rad；pitch RMS 0.115666 rad、peak 0.228161 rad；角加速度 RMS 6.143079 rad/s²；applied servo rate RMS 25.236246 °/s；residual servo RMS 9.455461°、wheel RMS 0.360529 rad/s。这是本失败回合的指标，不含未到达阶段，不构成稳定性优于 FSM 的结论，也不与不同 seed 的早期评估作改善比较。

实际保存 758 帧，首采样 tick8、末帧 tick6060；末步为真实 4-tick partial interval。物理 50.5 s，编码时长 50.533333 s（末帧显示量化 +0.033333 s）；无 PRE/POST 额外物理步或帧、无插值/加速，失败尾被保留。录像完整不能替代任务成功；本轮仅证明 FL 已真实越沿而未实际放置。
