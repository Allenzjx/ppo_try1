# Capture-retention run：episode5 RR carry 有界诊断

状态：仅对已完成的第六回合进行只读日志分析。不是整块训练最终报告，不新增训练计数，不选择下一修复或门禁。未运行 Python、Isaac 或测试，未修改生产代码或历史日志。

## 1. 固定来源、边界与真实结果

- Runtime：`68631e932c7deb08a7a3f2a2787b79fa7eb569ef`。
- Run：`runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189`。
- 只读取该 run 的 `completed_episodes.jsonl` 第六条及 `residual_and_projection_audit.jsonl` 对应 episode5；物理/目标细查仅限 P07–P09，未解析其他回合。
- Source global 54656；前五回合 `939+677+931+923+935=4405`，因此本回合首 global 为 `54656+4405+1=59062`，末 global 60080，共 **1019 decisions / 8152 physics ticks**。
- 天然 P01 策略回合，无教师前缀。阶段决策数 P01–P13 为 `1,187,3,1,148,227,1,1,450,0,0,0,0`。
- 真实终局为 **P09 / INCOMPLETE_CONTROLLER_BLOCKED / 67.9333333333 s**；task/full success 均 false。共同物理评价器 valid=true，physical failure/termination=null，不是机身碰撞等硬物理失败，也不是软件、传感器或视频执行错误。
- P09 在 tick4552 / 37.9333333333 s 进入，tick8152 / 67.9333333333 s 达到 30 s 阶段 deadline。当前未完成任务是 RR 的对应主动越沿/放置；从未 RR cross/place，不能将抬高、再次获资格或 AIR 当作成功。

精度范围：后文极值、分支计数和指令表均来自 **15 Hz 决策结束样本**，不是完整 120 Hz 原始几何极值。已保存任务事件具有原生 tick；紧凑逐 tick native 审计覆盖本回合全部 8152 tick，但不包含每个子 tick 的完整 pose/contact/geometry 字段。

## 2. P07–P09 任务与资格历史

| 事件 | Episode tick / time | 已记录依据 |
|---|---:|---|
| P06→P07 | 4536 / 37.8000 s | rear_approach=1 |
| P07→P08 | 4544 / 37.8667 s | RR/RL workspace=1，RR support=1 |
| P08→P09 | 4552 / 37.9333 s | RR workspace=1，RR load_ready=1 |
| RR 第一次 Q | 5642 / 47.0167 s | 向上净空增量 34.5458 mm，own joint motion 10.6450°，当时 above-top clearance +0.5661 mm |
| 第一次资格撤销 | 7565 / 63.0417 s | GROUND before crossing |
| RR 第二次 Q | 7774 / 64.7833 s | 向上增量 47.6246 mm，joint motion 12.1032°，clearance +0.0846 mm |
| 第二次资格撤销 | 7917 / 65.9750 s | GROUND before crossing |
| RR 第三次 Q | 8125 / 67.7083 s | 向上增量 17.1745 mm，joint motion 8.5547°，clearance +0.0378 mm |
| Deadline | 8152 / 67.9333 s | 第三次 active qualification 尚在，但无 RR cross/place |

终局 `lift_attempt_events` 包含三次 Q 和两次撤销；只读 `history.event_ticks` 会只看到首次 Q5642，不能据此省略后两次尝试。此前 FR Q/C/P=51/1508/1528，FL=1608/2581/2713；FL 历史 placed 不等于 P09 当前一直有支撑。whole-body initial 事件仅为诊断，不另算 hard qualification。

## 3. 实测接近、抬高与回退

P09 的 450 个决策结束样本中：

- **最近前沿：tick5864，48.8667 s，global59794，RR front −31.0573 mm、clearance +8.1393 mm，AIR 且 Q=true。** FL 此时真实 TOP、load fraction 0.0335444。
- **最高净空：tick5800，48.3333 s，global59786，clearance +30.6526 mm、front −55.0412 mm。** FL 此时 TOP、load fraction 0.0655756。
- 两个极值不在同一状态，不能合并成“距离最近且净空最高”的合成状态。最近样本仍未到前沿平面。
- 主接近在 tick5864 达到全回合最近点，随后 5872/5880/5888/5896 的 front 为 −33.1702/−37.0235/−38.4107/−41.4601 mm；是回退拐点，但之后有再接近，不是单调回退。tick6608 再到 −39.8042 mm、净空 +20.3097 mm。
- 最后一个 front 优于 −70 mm 的样本是 tick7008（−68.7925 mm）；此后一直未回到该描述性范围。该 −70 mm 只是报告分段，不是新判据。

| Tick | RR front mm | RR clearance mm | RR 当下 | FL 当下 / load fraction | Body forward mm |
|---:|---:|---:|---|---|---:|
| 4552 | −208.2510 | −50.0893 | GROUND，无 Q | AIR / 0 | 141.9611 |
| 5648 | −53.5935 | +5.4396 | AIR，Q | TOP / 0.02066 | 159.5693 |
| 5800 | −55.0412 | +30.6526 | AIR，Q | TOP / 0.06558 | 134.1032 |
| 5864 | −31.0573 | +8.1393 | AIR，Q | TOP / 0.03354 | 159.0393 |
| 6560 | −57.6171 | +20.9158 | AIR，Q | TOP / 0.07200 | 146.2254 |
| 7048 | −71.3954 | +17.0733 | AIR，Q | 该段并非始终无接触 | 136.6137 |
| 7560 | −142.5543 | −42.7989 | AIR，首次 Q 尚在 | AIR / 0 | 76.7604 |
| 7872 | −184.6967 | −20.5644 | AIR，第二次 Q 尚在 | AIR / 0 | 56.6311 |
| 8152 | −162.2198 | −0.6532 | AIR，第三次 Q 尚在 | AIR / 0 | 54.9766 |

7048→7560 的明显后退伴随 body forward 136.6137→76.7604 mm，而不只是腿部相对身体缩回。P09 早期 body forward 最大为 tick4624 的 213.1143 mm；终局为 54.9766 mm，相差 −158.1377 mm。仅是测量关联，不能从这些坐标反推出某一轮、关节或接触的独立因果。

FL 在 P09 的 450 个决策结束样本中，AIR 244、TOP 206、正载荷 206，最大载荷比例 0.389516。最高净空及最近前沿时 FL 都有实测 TOP 接触，因此“RR 未完成必然是 FL 全程没有支撑”不符合日志。晚期多个回退样本 FL 确实 AIR/零载荷；它是同期实际接触状态，不是由 CoM 位置猜测。

末态 FR TOP/load 0.498666，RL GROUND/load 0.501334，FL/RR AIR/load 0；body speed 0.0915644 m/s、body omega 0.0995062 rad/s，最大实测轮速 0.472432 rad/s。这不是全腿当前 TOP 或稳定完成状态。

## 4. Nominal geometry：执行分支与实际目标

450 个 P09 决策结束 receipt 的分支计数：

| 分支 | 样本数 |
|---|---:|
| identity_within_descent_allowance | 180 |
| projected_exact_forward | 91 |
| projected_relaxed_forward | 96 |
| degraded_bypass_infeasible_box_downward | 21 |
| degraded_bypass_infeasible_nonreversal | 3 |
| no_eligible_context | 59 |

391 个有 context 的样本均满足 source_control_tick=decision-end tick−1、dispatch tick 与 actual audit tick 一致，actual_mapping_matches_dispatch=true、setter_dispatch_targets_equal=true，绑定异常 0。它证明所记录投影目标确实按同 tick 路径下发，不证明任务完成；表也不是全部 3600 个 P09 子 tick 的分支直方图。

tick5800 最高净空和 tick5864 最近前沿均为 projected_exact_forward。后退早于晚期 degraded 分支：tick7000 后第一个记录到的降级样本在 tick7488（infeasible_nonreversal，front −124.5775 mm，clearance +1.6899 mm），不能把从5864/7008开始的回退全归于降级。tick7872 是 infeasible_box_downward、geometry correction 为零；这是如实旁路，不是仍保持几何投影保证。

### 轮目标与 RR residual 的时序

以下轮数组顺序为 **FL/FR/RL/RR**，单位 rad/s，均为 canonical **目标**，不是实测轮速。Actual=nominal+当前 projected residual（仍受既有映射/限速）；native float32 到达另由审计确认。RR residual 为 hip/knee canonical degrees。

| Tick | 四轮 nominal（每轮） | Actual wheel targets | RR residual hip/knee ° |
|---:|---:|---|---|
| 5856 | 0.300 | [0.6818, 0.0235, −0.0019, 0.0944] | 此行仅定位轮目标 |
| 5864 | 0.150 | [0.4118, −0.1194, −0.2719, −0.0007] | [−7.1519, −20.2000] |
| 5872 | 0 | [0.1774, −0.2320, −0.3997, −0.1447] | 此行仅定位轮目标 |
| 6560 | 0.300 | [0.5567, −0.0101, −0.0057, 0.2601] | [−8.8610, −20.2000] |
| 7000 | 0.300 | [0.5656, 0.1936, 0.0700, 0.2296] | [−8.3875, −9.5938] |
| 7560 | 0 | [0.2244, −0.1381, −0.2787, −0.0279] | [−4.9163, −20.2000] |
| 7872 | 0 | [0.2179, −0.1424, −0.3605, −0.0145] | [−4.0903, −10.0673] |
| 8152 | 0 | [0.2032, −0.1423, −0.3588, 0.1710] | [−8.1623, −16.2000] |

5648 之后上述样本的未几何修正 RR nominal 已为 [−6.9,−37.8]°，但 geometry correction、residual 和最终目标仍可变化。大幅 canonical inverse correction（例如 −40°）不等于瞬间关节跳变，实际 final targets 仍经既有 slew。

当前 `NominalMotionProvider._continuous_advisory` 在 P09/P12 对已有 active lift、当前 clearance≥0.015 m、front<−0.005 m 的后腿复用四轮 +0.3 建议；外层保留 3 rad/s²（120 Hz 下 0.025 rad/s/tick）的 slew。有限 source 已结束之后也可由该反馈再提出建议。5856→5864→5872 的 .3→.15→0 与该净空反馈及 slew 相符，**不是这时 P06 finite source 才结束**。决策结束观测晚于各 tick 的 command source，不能用这些 15 Hz 点宣称精确的阈值切换 tick。

P09 最后一个非零 nominal wheel 的决策结束样本为 tick7824（每轮 .175），其后直到8152 nominal wheel 均为零；但 actual wheel 仍因 residual 非零而非停轮。未把“有限 nominal 已为零”误判为无策略动作或 codec 问题。

## 5. 可严格分离的只是同状态一阶目标模型

用每个已保存 context 的 RR 两关节实测 q、固定基座 wheel-link Jacobian J，以及三个同状态、同 float32 dtype、经过实际 clamp/final slew 的目标分支：

- raw：`raw_nominal_native_targets`，原 nominal+冻结 controller、当前 PPO residual=0；
- geom：`geometry_nominal_native_targets`，几何修正 nominal+controller、当前 PPO residual=0；
- actual：`actual_native_targets`，同一路径加当前 PPO residual。

分别计算 `N=J(raw−q)`，`G=J(geom−raw)`，`R=J(actual−geom)`，故按定义 `N+G+R=J(actual−q)`。N 已包含冻结 controller，不能再称纯 source nominal。下表是 x/z **线性目标位移预测（mm）**，不是下一 tick 实测位移。

| Tick | N_x | G_x | R_x | N_z | G_z | R_z |
|---:|---:|---:|---:|---:|---:|---:|
| 5800 | −0.3336 | ≈0 | +0.3172 | −17.6571 | +1.9997 | +6.6078 |
| 5864 | −0.4466 | ≈0 | +0.4093 | −3.3591 | +3.3591 | +5.4080 |
| 6000 | −1.4468 | 0 | −1.8864 | −0.5618 | +0.5618 | +6.1699 |
| 6560 | −2.1578 | ≈0 | +2.6599 | −5.5911 | +0.0313 | +8.3796 |
| 7872 | −13.6215 | 0 | +2.8468 | −19.3716 | 0 | +1.0831 |

这能够说明：当前 residual 对一阶 x 的方向不是始终一致（例如6000为负、6560为正）；geometry 的名义保护并不保证全部 PPO 后的 full-body 行为；较高净空时投影可允许有限下降，故 projected 分支的 N_z+G_z 不必恒为零。

不能从该分解推导真实物理因果：模型只含同状态 RR 两自由度，忽略基座运动、其余腿、轮/接触耦合、动态跟踪及非线性余项。实际 q 和 previous target 已来自此前 PPO 轨迹；某 tick 因 final slew 得到 R=0，也不能解释为该 tick 的历史状态不受 PPO 影响。没有运行去掉 residual 的同初态反事实 rollout。本回合还跨真实优化更新，不能当成单一冻结 deterministic policy 的配对试验。

## 6. 结论与停止范围

日志支持的核心事实是：RR 在有效物理状态下三次取得真实抬腿资格，并曾同时有正净空与较近前沿，但最近记录仍在前沿后31.06 mm；随后伴随身体后退、轮目标正负混合、FL 间歇失载和两次落地撤销，最终 P09 deadline 未完成。几何投影既有可行分支也有诚实降级，早期最近点与后退不能简化为“投影未执行”或“FL 从未承载”。现有证据不足以为某一控制分量分配独立物理因果。

本报告不重分类失败、不恢复旧入口/clock 门、不要求新成功门、不预选下一参数或 nominal 修订；不将该回合计为 suffix/full success，也不将仍在运行的8192计划记成已完成。分析到此停止。
