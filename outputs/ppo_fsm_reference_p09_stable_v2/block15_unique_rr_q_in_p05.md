# Block15 唯一 RR Q 发生在 P05，并在进入 P06 前真实回地撤销

范围：已完成 run `train/20260910T1631133451770Z_gd4e46006b382_079314d0fcb0460bbd585da6715d4cb9`。复用之前报告的字节边界；2026-09-10 **17:16:01.225Z** 仅一次读取 `residual_and_projection_audit.jsonl` 的剩余 **bytes[78,475,261,146,295,300)、行1139–2048（910行，无半行）**。未重扫前1138行，未读physical流、checkpoint或活动视频。

目标是 **ep6：行1426–1750，global153618–153942，325 decisions**；source阶段 P01=2/P02=137/P03=3/P04=1/**P05=126/P06=56**，P07以后为0。

## 唯一 RR 尝试

真实事件为 **I1805 → Q1813 → 1848回地撤销**，均在P05。首次Q记录是行1652/global153844的 `history.lift_attempt_events`：

`qualified_measured_upward_lift, physics_tick=1813, upward_excursion_m=.008186675258457704, joint_motion_deg=2.2681931252399288, airborne_clearance_above_top_m=−.04240520274746905`。

该决策的1809–1816八个native source ticks均为P05、verified=true，故不是按末态标签猜发生阶段。`current_lift_valid=true` 只出现在 **1816/1824/1832/1840** 四个决策端点，AIR、motion_continuation_allowed和body_control_evidence均真，实际其他支撑证据为 **FR/RL**。Q成立是当时的功能状态，不是以AIR或大抬升单独替代可控证据，也不保证未来维持。

Q后的1816端点，RR front **−608.730955mm**、台面gap **−41.722119mm**；RR自身近期关节响应2.707433°、全身近期关节响应20.212217°。本次确有自身及全身响应，但这些量不能单因果归因，更不是以后必须先动RR自身关节的门槛。

1848出现原始 `qualification_revoked_ground_before_cross` 与 `current_lift_revoked_ground`（同一次物理回地的两个记录，不算两次撤销）。当前mode=GROUND，I/established/current均false，front **−601.250571mm**，bearing **22.878073N**。首Q event_ticks.RR=1813仍保留为历史，不能用历史tick冒充当前资格。全回合RR **C/P均无**。

## 为什么有 Q 却没有 P07

1. **Q发生时仍在完成FL任务。** P05只以真实`placed_FL`完成；RR Q不能替代FL越沿放置。1816/1824/1832时FL为OBSTACLE_AMBIGUOUS、within_top_xy=false、support=false、bearing_verified=false；不能把其记录的0受力当“已测得零载荷”，也不能虚构它在承载。FR TOP与RL GROUND是可验证支撑。1840时FL已AIR，仍不承载。FL当时还没有C/P。
2. **等FL完成时，RR的早期Q已失效。** FL新Q1834后C1963、P2150，**P05→P06在2152**。此时FL真实TOP、within_top_xy/support/bearing_verified真，bearing **2.774488N**；RR却已GROUND/currentQ=false，front **−577.593290mm**。并非阶段切换把有效Q清掉，而是早在1848回地撤销。
3. **P06正常接管后，实际后腿前送仍未达成。** 56个P06样本均current RR Q=false，`completion_values.rear_approach`始终0；RR/RL最接近前缘的记录仍分别为 **−463.417974mm / −462.358725mm**。生产`rear_approach=min(edge_proximity_RL,edge_proximity_RR)`，当前接近目标没有满足；当前有效RR抬升可以连续接管P06，但本回合这时已不存在。没有固定停稳、重抬或历史姿态门槛造成的Q拦截证据。

## 连续性与结尾

P05→P06为普通非done交接；下一决策的bridge保存全12 residual，无禁止通道丢弃/phase范围裁剪，最大servo/wheel residual差约3.55e−15/1.11e−16。全回合 **2595/2595 native ticks verified、2590 own-phase effect**，无episode内state write。Q发生在更早P05，因此没有“跨P06→07丢失Q”的原始事件需要补造。

末态 **tick2595 / 21.625s，P06 FALL**。FL历史P保留，但当前已GROUND、非TOP、within_top_xy=false，bearing14.734474N；FR AIR不承载，不能沿用早先双前腿顶面支撑。终态372无finite fallback，按当前schema与锁定障碍物底面回解float32 base_z≈**14.949568mm < 15mm**，gravity_z≈−.967559；保留真实安全失败，不松阈值。

另一个RR I位于 **ep7，行1918/global154110，P05 tick1339**，gain3.051701mm；该回合末历史只有这一I，未Q/C/P。这样尾段原始事件与全块 **I2/Q1/当前有效端点4/回地撤销1/C0/P0** 一致。

结论：全身协同在尚未完成FL放置时确实产生过有效RR抬升，但没有保持到可接管的后腿阶段，也没有形成前缘越过或放置。本证据不说明固定策略改善、不推断单一失控原因；未发现新的明确执行或判定缺陷，不生成新训练门禁。
