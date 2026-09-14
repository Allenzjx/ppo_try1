# Video11：P02 实测 RR 膝关节硬限中止

## 范围与真实结果

仅审计完成 run `video_eval/validation/20260911T0130039409655Z_gd4e46006b382_4c377fe16f9a405ea1289f599bc2a6f6`。2026-09-11 01:44:42 UTC 一次读取 source/video_policy_decisions.jsonl 的全部 165 行（文件 9,998,531 bytes）和 source manifest 选定字段；另各一次读取 physical/native 文件末尾 128/64 KiB，仅解析最后一行。未读 CP、媒体、巨型质量数据或其他运行。

确定性自然 P01，源 CP164736 的加载身份由主控独立核验，本诊断不重复加载。165 issued = P01 2 + P02 163；前 164 次返回、各执行 8 ticks，末次 P02 在 1312→1317 实际执行 5 ticks 后抛 PhysicalEndpoint（returned=false，无 step_info）。总 1317 ticks / 10.975 s，0 optimizer updates。真实结果是 `SAFETY_ABORT / HARD_JOINT_LIMIT / hard joint limit: rear_right_knee`，物理 VALID/VERIFIED，生命周期 DIAGNOSTIC_FAILURE；不是任务超时、录像裁切或成功。

## 决定性实测与命令证据

末 physical observation 与末 native audit 的 episode tick 均为 1317。物理 all_finite=true，RR knee 规范实测角 **−60.01819305141322°**、速度 −3.431128240639569°/s，命令 **−3.3718302922432346°**。相对实际膝关节下限 −60°，越出 **0.01819305141322°**；命令并未越限。实测与命令差 56.64636275917°，不能称为命令舍入或仅小幅目标超调。

| episode tick | RR knee 最终目标 ° | RR knee 实测规范角 ° | 实测来源 |
|---:|---:|---:|---|
| 256 | −3.305823 | −11.095250 | 当拍 measured joint-range margin 回解 |
| 768 | −3.955032 | −33.460211 | 同上 |
| 1024 | −3.864810 | −51.531526 | 同上 |
| 1312 | −3.362310 | −59.815418 | 同上，下限余量 0.184582° |
| 1317 | −3.371830 | −60.018193 | 原始 physical observation 双精度字段 |

所摘取轨迹端点的 RR knee nominal 均为 0°；全部 164 个正常决策的最终目标范围为 −3.993678…−0.413199°。实测持续远离目标是本次可证明的跟踪失败；上述记录不足以把原因唯一归给惯性、载荷、某个关节命令或策略某一通道。

末 native `verified / setter_dispatch_targets_equal / actual_mapping_matches_dispatch` 全 true，Full12 mask 全 1，实际来源是已有 write_data_to_sim 后的机器人 native target buffer。mapper nominal 的 RR knee 为 0°，combined post-mapper bias 为 −3.371830292°，合成后与原始 physical command 一致。不得把 nominal 当最终目标，或把 native 关节坐标直接当规范角。内部 native tick=1496 与 episode tick=1317 是不同计时域，未混作同一物理时间。

## FR 首个未完成任务与实际接触

新 FR I=17、Q=26；Q 记录实测 upward excursion 8.625359 mm。全段没有 FR C/P，也没有 FL 或 RR I/Q/C/P；RL 仅 I=7，没有 Q/C/P。P02 已有 FR 抬升，但净空与向前接近尚未满足：最后正常端点 completion_values 为 lifted_FR=1、clear_FR=0、approach_FR=0.777143；精确终态 FR 距前缘 **−65.211144 mm**、台面净空 **−7.523211 mm**，AIR、无 top contact、不在 top XY、support=false、有效载荷 0 N。首个未完成任务是 **FR 保持所需净空并接近前缘（P02），随后越沿/放置尚未发生**，不是把早期 Q 当越沿。

末 RR 为 GROUND、support/bearing_verified=true、15.966395 N、载荷比例 0.494351，有效；距前缘 −621.955645 mm、ground-relative lift=0、I/Q/current/C/P 均无。本次 RR 是实测硬限中止关节，不能因关节属于后腿就声称已进入后腿任务，也不能把瞬时承载证明为长期稳定。

## 连续性与证据边界

唯一普通转换在 tick16，P01→P02、termination=null。下一决策的 handoff receipt 显示 Full12 previous/carried residual 原值一致，forbidden/clipped channel 均为空，wheel residual jump=0，handoff_hold_used=true。tick16 与24 的 nominal wheel 都为 0，而 residual/final wheels 分别为 [−0.106801,−0.043823,−0.020401,+0.141174] 和 [−0.143558,−0.061281,−0.027329,+0.194430] rad/s：没有普通切换清掉 residual/wheel。源新阶段 nominal 关节请求变化不等于 mapper 重置。

前 164 正常决策累计 1312 native ticks，1312 verified/actual-effect，1311 own-phase-effect；164 个端点均 mask 全 1、setter/mapping 一致、no_in_episode_state_writes_verified=true。精确末 tick1317 同样 verified、mask 全 1、四类状态/外力写入计数全 0。未逐个解析中间 1313–1316 的 native 行，因此不将末端核验冒充这 5 拍的逐拍独立证明。

**本有界审计未确认新的 dispatch、mapping、mask 或跨阶段残差清零缺陷。** 保留真实实际关节跟踪失败和物理硬限，不改变阈值，也不从一次确定性失败推出 FSM/PPO 稳定性优劣。诊断不是后续 optimizer 的启动门禁。
