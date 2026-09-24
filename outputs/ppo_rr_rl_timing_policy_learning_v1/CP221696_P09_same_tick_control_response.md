# CP221696：P09 同拍控制响应（只读）

封存 source：`runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260923T2245371502769Z_gd7e97ee7b7e4_083d82482a6541f8a129462a60dd3dd0/source`。仅用 base Python 标准库／PowerShell 读取此 run，无模型、仿真或配置修改。

**结论：此窗口四轮都有指令并实际转动。FL nominal 的正向前送被 policy 负修正抵消、反转；不是 nominal 被 mask，也不是只给 RR 派发。RR 悬空轮转动不等于提供牵引。此证据不能单独证明 FL 反转是 RR 悬停的唯一原因。**

## 精确 plateau：episode tick 9656，80.4666667 s

对应 policy decision 1207，输入 tick 9648，作用区间 9649–9656；native dispatch tick **9835**（settle/prime 时钟偏移），物理读回 episode tick **9656**。以下 canonical 四轮顺序 **FL、FR、RL、RR**，单位 rad/s；raw 是无量纲 conditional mean，**不是**最终执行目标。

| 轮 / native ID | 源 N / 同拍 mapped N | 原始 raw / tanh | residual 许可 / scale | 原始 REQUEST = 本拍有效投影修正 | FINAL canonical | native 最后写入目标 | 步进后实测 canonical qdot | 当前接触 / bearing N |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FL / 8 | .300 / .300 | −1.245439 / −.847000 | 1 / 1.2 | −1.016399709 | −.716399709 | +.716399729 | −.770530343 | TOP / 1.756256，verified |
| FR / 9 | .300 / .300 | −.039038 / −.039018 | 1 / 1.2 | −.046821620 | +.253178380 | +.253178388 | +.114122443 | TOP / 13.475726，verified |
| RL / 10 | .300 / .300 | −.026807 / −.026801 | 1 / 1.0 | −.026800747 | +.273199253 | −.273199260 | +.086891502 | GROUND_AND_OBSTACLE / 10.612465，bearing 未验证 |
| RR / 11 | .300 / .300 | −.260173 / −.254458 | 1 / .6 | −.152674516 | +.147325484 | +.147325486 | +.146302864 | AIR / 0，非承载 |

native 轴号／符号不可与 canonical 方向混同：FL、RL 的 native 目标符号反转。IDs 来自本 run `height_diagnostics_startup.json` 的真实 `robot.joint_names` 顺序与按名称建立的 joint map。同拍审计读取 `robot._joint_vel_target_sim`（RobotAdapter 的单次 `write_data_to_sim` 之后、physics step 之前）；`setter_dispatch_targets_equal`、`actual_mapping_matches_dispatch`、`verified` 均 true。writer 为 `RobotAdapter.set_joint_velocity_target(...joint_map.wheel_ids)`，本拍不存在第二次执行器写入或 RR/轮辅助 owner。

全 12 residual 许可为 1，capture-owned 12 通道全 false，rear task assist OFF。表中有效修正使用 **同拍 `policy_headroom_evidence.effective_policy_residual_full12`**；它与实际 REQUEST 完全相等，candidate 与 FINAL 相等，本拍没有限速／投影再改变四轮。并非用独立重算 N 与 FINAL 做差猜测 PPO。HISTORY 已在原 raw conditional mean 内；前一拍请求与当拍请求均独立保存。直接 readback 也证明后三轮并未被屏蔽；FR/RL 实测低于 target，不据此断言是 mask 或特定负载原因。

RR 源 hip/knee = **−6.9/−37.8°**；同拍 mapper 为 **−8.15/−39.05°**；原 REQUEST = **+17.523388/−17.771172°**；FINAL = **+9.373388/−56.821172°**，实测 **+9.104713/−56.592522°**。本拍无 servo headroom clipping。RR 当前 Q=true、cross=true、AIR=true、TOP=false、force=0，gap **44.220213 mm**，front **99.878076 mm**；P09 source clock **5.4 s / 648 ticks**，等待真实 RR bearing，late group 尚未放行。与“绝对 hip −20°附近、近保持 FINAL knee”不是同一个候选动作。

## 抬升、越沿、等待与反转的先后

| 事件（episode tick / s） | RR gap / front mm | RR hip FINAL / actual ° | RR knee FINAL / actual ° | 说明 |
|---|---:|---:|---:|---|
| P07 入口 7008 / 58.4000 | −50.061 / −224.727 | 16.928 / 16.920 | −18.181 / −17.899 | N 四轮 +.3，FL REQUEST −1.004771，FINAL −.704771、实测 −.705506，反转已存在 |
| 首次读到 AIR 7077 / 58.9750 | −50.952 / −207.931 | 17.276 / 17.217 | −18.338 / −18.041 | 不是合格 lift；Q=false，不能把接触瞬失当有效抬升 |
| 本次 lift 资格事件 7277 / 60.6417 | −41.253 / −208.147 | 32.727 / 24.386 | −18.163 / −18.070 | Q=true，尚未越沿 |
| 首 late wait 7904 / 65.8667 | 46.677 / −39.550 | 9.192 / 8.930 | −57.041 / −56.819 | source clock 到648后等承载；此 endpoint 的新 N 建议+.3，但刚执行的 mapped N 仍0，不能混为同拍实际 PPO 抵消 |
| 首 RR 越沿 8356 / 69.6333 | 46.324 / +.0127 | 9.246 / 8.984 | −57.064 / −56.847 | Q/cross=true，仍 AIR、无 TOP；N 四轮+.3，FL FINAL−.713524 |
| plateau 9656 / 80.4667 | 44.220 / +99.878 | 9.373 / 9.105 | −56.821 / −56.593 | 仍等待真实接触承载 |

有界 P07→plateau 窗口中，反转最迟在 P07 入口就已存在；没有把该点冒称整集最早起因。P09 **7024–9656 的 2633 个 physics endpoint，FL FINAL 连续为负**（约21.94 s）。这早于 P09 源 wheel stop：endpoint7896/65.8 s 新 N 已为0，而同拍已执行 mapped N 仍+.3；**7897** 首次实际 mapped N 四轮0，**7905** 又实际+.3。源停止／新建议与实际执行有一拍因果次序，原始日志保留，未把这种合法延迟算成 policy 修正。plateau 前 N 可在0／+.3之间切换；无论该变化，已记录的 FL 负 REQUEST 约−1.01 足以压过+.3。

来源：同 tick `capture_assist_ticks.jsonl`、`native_tick_audit.jsonl`、`physical_observations.jsonl`；对应 `video_policy_decisions.jsonl` 的 request/nominal-provider diagnostics 与 lift/cross history。未独立重放 mapper，未推断未记录的轮地牵引方向；CoM 未用新的假定坐标轴计算。
