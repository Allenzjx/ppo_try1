# v8 首个 P07 课程回合：RR 捕获后再失载（只读）

## 范围与结果

源：`train/20260923T1045146351727Z_gd1871df37d6e_0a956b4862894edb85aa7a239aa27f0f`；runtime `d1871df37d6ea909657511d0e43e7435198f6ccd`。从 CP221056 开始，successful_nominal 真实前缀 645 decisions / 43 s 不计 PPO 信用；随后是 stochastic learner，回合内发生 1693+ 更新，**不是同一固定 CP 的自然 P01 确定性成功**，不能用约 69 s 捕获时间与旧自然评估 122 s 直接比较。

本分析只顺序读第一回合 852 条到 global221908 / tick11975，99.7916667 s，P12 `INCOMPLETE_CONTROLLER_BLOCKED`；第二回合未读。保存 67–78 s 的 166 条紧凑端点及事件，分析新增 0 physics / 0 PPO / 0 actor forward。reader PID68348 exit0，1.73 s；随后只读产物的小 helper 全部 exit0，PID68348 已确认不存在。

JSON：`v8_first_suffix_RR_capture_loss_readonly.json`，SHA256 `e73f0676d4272df20c41dc871d06385de192da1899eaf2614410f598ed4b9da3`。原流已读前缀 114655170 bytes，SHA256 `c3c76d688a2fbae1edbb171eed82ed31a2c8b329ee91cce20bf134001be38107`；不是对仍活跃整文件做哈希。

## 接触与阶段：入口确实有承载，不是只凭历史放置

| 事件 | episode tick / s | 实际证据 |
|---|---:|---|
| 首个确认连续 RR TOP streak | 8283 / 69.025 | 8288 的 TOP count=6；8280 仍 AIR。8281–8282 的短瞬态不可由端点排除 |
| RR placed | 8284 / 69.0333 | history event 的精确 tick |
| P10 / P11 / P12 entry | 8288 / 8296 / 8304 | 三个端点 RR 都 TOP + support，力分别 1.236 / 3.987 / 7.712 N |
| 首次确认 AIR onset | 8401 / 70.0083 | 8400 TOP，8408 AIR count=8、0 N |
| 首次确认再 TOP streak | 8410 / 70.0833 | 8416 TOP count=7；8409 未记录 |
| 后续强承载恢复 | 8664–8816 端点范围 | 8760 RR TOP 11.390 N，FL/FR TOP，RL AIR |
| 后一次 AIR / FL 同时失载 | RR 8818；FL 8822 | 8824 RR AIR count=7、FL count=3；两者均 0 N |
| RR 明确重新接地 | 9153 / 76.275 | `current_lift_revoked_ground` 精确事件；9159 precontext 已 GROUND；9160 3.772 N 地面支撑，无 TOP |

历史 RR Q/cross/placed 被保留，但不代表当拍承载。`bearing_verified=true` 仅是读数可核验，AIR 时也可能为 true；本报告承载同时要求 support、非 AIR、接触类型及力，未单用该位。RL 两次 Q 在 8322/8667 建立，8355/8822 因重新接地撤销，始终未 cross/place。

## 四轮：先合法脉冲内失载，后源停止仍持续反转

顺序 FL/FR/RL/RR，单位 rad/s；N 为日志中最终 nominal 建议，REQUEST 为实际经投影/限速的 residual 请求，不是离线 N 差分。完整 raw12、mask、prestate、native float32、四腿接触均在 JSON。

| tick | N 四轮 | REQUEST 四轮 | FINAL 四轮 | 实测 canonical qdot | RR front / 状态 |
|---:|---|---|---|---|---|
| 8280 | 0,0,0,0 | −1.081,−.036,.021,−.084 | 0,0,.021,−.084 | −.005,−.155,.270,−.083 | 99.33 mm / AIR；P09 floor |
| 8288 | 0,0,0,0 | −1.106,−.077,.075,−.096 | 同 REQUEST | −1.011,−.215,.293,−.072 | 100.09 mm / TOP；capture/release 区间 |
| 8368 | −.3,−.3,−.3,−.3 | −1.017,.173,−.045,−.154 | −1.317,−.127,−.345,−.454 | −1.271,−.246,−.274,−.462 | 68.56 mm / TOP |
| 8408 | −.3,−.3,−.3,−.3 | −.780,.016,−.051,−.172 | −1.080,−.284,−.351,−.472 | −1.032,−.325,−.212,−.471 | 66.26 mm / AIR |
| 8632 | 0,0,0,0 | −.768,.060,.080,−.059 | 同 REQUEST | −.728,−.024,.217,−.058 | 45.03 mm / AIR |
| 8688 | .3,.3,.3,.3 | −.996,−.265,.048,.027 | −.696,.035,.348,.327 | −.427,.032,.349,.223 | 49.02 mm / TOP |
| 8720 | 0,0,0,0 | −.916,−.269,−.057,.012 | 同 REQUEST | −.851,−.215,−.055,.018 | 47.87 mm / TOP |
| 8760 | 0,0,0,0 | −1.088,.053,−.182,−.073 | 同 REQUEST | −.753,.037,−.181,−.138 | 40.34 mm / TOP |
| 8832 | 0,0,0,0 | −.909,−.228,.077,−.204 | 同 REQUEST | −.909,−.334,.205,−.205 | −.43 mm / AIR |

P12 authored −.3 pulse 在 8368 首端点可见，8624 仍有，8632 首端点已 stop；contract 的源时间是 .466667→2.666667 s。P12 入口 tick8304 后源第一步在8305，量化源 onset/stop 为8361/8625；这两个精确派发 tick 是现有 source clock 推导，端点只直接约束在前述 8-tick 窗口，未伪称另有逐 tick ACK。

8688–8712 的 +.3 **不是第二个 authored pulse**：`semantic_supervisor.py:2550–2557` 对 RL active lift/approach 的 P01-derived rolling 建议；FL 负 residual 把它取消成负 FINAL，实测也负。8720 再变 0 后，RR 在 .9333 s 内从 front +47.866 退到 −.431 mm；FL 负 FINAL 与实测持续存在。退回还伴随 RL 关节/承载和 RR captured-follow 变化，不能仅凭相关性宣称 FL 单轮是唯一力学原因。

native 示例 tick8760 wheel targets `[+1.088495016,+.052598342,+.182040259,−.073165677]`，与 canonical 符号映射 `[−,+,−,+]` 一致；mask12 全 1，最终独立写入证据为 `robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim`，verified=true。这些是实际 target，不是角速度；实测 qdot 另列。该流未保留 actuator joint IDs 与独立 native qdot，均不补造。

## RR 跟随及剩余 RL 节点

| tick | RR N hip/knee | RR REQUEST hip/knee | RR FINAL | RR actual 前一 tick | 含义 |
|---:|---|---|---|---|---|
| 8288 | −6.9/−37.8 | 20.787/−17.433 | .919/−36.823 | .690/−36.672 | HOLD 接触 |
| 8296 | −6.9/−29.3 | 20.541/−17.092 | .673/−32.032 | .459/−35.469 | CAPTURED_FOLLOW，P10 膝源实际展开；仍有跟踪过程 |
| 8304 | −6.9/−27.2 | 20.684/−17.480 | .816/−26.270 | .148/−32.796 | 源与 residual 增量连续进入，不是名义 P10 被吞 |
| 8400 | −6.9/−27.2 | 17.838/−14.158 | −2.030/−22.939 | .116/−23.367 | 最后 TOP 端点 |
| 8408 | −6.9/−27.2 | 18.932/−15.693 | −.935/−24.416 | −.588/−23.635 | 转 DESCEND，再捕获 |
| 8760 | −6.9/−27.2 | 15.383/−12.811 | −4.485/−20.317 | −6.811/−22.513 | 再承载 CAPTURED_FOLLOW |

当前 N、REQUEST、pending assist target、实际 FINAL 和 actual 不能混作同一量；后续 source/REQUEST 积分确实作用，不等于证明其方向始终有利。P10 source 膝 +10.6°、策略波动和全身脉冲共同发生，无法由此分离首失载单因。

8401 之后剩余 RL 源不是已全部结束：N 先保持 hip .5/knee35.3，8640/8648/8656 在 RR 仍 AIR 时 hip 新节点进入 1.6/4.8/19.6；再 TOP 后到30.2/31.2，knee依次30/5.6/−13.4/−18.7，hip返回到−10.1。8824/8832/8848 RR AIR 时仍有 hip10.1/7.9/−.5等节点。`_sequence_permission` 在非 P07–P09 直接 true（supervisor:2073），所以没有持续 RR current-bearing 的 RL 新卸载许可；但初始所有 entry 都有承载，入口 guard 本身不会改变这次首失载。

同期 N_FL 始终 −18.5/−31.4，FINAL_FL knee 长时间 −58；N_FR到3.7/31.1，真实请求使 FINAL_FR knee约−20…−35。是源、residual 与投影合成后的构型，不应把正/负角符号直接解读为力学好坏。没有从本精简流另造 mass-CoM/body 位移因果。

## 最小下一步候选，而不是已证修复

优先可以检验**已停止源后的 RR 接收/保持窗口**保留 current-bearing 支撑轮非负/适量前送包络：scope P10–P12、RR 当前合法 TOP 或合格合法 AIR、RL 未 placed；支持与几何不足则让权，fresh source owner 和合法 −.3 pulse/stop 优先。目标是这次已再捕获后 8720→8832 的持续反向，不声称解决发生在 −.3 pulse 内的8401首失载，也不能在 RR 已出 XY/已 GROUND 后靠历史 placed 强行推进。是否足够须同版本真实验证。

必要实现边界：现 `semantic_rr_carry_wheel.py:52–142` 的 source/ACK proof **固定 P09**，P12 authored pulse 活跃时仍显示旧 P09 held zero；不能只删 `P09_only` 就把旧 P09 stop 当作 P12 stop。必须辨认当前/未来相关 wheel owner，保持 P12 源脉冲与 stop 优先，再证明邻接实际 ACK 已提交适用 stop。当前 envelope P09-only 在 P12 AIR/再捕获时关闭是已证控制范围，不是 mask 丢失。

RL owner-lane 可作为更后续、有状态可观测迁移的方案：若要暂停，只暂停未消费的新 RL 卸载节点，保留 RR capture-follow、FR/FL准备、wheel stop；本次不应先扩张成整 P12 暂停或固定角度门槛。没有改 runtime/config/tests，没有新增门禁。
