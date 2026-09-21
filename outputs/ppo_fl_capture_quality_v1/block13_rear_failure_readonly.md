# Block13 两次后腿碰障：有界只读诊断

仅读取新run `20260918T0953571182841Z_ge73542cb57ad_fb762812ab4e4b52a77c8707d38e5d38` **首679条learner记录，止于global184999**；每个episode取首次P09、碰障前完整decision、terminal共3帧。没有读取第三个活动前缀/episode尾部、没有改代码或启动物理/optimizer。完整12通道请求与执行证据在[JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/block13_rear_failure_readonly.json)。

## 结论先行

不是“只有FR新噪声仍在破坏任务”或“RR始终无法抬起”的证据。第一episode真实RR自由抬升Q6528，随后6557触地撤销，尚未越沿就失去carry；第二episode没有Q。碰障前两次N的P09 source clock都在80等待 `current_free_lift_before_pending_knee`，这是对**当前RR未保持自由抬升**的正确保护，不应通过放宽资格或强行释放后续膝/轮动作解决。

同一实际前态中，仍可见多通道策略修正：RL hip抵消部分正向N辅助、RL knee保持正偏置，RR hip正偏置而RR knee负偏置；N四轮已停时policy仍请求四轮差速。这些变化不仅存在于单次随机创新，也存在于base输出和含HISTORY的conditional mean，**但conditional mean绝不能全称为网络学得的base mean**。这些是应继续分析的实测动作分配，不足以从6帧判定单一致碰动作或新修复方案。

## 实际结果与源动作状态

- 第一episode：439 learner决策，FL placed5415；首次P09请求global184705/end6264。RR initial6518、自由抬升qualified6528（unsupported rise 8.343956mm），6557 `qualification_revoked_ground_before_cross`；global184759/tick6694（55.783333s）机身/障碍碰撞，RR未crossed/placed。
- 第二episode：240 learner决策，FL placed3230；首次P09 global184958/end4776；RR无qualified事件；global184999/tick5098（42.483333s）同类碰撞。terminal的2个AIR样本/1.363922mm上升不能称自由抬升资格。
- 碰障前tick6688/5096均为VERIFIED：FL AIR不承载；FR在TOP承载11.101/12.428N，RL在GROUND承载13.926/12.684N，RR仍GROUND承载3.512/4.928N。不能把FL在空中留空间说成FL承载。
- 第一terminal为CONTACT_BEARING_UNVERIFIED且support=0；不能据这条包推断四腿物理上同时失去支撑。前一完整VERIFIED帧才用于上面的载荷说明。

两次前terminal，P07/P08源层已执行，P09 current-free-lift guard在source80正确hold；RR hip N仍是55.6°，待发knee waypoint尚未获准。普通P09入口都有1个前阶段原目标保持tick，之后7个当前请求tick；未见跨阶段清零/重抬的证据。

## 碰障前全8关节：同次派发账本

每格为 **mapped/geometry N / filtered REQUEST / final / 下发前实际q**，单位deg。源nominal向量两次均为
`[38.6,-13.4,0,31.1,28.2,0,55.6,0]`，顺序FL hip/knee、FR、RL、RR。N与final对照是当拍执行账本；严格同前态反事实只用JSON中已记录native counterfactual及其delta，不能把另外一条zero同tick代入。

| 关节 | episode1 tick6688 | episode2 tick5096 |
| --- | --- | --- |
| FL hip | 38.600 / -5.385 / 33.215 / 33.526 | 38.600 / 6.825 / 45.425 / 46.457 |
| FL knee | -12.150 / -2.396 / -14.546 / -18.408 | -12.150 / -6.062 / -18.212 / -17.512 |
| FR hip | 0.000 / 1.028 / 1.028 / 1.950 | 0.000 / 2.159 / 2.159 / 2.088 |
| FR knee | 32.805 / 1.683 / 34.487 / 32.724 | 32.641 / 3.388 / 36.029 / 37.032 |
| RL hip | 28.200 / -4.965 / 23.235 / 22.802 | 28.200 / -4.471 / 23.729 / 24.166 |
| RL knee | 0.000 / 7.592 / 7.592 / 5.401 | 0.000 / 7.030 / 7.030 / 4.151 |
| RR hip | 56.965 / 10.996 / 67.961 / 64.499 | 61.850 / 7.605 / 69.455 / 62.640 |
| RR knee | 0.000 / -13.131 / -13.131 / -12.548 | 0.000 / -8.753 / -8.753 / -9.150 |

FR knee在这两帧仅有+1.683/+3.388°REQUEST，final34.487/36.029°，不是旧cap扩大后几十度的创新跳变；不能因此断言FR一轴无影响。RL hip final23.235/23.729°低于N28.2°；RL knee final7.592/7.030°而N0。RR目标/实际跟踪还分别存在约3.462/6.815° hip差，ACK通过不等于物理完全跟踪。

上述actual q由日志每轴 `nominal_deg−current_actual_canonical_error_deg`计算，JSON保留native rad；它在最后派发前测得，接触/速度是decision结束后的状态，两者并非同一积分时刻。训练日志没有这几帧的body z、CoM高度或机身碰撞盒垂直余量，**没有伪造“下沉多少”或由关节角直接断言缩短支撑高度**。

## Base、HISTORY、conditional与创新分开

下表为同两帧；前三列是raw latent，创新为实际raw sample−conditional μ；“μ物理”仅cap×tanh(μ)的未滤波参考，不是另跑policy的闭环动作。REQUEST是实际过滤/投影后的记录。

| episode / 轴 | base μ | HISTORY center | conditional μ | raw创新 | μ物理 deg | REQUEST deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / RL hip | -0.412 | -0.246 | -0.263 | 0.053 | -6.160 | -4.965 |
| 1 / RL knee | 0.194 | 0.192 | 0.192 | 0.022 | 6.834 | 7.592 |
| 1 / RR hip | 0.449 | 0.300 | 0.315 | 0.187 | 7.322 | 10.996 |
| 1 / RR knee | -0.367 | -0.404 | -0.401 | 0.018 | -13.698 | -13.131 |
| 2 / RL hip | -0.225 | -0.176 | -0.181 | -0.008 | -4.287 | -4.471 |
| 2 / RL knee | 0.215 | 0.150 | 0.157 | 0.041 | 5.592 | 7.030 |
| 2 / RR hip | 0.335 | 0.315 | 0.317 | 0.011 | 7.357 | 7.605 |
| 2 / RR knee | -0.276 | -0.267 | -0.268 | 0.020 | -9.429 | -8.753 |

RL hip/knee与RR hip/knee的符号在base和conditional均存在，但HISTORY占.9，且随机创新、残差rate限制、N补偿及跟踪各自仍有作用，不能把全部实际偏置归为“学到的均值”。第一帧RR hip正创新较明显；两次RL hip负REQUEST均不是因残差mask或名义N丢失。FL的请求也显著不同（第一前terminal hip−5.385/knee−2.396，第二+6.825/−6.062°），因此不存在已证明的唯一同轴失败模式。

## 四轮：N停止后策略仍有动作，不是mask

按FL/FR/RL/RR顺序，目标为canonical rad/s、实测为评价器原记录rad/s；native actuator轴符号另存JSON，不由正负号直接推断各轮地面牵引。

| episode / end tick | N target | 最后final target | 实测角速度 |
| --- | --- | --- | --- |
| 1 / 6264 | [0.295, 0.295, 0.295, 0.295] | [0.278, 0.334, 0.603, 0.189] | [0.278, 0.053, 0.567, 0.189] |
| 1 / 6688 | [0.000, 0.000, 0.000, 0.000] | [0.665, -0.038, 0.391, -0.079] | [0.666, -0.588, 0.175, -0.075] |
| 1 / 6694 | [0.000, 0.000, 0.000, 0.000] | [0.575, -0.027, 0.307, -0.075] | [0.574, -0.028, 0.305, -0.074] |
| 2 / 4776 | [0.206, 0.206, 0.206, 0.206] | [0.670, 0.344, 0.365, 0.092] | [0.670, 0.287, 0.424, 0.057] |
| 2 / 5096 | [0.000, 0.000, 0.000, 0.000] | [0.240, 0.212, 0.318, -0.077] | [0.238, 0.301, 0.272, -0.056] |
| 2 / 5098 | [0.000, 0.000, 0.000, 0.000] | [0.210, 0.211, 0.316, -0.074] | [0.210, 0.165, 0.209, -0.075] |

入P09时N仍有四轮前送；到两次碰障前N均0，但policy仍保留四轮非零差速。例如第一前terminal FL+.665/RL+.391、FR−.038/RR−.079，FR实测−.588与其目标并不相等，可能包含接触/负载/滚动耦合；没有把它称为漏写或被屏蔽。合法源stop与独立安全stop不是同一语义，本审计不据此永久禁掉wheel residual。

6帧所有许可mask=1、headroom clipped_indices为空；所含40个native tick全部verified，最后setter与实际native缓存一致，未观察到错误索引或后写覆盖。第二terminal首tick有11个而非12个counterfactual差异通道，但它仍verified；“有一通道当拍native差异为0”不等于mask。最多能排除这些已检查帧的接线异常，不能把未采样时刻或物理跟踪一并认证。

## B_control的事件对齐参考，非同拍反事实

保留成功B `20260918T0555572376368Z_gf2e552406ea7_1894338b350241fcbbac4868c5f7f3fb`（73.808333s）。仅取其P09入口5184、RR Q后的5440、RR越沿/放置后的6160三帧；不是新B重跑，也不把f2e与e735 kernel的版本差异隐藏。

- 入口：同样FL AIR、FR TOP、RL/RR GROUND，N四轮.3，8关节final `[46.100, -12.150, 0.000, 45.959, 6.900, 0.000, 0.000, 0.000]`；所以“FL在P09入口AIR”本身不是失败。
- RR Q后5440：source P09才33tick，RR自由AIR且current_lift_valid=true，FR/RL分别12.639/15.785N支撑。final `[38.600, -12.150, 0.000, 29.994, 28.200, 0.000, 13.800, 0.000]`，轮目标全0，RR在无额外policy偏置的真实轨迹继续抬升。不能用这组不同状态关节差值当作当前policy每轴因果效应。
- 6155实际RR跨沿且放置；6160进入P10，RR current_valid=true、TOP承载9.688N。其P09 source已到654而不是卡在80。这说明成功参考保持了可持续carry/前送，不能证明照抄其关节数值是唯一解决办法。

本报告只给已有证据和缺失项；**不改qualification，不改reward/mapper/支持组合，不凭6帧调整动作。** 后续更改应等当前run合法封存，再由根任务结合完整可用证据决定。

