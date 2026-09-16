# C172544：四轮 nominal 完整，策略抵消仍存在

C172544 自然 P01、all12 deterministic 正式评估在 **723 tick / 6.025 s / P02 RR knee HARD_JOINT_LIMIT** 结束：RR 目标 −4.347184°，实际 −60.025440°；FR 获得 Q（tick24），尚未 C/P，body collision=false。不是完整成功，也不是后腿失败。

本分析仅使用已封存 C172544 前723 tick、旧 C171520 前693 tick、同 4a58 成功 zero 前723 tick，以及既有代码审计。三者 tick0 的实际 q/qd、base、CoM、接触和几何完全相等。C172544 与 zero 的 source N Full12 在723个共同tick完全相等；与旧 C 在693个共同tick相等。zero 原完整结果仍为8857 tick、P13成功，本分析没有重新裁定它。

版本不能混称：旧 C171520 是 eec6 / `history_conditioned_heteroscedastic_log_v1`；新 C172544 是 4a58 / `history_conditioned_heteroscedastic_log_temperature_v1`，经过 τ=.5 的1024决策、8次更新。两次正式评估都使用条件均值，不采样 σ；因此不是“只改变温度”的因果试验。

## Tick400：同一物理时钟的完整四轮链

轮速单位 rad/s；canonical 统一向前为正。四轮 N=mapped N 均为+.3，controller bias均为0。

| 通道 / Full12 index / live joint_id | raw | projected residual | 最终 canonical target | 最终 native target | 实测 native qd | 实测 canonical qd | ground normal N |
|---|---:|---:|---:|---:|---:|---:|---:|
| FL / 8 / 8 | −1.348595 | −.524233 | −.224233 | +.224233 | +.045131 | −.045131 | 11.5157 |
| FR / 9 / 9 | −.519091 | −.286199 | +.013801 | +.013801 | +.012796 | +.012796 | 0，AIR |
| RL / 10 / 10 | −.462615 | −.259328 | +.040672 | −.040672 | −.044501 | +.044501 | 3.4489 |
| RR / 11 / 11 | +1.188631 | +.498092 | +.798092 | +.798092 | +.691982 | +.691982 | 13.8061 |

FL/FR/RL/RR exact joint names 是 `front_left_ankle/front_right_ankle/rear_left_ankle/rear_right_ankle`，接触对象是对应 `*_wheel`，canonical→native signs=`[-1,+1,-1,+1]`。上表 native qd 是**同 tick `robot.data.joint_vel` 原生读数**，不是用目标冒充实测；全部92个新C真实height样本的符号交叉校验误差为0。其他没有height行的120Hz tick，只把 native qd 标为由实测canonical按符号推导，未伪补原生采样。

同 tick400 的对照：

| 运行 | 最终 canonical targets，FL/FR/RL/RR | 实测 canonical qd | RR target / actual ° | FR top-gap mm | base-origin / CoM 高 mm |
|---|---|---|---:|---:|---:|
| C172544 | [−.22423,.01380,.04067,.79809] | [−.04513,.01280,.04450,.69198] | −4.371 / −28.388 | 73.624 | 66.922 / 150.424 |
| C171520 | [−.18902,.02475,.00641,.81040] | [−.05643,.02486,−.01693,.68424] | −5.215 / −31.016 | 63.164 | 63.237 / 146.614 |
| same4a zero | [.3,.3,.3,.3] | [.24424,.29813,.34175,.31376] | 0 / −.607 | 100.780 | 72.502 / 164.641 |

三者此刻 FR 都 AIR、实测承载0、Q=true/C=P=false；FL/RL/RR ground pair真实有效。FR轮在空中转动不能被称作承载推进。

## 抵消不是漏传 N，也不是通过 mask 就算合理协同

新C tick41–723的683个正N tick：前三轮raw全为负、RR全为正；FL最终目标负向682 tick，FR绝对目标<.05为579 tick，RL为515 tick，RR高于N为683 tick。与旧C同类的**目标层RR主导正转**仍存在。它不等于只有RR有实际牵引，也不说明任何差速/停轮都错误；.05仅为描述阈值，不是任务门禁。

既有链路 `semantic_backend → phase_action_masks_v2 / action_projection → semantic_residual_adapter → command_batch → all4 velocity target setter → write_data_to_sim` 的本次逐tick复算：

- source N→mapped N四轮最大差0；wheel controller bias=0。P01/P02 residual caps=.6，仅缩放 residual，不乘整个N；tanh后每tick最大变化.015。
- 计入既有P01→P02首tick继承hold，723tick的projected复算最大误差1.11e−16；final native按符号还原后与N+residual组合最大误差2.90e−8（float32写入）。全部723tick实际setter/mapping/counterfactual验证通过，没有发现四轮遗漏、错误mask对象、符号或单位错配。
- ordinary transition不清零：tick16→17继续上一projected残差。raw每8tick保持；源代码每physics tick用实际raw更新HISTORY。完整372输入未写入正式video日志，故独立重建网络实际HISTORY tensor仍未知，未额外forward。
- live joint_names和精确resolver支持ids8–11；原native audit没有独立逐行保存完整ACK id数组。不能把代码支持的解析说成额外实测字段。

旧零残差float32 +.3写成.3000000119不是策略增强，本报告计数使用1e−6容差排除此数值尾差。

## 首分歧、接触与高度先后

相对zero，新C在tick1已有all12最终目标差异，实际q/qd、接触力和base也开始分歧；不能只挑wheel断言单一原因。相对旧C，四轮**tick1最终目标相同**（[−.015,−.015,−.015,+.015]），但伺服目标已不同、物理响应已分歧；四轮目标首次差异分别在tick5/2/2/6。所以“轮目标先改变导致全部首分歧”不成立。

新C的描述性连续12tick阈值：RR误差≥1°起于27（确认38）；base原点比初态降低≥3mm起于38；四轮N+.3开始41；RR≥5°起于63；FR top-gap在97达101.662mm，191起持续低于峰值5mm；CoM比初态降低≥3mm起于379。RR始终有真实ground contact，FR自tick2为空中，不能说RR早期失去地面支持。尚未完成各腿载荷、关节驱动力矩与全身运动的因果分解。

成功zero的base原点降低3mm更早（tick23），却在此723tick前缀没有RR误差≥1°持续12tick；CoM也未持续下降到初态以下3mm。**base原点下降并不等同CoM下降，更不是单独的故障判据。** 新C末tick723 FR top-gap仍+0.320mm，却front-distance=−181.033mm且未C/P；不能说必须FR负净空才会触发RR失败。旧C末tick693为−5.570mm，两次终态不能强配为同一状态。

## 当前最有证据支持的下一方向

保持all12、动作范围、N、reward、HISTORY/ρ和τ=.5不变，从最新172544继续自然P01完整轨迹训练，让真实P02终止进入既有信用路径，再正式重载均值评估。此前1024块只有484个P02样本、1次真实P02终止，增加当前失败附近的on-policy覆盖比再次改变多项因素更可解释；不保证2048就能恢复成功。

本结果不支持永久mask四轮、不支持直接改gain/物理参数，也不支持把σ定为根因。真实轮速和接触力已记录，但没有独立测得瞬时关节驱动力矩；ACK/公式正确只是执行链证据，不是学习协同合理性的证明。详见同名JSON及其中封存源路径。
