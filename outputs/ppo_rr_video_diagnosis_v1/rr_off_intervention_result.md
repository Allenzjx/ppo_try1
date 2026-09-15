# RR6/7 raw 关闭干预：真实结果

这是已完成的唯一闭环诊断，不是训练或成功评估。源运行 `20260914T0503017583189Z_g4b2c038887c4_0d7983dafbee4b78ad64ba31363625f4` 于 2026-09-14 05:06:06 UTC 完成；未修改生产代码、reward 或控制参数，新增 PPO 样本/updates 均为 0。

## 初态与干预核验

D 与 B0 的实际初态完全一致：11组关节位置/速度、native首步前关节值、standing偏移、wheel速度、root位姿/速度及CoM位置/速度数组逐元素差值均为0；实际contact pairs相同，CoM均有效。这不是仅凭seed相同的配对声明。

- 全部998个120 Hz实际tick连续、精确按episode tick对齐。998/998 native verified、setter/dispatch一致、mapping一致、previous-ACK核验通过；没有episode内状态写入。
- RR hip/knee 的 dispatched raw、projected residual、effective residual、同前态native policy差、N请求、mapped N及最终canonical target，在998/998 tick中均严格为0。
- 其余通道每个tick均有非零实际有效控制。125/125次决策中其他10个raw值精确等于actor输出；RR原始actor latent本来非零，但dispatch置0。125/125 HISTORY检查精确沿用前次masked dispatched raw，生产phase mask保持全部12项为1。
- N反馈没有关闭：mapper补偿在961/998 tick非零，首次为tick38；峰值FR knee=3.75°、RL hip=1.25°。post-mapper controller bias为0，不等于整个N反馈为0。

## 终止与跟踪误差

P01采样16ticks，P02采样982ticks；tick998 / 8.316667s在P02发生 `SAFETY_ABORT / HARD_JOINT_LIMIT: rear_right_knee`，任务未完成。该次运行判定 `run_validity=VALID`。

| 同tick终点量 | RR hip | RR knee |
|---|---:|---:|
| N / mapped N / final target (°) | 0 / 0 / 0 | 0 / 0 / 0 |
| actual (°) | -1.301465466 | -60.018800938 |
| actual minus final (°) | -1.301465466 | -60.018800938 |

RR knee误差绝对值首次持续12个采样点达到1°/2°/5°，分别始于tick13/31/50，确认于tick24/42/61。这些是诊断阈值，不是新增任务或训练门槛；12个端点首尾跨度为11/120秒。

终点base原点高度0.051117666m，真实CoM高度0.117693788m；标定后body roll/pitch为0.074820112 / -0.117045450rad。999条physical记录（含tick0）的body collision detected、real_pair_active及persistent均为false，无缺失。小倾角不能脱离沉降、关节越限与任务结果解释为更稳定。

## 实际驱动读回与结论边界

首个policy action前的实际PhysX getter读回：8个servo的K=600、D=60、maxForce≈2.700000048Nm、maxSpeed≈4.999999523rad/s。读回前后clock不变，probe没有setter/reset/physics step/update。它们是有效配置读回，不是瞬时关节力矩，也不证明实际关节跟踪成功。运行回执确认frozen model不变，125次HISTORY检查通过。

本次受控干预表明：**直接RR残差不是该RR膝越限失败的必要条件；在其余10个policy通道保持活动时，全身闭环交互仍足以复现失稳。** 不能据此认定所有PPO都无关，也不能定位某个其他单通道或断言reward动机。瞬时关节驱动力矩未测得；已记录接触力，但本次尚未完成各腿承载与关节力矩的时序因果分解，因此具体力/力矩机理仍未被本诊断证明。

同前态native反事实的范围仅为 `same_pre_tick_state_without_current_ppo_residual`；不是独立B0重演。干预后其余policy动作随新的真实状态与masked HISTORY闭环变化，因此不是C0其他10通道的开环原样回放。

逐字段数值、来源、单位和核验计数见同目录 `rr_off_intervention_result.json`。原B0/C0/D日志和媒体均未改写；未新增仿真实验。
