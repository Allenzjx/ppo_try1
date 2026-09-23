# RR capture height comparison — sealed 26db vs historical accepted N

只读、非严格配对；0 physics/forward/optimizer。C 是26db/feedback-v2、CP220544的封存失败视频（104.0333s、RR未placed），**不是当前正在训练的v3**。N 是既有成功零残差参考；RR cross+placed=6155、P10 entry=6160。两条轨迹的到达时刻、动作和全身状态不同，不作因果反事实或稳定性结论。

仅各读一次两份 height_diagnostics.jsonl（18.38/12.97MB），保留10个C、6个N实拍；复用8行关键表及已审N事件。高度日志是每8 physics tick采样，9516/9555/10755事件和所选9512/9552/10760高度行**不假装同拍，也不插值**。全部xyz、命令/实测关节及原四元数保存在同名JSON。phase_metrics.csv是阶段汇总，不能补成逐拍contact。

## 1. 实测关节，单位deg

| source tick / phase | FR hip/knee | FL hip/knee | RR hip/knee |
|---|---:|---:|---:|
| N 6152 / P09 | 0.51/30.26 | 38.95/-12.02 | -8.23/-39.43 |
| N 6160 / P10 | 0.31/30.12 | 37.85/-13.05 | -8.06/-39.36 |
| N 6248 / P12 | 3.84/29.87 | -21.18/-31.52 | -6.62/-27.35 |
| C 9552 / P09 | 9.48/-22.28 | 31.93/-43.39 | 17.90/-45.19 |
| C 9856 / P09 | 9.22/-21.45 | -24.49/-58.46 | 13.08/-45.18 |
| C 10760 / P09 | 9.57/-21.50 | -24.48/-58.45 | -1.92/-45.12 |
| C 12484 / P09 | 9.59/-21.41 | -24.43/-58.44 | -2.05/-45.12 |

C终态FL knee命令−58.000、actual−58.435°，确实接近原−60°硬下限（实测仍有约1.565°负向角余量；不是可用Cartesian下降距离）。N capture时FL knee约−13°，后续P12约−31.5°；C更早到负端的事实成立，但不能仅凭角符号决定它对RR高度的因果方向。

C的FR knee约−21°，N capture时约+30°，构型明显不同。**日志没有FR/FL hip世界关节安装点、膝pivot或经核验的连杆伸展率，不能宣布FR“高/直”或把它认定为唯一障碍。**

## 2. 世界高度与姿态

RR hip取USD joint localPos0经当拍父link pose变换；wheel取真实link origin及独立当前collider minimum，不是COM，也不是接触点。下表高度/垂直间距单位mm，roll/pitch单位deg（由原wxyz归一化按标准XYZ公式计算）。

| source tick | RR hip z | RR wheel center z | RR collider min z | hip−center垂直间距 | body origin z | roll/pitch |
|---|---:|---:|---:|---:|---:|---:|
| N 6152 | 197.77 | 98.29 | 49.70 | 99.48 | 52.01 | -11.54/-3.49 |
| N 6160 | 198.88 | 98.45 | 49.73 | 100.42 | 55.11 | -11.09/-3.57 |
| N 6248 | 235.24 | 98.69 | 48.96 | 136.55 | 123.90 | -4.56/-5.73 |
| C 9552 | 205.61 | 126.95 | 77.52 | 78.67 | 72.05 | -7.14/-1.92 |
| C 9856 | 234.78 | 159.68 | 110.61 | 75.10 | 78.34 | -9.51/2.68 |
| C 10760 | 231.74 | 145.12 | 96.10 | 86.62 | 77.28 | -9.16/2.46 |
| C 12484 | 231.50 | 144.81 | 95.70 | 86.69 | 77.21 | -9.19/2.36 |

- C9552→9856，与late全身重配置/hip搜索同窗：RR hip升29.16mm、轮心升32.73mm，而body origin只升6.29mm。旋转与其他腿姿态不能被“机身z”一个标量替代，也不能把这次上升全归于RR hip。
- C9856→10760：RR hip下降3.04mm，hip−轮心垂直间距增加11.52mm，轮心下降14.56mm；RR knee实测几乎不变。说明后半段确有下降，却不能从相关轨迹单独识别负hip的净因果贡献。
- C终态对N6160：RR hip高32.62mm，当前垂直间距短13.73mm，轮心合计高46.35mm。**这是精确的几何差分，不是证明还需移动哪个关节多少度。**
- N6248已经保持RR接触时hip z235.24mm，甚至略高于C终态231.50mm，但其hip−轮心垂直间距136.55mm，而C只有86.69mm。因此“RR hip世界高度过高”不是充分解释；不同RR膝/全身构型同样关键。此行是N后捕获P12，不能当作C未捕获P09的唯一入口姿态。
- mass-weighted COM：N6160=(0.703436,−0.118294,0.154196)m；C终态=(0.728697,−0.091489,0.164266)m。它们在不同世界位置/任务状态，不据此宣布CoM转移方向正确或计算支撑三角margin。

## 3. 接触与有效支撑高度：只给实际记录

C终态12484的封存physical evaluator（非图像推断）：

| leg | 当前接触/支撑 | bearing force N | wheel collider minimum world z mm |
|---|---|---:|---:|
| FL | TOP，support=true，bearing_verified=true | 3.306 | 49.979 |
| FR | TOP，support=true，bearing_verified=true | 13.738 | 48.604 |
| RL | GROUND，support=true，bearing_verified=true | 11.491 | −1.058 |
| RR | AIR，support=false，无TOP/ground | 0.000 | 95.703 |

FR轮心z97.660mm，FL轮心z102.252mm；FR并非悬在高处而不接地。TOP平面50mm与RR实测gap45.703mm一致。表内collider最低点不是已测contact point；毫米级穿透/几何偏差保留原值，未改写为理想平面。没有虚构平均支撑平面或支撑多边形margin。

高度流没有逐拍contact/force。C另外各高度行的四腿当前承载、N各高度行的四腿承载与接触力均未从几何补造；N6155真实placed事件单独引用原事件记录，不能替代这些缺测量。8行关键表中的RR均AIR/未placed。

结论：**FL接近负端、RR当前垂直间距较短以及late时RR安装点抬高有证据；FR构型不同有证据，但FR“高/直阻碍”的因果解释和剩余可达下降行程尚无足够量。** 本报告不建议热改控制，不把当前knee候选说成已经成功。

Sources:
- C: runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0356402230589Z_g26db2a1946e8_c1a7c95ca1db41d69882389e1e97cd8d/source/
- N: runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source/
- C exact milestones: video_review/CP220544_RR_feedback_v2_g26db2a1946e8_review_corrected_v2/rr_capture_selected_response.{md,json}
- Existing N timing: outputs/ppo_p05_hip_only_continuation_v1/CP218496_RR_capture_source_readonly.json accepted_N_ref

CPU extraction已结束，无生产编辑、仿真或学习。
