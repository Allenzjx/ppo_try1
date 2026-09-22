# CP194560 deterministic：FR 捕获窗口预览

标注 **PPO+LIMITEDAUX**（checkpoint 训练谱系由主代理提供，不作辅助训练的单独因果归因）。真实 FR qualified/cross/placed = **23/1785/1795**；仅读取当前 run 的已完成物理前缀 **0–1795**。整回合仍在继续，本报告不判全程成功或稳定性胜出。

物理 seed=4001，deterministic conditional mean；与 CP192512 的已记录 runtime 文件 hash 完全相同。B、CP192512 数字直接复用既有 JSON，没有重读旧 raw。B 与新 run 的历史代码差异仍按旧报告保留，不伪称所有运行文件相同。

| FR准备→捕获指标 | 已保存 B | CP192512 det | CP194560 det PPO+LIMITEDAUX |
|---|---:|---:|---:|
| 时长 (s) | 12.5167 | 14.4833 | **14.9583** |
| roll RMS (rad) | .231919 | .201981 | .189668 |
| pitch RMS (rad) | .183521 | .173810 | .180312 |
| roll/pitch 导数联合 RMS (rad/s) | .117889 | .098530 | .092517 |
| 峰值 tilt (rad) | .309464 | .285290 | .280743 |
| body collider 最低点：窗口最小/平均 (mm) | 91.272 / 94.389 | 93.161 / 96.113 | 94.868 / 99.364 |
| 捕获时 body collider 最低点 (mm) | 106.170 | 103.405 | 104.940 |
| FR 首次 AIR ≥15mm 至 cross 前：gap 最小/平均 (mm) | 15.330 / 98.762 | 15.905 / 89.006 | 15.267 / 89.569 |
| 真实 base 前向位移 (mm) | +179.136 | +170.891 | +172.560 |

相对 CP192512：rate RMS −6.10%、peak tilt −1.59%、roll RMS 降低，但 **pitch RMS 上升**；时间增加 **0.475 s（3.28%）**。body 最低点的窗口最小值 +1.707 mm、捕获值 +1.535 mm，FR safe 窗平均 gap +0.563 mm。因此本次不是通过更低机身换取这些指标，但存在减速代价，不能声称等速下已改善。

相对 B：时间增加 **2.442 s（19.51%）**，rate RMS −21.52%、peak tilt −9.28%；body 窗口最小值高 3.597 mm，然而捕获时低 1.230 mm，FR safe 窗平均 gap 少 9.193 mm，base 总前送少 6.576 mm。改善与代价同时保留，不把更慢的全窗口平均直接当速度无关的优胜。

导数 RMS 使用真实相邻 120 Hz、角度 wrap 后的时间积分 `sqrt(∫((roll_dot²+pitch_dot²)/2)dt/T)`，没有再除以时长。body 数值是实际 collider 最低世界 z，不是 base_z、CoM 或精确障碍净空。首次 safe 到 cross 前的 gap 区间沿用旧定义，不宣称期间每帧都 AIR。未测安装点高度仍为 null。

同名 JSON 保存全部原指标和已消费前缀摘要：physical 1796 完整行，stage 7 完整行；不是仍增长文件的全文件 hash，也没有伪造最终 manifest。未读取 tick >1795、未改生产、未启动物理。
