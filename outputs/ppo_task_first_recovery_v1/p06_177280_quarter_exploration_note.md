# 首个 0.25 温度 P06 块：有推进，仍有局部宽创新

已封存 run `20260916T0619126997844Z_gfc14a68c037a_1cc1b227105648c8a0795399b5da8062`，rollout `001350`：128 个 P06 decisions、1 次 PPO update / 20 optimizer steps，CP177280。成功 N 从 P01 roll-in 到 P06；学习段连续 8.5333 秒至 tick3704，无 terminal、无任务失败，但尚无 P07–P13 样本，不是后腿通过或新完整成功。主线同设置继续 512，不以本分析为门禁。

只分解已保存 `raw = (conditional_mean − .9×history) + .9×history + innovation`；未 forward actor/critic、未重算或替换原 GAE/returns/advantages。128 拍 raw 与原日志误差0，HISTORY 与前一 raw 误差0，roll-in 后初始 HISTORY 全0。

| 描述指标 | 旧0.5首个失败段（80拍） | 新0.25连续段（128拍） |
|---|---:|---:|
| 至少一维 `abs(raw)>3` | 9 / 80 | 6 / 128 |
| servo 至少一维 `abs(tanh(raw))>.95` | 20 / 80 | 4 / 128 |
| wheel 至少一维 `abs(tanh(raw))>.95` | 13 / 80 | 14 / 128 |
| wheel raw共同 / 差速分量RMS | .439 / .747 | .377 / .389 |
| 结束状态 | tick3317机身碰撞 | tick3704非终态P06 |

新6拍高raw为 FL wheel 5拍、FL hip 1拍。不能声称饱和已消除：状态相关有效sigma峰值仍达 FL wheel **2.52769**、FL hip **2.34842**（平均各约 .198），FR knee 平均 .13092 / 峰 .55286。温度减半只保证**同权重、同观测**有效sigma减半；不同物理轨迹不能据此推断每拍sigma必然低于旧运行。

首个 `abs(raw)>3` 为 tick3424 的 FL wheel：base贡献 .03023、HISTORY贡献1.75720、innovation1.24492（有效sigma1.74444，约.714σ），raw3.03235。不是base mean学出了3的输出。tick3432 FL hip 为 `.00200 − 1.88523 − 2.16914 = −4.05238`；下一拍创新 +6.61411（2.816σ）使raw回到+2.97224。由于真实slew，FL hip target仅由−9.26261°回到−5.26261°，不是当拍已执行到正向大角度。必须继续区分 raw、HISTORY、投影/执行历史与实测。

支撑4/3/2腿分别47/47/34拍，未出现少于2腿；首次低于4腿 tick2760（FL暂未承载），首次2腿 tick2808（FR/RL承载）。两腿支撑不自动失败，也不能虚构空中FL在承载。FR knee target峰103.58437°（tick3024），机身底部world-z最小 .08601m（tick3040）；这是世界高度，不是障碍净空。后续恢复，末拍FR knee target44.75450°，实测48.71318°（前1物理tick），机身底部z .14060m。

末拍四轮顺序 FL/FR/RL/RR：target `[.31164, .32896, .63053, .26312]` rad/s，实测 `[.32638, .35642, .63073, .27104]` rad/s；不是仅RR有指令。机身前向坐标从−.21110到+.05995m，推进约.27105m；RR前缘距离从−.52425到−.27840m，仍未越沿。未依据轮速推断净牵引。

局限：旧源网络为CP176768，新为已完整成功的CP177152；RNG、物理轨迹及期间执行修复均不同，窗口长度也不同。上述频度下降和无碰撞是**本次实际观察，不是严格温度因果证明**。支持保持唯一0.25候选继续采后腿数据，不支持此时叠加reward/rho/scale/mask修改，也不保证后续512通过。

详细逐通道数据：`p06_177280_quarter_exploration_decomposition.json`。复用脚本 `check_p06_exploration_176896.py` 仅增加输出侧run/rollout/temperature参数与episode边界推导；原0.5证据文件未覆盖，生产及训练现场未改。
