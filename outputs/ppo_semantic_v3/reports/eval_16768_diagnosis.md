# C16768：P01 评估的 RR 连续窗口（只读诊断）

来源：`runs/ppo_semantic_v3/validation/20260906T0454524190796Z_ga475bad8f9a8_dc53cb90c56147619d5c3b654f8211ab`。
checkpoint 16768；运行版本 a475bad8f9a8；seed 2001；确定性当前策略从 P01 开始。
741 decisions / 5926 physics ticks，49.383333 s 结束，评估中 optimizer updates=0。
原结果保留为 `TASK_FAILURE_WHEEL_ONLY_CLIMB`，不是成功或新判定结果。

## 核心结论

这次不是“已有合法抬升、跨线时微小负净空”的边界案例。
整个 episode 的 RR 没有 `whole_body_initial_clearance` 或 `qualified_measured_upward_lift` 事件；
RR active_lift / front_edge_crossed / placed 始终为 false。
有接触分类 AIR 样本，但 AIR 本身不等于真实有效抬升，更不等于跨越资格。

P06→P07 / P07→P08 / P08→P09 分别发生于 tick 5240 / 5248 / 5256。
下表的净空均为轮底相对障碍顶面；前沿距离为轮心 x 减前沿 x。

| 时刻 / tick | 实际事件 | 前沿距离 mm | 净空 mm | 接触证据 |
|---|---|---:|---:|---|
| 23.941667 / 2873 | P06 后首个 RR AIR 分类样本 | -487.037 | -50.187 | ground/obstacle 都 inactive；非合格抬升 |
| 43.900000 / 5268 | P09 首个 RR AIR 样本 | -186.283 | -50.242 | 两接触 inactive；无初始/越顶资格 |
| 44.908333 / 5389 | P09 接触障碍前的最大净空 | -161.336 | -47.990 | AIR；轮底仅约高于地面 2.01 mm |
| 45.608333 / 5473 | 再次 ground 接触 | -49.777 | -49.987 | ground 3.517 N，obstacle 0 |
| 45.616667 / 5474 | 首次障碍接触，亦为最大障碍力 | -49.608 | -50.058 | ground 3.084 N + obstacle 32.725 N |
| 45.816667 / 5498 | ground 消失，保持障碍接触 | -49.493 | -50.241 | obstacle 16.633 N |
| 49.200000 / 5904 | 全 P06–P09 AIR 样本最大净空 | -4.626 | -2.339 | 两接触 inactive，仍未达顶面 |
| 49.383333 / 5926 | 首次轮心跨线并终止 | +0.401 | -2.220 | obstacle 13.196 N；RR load fraction 0.475 |

P06 起窗口 3231 个 raw samples，430 个 AIR 分类样本；其中很多是地面附近短暂接触切换。
窗口没有非有限值或机身碰撞。最大净空发生在末帧，仍低于顶面 2.220 mm。
先在轮心距前沿约一个轮半径、轮底位于地面时接触障碍，然后保持障碍接触抬高，
最后无合格抬升历史跨线。这支持现有失败判断，不适用 qualified-AIR pending 豁免。
“立面接触”是结合轮几何与 exact body-pair 力的解释，不声称有独立验证的接触点/平面分类；
也不能仅凭接触力把运动的全部动力来源归结为轮驱。

## 动作与原生执行：没有 cap 饱和证据

P09 共 84 decisions / 670 ticks。逐 tick native audit 670/670 verified，
670/670 有可分辨 native target effect；没有 in-episode state write。
全部 12 通道、所有 670 ticks 的 phase-cap 饱和次数为 0。

- RR hip 实际 projected residual：[-2.460795, -1.312281]°，最大为 ±24° cap 的 10.253%。
- RR knee：[-7.648338, -4.749347]°，最大为 ±36° cap 的 21.245%。
- 其余通道最大 cap 使用比例也低于 15.4%。
- P09 首决策末：nominal RR 1.6°/0°；residual -2.460795°/-7.648338°；
  dispatch target +0.389205°/-7.648338°。
- 终止帧：nominal -6.9°/-37.8°；residual -1.312281°/-4.749347°；
  dispatch target -4.462281°/-36.299347°。实测 q 则是 -6.058148°/-40.667096°，
  必须与 target 区分。

因此当前确定性 actor 并未用尽现有工程残差域；这不是“本次必被 cap 卡住”的证据。
它也不证明该域足够覆盖所有失败后的重试。该能力问题与本次实际采样幅度应分开讨论。

## P09 连续前缀入口：最少记录，不新增入口门

1. 来源与时钟：A/C teacher、seed、requested/actual phase、offset、actual physics tick、
   episode elapsed/remaining time；prefix 与当前 policy credit 分账。
2. 接管连续性：刚实际执行的 nominal / residual / native targets，mapper 当前状态与 tracking，
   同 tick ACK；已有 action/mapper/contact 历史不清零、不恢复旧姿态。
3. RR 当前证据：front distance、clearance、前向/垂向速度、ground/obstacle active 与 force；
   initial-clearance / qualified-lift / crossed / placed 及最后 ground 撤销事件。
4. 准备和支撑：四腿当前 contact/load，尤其 FL 真实支撑与否；body/CoM 位置、速度/角速度，
   实测 q/qdot。记录不同可继续构型，不要求固定关节角、固定载荷或固定对角支撑。

P09 前缀可以在还未 AIR/qualified 时开始；不能以取得资格作为新 gate，也不能将 A 形成的
入口记作 PPO 学会了准备。沿用当前真实失败、200 s 总时钟和 suffix/full 标签。

本报告仅使用 PowerShell 读取已结束运行文件；没有执行 Python、Isaac、训练或重分类旧 run。
