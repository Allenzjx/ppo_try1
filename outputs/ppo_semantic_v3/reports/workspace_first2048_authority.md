# Run 29：首 2048 已优化决策的 P06 前轮实际动作范围

固定范围 **global 103169–105216**；对应已记录 updates 772–787。只读该范围的 2048 条记录，未读取其后 tail。本文不重复迁移、checkpoint 哈希或整块账本。

来源：[run 20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c)。其中按动作 source phase 计 **P06=722 条**：episode 0 的完整 P06 段 600 条，episode 2 的有界 P06 尾段 122 条；episode 1 未进入 P06。以下计数均为这 722 个 decision 末记录，不冒充逐 120 Hz tick 统计。

## 结论与字段边界

**新 ±1.2 范围被实际使用，但越过旧 ±0.6 的部分全部在负向。** 当前 P06 两前轮 nominal 都为 **+0.175～+0.3 rad/s**，没有负 nominal 样本，因此这个窗口不能验证对 −1.07 等负 nominal 的抵消能力。实际观察是：正向 nominal 经策略残差后多数成为负向净指令。

Canonical channel 8=FL、9=FR。raw/mean/std 是 PPO 高斯 latent；`projected_residual_full12` 是已有过滤/速率限制后的请求；`actual_drive_target_full12` 是实际 filtered canonical drive，不是 float32 native buffer。另以 `actuator_target_effect_audit.actual_native_targets.wheel_velocity_rad_s` 核对原生轴目标：FL 轴符号取反、FR 同号。

## 1. P06 分布与实际命令范围

| 范围 min…max | FL / channel 8 | FR / channel 9 |
|---|---:|---:|
| sampled raw latent | −1.464196…+0.833634 | −1.688202…−0.035770 |
| old distribution mean | −0.585166…−0.243281 | −0.975708…−0.289740 |
| old distribution std | 0.241609…0.425042 | 0.181960…0.304639 |
| projected residual (rad/s) | −0.925739…+0.006337 | −1.110435…−0.135474 |
| nominal (rad/s) | +0.175…+0.3 | +0.175…+0.3 |
| actual filtered canonical drive (rad/s) | −0.625739…+0.306337 | −0.810435…+0.044526 |

这些 min/max 不一定来自同一决策，不可交叉拼成合成控制状态。实际 722 条中，两前轮 controller wheel bias 全为 0，mapper wheel nominal 与 nominal 相等；headroom effective residual 与 projected request 相等。每条均满足 `actual canonical = nominal + projected residual + controller bias`，本次 double 重算最大差为 0。原生轴 float32 目标与相应 canonical 值经符号转换后的最大差分别为 2.970e−8、2.975e−8 rad/s；这不是把 canonical double 当成 native readback。

| 实际计数 / 722 | FL | FR |
|---|---:|---:|
| mean > 0 | 0 | 0 |
| sampled raw > 0 | 72 | 0 |
| projected residual > 0 / < 0 | 1 / 721 | 0 / 722 |
| projected residual > +0.6 | 0 | 0 |
| projected residual < −0.6 | 231（31.99%） | 664（91.97%） |
| actual canonical > 0 / = 0 / < 0 | 60 / 0 / 662 | 3 / 0 / 719 |
| actual canonical 在 ±0.02 内 | 28 | 2 |

负 nominal 的分母为 **0**，所以“负 nominal 被抵消至 0/正向”的条件频次是**未覆盖 / 不适用**，不是 0% 成功率。上表净正向或近零次数发生在正 nominal 下，也不是对负 nominal 的成功抵消。

## 2. 同条记录的数值例

| global / episode / tick | 通道 | raw / mean / std | nominal | projected request | actual canonical | 原生轴目标 float32 |
|---|---|---|---:|---:|---:|---:|
| 103779 / 0 / 4888 | FL | −1.102030 / −0.526557 / 0.362756 | +0.3 | −0.9257391640 | −0.6257391640 | +0.6257391572 |
| 103669 / 0 / 4008 | FR | −1.688202 / −0.932956 / 0.277109 | +0.3 | −1.1104346480 | −0.8104346480 | −0.8104346395 |
| 105126 / 2 / 2960 | FL | +0.436894 / −0.334554 / 0.365896 | +0.3 | +0.0063374103 | +0.3063374103 | −0.3063374162 |

前两例确实使用旧 ±0.6 请求域之外的负向幅度，并非 nominal 独自导致净负向。第三例说明正 raw 不能直接当作正的未滤波 `1.2*tanh(raw)` 已全部下发：实际请求有已有历史与 slew，须读取 projected 字段。

近零净指令也有实际记录：g103516 FL 为 `+0.175 − 0.1809343887 = −0.0059343887`；g105095 FR 为 `+0.175 − 0.1825325521 = −0.0075325521`。不是精确 0，也不代表实测轮速已停止。

当前配置的 P06–P13 前轮请求 cap 仍是对称 ±1.2，非软件只允许负向。本窗口能支持的描述仅为：**新增幅度在实际样本中只被负向使用，且所有记录 mean 均负；没有观察到利用 >+0.6 去抵消负 nominal。** 不能据此证明正向动作不可达、唯一失败原因或未来策略行为；也不提出新硬 gate。

## 3. 后腿 workspace 与事件简述

下表均为 P06 decision 末真实 front distance，单位 m；“最近”是该有界段内最大 x，不是同时达到双腿最近。

| P06 段 | RR 首 / 最近 / 末 | RL 首 / 最近 / 末 |
|---|---|---|
| ep0，g103516–104115，600 条 | −0.576916 / −0.503478 / −0.875894 | −0.580979 / −0.552545 / −0.723719 |
| ep2，g105095–105216，122 条 | −0.534538 / −0.501065 / −0.575989 | −0.562694 / −0.534344 / −0.568200 |

这些最近值仍未达到原硬 workspace 下界 −0.22；两段未记录 RR/RL hard Q/C/P，历史事件 tick 均为空。ep0 在 tick 7576 / 63.133333 s 以 P06 `INCOMPLETE_CONTROLLER_BLOCKED` 终止；当时 RR 虽为 AIR，但 clearance −47.935 mm，不等于合格抬升。ep2 的固定窗口末 tick 3680 / 30.666667 s 仍 P06、非 terminal，两后腿均 GROUND，RR/RL load 分别 0.154990 / 0.450777；不推测后续结果。

数据呈现了短暂接近后又远离的状态变化及明显负向前轮指令，但没有动作反事实轨迹，不能把 workspace 回退全部归因于单个轮通道或本次 soft-Phi 修订。

只进行了 PowerShell 源码/固定日志读取并新增本报告；未修改 production/config/tests/master，未运行 Python/PT/GPU/Isaac，未读取 global 105216 后的记录。完成后停止。
