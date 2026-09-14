# Block18 episode1：RR 膝实际硬限，非越界目标

固定来源：`train/20260910T1921276505520Z_gd4e46006b382_2f570e3174ba4fe3a1035ca1c6162b8b`。仅核对 `completed_episodes.jsonl` 第二条及 `residual_and_projection_audit.jsonl` 第440–443条；前439条只用于定位，未读取后续回合、物理大流、Torch、checkpoint 或其他运行数据。第二条 completed record 的 `terminal_info` 与第443条 `applied_audit` 完全一致。

## 终止帧与越界依据

本回合109条 on-policy 决策；最后 global155451 / P12 / episode tick8450 / 70.416666667 s，原因 `HARD_JOINT_LIMIT`，监督器 `SAFETY_ABORT`、source=`HARD_JOINT_LIMIT`。它是独立安全中止，任务未完成，不重分类为 BODY 或 wheel-only。P10教师前缀在 tick7584 / 63.2 s 交接，前缀不进 PPO；本回合实际 on-policy 为866 ticks / 7.216666667 s，而不是70.4167 s全由策略完成。

最后一条只执行 **2 ticks**（8449–8450），不是原计划8 ticks。terminal372维 policy/critic 完全相同、全部finite，`terminal_observation_finite_fallback=false`。从其 `actual_joint_position_deg` 固定90°尺度回解：

- **RR knee ≈ −60.110283494°**，低于真实硬限−60°约 **0.110283494°**；其他7个servo均在各自硬限内。RR膝速度≈−84.029623°/s。
- 这是落盘float32观测回解，不冒充缺席的raw双精度关节读数；该特征未触及schema clip。生产判据为 `not(lower <= measured_position <= upper)`，膝范围[−60°,210°]（`sensing/guard_state.py:365–373`）。
- terminal `time_outs=false`、`terminal_bootstrap_allowed=false`；本次并非P12 deadline：stage_age7.083333s，effective_limit31.225s（30+1.225）。

## 实际命令端点（RR knee，角度均为度）

| Global / tick | 实际执行ticks | raw latent（无单位） | nominal N | filtered residual | final drive target | terminal |
|---|---:|---:|---:|---:|---:|---|
|155448 / 8432|8|−0.050541|−27.2|−0.512360|−27.712360|false|
|155449 / 8440|8|−0.028075|−27.2|−1.010446|−28.210446|false|
|155450 / 8448|8|+0.040812|−27.2|+1.468419|−25.731581|false|
|155451 / 8450|2|−0.070187|−27.2|+0.468419|−26.731581|true|

末端8个final servo target均在审计的2°reserved safety limits内；RR膝目标距−60°下限仍约33.268419°，但实测比该目标低约33.378702°。四个端点没有RR膝headroom裁剪；最后两条列出的额外headroom裁剪索引5是RL knee，不是RR knee。最后2个native ticks均verified且有真实target effect。raw latent、经过历史/限速的residual、final目标、实际关节状态是不同量；负raw与尚为正的filtered residual不等于动作概率或符号错误。这些端点能证明实测越界而目标未越界，不能单独确定接触负载/惯性/追踪误差的因果份额，也不是完整逐tick控制器重放。

## 终态机身与真实接触（有限范围）

由terminal372中的锁定障碍bottom0/top0.05相对高度交叉回解，base z≈0.082511142 m（两条回解差7.45e−10 m）；gravity_z≈−0.976629913，不满足既有FALL的z<0.015或gravity_z>−0.30分支。该高度同样为float32特征派生，不是新raw物理采样。

- FL：TOP、support=true，承载23.714203 N，load≈40.113687%、valid=true；front+40.842835 mm。
- RR：GROUND、support=true，承载35.403282 N，load≈59.886313%、valid=true；front−76.351490 mm，top gap−50.205409 mm。
- FR与RL均AIR、support=false、承载0；RL front−264.434662 mm、top gap−21.195436 mm。四腿bearing_verified均true。

因此不能把悬空FR/RL虚构为支撑，也不能从这四个端点推断不可测CoM或完整载荷转移因果。终态body速度标量0.256445 m/s、angular speed1.299858 rad/s，仅描述该终态，不作全程稳定性比较。

计数截止仍采用主控提供的固定边界：本块443已采样、384已优化，59未优化；本诊断没有追读新的optimizer记录或活动流。训练正常继续，本结果不构成新optimizer门禁；无生产、参数、主报告或训练进程更改。
