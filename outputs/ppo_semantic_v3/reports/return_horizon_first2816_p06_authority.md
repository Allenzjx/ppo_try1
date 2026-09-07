# #33 前 2816 已优化决策中的 P06 前轮控制能力

固定读取 **global 109825–112640**，仅汇总 source phase=P06 的 **840 条**记录；未读取 112640 之后、checkpoint/tensor 或其它新运行。来源 run `20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8`。仅 PowerShell/JSON 只读及本报告写入，未运行 Python/PT/pytest/Isaac，未改生产或 master。

## 结论

本窗口**不是完全单向的 raw 探索**：FL/FR raw latent 都采到过正负值，实际 canonical 净轮命令也各自出现正负方向。但实际过滤后政策请求明显偏负：FL 840 条中仅 7 条为正，FR 840 条全部为负；两轮旧分布 mean 全部为负。扩大到 ±1.2 的实际超旧 ±.6 用量也仅发生在负侧。

P06 nominal 前轮指令在这 840 条中始终为正（+.175 至 +.3 rad/s）；因此不能用本窗口检验对负 nominal（例如 −1.07）的取消能力。正净命令不必意味着正政策请求，它也可能是正 nominal 未被负请求完全抵消。上述为已执行动作链描述，不是 gamma 修订的配对改进证明，不推断唯一失败原因，也不提出 std 修改或新训练门。

## 1. 样本范围与当前前驱准备

| run 内 episode（零起算） | 本窗口 P06 globals | P06 样本 | 首/末观测 tick（sim time） | 本窗口末状态 |
|---|---|---:|---|---|
| 2 | 111454–112053 | 600 | 2656（22.133333 s）→7448（62.066667 s） | P06 `INCOMPLETE_CONTROLLER_BLOCKED`，真实 terminal |
| 3 | 112401–112640 | 240 | 2784（23.2 s）→4696（39.133333 s） | P06 nonterminal，截在固定边界 |

episode 编号由固定范围内的真实 terminal 顺序确定。840 条均实际执行 8 ticks，共 **6720/6720 native ticks verified**；详细末 tick 的实际映射/下发相等检查全真。这里只提取 P06，未把同一范围其它阶段样本加入以下统计，也未把 episode3 后续尾段补入。

## 2. mean/std、raw、请求与实际命令不是同一量

`old_distribution_mean_full12/std_full12` 是各条真实采样时保存的行为分布参数，不是当前 checkpoint 固定 mean 评估。这 840 条跨在线更新与两个实际 episode；汇总不等于一张固定策略的分布。mean/std/raw 的单位是未界定 Gaussian latent；request/drive 的单位为 rad/s。

| 量：min…max；括号为样本均值 | FL（channel8） | FR（channel9） |
|---|---|---|
| old raw mean | −.496389…−.252274（−.397646） | −.925710…−.101193（−.606104） |
| old raw std | .276953….495292（.404015） | .179950….351419（.268951） |
| sampled raw latent | −1.953692…+1.063831（−.378015） | −1.539173…+.231894（−.608759） |
| `1.2*tanh(raw)` 未滤波数学建议 | −1.152727…+.944551（−.380805） | −1.094378…+.273390（−.617633） |
| 实际 projected/filtered requested residual | −.964227…+.104938（−.421987） | −1.040408…−.067887（−.636554） |
| nominal wheel | +.175…+.3（+.299702） | +.175…+.3（+.299702） |
| actual canonical 净命令 | −.664227…+.394031（−.122285） | −.740408…+.232113（−.336851） |
| actual native float32 机械轴目标 | −.394030631…+.664227366（+.122285） | −.740407526…+.232113123（−.336851） |

当前 P06 cap 确为 front FL/FR=1.2。未滤波 `tanh` 列只作表达对照，**不是**已执行请求或另一条物理轨迹。其正向幅度不能替代经过历史滤波、slew/phase bridge 后实际请求的统计。

本窗口每条 wheel 的 requested=post-mapper effective residual，差为 0；`actual_drive_target_full12 = nominal + requested` 的 wheel 分量误差为 0，`applied_action_full12` 与 actual drive 的 wheel 分量也相同。这一相等仅针对上述 wheel 通道，不能推广到八个 servo。

原生 readback 使用 audit 的 `actual_native_targets.wheel_velocity_rad_s`：FL 机械轴符号相对 canonical 取负，FR 同号；与 `float32(sign*actual_canonical)` 逐条相等。所以上表 FL native 正负不能直接解释为 canonical 的前后方向；canonical double 命令也没有被冒称为 native float32 缓冲区。

## 3. 实际正负用量

下表正/负均为严格 `>0/<0`；这些量本窗口精确零计数均为 0。

| 840 条 P06 末采样 | FL 正 / 负 | FR 正 / 负 |
|---|---:|---:|
| old mean | 0 / 840 | 0 / 840 |
| sampled raw | 144 / 696 | 17 / 823 |
| filtered requested/effective residual | 7 / 833 | 0 / 840 |
| actual canonical 净命令 | 211 / 629 | 14 / 826 |
| native float32 机械轴目标 | 629 / 211 | 14 / 826 |

actual canonical `abs(command)<=.02` 的近零样本：FL=52，FR=11。它们不是精确静止，更不是完整任务受控停止。实际 positive raw 占比 FL=17.14%、FR=2.02%；positive filtered request 占比 FL=.83%、FR=0%；positive net command 占比 FL=25.12%、FR=1.67%。这里是每轮分别的统计，不据此声称两轮同时前送的持续时间或 body 必然前移。

超旧 cap 的实际请求：FL `<−.6` **143/840（17.02%）**，FR `<−.6` **523/840（62.26%）**；两轮 `>+.6` 均 **0**。这证明实际用过扩大后的负侧幅度，不证明正侧动作空间不存在、也不证明策略已经探索到正侧所需的持续轨迹。

三个不同的实际数值例：

- g111670：nominal FL/FR 均 +.3，请求 −.964227/−.935145，actual canonical −.664227/−.635145；对应 native float32 +.664227366/−.635145247。这是两轮净反向的实际样本。
- g112420：FL raw=+.750134、mean=−.350013、std=.422590，但 filtered request 仅 +.094031；nominal +.3，actual canonical +.394031，native −.394030631。不能将该 raw 的未滤波 `tanh` 幅度当实际命令。
- g112603：FR raw=+.115505，filtered request 仍 −.067887，nominal +.3 后 actual canonical +.232113。正净命令在这里来自名义与历史过滤后请求的合成，不是该时刻已得到正政策残差。

## 4. body 与两后腿的真实几何/接触

`body_forward_m` 是 evaluator 的 `base_position.x − obstacle_front`，不是本阶段累计位移。840 个末观测的范围为 **−.281754…−.167082 m**。

| 当前后腿末观测 | RR | RL |
|---|---|---|
| front distance 范围（m） | −.661616…−.451060 | −.527951…−.432533 |
| clearance 范围（m） | −.051675…−.042389 | −.051189…−.048532 |
| GROUND / AIR / TOP 样本 | 722 / 118 / 0 | 840 / 0 / 0 |
| obstacle-pair-active 样本 | 0 | 0 |
| load fraction 范围 | 0….532796 | .133968….659735 |
| 当前 initial-clearance=true 样本 | 17 | 0 |
| hard qualified / crossed / placed 样本 | 0 / 0 / 0 | 0 / 0 / 0 |

两后腿 within_lateral_span 均为 840/840，但最近 front 仍未达到原硬 workspace 下界 −.22 m。RR 的 118 个 AIR 末观测和 17 个 initial hint 不等同于硬 lift qualification；最高该处 AIR/全样本 clearance 仍为负，所有已记录硬 Q/C/P event map 均为空。接触计数是 decision 末状态，不是全 120 Hz 状态持续时间。

同一 episode 内首末 P06 观测的变化（不跨 reset 混算）：

| episode | body_forward 首→末（m） | RR front 首→末（m） | RL front 首→末（m） |
|---|---|---|---|
| 2 | −.219593→−.271691（−52.098 mm） | −.497705→−.658168（−160.463 mm） | −.527951→−.485279（+42.673 mm） |
| 3，固定窗口未终结 | −.223897→−.178918（+44.979 mm） | −.505756→−.471404（+34.353 mm） | −.527266→−.437378（+89.888 mm） |

因此该窗口包含不同方向的真实几何变化，不能把负 mean/轮命令简单等价为所有身体点只后退，也不能把 episode3 局部接近称为完成。数据支持“名义与政策合成后的探索显著偏负、后腿尚未进入硬准备 workspace”，不支持唯一归因、参数修改结论或新 gate。

来源：[固定 run decision audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8/residual_and_projection_audit.jsonl)。cap 与几何字段解释只读参考 [execution_profile.yaml](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/execution_profile.yaml:31) 和 [semantic_supervisor.py](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:539)。
