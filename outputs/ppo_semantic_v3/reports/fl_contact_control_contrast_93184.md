# C93184 FL 近接触控制对照：固定只读窗口

结论：C93184 的 P05 末段确有可观测的 **FL hip mapper 反馈反向抵消正残差，并形成小幅目标/关节往复**。这不是仅由 `r/9` 玩具模型推出：用真实上一 tick 的关节 q、当时 nominal 及上一 mapper 偏置，16 次反馈更新逐次符合现有 gain8、±10°、单次1.25°更新公式。选中样本的 requested residual 没有被 headroom 再裁减。**但这不能证明该抵消造成未接触**：knee 的 mapper 偏置反而与正残差同向；#27 三个完成回合也在 hip 反向偏置存在时实际放置了 FL，且路径、动作、支撑和策略采样并不相同。

本报告仅读取日志/源码，未运行 Python、Isaac、GPU、tensor load、训练或修改生产文件；不重复 checkpoint hash，不改变原任务结果。

## 1. 来源、截止与量纲

- 评估：[C93184 run](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/validation/20260907T0009333548816Z_g2677995544c9_869bd7c54b5f4b688d9582b8a4e507ae)，HEAD2677995544c9、seed2001、自然 P01 fixed mean。正式结果为657 decisions /5256 ticks /43.8s，P05 `INCOMPLETE_CONTROLLER_BLOCKED`；physical evaluation valid、task success false、optimizer0。FL Q1731/C2763/P缺失。原结果与全评估计数见[既有报告](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_93184_diagnosis.md)，本报告不重新分类。
- 评估局部读取 `native_tick_audit.jsonl` 与 `physical_observations.jsonl`：t2720–2856、4232–4272、5232–5256，共203条实际下发/末态；另取各窗口所需上一 tick 实测 q。不是整段5256 ticks的第二次全量物理审计。
- 训练：[当前 #27 run](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0017590892338Z_g2677995544c9_bf91484d9ac74584b18ac862afe2f22c)，只取已完成 episode0–2 内22条近放置 decision-end 样本；三回合合计2854 decisions，结束 global96038。已记录 update716/global96128 包含这些样本；不统计当前之后的采集尾部，也不预判整个8192块。

以下角度均为 canonical、相对 standing 的度数，二元组按 `[FL hip, FL knee]`。`n`=authored nominal；`m`=`native_drive_target_full12` 中 mapper 输出的 canonical 值；`β=m−n`=在 nominal 已稳定且排除 nominal stepping 后可识别的内部 mapper 偏置；`c`=`controller_drive_bias_full12`，是另一个 post-mapper controller bias，不能把它与 β 混为一谈。`r`=`projected_residual_full12`，`r_eff`=headroom evidence 的 effective residual；`d`=实际 filtered canonical target。q−/q+来自本次下发之前/之后的真实物理观测。

选中203条评估及22条训练样本均为 FL `c=[0,0]`、没有 nominal geometry adjustment、FL `r_eff=r`，所取 native audit 均 verified / actual_mapping_matches_dispatch。d不是左/后轴符号转换后的 float32 native readback；真正下发值另见 `actual_native_targets.servo_position_rad`，含 standing/sign/radian 转换。

## 2. C93184：请求、反馈、关节与间隙

| tick | n | m | r = r_eff | d | 实测 q+ | FL gap / 接触 |
|---:|---|---|---|---|---|---|
| 2764 | 48.2, −25.4 | 49.45, −25.4 | 6.31076, .96729 | 55.76076, −24.51436 | 55.53578, −32.04760 | +76.138mm / AIR |
| 2832 | 22.8, −13.4 | 21.55, −12.15 | 6.01330, 1.02432 | 27.56330, −11.12568 | 32.39217, −11.26485 | +29.801mm / AIR |
| 2856 | 22.8, −13.4 | 14.05, −12.15 | 6.26334, 1.23108 | 20.31334, −10.91892 | 23.59284, −11.03063 | +4.619mm / AIR |
| 4261 | 22.8, −13.4 | 16.62545, −12.15 | 6.30676, 1.58320 | 22.93221, −10.56680 | 23.55033, −10.57056 | +2.079mm / AIR |
| 4264 | 22.8, −13.4 | 16.62545, −12.15 | 6.30676, 1.58320 | 22.93221, −10.56680 | 23.40180, −10.57060 | +1.549mm / AIR |
| 4265 | 22.8, −13.4 | 17.87545, −12.15 | 6.30884, 1.58459 | 24.18221, −10.56541 | 23.45715, −10.57041 | +1.665mm / AIR |
| 5256 | 22.8, −13.4 | 16.61376, −12.15 | 6.32794, 1.58195 | 22.94169, −10.56805 | 23.41227, −10.57142 | +2.411mm / AIR |

早期 nominal 在动：t2764 knee 真实 q 与目标相差约−7.53°，不能套用静态反馈稳态推导。t2832以后 hip nominal 已到22.8°，mapper 输出继续下降；t2856的 βhip=−8.75°，正请求+6.263°后最终目标仍比 nominal 低2.487°。这一段反向补偿同时产生实际降低 hip 目标的作用，**不能仅凭“抵消正残差”认定对接触有害**。

近最小 gap 的 t4232–4272：hip 请求只在6.306724–6.308843°变化，而 βhip 在−6.174549°和−4.924549°之间往复；真实 qhip 为23.401391–23.605923°。gap 为+1.549242至+2.235522mm，所取原始 obstacle/ground pair均 verified、inactive、0N。末段 t5232–5256仍有同类循环：请求6.327937–6.331595°，βhip为−6.186242/−4.936242°，qhip为23.412275–23.618184°，gap为+2.410625至+3.074308mm。

t5256的明确分账为 `22.8 − 6.186241827 + 6.327936546 = 22.941694719°`：+6.328°请求完整通过 headroom，但净目标相对 nominal 仅+.141695°。实测 q 相对 nominal 为+.612275°，相对目标仍高+.470580°；这是实际观测，不是 r/9 近似。**knee并非同样被抵消**：βknee=+1.25°，与 r=+1.581952°同向，最终目标比 nominal 高2.831952°，实测 q 高2.828581°。

## 3. 逐次反馈核对及同历史反事实边界

冻结 [ServoTargetMapper.advance](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/infrastructure/servo_target_mapper.py:106) 使用真实 measured physical joint、nominal tracking reference；FL canonical sign为正。nominal已稳定时：

`β_next = β_prev + clip(clip(8*(n−q−), −10, +10) − β_prev, −1.25, +1.25)`。

在上述两个稳定窗口实际发生的16次 hip mapper 变化，用原始日志双精度数字和真实 q−重算，结果与记录 β 逐次相等（本次 PowerShell double 计算最大绝对误差0）。例如：

- t4261：q−=23.605625411782°，βprev=−4.924548543499°；gain8期望−6.445003294257°，受1.25°更新限制得到−6.174548543499°，正是记录值。
- t4265：q−=23.401798906571°，βprev=−6.174548543499°；gain8期望−4.814391252565°，更新得到−4.924548543499°，也是记录值。

变化每4个物理 tick一次。此运行 native dispatch tick=episode tick+179，例如4261→4440、4265→4444；不是拿 episode tick 对4取模。其他相邻稳定样本的 m 保持不变。这证明实际记录符合 q 驱动的反向反馈和更新限速，而非仅凭 toy equilibrium 猜测。

[Semantic residual dispatch](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_residual_adapter.py:85) 仍是一次 mapper → nominal-only geometry → headroom → 原 hard/slew → write。请求通过 headroom 不等于实际 native 目标相对零请求每次都增加整个 r：[native audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/actuator_target_effect.py:159) 的零当前残差分支共用当前 mapper、相同上一 final target，并各自再过同一限速。

t4261 hip 的 actual−zero-current-r native差为0，两分支同落在下降 slew边界；t4264约+1.25°，t4265约+2.5°（一边上升1.25°、另一边下降1.25°）。稳定两窗口分别5/41和3/25条 hip同 tick effect为0，不能称这些请求未送入 headroom。t4265最终 dhip=24.182212°还比未限速 `m+r=24.184294°`低.002082°。这里的零残差反事实**保留由先前真实轨迹形成的 mapper 偏置**，不是一次不带反馈/不带策略的独立物理运行。

## 4. #27 已完成三回合的实际 FL 放置对照

这些是同HEAD、seed1001的随机训练，策略随 update更新；不是与 seed2001 fixed-mean C93184配对的同状态实验。episode0/1/2分别954/941/959 decisions，均后来P06 incomplete，但FL Q/C/P分别1710/2735/2832、1630/2665/2728、1765/2774/2868。下表只比较已发生的FL子任务。

| 已优化样本 | m / βhip | r_eff [hip,knee] | d [hip,knee] | gap / front | 当前接触证据 |
|---|---|---|---|---|---|
| ep0 g93538/t2832 | [14.05,−12.15] / −8.75° | [7.69604,6.02685] | [21.74604,−6.12315] | +.580 / +21.293mm | TOP2、pair active、XY内、load .113525 |
| ep1 g94479/t2728 | [21.55,−12.15] / −1.25° | [.35220,3.37277] | [21.90220,−8.77723] | +.457 / +26.615mm | TOP2、pair active、XY内、load .084206 |
| ep2 g95438/t2872 | [16.55,−12.15] / −6.25° | [5.93248,3.99733] | [22.48248,−8.15267] | +.314 / +43.788mm | TOP6、pair active、XY内、load .485286 |

三行 n均[22.8,−13.4]，c均0、FL headroom无裁减；load为记录的归一化载荷，不是N。ep2的正式placed为t2868，位于最后8tick区间内；表中目标/native是t2872末tick，不冒充2868精确事件时的驱动或实测q。ep0/1事件恰在决策末tick；三条 action/native source phase都是P05，末态已转P06。

成功接触之前的 gap 确实下降：ep0 t2816/2824/2832为8.248/4.575/.580mm；ep1 t2720/2728为16.962/.457mm；ep2 t2856/2864/2872为21.084/6.260/.314mm。这些样本的 hip反向偏置没有消失，ep0放置后甚至到−10°且仍TOP。故不能将反向偏置单独当成失败分类器。另一方面，训练 knee请求在所列放置行约3.37–6.03°，区别于C93184末段约1.58°；双关节组合并不相同，不能把差异只归因于hip反馈。

实际轮命令也不同，按FL/FR/RL/RR的 canonical rad/s：C93184 t4264为[−.06887,−.03809,+.01084,+.06625]；三个放置比较行为[−.09474,−.05345,−.00340,+.03925]、[+.03746,−.11599,+.06315,+.03391]、[−.03688,−.07357,+.08607,+.06564]。对应总support数分别常见3（C t4265短暂2）、4、4、2，不是固定同一支撑实验。ep1 t2736已再次AIR/load0，虽然placed历史保留；不能把历史placed当持续当前TOP。

## 5. 缺失证据与唯一下一步假说

当前训练 JSONL 的这些样本有请求、mapper/最终目标、native验证及每decision的8tick摘要，却没有逐tick实测关节q，也没有完整物理 raw/native tick文件；本次没有读取tensor。因而不能对训练成功样本复算相同 gain8反馈式、假定q跟随目标，或确定t2868的精确驱动。评估native审计也没有逐条持久化全部 mapper private tracking/nominal-reached/retirement flags；本文对稳定hip段的判定由真实n/m/q与源码更新律共同支持，不把膝部不变的+1.25°直接称为持续active feedback。

未有同一实测状态下更改反馈参考后的真实动作 counterfactual，也未控制base姿态、关节耦合与接触求解。因此不能给出“关闭/改写反馈即可落地”、正hip或正knee必然产生某个world-z方向、任务成功概率或首要因果占比；本报告不提议在当前训练中修改实现。

**仅保留一个可检验假说：在FL已Q+C且XY内的毫米级无接触窗口，nominal-only tracking会使小幅、持续的hip残差变化产生部分反向mapper响应，降低最终关节/间隙对该请求的持续响应幅度。** 下一次若决定检验，应以匹配的真实pre-tick状态、同一nominal及其他通道/支撑条件，对比短窗口请求变化后的β、最终目标、实测q和gap联合响应；仅比较r与q差值或静态代数不足以确认，更不能据此认定应修改控制。现有数据支持“存在反向反馈”，尚未支持“它独自造成FL不放置”。

报告完成后停止；live #27、生产、配置、测试、原始结果与master均未修改。
