# C101376：rear workspace 准备势函数的远距平段

只读诊断与一个未实施的软进度候选。未修改生产、硬判定、配置或现有报告，未运行 Python/PT/pytest/Isaac/GPU；当前 #28 不受影响。

## 1. 实际范围与结论

来源：[已完成 C101376](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/validation/20260907T0108106916731Z_g2677995544c9_3cfbdf407efc46619d280462e8ac47a4)，HEAD2677995544c9、seed2001、自然P01 fixed mean。正式结果966 decisions/7728 ticks/64.4s，P06 `INCOMPLETE_CONTROLLER_BLOCKED`、physical valid、task false、optimizer0。原结果见[既有诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_101376_diagnosis.md)。新增 tracking-reference 控制不属于这次评估。

本次流式读取已完成决策JSONL，固定检查P06的600个决策末态（t2936至7728）；不重扫120Hz raw，不把决策末态最值称为未检查中间tick的全轨迹最值。只读diff确认当前 `semantic_supervisor.py`、v3 task spec和reward config与本评估HEAD相同。

**成立的结论：在这600个实际末态，以及各状态保持其他输入不变的小幅接近/后退扰动下，RR/RL现有 workspace 准备分量均处于0平段，没有对接近方向的局部稠密变化。不能因此称整个势函数或reward为0。**

| P06末态统计 | RR | RL |
|---|---:|---:|
| 首front距离，t2936 | −.532472256m | −.541348366m |
| 最近front距离，各自独立最值 | −.492873205m | −.530957533m |
| 最近时距workspace下界−.22m | .272873205m | .310957533m |
| 终点front距离，t7728 | −.823927995m | −.855566439m |

两腿600/600均 `within_lateral_span=true`；600/600 `rear_approach=0`；全窗口前腿FR/FL历史placed为真、后腿RR/RL历史placed为假。以上最近位置不是同一时刻的配对构型。

## 2. 为什么准备分量确实没有接近方向变化

[predicate](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:650) 对workspace采用当前区间 `[a,b]=[−.22,+.06]m`。lateral内时，区间内返回1，否则 `clip(1−d/.25)`，其中 `d` 是距区间的距离。因此 `x≤−.47m` 时该值为0。此轨迹中RR/RL各自的最近决策末态仍比−.47m更靠后22.873/60.958mm；不是仅因两腿取min才看不见其中一条腿改善——**两条腿各自都为0**。

[physical_potential](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:664) 的固定分账为：

- RR的前驱FR/FL已placed，故RR进入完整未placed分支，其中workspace占腿进度 `.1*W_RR`。
- RL因RR未placed，进入predecessor未完成分支；当前v3 preparation opt-in仅给 `.1*W_RL`，没有偷给unload/lift/carry/capture。
- 全局系数是 `.85/4`，所以每条后腿的workspace份额为 `.02125*W`。此窗口两项均精确0；较远范围内单独改善x但仍未超过−.47m，不改变这两项。

这不排除离开平段之后出现进度，也不代表策略无法通过其他状态/动作获得回报。仅`rear_approach=0`本身不足以推出两腿都平；这里结论还使用了两条腿各自的实际front距离与lateral状态。

## 3. 其余信号确实仍在变化

同600条日志中，总 `task_progress_potential` 最小.36125、最大.44625；RR `load_ready`按现support/载荷公式复算范围0–1，首值.649202194、末值1。RR支持准备仍可贡献 `.02125*load_ready`；不能把它与RL前驱门控的零unload混淆。

已placed的前腿保留 `.8+.2*current_capture_retention`，不是永久固定1。首P06两前腿outside distance均0、clearance在允许保持区域；至t5328，FR/FL outside为.081388/.161018m，至终点为.255285/.335181m。现保持项会随当前区域/高度改变，且历史placed仍保留。

| tick | 日志总Phi | shaping | 总reward |
|---:|---:|---:|---:|
| 2936 | .438795547 | +.058007844 | +.053761668 |
| 5328 | .384184678 | −.010230171 | −.011806536 |
| 7728，终止 | .361250000 | −1.806250000 | −41.807814556 |

600条shaping及总reward均非0。末条的物理任务快照Phi仍.36125，但reward对终止采用 `Phi_next=0`，所以shaping为 `5*(0−Phi_before)`，另含−40失败事件等；不能用末快照Phi直接替代terminal reward的零后势。其余body stability、contact quality、actual-command smoothness及时间成本也不因workspace平段而消失。

## 4. 唯一候选：只替换软势函数内的workspace距离项

**未实施。** 保留所有`predicate`、entry、completion、rear_approach、TaskEvaluator Q/C/P和A/B/C共同硬成功规则不动。仅在新版本 `physical_potential` 的两处workspace/preparation份额中，使用：

```text
d(x) = max(a−x, 0, x−b)
W_soft(x) = .25 / (.25 + d(x))       （当前lateral span内）
W_soft(x) = 0                        （lateral span外，保留现语义）
```

复用原.25m尺度，不引入某个历史入口、固定关节构型或新距离阈值。区间内及边界d=0为1；区间外理论上严格在(0,1)，靠近区间连续增加、远离连续减小，远距离渐近0而不在.25m处硬截零。区间外对距离的导数为 `−.25/(.25+d)^2`；仍有远处梯度变弱，以及原lateral门外无进度的限制，不声称消除所有平段。

候选值仅是对已有记录的公式计算，不是新物理执行：

| 实际点 | 旧W | 候选W | 候选全局workspace份额 `.02125*W` |
|---|---:|---:|---:|
| RR首点 | 0 | .444466367 | .009444910 |
| RL首点 | 0 | .437561416 | .009298180 |
| RR最近点 | 0 | .478127388 | .010160207 |
| RL最近点 | 0 | .445666535 | .009470414 |
| RR终点 | 0 | .292764731 | .006221251 |
| RL终点 | 0 | .282305188 | .005998985 |

当前轨迹从首点到终点明显后退，因此这两项的候选总贡献将从.018743090降至.012220236，并非一律给更多正reward。仍用原PBRS `5*(.995*Phi_next−Phi_before)`；固定非terminal状态得到负的折扣势差，不会仅因静止在远处得到正shaping。接近提高后势及其相对shaping，但极小接近也不保证抵过折扣差/时间或其他成本。

**浮点与门控：不能直接把这个reciprocal值替换公共predicate。** 极小正d可能使双精度比值舍入到1，更不用说float32观测；公共predicate又被`>=1`用于entry/completion。候选严格放在势函数内部，硬workspace仍由原明确区间分支判断，因此不会因新软比值舍入而提前触发完成。soft值即使舍入到1也不授予历史或成功；它也不应成为新的nominal调度输入。

同一个helper应覆盖所有未placed腿原有的workspace份额，而不只特殊奖励某一条后腿；已placed retention、其余unload/lift/carry/capture权重与顺序均保持。新的soft W不增加reward family，整体Phi原有[0,1]界仍保持。

## 5. 版本边界和唯一检验目标

观测仍为324列，但已有task-progress-potential标量会变化；物理动作、传感器、硬事件/成功条件不变，reward MDP仍已改变。若采纳，应显式新MDP迁移并使用fresh rollout，保存来源与新势函数版本，保留旧评估/训练结果；不能按同MDP exact resume或旧reward继续声称无变化。旧spec不启用时必须保留旧行为。

唯一可检验目标是：**在真实记录中的远距/lateral有效/未placed准备状态，只改变当前到workspace区间的距离，是否能让原 `.1` 准备份额连续、单调响应，同时硬entry/Q/C/P/成功和历史完全不变。** 这可先用同状态公式正反例核验，再在未来实际训练中记录该分量是否被触及；不将公式响应等同于推进、放置或任务成功，也不将本次P06失败全归因为这一平段。

固定范围诊断结束。仅新增本报告，源数据、master和运行中#28均未改变。
