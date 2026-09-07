# Block 29 final：workspace soft-potential，checkpoint 107264

正式结果为 **SUCCEEDED（训练预算执行完成，不是物理任务成功）**。只读审计完成运行 [20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c)，HEAD `f1a9bbf650b1b8f80d5169e09173fcfc68797d99`，N1 / seed 1001 / natural P01 / full_episode。源结果保留；没有读取 C107264 评估或任何后续课程。

## 1. 完成账本与保存

| 项目 | Source / initial | 本块新增 | Final |
|---|---:|---:|---:|
| global policy decisions | 103168 | 4096 | 107264 |
| PPO updates | 771 | 32 | 803 |
| lifetime optimizer steps | 15420 | 640 | 16060 |
| full_episode 消费预算 | 45312 | 4096 | 49408 |
| phase_suffix 消费预算 | 47744 | 0 | 47744 |

planned/requested/actual 均为 4096，unused=0、rounding_overrun=0、wall_time_s=2110.9012369001284。v3 origin 10112 不变；自 origin 的 29 个最终训练块累计新增 **97152 决策 / 759 PPO updates / 15180 optimizer steps**（origin 自带 44/880）。现有 full/suffix 100000 各自预算余量为 50592 / 52256；本报告不据此启动或预记下一块。

这是 workspace Phi 改变后的显式 NewMdpWarmStart，不是保留旧 Adam 的普通 exact-MDP resume。Source→initial actor（含异方差 std）、critic、identity normalizer 与完整 RNG 记录相等；initial 不加计数/预算，新 Adam 初始 3e−5、旧 rollout/physical state 不继承、合法 P01 reset。初始迁移详见 [首 128 固定报告](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/workspace_potential_first_103296.md)，不重复其大 metadata 或哈希计算。

| 实际保存 global | PPO / optimizer | full 消费预算 | roundtrip / normalizer 等于 source / actor 等于对应 update |
|---:|---:|---:|---|
| 103296 | 772 / 15440 | 45440 | true / true / true |
| 104192 | 779 / 15580 | 46336 | true / true / true |
| 105216 | 787 / 15740 | 47360 | true / true / true |
| 106240 | 795 / 15900 | 48384 | true / true / true |
| 107264 | 803 / 16060 | 49408 | true / true / true |

五次保存的 target runtime 相同、suffix=47744、source ledger=45312/47744、origin=10112。最终 [sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000107264_manifest.json) 与 [last pointer](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/checkpoint_last_pointer.json) 所记 checkpoint 路径/SHA 一致：`54449971c1233785133c33b4f362f85e6d449b7c73be4d4ea589d2db54e288d0`；pointer 记录 sidecar SHA `ab23007a4df8aa5be7598e5b738eaf29cad7c0912ec8de2625da993b9f1dda63`。本审计未重复哈希 PT 或 sidecar。

## 2. 全部实际阶段样本与回合分账

P01–P13 source-phase 向量为 **[6,931,22,5,1383,1749,0,0,0,0,0,0,0]**，总和 4096，与 final telemetry 相等。P07–P13 的 0 是未访问的样本数量，不是质量或成功评分。

| Episode | Global 范围 | 决策 / ticks / 秒 | P01/P02/P03/P04/P05/P06 样本 | 最后阶段与结果 |
|---|---|---|---|---|
| 0 | 103169–104115 | 947 / 7576 / 63.133333 | 1/196/5/1/144/600 | P06 INCOMPLETE |
| 1 | 104116–104756 | 641 / 5128 / 42.733333 | 2/185/3/1/450/0 | P05 INCOMPLETE |
| 2 | 104757–105694 | 938 / 7504 / 62.533333 | 1/183/5/1/148/600 | P06 INCOMPLETE |
| 3 | 105695–106324 | 630 / 5033 / 41.941667 | 1/173/4/1/451/0 | P05 INCOMPLETE |
| 4 | 106325–107264 | 940 / 7520 / 62.666667 | 1/194/5/1/190/549 | P06 **nonterminal tail** |

4 个 `completed_episodes.jsonl` 记录与对应 terminal audit 的 decisions/tick/duration/phase 一致，均为 `INCOMPLETE_CONTROLLER_BLOCKED`、task_success=false、full_task_success=false、physical valid=true / hard failure=null。完成回合合计 3156 决策；尾段 940 不是第五个失败，也不是成功。全窗口 4096 行 physical validity 均 true，无 finite fallback。

唯一短间隔是 **ep3 / g106324 / P05**：末决策只运行 **1 tick**，结束 tick 5033，仍是有效物理下的 incomplete，不是碰撞。故总 ticks 为 `4096×8−7 = 32761`；其余 4095 条均为 8 ticks，没有日志缺口。

## 3. 实际硬事件与首个未完成环节

Q/C/P 分别指真实 evaluator 的 hard qualified lift / front crossing / placement 历史，tick 均为各自 episode 时钟；“—”表示没有该事件，不用初始 clearance 代替 Q。

| Episode | FR Q/C/P | FL Q/C/P | RR / RL 硬 Q/C/P | 首未完成或尾段缺口 |
|---|---|---|---|---|
| 0 | 52/1587/1611 | 1729/2749/2775 | 全部 — | P06 rear_approach=0 |
| 1 | 55/1500/1515 | 1612/—/— | 全部 — | FL 尚未 crossed/placed，placed_FL=0.6256705552 |
| 2 | 60/1475/1508 | 1610/2636/2702 | 全部 — | P06 rear_approach=0 |
| 3 | 52/1397/1419 | 1508/2669/— | 全部 — | FL 仅历史 crossed、未 placed，placed_FL=0.7 |
| 4 tail | 55/1558/1600 | 1694/2730/3123 | 全部 — | 尚在 P06，rear_approach=0，不是终局 |

两次完整 P06 终局 RR/RL front distance 分别为 −0.875894/−0.723719 m、−0.783124/−0.629583 m；都未到原硬 workspace 下界 −0.22。新 reciprocal 是 soft potential，不将公共 rear_approach=0 改成硬完成，也不表示整 reward 为 0。

历史 placed 不等于当前支持：ep0/2 的 FR 在终局已 GROUND、FL AIR/load 0；ep1/3 的 FR 仍 TOP，但 FL 均 AIR/load 0。ep3 FL 虽历史 C=true，当前 front −0.0614522 m、clearance +0.0162602 m，不能据历史 crossing 称其当前已在可放置区。

最终非 terminal tail 的 RR/RL front 为 **−0.6417162384 / −0.4799942973 m**，都 GROUND、load 0.0479227600 / 0.4834657616；FR GROUND/load 0.4686114784，FL AIR/load 0、clearance +0.0341335287 m、front −0.0143765773 m。support_count=3 没有替代后腿资格或台面支持证据。没有本块 rear lift/cross/place、full success 或 suffix success。

首 2048 的前轮实际范围已有 [独立固定审计](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/workspace_first2048_authority.md)：新增幅度在那个固定窗口只被负向使用。该报告的 722 个 P06 样本不能直接推广成完整 1749 条的统计；本报告没有重新作或伪造全块方向归因。

## 4. Native、状态写入、时钟与 PPO/GAE

逐条核对全部 **4096 rows / 32761 compact native tick receipts**：

- 32761/32761 ticks verified，native effect=32761，own-phase request effect=32738；4096 个末 tick 的 setter/dispatch 相等、actual mapping 重建、same-tick counterfactual 标志均通过。
- 4096 个末 tick 的 previous-ACK tracking-reference 独立验证通过，headroom mode 均为当前版本；这不是声称每个 channel 每个 tick 都启动了 reference correction。
- 四类 in-episode root pose / root velocity / force-or-impulse / gravity 写入均为 0，4096 个 no-state-write 验证为 true。episode reset 是合法物理 reset，不属于禁用的 episode 内写入。
- 每次 reset 后 episode tick 从 1 连续计数；native command tick 与 episode tick 恒差 +179，直到对应 terminal/tail；无 global、episode 或 native tick 缺口。
- 23 次 source→end phase transition 全为 nonterminal、允许正常 bootstrap，不因阶段标签切换切断 GAE。time_outs 全 false；4092 个 nonterminal 允许 bootstrap，4 个真实终局都禁止 bootstrap 且 next potential=0。

源码路径仍是已安装官方 RSL PPO 的 `compute_returns`：gamma=.995、lambda=.95，TD/GAE 都用真实 done 的 `1−done` mask；rollout 尾部若 nonterminal 使用当前 obs 的 value，stage transition 不是 done。没有 peer truncation、教师 transition 或 timeout 补偿。末个 940 决策物理尾段虽未终结，其全部样本已进入最后完整 rollout/update，不等于未优化 tail。

所有 raw/mean/std 均有限 12 维，std>0，old LP/value/reward 有限。double Gaussian LP 重算与 float32 日志最大绝对差约 1.65055e−6；生产循环另有 storage raw 和 mean/std 的逐值相等检查。本报告未 PT-load 或重新计算 live returns tensor，因此不把日志数值复核冒充独立 bitwise GAE 认证。

全部 PBRS `5*(.995*Phi_after−Phi_before)` 与日志重算差为 0；同回合相邻 Phi 链误差为 0，reset 边界不强行连接。四个终局 terminal event 均 −40。日志 RSL float32 reward 总和为 −214.6312136032，与环境 double episode-return 分账存在正常精度差，不混用。

32 个更新均 actor_changed、finite_nonzero_gradient=true。源 actor → update 772 → 31 个相邻连接 → update 803 → final sidecar 的哈希链全部相符；每次恰 128 个决策/20 optimizer steps。gradient norm 范围 1.0003083272–1.4142136901。最终 actor `78064cd4…9f9a26`、critic `378e31ac…7688fb`，normalizer 不变。记录的 update-end LR：31 次 1e−5，update 780 为约 1.5e−5；不能据此断言所有 640 minibatch 的 LR 恒定。

本块 5 次自然 P01 起点，teacher/prefix credit=0，teacher roll-in ticks=0，全部 4096 优化样本属于当前策略 full_episode 预算。新 MDP 只在 initial 边界发生，块内 runtime/observation schema/固定 curriculum 不切换。

## 范围与结语

这是一块实际训练完成、未取得物理全任务成功的账本。既有初始迁移/首 128、前 2048、所有 A/B/C 结果和 failed receipts 保持原样；新旧 same-state logical action comparison 不证明新旧 Phi、完整 observation 或物理轨迹等价。

仅通过 PowerShell 读取完成 run29、侧车与源码，新增本报告并在主报告末尾追加 final 段。未运行 Python/PT/GPU/Isaac，未改 production/config/tests，未重复大文件哈希。C107264 及下一课程未读取、未预填。完成后停止。
