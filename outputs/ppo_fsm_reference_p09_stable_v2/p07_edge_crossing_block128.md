# P07 128 决策块：RR 边缘重试、越沿与未放置

证据固定为 `20260910T0647306226372Z_g06716a88bcc9_c6cfcadd350b418a9969f7dfc35406a5` 的最终 128 行 audit（11,054,957 bytes），6 行 completed_episodes、1 行 optimizer_updates、training_manifest；对照快照 `20260910T082308229118Z.json`。重点为 episode6 第72–128行，g141640–141696，共57次策略决策；没有扫描其他旧运行或重跑仿真。

**结论：取得真实 RR 边缘抬升、脱离接触及一次越沿事件，未取得放置。** episode6 最后是 tick6368/53.066667s，P09、terminal=false、termination=null；这是收满训练块后的未完成后缀，不是第7次任务终止，更不是完整成功。之前6回合均 BODY_COLLISION，原结果保留。

## 两次尝试必须分开

| episode6实际时刻 | RR事件／当前状态 |
|---|---|
| 5995 /49.958333s | 第一次 I：真实增高3.410mm、全身实际运动22.282°。 |
| 6001 /50.008333s | 第一次 Q：增高8.601mm；6008–6048有6个决策末端当前有效AIR。 |
| 6055 /50.458333s | 回地；Q及当前有效性撤销，未C/P。 |
| 6128–6144 | FRONT_WALL末端接触，增高0.421→1.104mm，无I/Q；此时旧资格已撤销，不能写成同次有效AIR直接碰边接续。 |
| 6152–6160 | GROUND_AND_OBSTACLE；随后还有地面重试，仍无当前Q。 |
| 6240 /52.000000s | 边缘新I：增高3.338mm；FRONT_WALL，实际轮端移动3.294mm、全身实际运动4.638°。 |
| 6249 /52.075000s | 新Q：增高8.297mm、RR关节运动6.532°；不是等AIR计时。6256末端仍FRONT_WALL、air_duration=0，Q/current_valid=true、允许连续运动。 |
| 6264–6296 | OBSTACLE_AMBIGUOUS接触继续；不是纯AIR，也不能伪定精确接触面。实际边缘调整响应为true，未清掉本次Q。 |
| 6304 /52.533333s | 已真实AIR，沿同次尝试继续；台面净空仍−14.841mm、前缘距离−33.342mm。 |
| 6328 /52.733333s | 历史C首次成立；AIR、前缘+0.973mm、台面净空48.437mm，尚未TOP接触／放置。 |
| 6344→6368 | 前缘最高+3.577mm后退至−23.148mm；末端within_top_xy=false、AIR、Q/current_valid仍true、历史C=true、P=false。 |

`history.event_ticks.active_lift.RR`仍保存首次Q6001；本次重试起点应读完整事件中的6249，不能把旧首Q当作新尝试起点。I6240事件的 `evidence=active_air_or_early_top` 是既有泛化标签，不是Q6249的证据字段；本报告以同期明确contact_mode解释边缘状态，不把该字符串当纯AIR证明。64c0324消费者核查未发现生产读取此字符串或首次Q时间戳来判定当前资格；控制、reward与372维观测使用当前接触、几何、实际响应及本次成立状态，视频按真实物理观测独立重放。这里是日志标签/首次事件索引的局限，不是已证实的AIR误判或固定计时门。

## 实际全身协同，而非只看发出的 RR 角度

6240→6328，实测canonical：RR hip −5.611→−17.080°、knee −50.358→−35.133°；RL knee +2.583→+23.710°；FR knee +40.417→+16.358°；FL hip +60.462→+51.951°。同段body前进坐标+155.609→+223.410mm；CoM位置另有独立测量，不能把它冒作body速度。

当时主要实测支撑为FR/RL；FL仍AIR、不承载，直到6336起决策末端才见FL重新实际支撑。不能称“FL悬空已经承载”或“抬FL必然抬RR”。6240→6296边缘过程RR载荷valid=false，但独立几何／CoM角色valid与motion_fraction仍保留；并未把未知负荷解释为确定零载。

6328的RR nominal hip/knee为−6.9/−37.8°，projected residual−14.686/+5.628°，最终target−20.336/−33.422°，实测−17.080/−35.133°。wheel最终FL/FR/RL/RR为+0.281/−0.204/+0.233/+0.855rad/s，实测+0.283/−0.396/+0.456/+0.853。指令、限速／反馈后的目标及实测不能合并成一个量；真实前送不只来自RR单关节。

边缘段没有额外统一wheel强推；日志reason为source_and_residual_adjustment_continue，fixed_lift_timer_gate与above_top_15mm_action_gate均false。后续AIR安全接近条件成立时才出现额外rolling建议。当前有效抬升来自几何、实际响应、其他支撑与既有安全状态，非持续时间达标；边缘/AIR时长分列，仅诊断。

仍有明显未解决问题：越沿时body角速度模达到2.312rad/s；末端虽降至0.170rad/s，body也自6328后退约23.0mm，RR离开放置区。126.702mm最大地面相对抬高、历史C或较长AIR都不能代替受控捕获／放置。首个未完成任务现为 **P09在当前可用位置完成RR真实放置**；P10–P13本块仍无策略样本。

## 训练记账与边界

全块P07=7/P08=7/P09=114；新RR I=5、Q=3、回地撤销Q=2、C=1、P=0；25个决策末端当前有效（其中episode6为21）。所有这些探索发生在同一个更新前策略下，不能称optimizer已改善策略。之后确实完成新增128 policy decisions、1 PPO update、20 optimizer steps，累计141696/1072/21440，checkpoint为 `checkpoints/history/checkpoint_step_000141696.pt`。训练lifecycle成功仅表示更新／保存完成，不表示越障成功。

本报告只读分析；未改生产、动作范围、reward或仿真。下发验证能证明控制链产生目标影响，不足以证明接触冲量因果；训练audit缺全量物理流，不能离线重验每个tick姿态／力。当前新checkpoint还须重载自然P01评估，不能把教师后缀事件升级为完整PPO成绩。
