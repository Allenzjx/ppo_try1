# Block19 已完成终止：固定4条，不跟踪活跃回合

读取时间 **2026-09-10T20:43:17.6670342Z**。唯一数据源：`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T2021542525340Z_gd4e46006b382_6707fa943a204563ab1699572bde18a9/completed_episodes.jsonl`，一次读取后固定为4条（episode_index 0–3）。已完成记录合计1153决策，不代表该时刻总采样、优化或保存计数。本报告未等待/追踪活跃回合，未读audit、physical、optimizer或checkpoint。

首回合ep0为580决策、P09/tick4640/38.666667s，`INCOMPLETE_CONTROLLER_BLOCKED / RR_FIRST_ORDER`；本报告不重做该次顺序判定诊断。以下是其余全部已完成回合，索引从0起。

| Episode | 决策数 | 终止phase / tick / 秒 | 末决策实际ticks | 环境reason | supervisor reason / source | FL C/P tick | RR C/P | RL C/P |
|---|---:|---|---:|---|---|---|---|---|
|1（第二回合）|150|P05 / 1193 / 9.941667|1|FALL|SAFETY_ABORT / PHYSICAL_SAFETY|无 / 无|无 / 无|无 / 无|
|2（第三回合）|259|P06 / 2067 / 17.225000|3|FALL|SAFETY_ABORT / PHYSICAL_SAFETY|1496 / 1952|无 / 无|无 / 无|
|3（第四回合）|164|P05 / 1305 / 10.875000|1|FALL|SAFETY_ABORT / PHYSICAL_SAFETY|无 / 无|无 / 无|无 / 无|

C=已记录前缘越过，P=已记录受控顶面放置；“无”同时核对了history布尔值false及event_ticks无事件，不把初始抬升当C/P。ep1与ep3只完成P01–P04，首先未完成P05 FL越沿/放置；ep2完成P01–P05，首先未完成P06当前任务，不能称已完成后腿越障。三者task_success/full_task_success均false。

## 实际终态接触与时间门

- ep1：FL AIR且不承载，front−107.054 mm；FR TOP实支撑，RR/RL GROUND实支撑。P05 age1.608333s < effective32.171868s。
- ep2：FL **TOP实支撑**，front+112.062 mm、top gap−0.679 mm，与已记录FL P1952相符；FR/RL AIR无支撑，RR GROUND实支撑。P06 age0.958333s < effective40s。FL真实放置不等于后续任务或全程成功。
- ep3：FL AIR不承载，front−155.165 mm；FR TOP实支撑，RR/RL GROUND实支撑。P05 age1.208333s < effective30.967450s。

三回合所有腿的bearing_verified及load_fraction_valid均true，末physical run_validity=VALID、physical_evidence_status=VERIFIED。这表示当前证据可用，不表示任务成功或安全通过。三次均为FALL独立安全终止，不是BODY/wheel-only任务违规、不是deadline，也不是某个视频/文件失败。末决策仅执行1/3/1 tick，不按完整8 ticks补算；`time_outs=false`、`terminal_bootstrap_allowed=false`。

## 硬限/FALL证据的真实范围

这4条completed记录的terminal_info **均不包含terminal_observation policy/critic的372维数组**，只记录 `terminal_observation_finite_fallback=false`。因此本次不能在“只读completed记录”的范围内回解base_z、gravity_z或逐关节float32终态，也不能声称独立确认FALL究竟走高度还是姿态分支。环境记录明确是FALL；physical evaluator的较宽文字为“fall or physics explosion”，不能用该笼统文字把具体环境结果改称数值爆炸。

现有终态transfer-role的 `receiver_workspace_state.joint_range_margin_deg` 覆盖8个servo，三回合记录的全部正/负方向margin均为正；最小分别为32.328636°（FL knee下界）、56.756519°（RR knee下界）、38.380134°（FL knee下界）。这是记录中的距离诊断，不是新增raw关节测量；与三个环境reason均非HARD_JOINT_LIMIT相符，不捏造某膝越界。没有读取额外audit来补充缺席的FALL数值。

结论：第二、第三、第四回合是不同真实状态下的未完成任务及FALL安全终止；仅凭这些终态没有证明新的判定器、执行链或学习消费者缺陷，也不推断不可测CoM或失败因果。生产、参数、主ledger、checkpoint与训练进程均未修改，本报告不增加训练门禁。
