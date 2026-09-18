# CP178432：封存 P06 后缀训练的 RR carry / 交接诊断

## 结论

本块 **512 decisions、4 PPO updates（1356–1359）、80 optimizer steps** 真正执行并封存，但没有完成任务：成功N前缀接管后，同一episode的本块学习段为 tick2688–6776，仍非终态、终止原因为空，最终 **P12 / RL 未越沿放置，RR 已退回 ground**。`SUCCEEDED` 是训练执行状态，`full_task_success=false`；不能称后缀成功，更不能称 P01 完整 PPO 成功。

新 AIR 资格逻辑在实际训练中记录了真实自由抬升、落地撤销、再抬升；新窄源许可也实际执行。它们**没有证明持续 carry 或放置后守住平台**。已定位的后段失守是实际源动作、支撑/机身状态与非零策略共同作用的轨迹；未从这些证据确认 mask、遗漏 stop 或非法 owner 覆盖错误。不能仅凭单次轨迹删除原 N 的全部反向动作。

源 run：`20260917T0454003203604Z_g6c2121b68654_7079dc3b6d4f471d8d3dfe5404d49924`；封存 `2026-09-17T05:10:23.323174+00:00`。主窗口4264–6776，共315个原保存决策快照；另仅列4000–4256的源许可前驱上下文。源记录每8物理步一个决策快照，事件列表保留真实物理 tick；没有逐步物理日志，不能把快照间未知接触补成连续 AIR。servo actual 是最后 dispatch 前一物理步；wheel qd、事件与当前 evaluator 是决策末步。原 CP177152 标签/证据不变。

## 1. 不是一条连续的抬升

| 真实事件 tick | 记录 |
|---:|---|
| 4001 | RR 首次 qualified，unsupported AIR gain **8.058 mm** |
| 4376 | ground：撤销 qualification/current lift，尚未 crossing |
| 4640 | 新 AIR qualified，gain **8.174 mm**；基点4626 |
| 4668 | ground：再次撤销 |
| 5108 | 新 AIR qualified，gain **8.517 mm** |
| 5259 | ground：再次撤销 |
| 5434 | 新 AIR qualified，gain **8.360 mm**；基点5403 |
| 5726 / 5731 | crossing / placed 事件 |
| 6053 | ground：当前 lift 撤销；明确保留已发生的 crossing history |

因此4280（AIR，freegain78.173 mm）与4656（AIR，9.858 mm）属于不同 AIR/attempt 上下文：不是同一 AIR 段仅减了68 mm。4280 的 reference tick3968，4656为4626。两次 ground 撤销之间的新资格用本 AIR 段自由增高，不是旧 ground 总高度。

最后一次5434资格到crossing也不是全程无接触：5496快照 gap−12.119 mm；5512为 AMBIGUOUS 接触、反力4.400 N、bearing/load validity=false；不能将记录中的数值0解释为真实无载。5520已回 AIR，reference tick5516，此后 AIR 自由增高到5720为21.720 mm、gap+3.262 mm，随后越沿。该序列包含真实自由增高和允许的接触延续，不等于旧 CP177152 的“接触增高直接新建资格”；但有限快照无法唯一拆分机械贡献，也不构成最终稳定成功。

FL 也不能被统一描述为一直承载或一直悬空：5440仍 AIR、0 N，RR实际其他支撑为FR/RL；5504 FL有2.656 N、5736为6.629 N。5440过去0.5 s的 CoM→FL 投影是−22.092 mm，5488转为+27.277 mm、5736为+20.239 mm。运动与承载分别记录，不虚构“抬FL必然抬RR”。

## 2. 窄源保护确实参与，但不覆盖已开始的整个前送

保存了7个新就绪诊断，均 `current_evidence_fresh=true / ready=true`，没有 wait：

- 六个 pending knee：4104–4144，真实 gap **115.794～137.022 mm**，其他已验证支撑3个；不满足低于15 mm的下降等待条件。
- 首次正向四轮组：4256，AIR/current-valid，gap **30.004 mm**、实测 bottom-vz **+0.04383 m/s**，允许继续。4264已见四轮 N 均+0.3。
- 4304下降到13.570 mm、4328到0.471 mm时，原 knee waypoint 已消费，没有待执行 knee 事件；4376才落地。因此这是**启动后 carry 丢失**，不是新 pending guard 漏掉一个当时应该等待的源组。
- 落地后原有限四轮脉冲继续，4672快照显式 stop 已使 N四轮=0。旧 late-reconfiguration guard 随后将P09 source clock保持648，直到5728首次保存恢复推进。不能把它与新 free-lift pending guard 的“0次等待”混为一谈。

新规则是窄事件许可，不是每tick强制停车/保持净空的控制器。当前记录不能宣称它已经解决所有后续下落；也不能因为它未做设计外的持续干预，就判定实现错误。

## 3. 放置后失守的时间线

| tick | RR / 阶段与源动作 | 支撑、运动事实 |
|---:|---|---|
| 5728 | 越沿后P09 late全身组恢复；四轮N `[-1.07,0,0,0]` | RR仍AIR，front +3.61 mm |
| 5731 / 5736 | RR placed；P09→P10 | 5736 TOP、front+14.61 mm、RR bearing2.76 N；capture只退休已有RR servo owner，保留后来协作 |
| 5744 | P10→P11；RR nominal knee已从−37.8变−29.3° | FR已无支撑；RR14.82 N、FL11.73 N、RL2.52 N |
| 5752 | P11；RR仍TOP但current-valid=false | 其他已验证支撑仅FL，**不是RR已经落地**；RR仍14.06 N |
| 5760 | RR front达到快照峰值+19.41 mm | CoM x速度已−0.0178 m/s，此后主要回退 |
| 5896 | P11→P12 | RR仍TOP、front+10.36 mm；CoM x速度−0.0799 m/s；回退早于P12四轮反向组 |
| 5944 / 5952 | 首次保存P09显式stop后四轮N=0 | stop没有被吞掉 |
| 5960 | P12 authored四轮N均−0.3 | final四轮 `[-.307,-.734,-.219,-.551]`；policy对FR/RR进一步负向修正 |
| 5976 | 首次保存RR已AIR且front<0：−4.56 mm | RR qd−.568 rad/s，CoM x速度−.1551 m/s |
| 6053 / 6056 | 实际ground撤销 / 首个保存GROUND快照 | 6056 front−47.38 mm、gap−50.00 mm |
| 6776 | P12尚未完成 | RR ground、front−130.729 mm；RL未cross/place |

源码和 contract 的有限核对支持动作来源：P09源5.4 s有late全身原子组、7.2 s显式四轮stop；P10源只改变RR knee；P12源0.4667 s至2.6667 s有四轮组，原wheel integral均为负。`_continuous_advisory` 让未完成前驱的已触及通道继续，较新实际修改的通道取得所有权；capture退休旧servo owner，不永久锁死后来P10动作。记录中的这些持续值与该语义相符，不能称为mask泄漏。

放置后的当前支撑先发生变化，随后回退；新P12负向轮组又出现在离台前。单轨迹只能确定这一时间顺序，不能把全部回退归因于P12、FL轮或RR knee某一个通道。RR nominal knee虽然允许后续协作，但“历史placed已真”不等于后续负向前送适合当前支撑与运动趋势。

## 4. 策略不是零残差，也不是本段raw饱和失控

5736–6056段 raw最大绝对值 **0.81735**、保存conditional mean最大 **0.74971**、有效sigma最大 **0.31709**。没有这一窗口的|raw|>3饱和证据，不能据此再盲降温度。

5736 FR knee N=31.1°，最终target=−1.881°，actual=−3.305°；5744 actual仍−1.896°而FR支撑已丢失。实际全身构型显著不同于source N；但 projected residual包含最终slew/历史相对计算nominal的有效偏移，不应把target−独立重算N全部说成当拍raw的作用。

5960四轮raw为 `[-.389,-.379,+.082,-.476]`，保存conditional mean `[+.095,-.278,+.060,-.232]`，innovation `[-.484,-.101,+.022,-.245]`。这是已有FR/RR负向条件均值加当前innovation；5976 RR raw−.504、mean−.490，负向已进入HISTORY条件分布。同期N本身也为−.3，因此不是policy单独“把正向N抵消成后退”。

更新边界在ticks3704、4728、5752、6776；5752样本产生于第三次更新之前，之后样本使用更新后的网络。这是持续随机训练轨迹，不是最终CP178432固定网络的确定性评估，也不是严格策略因果对照。未重算任何policy forward、GAE、return或advantage。

315个快照的12维mask全部1，dispatch audit全部verified。它们支持“当前窗口没有被许可mask关闭或记录内派发不一致”，不证明全部物理跟踪正确。比如RR final为小正轮速时其TOP实测qd仍可为负，不能只用ACK判推进方向。

## 5. 有据的下一步与边界

先完成正在运行的正式自然P01评估；本轨迹只作后腿真实样本，不替代完整评估，也不作为阻止optimizer的门禁。

若正式评估重复该失守，优先窄查**当前支撑/后退趋势与P09 late→P10 knee协作→P12反向组的接续条件**，并用成功zero对应交接作正例。保持连续动作、全12 residual、合法stop；不要直接永久禁用负轮速、恢复固定旧膝角、引入固定姿态或要求停稳。当前证据尚不足以选定单个控制修复。

学习信号另需注意：6056 RR已ground，但当拍task-progress reward仍为+0.03001，terminal_event=0，质量epsilon=0。这不自动证明reward公式错误（潜势同时含其他任务状态），但说明该失守没有立即终端失败信号，不能仅凭历史placed信用解释为已守住后腿。未重算原GAE；是否需要回退相关信号应结合正式完整结果再决定。

机器可读逐快照与原事件：[training_RR_carry_178432.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/training_RR_carry_178432.json)。分析器：`analyze_training_rr_178432.py`。所有新增文件位于outputs，生产/配置/当前Isaac未改。
