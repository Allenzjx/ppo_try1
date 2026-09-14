# P06 block9 第二回合：BODY_CONTACT 与完整 rollout 尾终态

**范围/结果。** 仅一次固定扫描同 run `train/20260910T1155110470637Z_g7db0d17f398d_94db61a60fb54319aef3eb1d71d97e00` 的 audit 前256行，分析第149–256行及 completed_episodes 第2行。episode1、seed1001、global146069–146176，108决策；原始字节范围11,556,291–20,530,510（含端点，8,974,220 bytes），SHA256 `c7a9370c855e675b79187ff852e858b5dec5470b253874b634c06b70bc4f6bb2`。没有读取第三前缀或后来策略。

教师实到 P06 tick3584/29.866667 s 后执行858个策略物理tick，末态 tick4442/37.016667 s，**P09 BODY_COLLISION，task_success=false**。实际发出动作阶段：P06=68/P07=1/P08=1/P09=38；本块前两回合合计 P06=204/P07=2/P08=2/P09=48，共256。教师时间/历史不计策略信用。

## RR 事件与连续交接

| tick / 时间 | 实际动作阶段 | 事件/交接依据 |
|---|---|---|
| 4121 / 34.341667 s | **P06** | 新 I，excursion3.467 mm |
| 4126 / 34.383333 s | **P06** | 新 Q，excursion8.497 mm，台面净空-41.480 mm |
| 4128 / 34.4 s | P06→P07 | 当前有效抬升连续接管；rear_approach仅0.178，不是“已完成靠近前缘” |
| 4136 / 34.466667 s | P07→P08 | 同次 current functional lift，role_prepared_RR=1 |
| 4144 / 34.533333 s | P08→P09 | 同次抬升继续，transfer_ready_RR=1；edge_proximity_RR仅0.100 |
| 4201 / 35.008333 s | P09 | 真实 GROUND 撤销 Q/current lift |
| 4374 / 36.45 s | P09 | 新 I，excursion3.809 mm |
| 4379 / 36.491667 s | P09 | 新 Q，excursion8.843 mm |
| 4442 / 37.016667 s | P09 | BODY_CONTACT；AIR但current validity/continuation均false |

本回合 RR I2/Q2/回地撤销1/C0/P0，17个决策末态 current_lift_valid=true；不是悬停时长门槛。两次 I 的全身实测关节运动6.397°/15.524°、Q时RR自身运动4.429°/16.812°。前驱真实动作已产生抬升，而其向前运动和后续承载尚未完成，不能把提前阶段接管称为前缘/放置成功。第一次Q在P06，不能按记录末相P07或终态P09倒推发生阶段。

P06/P07/P08边界RR residual连续从[-5.818,5.539]→[-2.318,3.013]→[1.182,0.594]°，final从[-7.168,6.494]→[-3.668,3.967]→[-0.168,1.549]°；四轮仍有非零目标，Q在交接中保持直到真实回地。858/858 native effect verified、own-phase855、差额3为普通handoff hold，禁止的episode内state write=0；未发现新增reset/下发失效。

终态RR AIR、台面净空+93.036 mm，却还在前缘-204.909 mm、XY=false；较大离地不是越沿。FL AIR/有效载荷0，不能称它承载；FR TOP和RL GROUND为当前有效支撑，载荷比例0.571/0.429。history仍Q=true但当前body-control=false、continuation reason=`physical_safety_abort`，准确保留“曾经成立、现在不再有效”的区别。

## 机身碰撞证据的层次

terminal audit与completed一致：physical evaluator `valid=true/run_validity=VALID/physical_evidence_status=VERIFIED`，`TASK_FAILURE_BODY_COLLISION`、source=`BODY_CONTACT`、reason=`central body/obstacle collision`。代码 `semantic_supervisor.py:471–475` 只在 authoritative `observation.body_collision.detected` 为true时产生该任务失败；`sensor_reader.py:285` 接入 `BodyCollisionDetector`，该探测器只接受 verified exact base_link/Obstacle 活跃接触对，并要求连续≥2tick或live穿透≥1 mm。腿或wheel接触不能触发这一BODY条件。

**传感细节限制：** 本训练audit没有持久化原始 BodyCollisionStatus.real_pair_active/persistent/penetration 数值或原始BODY接触对，所以这里只能确认真实执行链消费了该权威检测结果，不能独立指定此次究竟为持续接触还是穿透分支。已保存body AABB min=[0.377146,-0.307422,0.050007725] m、max=[0.716988,0.080728,0.292741] m；AABB高度不是signed penetration，也不能用轮子无接触否定BODY碰撞。

终态372维policy/critic一致、finite_fallback=false。按既有schema及locked bottom0回解base_z≈0.025774449 m，gravity_z≈-0.882868588；不支持把这次BODY判定改成上一回合的低高度FALL。无原始接触细项不等于传感器被证实错误；保留原任务失败，不作阈值调整。首个后腿未完成任务仍为RR受控前送、越沿及放置。

## 尾拍与已保存更新：证据分层

前回合末20 + 本回合108 = 完整128 rollout，本回合末决策确为该rollout第128拍，虽然它因BODY接触仅执行2个物理tick。固定前两条 optimizer 日志：1106/global146048、1107/global146176均20步、finite_nonzero_gradient=true且actor hash改变；1107梯度norm约1.000–1.414。因此前256条现已优化，不再把上一回合20条描述为仍待更新。

不能仅靠整除断言deferred分支。本次另有主控直接观测：2026-09-10 12:17:23.7626987Z同时见terminal4442、optimizer1107/global146176，且下一教师前缀仍P02/t7.6 s；主控已核对146176 checkpoint/manifest hash一致与roundtrip true、source7db。**这证明checkpoint已落盘早于下一完整P06前缀完成**；旧eager路径必须先完成约29.8667 s前缀，才可返回末step并更新，故结合当前代码支持实际走了deferred尾拍分支。

`semantic_training.py:1208–1224`的终态回调先写两条终态证据并fsync；adapter随后设pending，collector完成returns/update及pending强制保存，再于下一rollout入口reset。audit/completed本身未保存`terminal_reset_deferred`字段，且没有独立逐事件timestamp trace；因此**没有直接时序记录单独证明“reset开始之前”每项操作的顺序**，该更细顺序依据代码，已观测时序严格限于“保存早于前缀完成”。本块第一回合是非尾eager，本回合是首个有上述实训佐证的尾终态，不把它称为任务/学习成功。

仅新增本报告；无生产、主报告、checkpoint或进程写操作，未读取后续活动物理数据。
