# Block10 自然 P01 首回合：FL 已真实放置，RR 回地后实际膝硬限中止

**固定证据。** Run `train/20260910T1233487014760Z_g7db0d17f398d_9984d1c7ed954d8fb2769e551a23fc6e`，仅一次解析 audit 前442行及首条 completed；episode0/seed1001，global146305–146746，物理tick0–3529/29.408333 s。原始前缀33,272,974 bytes，SHA256 `084437ccb4cae203d7025f1ca477df19e4eacca911a82018e3e1cc09f2b8d404`。这是自然P01随机策略训练，无教师前缀信用；未读取后续回合。

实际发出动作阶段数：**P01=3/P02=107/P03=6/P04=6/P05=146/P06=117/P07=1/P08=1/P09=55**，P10–P13=0。终态与completed均为 **P09 HARD_JOINT_LIMIT，task_success=false**，不是全程成功，也不是固定策略评估。

## 真实终止：RR knee 实测低于硬限

物理 evaluator明确记录 `hard joint limit: rear_right_knee`，source=`HARD_JOINT_LIMIT`，physical evidence VERIFIED。终态policy/critic各372维且完全一致、finite_fallback=false；按当前schema offset38/index7/scale90回解 RR knee **−60.015891°**，低于−60°硬限，速度约−1.380089°/s。最终实际下发目标仍为 **−58°**（最后四个决策末态均如此），不是命令越界或单纯舍入；nominal−37.8°、projected residual−25.929°的请求经过最终限制后发出−58°，实测响应却已越限。这里是保存的实测float32观测回解，不冒称原始双精度传感流复核。

同一终态从locked底面与relative plane回解base_z≈0.054935478 m，gravity_z≈−0.984511793，未显示高度/翻转FALL条件。原硬限安全中止独立有效；不因命令被限住或其他任务有进展而放宽阈值。控制响应受动力学/载荷等影响，此片段不支持指定单一原因。

## FL：确实放置，但承载不是全程连续

FL I1012/Q1014后，1490真实回地撤销；1526/1561再有I，第二次Q1567，随后 **C2105（17.541667 s）/P2139（17.825 s）**，P05→P06于2144（17.866667 s）由真实 placed_FL=1通过。history首次Q时间1014不能替代第二次尝试的Q1567。

P06入口FL当前TOP/support=true、有效载荷比例0.462；RR抬升及P06→P07→P08→P09时，FL却是 **AIR、载荷0、support=false**。终态FL重新为TOP，距台面仅+0.008956 mm、在前缘+343.781 mm、XY有效，bearing force **12.754680 N**、load fraction **0.432890**且验证有效、support=true。因此可确认末态FL实际承载，不能说它从P2139起始终承载，也不能用历史P虚构悬空期间的载荷。

## RR：准备阶段已创造抬升，但未完成越沿放置

| tick / 时间 | 当前动作阶段 | 证据 |
|---|---|---|
| 3070 / 25.583333 s | P06 | I：excursion3.667 mm，全身实测关节运动59.308° |
| 3075 / 25.625 s | P06 | Q：excursion9.037 mm，RR自身实测运动17.047° |
| 3080 / 25.666667 s | P06→P07 | 当前有效抬升continuous takeover；rear_approach=0.795，不冒称靠近任务满分 |
| 3088 / 25.733333 s | P07→P08 | 同次Q、role_prepared_RR=1，当前AIR有效 |
| 3096 / 25.8 s | P08→P09 | 同次Q、transfer_ready_RR=1；台面净空+5.602 mm、仍在前缘−261.296 mm |
| 3347 / 27.891667 s | P09 | 实际GROUND撤销Q/current validity；之后没有新I/Q |
| 3529 / 29.408333 s | P09 | 实际硬限终止，RR仍GROUND、C/P均false |

本回合RR I1/Q1/回地撤销1，34个决策末态current_lift_valid=true，**C0/P0**。末态距前缘−51.567 mm、台面净空−50.060 mm，已回地而不是保持离地越沿；GROUND bearing16.511965 N、有效载荷比例0.560411。此时FR AIR、不承载；RL虽GROUND但载荷仅0.006699，support=false。没有把接触、CoM方向或历史资格替代实际支撑。

三次后腿交接时RR final[hip,knee]从[−19.110,12.237]→[−19.437,8.737]→[−18.880,5.237]°，residual未清零且wheel目标持续非零，同一次抬升直到真实回地才撤销。全回合3529/3529 native-effect tick verified、own-phase3521，差额8为普通交接hold；禁止的episode内state write=0，未发现新的下发/状态重置缺陷。首个未完成后腿任务仍是RR保持有效抬升并越沿、受控放置，原物理安全失败保持。

**更新边界。** 固定前三条optimizer记录1109/1110/1111对应146432/146560/146688，各20步、finite_nonzero_gradient=true且actor参数改变；回合跨过这些更新，甚至RR I3070与Q3075分处第三次更新边界两侧。本证据切点共442采样、384已优化、末58尚未优化；不据此提升官方checkpoint指针。不能将本次FL放置称为相对不同条件video3的确定改善，更不能把P09到达视为完整成功。仅新增本报告，无生产、主报告、checkpoint或进程改动。
