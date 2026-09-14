# Block21 ep0：P08 机身碰撞，未建立 RR 抬升

固定范围：run `train/20260910T2134137184858Z_gd4e46006b382_fd72960a0c724c8eb08ff331353fc15d` 的 policy audit **前153行、bytes [0,11628661)**，global158081–158233；单次解析时间 2026-09-10 21:51:34.275–21:51:34.330 UTC。另仅读取 `completed_episodes.jsonl` 第一行；未解析后续活动回合、物理大流或 checkpoint。

真实信用起点为教师初始化 **P06/tick3584/29.866667s**，并非自然 P01 策略回合。153个新策略决策中 P06=139、P07=6、P08=8；执行1221物理ticks，末决策4800→4805执行5ticks。1221 ticks全 verified/native effect成立，own-phase request effect=1219；所有记录确认无 episode 内状态写入。

## 交接和首次未完成任务

- **4696：P06→P07**，`rear_approach=1`，入口valid，done=false。
- **4744：P07→P08**，`edge_proximity_RR/RL=1`、`role_prepared_RR=1`，入口valid，done=false。
- 下一决策4704、4752的 jump audit 均为全12通道 previous residual 与 carried residual 相等，forbidden/phase-scale-clipped列表为空，使用已有handoff hold。没有普通切换清零residual或native下发失效的证据；4752 nominal四轮为[0,−0.63,0,0]，final四轮仍为[−0.489839,+0.432044,−0.900733,+0.479912]，不能把nominal零通道当成执行清零。
- 碰撞前的 **4800/P08** 仍入口valid，但 `edge_proximity_RR=1`、**`transfer_ready_RR=0`**；终态亦未完成该条件。第一未完成的后腿阶段任务是P08的当前载荷转移准备，随后RR有效抬升、越沿和放置也都未发生；不是已到P09的抬升成功。

## 当前接触不等于教师历史

教师FL Q2417/C3115/P3583都早于信用起点，不能计为本次策略新事件。首记录3592的FL确为TOP/support=true、verified bearing=3.330N；首次记录AIR在3608，此时bearing=0。之后4696和4744又实际TOP承载13.770N、15.786N，不能把整个回合概括成FL一直悬空，也不能用P历史证明它一直承载。

RR全回合**新I/Q/C/P=0**，历史亦未建立Q；153个端点中GROUND150、AIR3，短AIR没有形成有效I/Q。4744的RR仍GROUND、bearing=14.510N、front−206.314mm、台面净空−49.999mm。末4805的RR仍GROUND、front−202.291mm、净空−50.246mm、bearing=10.597N；FL/FR/RL当下都AIR且support=false、bearing=0，RR的other-support列表为空、body_control_evidence=false。末态RR载荷比例1是当时已测腿部载荷分配，不代表稳定的单腿支撑或碰撞因果。

本回合仅另见RL初始净空事件4713（P07）、4762（P08），没有RL Q；不可误算为RR事件。FL历史Q/C/P保留，未产生新的策略FL I/Q事件。

## 原始终态保持失败

completed首行：seed1001，153decisions，40.041667s，`BODY_COLLISION`，task/full success=false。物理评价明确 `TASK_FAILURE_BODY_COLLISION` / **`BODY_CONTACT`** / `central body/obstacle collision`，tick4805，run VALID、physical evidence VERIFIED。所读audit和completed终态没有具体collider名称、原始接触pair/力或signed BODY distance，因此不从body AABB或姿态虚构具体碰撞部件。终态372观测存在且finite_fallback=false，但本报告未把姿态回解代替碰撞传感证据。

这是原样保留的真实任务碰撞失败，不是完整成功；本次固定范围未发现新的明确执行缺陷，不新增训练门禁、不改变安全/任务标准或任何生产配置。
