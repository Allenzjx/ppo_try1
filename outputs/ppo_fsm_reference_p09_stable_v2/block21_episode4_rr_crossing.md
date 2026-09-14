# Block21 ep4：新 RR 越沿成立，放置前机身碰撞

固定读取同 run `train/20260910T2134137184858Z_gd4e46006b382_fd72960a0c724c8eb08ff331353fc15d` 的 audit **行749–914，bytes [58643414,71867459)**（仅跳过此前文本、不重新解析）；2026-09-10 22:28:11.969–22:28:12.170 UTC 单次解析166行，global158829–158994。另读取completed第五行；未读取后续活动数据、checkpoint或大物理流。

真实教师接管点为 **P06/tick3584**，prefix_attempt_index=4；教师FL Q2417/C3115/P3583、FR历史均不计策略信用。起点RR I/Q/C/P未成立。新策略166决策：P06=139/P07=1/P08=1/P09=25，执行1323物理ticks；最后4904→4907执行3ticks。1323 ticks全部native verified，own-phase effect=1320，全部记录无episode内状态写入。

## RR 事件链：两次尝试，一次实际越沿

| 事件 | 真实 tick / 来源阶段 | 证据 |
|---|---|---|
| I → Q（尝试1） | 4691 → 4696，P06最后决策 | Q向上变化8.689228mm；4696 AIR/current Q=true，front−269.272mm、台面净空−41.812mm |
| 继承 | P06→07=4696，P07→08=4704，P08→09=4712 | 三次明确记录 `qualified downstream motion already active; continuous takeover`；同一尝试current Q连续true |
| 回地撤销 | 4724，P09 | qualification-revoked与current-lift-revoked是同一次回地，不能重复计数；4728 GROUND/current=false |
| I → Q（尝试2） | 4764 → 4773，P09 | Q向上变化8.744932mm；4776 AIR/current=true |
| **C** | **4840，P09，line905/global158985** | `event_ticks.front_edge_crossed.RR`首次出现4840；前一端点4832仍C=false/front−11.142mm；4840为AIR/current Q=true、front **+1.443211mm**、台面净空 **+57.477652mm** |

因此本回合新 **I2/Q2/回地撤销1/C1/P0**。C不是仅从终态历史boolean推断。第二次Q从4776至4904的全部17个保存端点均当前有效；连同第一次4个端点，共21个current-valid端点。历史 `event_ticks.active_lift.RR=4696` 是首次Q，不应用它替换第二次Q4773。

碰撞前4904 RR仍AIR/current=true、front+35.520557mm、净空+33.991390mm，C=true/P=false。终态4907仍AIR、front+36.113027mm、净空+31.650694mm；`lift_established`及C历史保留，current=false、motion_continuation=false，明确原因 **`physical_safety_abort`**。这不是第二次回地：末次ground-revoke仍是4724。全回合P始终false；第一尚未完成的RR任务是**真实受控放置**，不是初始抬升或前缘越过。

## 实际支撑与动作继承

4696/4704/4712的FL均AIR、support=false、bearing=0，不能用教师P历史虚构当前承载；此时真实其他支撑是FR TOP与RL GROUND，均verified。4696 FR/RL bearing=13.120/14.455N。到C4840，FL已实际TOP/support=true、bearing=3.537N，FR TOP=14.132N、RL GROUND=11.714N；这是对应时刻的支撑，不是整个动作链一直成立的固定组合。

4904 FL仍TOP、3.334N；4907碰撞终态FL转AIR且bearing=0，FR TOP=8.861N、RL GROUND=13.115N仍verified。该同时变化不足以认定FL失载是机身碰撞的单一原因。

三次普通交接都done=false；下一决策的全12通道 previous/carried residual逐项相等，无forbidden或phase-scale clipping，已有handoff hold按记录生效。4704/4712/4720的nominal四轮都为+0.3，residual和final轮命令持续非零，例如4720 final=[+1.181266,−0.865835,−0.521053,−0.300000]。这是端点与native审计证据，不虚构未读取的逐物理tick动作曲线，亦没有发现清零或无效下发的新证据。

## 终态与限制

completed第五行确认episode_index=4、seed1001、166decisions、40.891667s、BODY_COLLISION、full_task_success=false。物理评价为 `TASK_FAILURE_BODY_COLLISION` / `BODY_CONTACT` / `central body/obstacle collision`，tick4907、VALID/VERIFIED。所读记录未给具体collider、原始接触pair/力或signed BODY distance，不由姿态猜测具体碰撞部件。

这是教师P06前缀后的新策略RR越沿事件，**不是自然P01完整成功，也不是完整后缀成功或固定策略性能提升证明**；RR放置未完成且原机身碰撞失败保留。未修改生产、配置、标准、checkpoint或主ledger。
