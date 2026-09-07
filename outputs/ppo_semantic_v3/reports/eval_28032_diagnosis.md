# C28032 自然P01确定性评估：RR通过后失去放置支撑，RL越沿前落地

结论：保留正式结果 **P12 INCOMPLETE_CONTROLLER_BLOCKED**。这次自然P01确定性策略确实获得RR合格抬升、cross与placed；随后RR退回前沿后方并重新落地。RL也曾合格抬升，最高无接触顶面净空约21.624mm，但尚远在前沿后方，越沿前落地使资格被正式撤销，最终无RL cross/place。**这是RR子目标进展，不是完整任务成功。**不重分类原结果、不据此变更任何判定或控制规则。

## 来源、实际计数和独立终止

- Run：`runs/ppo_semantic_v3/validation/20260906T0735108936738Z_g64abc5357d00_b94fa750838144058479e4640e9a831d`。
- HEAD=`64abc5357d00763419fe51c8126db8454fcf7697`，checkpoint28032，SHA256=`9e1cc652ae19cbee87f0612051b62675a6397671b92d5ff8ccb18da344b355cb`。seed2001，`from_phase=P01`，deterministic policy，无教师prefix、无迁移文件，evaluation optimizer updates=0。
- 正式 **1203个policy decisions /9617个physics ticks /80.141666667s**。前1202决策各8ticks，最后1tick真实终止；raw observations为9618行（含tick0），全部finite。1203行策略audit合计9617 verified/native-effect ticks，所有in-episode root pose/velocity/force/gravity写入均零。
- `task_success=false`、`controller_task_success=false`，`window_ended_before_task_terminal=false`，因此不是3000决策窗口提前截断。最后`time_outs=false`、`terminal_bootstrap_allowed=false`。
- 最后controller因P12阶段30秒期限阻塞；独立physical evaluator同样success=false、final_region=false，但其hard termination reason为null，不能捏造额外WHEEL_ONLY_CLIMB或身体碰撞。`physical_failure_is_not_interface_failure=true`。

| policy source phase | 决策数 |
|---|---:|
| P01 | 1 |
| P02 | 190 |
| P03 | 4 |
| P04 | 1 |
| P05 | 148 |
| P06 | 314 |
| P07 | 1 |
| P08 | 1 |
| P09 | 90 |
| P10 | 1 |
| P11 | 1 |
| P12 | 451 |
| P13 | 0 |
| 合计 | 1203 |

P08→P09 tick5280 /44s，P09→P10 6000 /50s，P10→P11 6008，P11→P12 6016 /50.133333s。普通阶段转换没有结束episode。P12最终stage age30.008333s，完成目标`placed_RL`仍未满足；0.261579是记录的连续进度，不是完成布尔值。

## RR真实qualified、接触越沿与placed

以下表直接对齐120Hz原始轮bottom geometry、精确body-pair接触和判定历史。front=bottom.x−当次obstacle.front_x；clearance=bottom.z−top_z，单位mm。载荷N来自真实接触传感器，不从placed历史推断。

| RR事件 | tick /时间s | front mm | clearance mm | 当前接触与FL载荷 |
|---|---|---:|---:|---|
| P09首物理tick | 5281 /44.008333 | −182.518 | −50.573 | RR GROUND 4.238N；FL AIR 0N |
| 首initial | 5350 /44.583333 | −218.114 | −46.187 | RR AIR；FL AIR 0N |
| 再次initial | 5777 /48.141667 | −29.081 | −11.781 | RR AIR；FL AIR 0N |
| **hard qualified** | **5820 /48.500000** | **−21.374** | **+0.442** | RR AIR；FL AIR 0N |
| P09最高无接触净空 | 5928 /49.400000 | −5.727 | +5.966 | RR AIR；FL obstacle 1.071N |
| qualified后首次障碍接触 | 5956 /49.633333 | −7.527 | −2.427 | RR obstacle 6.079N，无GROUND；FL obstacle 3.365N |
| **cross** | **5998 /49.983333** | **+0.137** | **−1.871** | RR obstacle14.885N；FL obstacle14.697N |
| **placed** | **5999 /49.991667** | **+0.564** | **−1.849** | RR obstacle14.205N；FL obstacle14.101N |

RR qualified事件同时记录实测upward excursion15.659317mm、joint motion1.107194deg。cross时并非AIR：它是已有真实资格之后的接触越沿链；报告保留共享判定器的cross/placed记录，不把约毫米级负净空重新判成wheel-only。决策末tick6000，当前evaluator明确RR TOP，FL亦TOP/load fraction0.496327。

但通过前沿的余量很小，且后续没有保持放置：

| placed后的真实演变 | tick /时间s | RR front /clearance mm | 接触证据 |
|---|---|---|---|
| 进入P12前最后tick | 6016 /50.133333 | +1.746 /−1.755 | RR仍obstacle14.062N；FL15.745N |
| **首次退回前沿后方** | **6028 /50.233333** | **−0.151 /−1.752** | RR仍obstacle13.623N；FL13.875N |
| root最初关注的52s | 6240 /52.000000 | −32.343 /−12.393 | RR仍obstacle13.811N，但evaluator已非TOP；FL TOP/load0.493188、13.839N |
| placed后首次无障碍接触 | 6329 /52.741667 | −46.339 /−32.593 | RR AIR；FL obstacle5.958N |
| **首次重新GROUND** | **6356 /52.966667** | **−48.902 /−49.299** | RR ground20.833N；FL obstacle18.003N |

到RL hard qualified的6387时，RR已经GROUND。因此旧RR placed=true不能当作RL尝试期间仍由RR顶部承载的证据。RR跨过前沿后完成的历史字段保留，是事件账本语义，不会保证当前接触。

## RL确曾合格抬升，但越沿前GROUND撤销

RL早在自然P01的tick11已有initial诊断；它不等于P12完成资格。后腿接续段又出现initial6007、6077、6363；真正hard qualification只有6387，随后被明确撤销。

| RL事件 | tick /时间s | RL front mm | clearance mm | 实际证据 |
|---|---|---:|---:|---|
| **hard qualified** | **6387 /53.225000** | **−238.687** | **+1.918** | RL AIR；RR GROUND12.838N，FL obstacle14.850N |
| **P12最高无接触净空** | **6402 /53.350000** | **−248.967** | **+21.624** | RL AIR；RR GROUND13.272N，FL obstacle14.471N |
| **资格撤销** | **6445 /53.708333** | **−178.612** | **−50.191** | RL GROUND5.803N，obstacle0；事件名`qualification_revoked_ground_before_cross` |
| 后续initial | 6468 /53.900000 | — | — | 新离地尝试，之后没有新的hard/cross/place事件 |
| P12首次RL障碍接触 | 6561 /54.675000 | −51.323 | −50.131 | RL ground2.732N +obstacle1.768N；发生在资格撤销之后 |
| **正式终止** | **9617 /80.141667** | **−50.526** | **−50.454** | RL ground6.414N +obstacle4.947N；无cross/place |

RL qualified事件记录upward excursion53.409692mm、joint motion21.038512deg。它的确抬得足够高，但高点仍距前沿约249mm；未把这次离地转化为越沿和稳定放置。终止active_lift_RL=false、cross_RL=false、placed_RL=false；不能把保留的旧event tick6387误读成当前资格。

## 当前接触与FL承载不能由历史替代

- 全P09窗口5281–6000共720ticks：RR AIR254ticks；FL AIR569ticks、obstacle接触151ticks。RR initial/qualified时FL无实际载荷，到RR跨线放置时FL才承担明显障碍载荷。不是FL一直稳定承载。
- P12窗口6017–9617共3601ticks：RL AIR445ticks；FL obstacle3579ticks、AIR22ticks。RL最高点时FL确有14.471N支撑，但当时RR已经在地面，不能宣称后部转移有持续RR TOP支撑。
- 最后RR GROUND/front−63.632mm/clearance−49.474mm/load0.200458；RL GROUND且障碍接触/load0.374668；FL当前TOP/load0.215841（障碍6.545N）。final_support_available=true，但final_region=false、success=false。
- 全身CoM.x在RR placed时为0.703298m，52s为0.691965m，RR首次GROUND时0.669238m，RL最高点0.658373m；这是实际后退伴随支撑变化的诊断，不建立额外CoM门槛。若support margin在部分时刻为null，保留不可用，不填零或解释成已通过。

## 控制空间：只报告实际命令，不把nominal当机械角度

P09 RR projected residual绝对峰值hip1.931665deg/knee9.405639deg；P12 RL峰值hip2.267327deg/knee2.195876deg。它们远低于当前24/36deg通道幅度，未显示这些后腿通道整体饱和；这不证明更大动作必能成功，也不锁定唯一历史姿态。

| 决策末tick | 逻辑nominal（hip/knee deg） | projected residual | 实际native drive target |
|---|---|---|---|
| RR5824，刚qualified | −6.9 /−37.8 | −1.362889 /−8.939246 | −9.512889 /−47.989246 |
| RR6240，放置后退回 | −6.9 /−27.2 | −0.041498 /−6.694315 | −6.115217 /−27.334439 |
| RL6392，刚qualified | 30.2 /35.3 | −2.187485 /−1.145016 | 29.262515 /35.404984 |
| RL6448，GROUND撤销后 | 31.2 /−2.8 | −2.227253 /−1.574836 | 30.222747 /−5.624836 |

native target含真实mapper、原控制修正、硬限/速率限与独立残差组合；不能用nominal+residual简单相加替代它，更不能把native target当已达到的测量关节位置。本报告只是事实性定位：**当前自然P01策略首次复现RR事件链，但RR放置保持以及RL高点之后的前移/放置仍未完成。**先前checkpoint评估可能使用不同MDP版本，不能把跨版本差异归结为单纯权重提升的配对结论。

仅PowerShell只读分析并写此独占报告；原始run、checkpoint、生产代码与判定结果均未改变，未运行Python/Isaac或提交。
