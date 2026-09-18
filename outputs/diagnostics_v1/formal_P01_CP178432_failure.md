# CP178432 正式 P01 评估：第一未完成任务为 P05 的 FL 放置

## 结果与终止原因

完整自然P01评估没有成功，**未进入RR阶段**。tick5907 / **49.225 s** 在P05结束：FR已放置，FL在2106已cross但始终未placed；已完成阶段仅P01–P04。`placed_FL=0.85`是进度值，不是布尔完成。

- 正式run：`20260917T0510347247858Z_g6c2121b68654_35383f247031424e9b890de765bc2b13`，checkpoint CP178432，确定性conditional mean，from_phase=P01；运行封存时间 `2026-09-17T05:27:37.554665+00:00`。
- manifest：`lifecycle=DIAGNOSTIC_FAILURE`、`physical_task_success=false`、`success_candidate=false`；发布验收错误为 `SemanticVideoError: episode did not meet common physical task`。这是任务未完成，不是录像损坏。
- 最后**确实返回的**短decision为5904→5907，共3物理步。其 semantic task 原始终止种类 **`INCOMPLETE_CONTROLLER_BLOCKED`**，来源 **`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`**；P05 stage_age=**37.225 s = 30 s + 7.225 s 有限进展宽限**，不是200 s整场上限或外部中断。
- 物理evaluator自身 `termination_reason=null / termination_source=null`，`run_validity=VALID / physical_evidence_status=VERIFIED`；没有以碰撞、跌倒或数值异常结束。最终 `final_controlled=true`只是速度条件，同时region=false、final_support_available=false，不能拿它当成功。

此前P06后缀训练到RR cross/place再回退的证据仍成立，但不能覆盖本次完整评估先失败在FL捕获的事实。后缀报告单列：[training_RR_carry_178432.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/training_RR_carry_178432.md)。

## 数据范围与真实接触

只分析已封存2500–5907：**3408个逐物理步观察和同拍派发记录、427个原决策快照**。无新仿真、无policy forward、无GAE重算；未扫描其他Recording。

3408步FL ground/obstacle pair **全部不活跃，force为0**，没有TOP接触。FL末gap **+3.470 mm**、front **+197.938 mm**，AIR、load_fraction=0且valid=true；consecutive_top_samples=0。窗口最小gap **+1.387 mm（5805）**。不能因几何已在顶面区域、净空很小，或传感器保留一个旧contact_point坐标就声称承载。

3000–5907段gap为 **1.387～6.641 mm**；base z在92.222～94.565 mm之间，CoM向前移动 **81.879 mm**、z净变化仅−0.077 mm。末真实支撑为FR 11.975 N、RL 14.348 N、RR 2.348 N，FL没有支撑。这是全身维持悬空构型并缓慢前移，不是已着地但判定器漏发placed。

## 源动作是否遗漏、错误hold或被mask

实际源N继续执行下降/回收序列，而非从一开始就停住：

| tick | FL nominal hip/knee（deg） | 说明 |
|---:|---|---|
| 2500 | 48.2 / −36.7 | 源四轮N仍为+0.3 |
| 2537→2569 | knee −30.4→−21.9→−18.7→−16.6→−13.4 | 真实源waypoint持续消费 |
| 2559 | 同时段四轮N归0 | FL front在2558到达+100 mm；既有pending-capture前送辅助不再许可 |
| 2577 / 2585 | hip 45.0 / 24.9 | hip回收目标继续下发 |
| 2609→5907 | 22.8 / −13.4 | 有限源最后请求合法保持 |

P05源本来有限；endpoint不是任务完成，也没有自动补一条“继续下探到碰到顶面”的目标。已核实所有决策中 `capture_owner_hold.captures` 仅有FR，**没有FL capture owner、没有suppressed FL swing**；所以此处是**源端点保持**，不是伪造FL已放置后把它锁住。

当前P05 pending-capture wheel辅助要求真实pending、至少2个其他支撑、gap≥−15 mm、front<+100 mm等条件。它曾延续前送，但2558越过其前方范围后不再提供+0.3。末front+197.9 mm不满足辅助范围，不能说其它wheel因此被mask。P05没有P09/P12那类nominal geometry投影；此窗口controller bias全12维始终0。

3408步实际 residual mask全1，派发/目标映射一致性audit均通过；并有实测跟踪证据如下。这排除了窗口内已记录的许可屏蔽/派发不一致，不等于仅凭ACK就认定物理推进正确。

## 同拍执行链与实际跟踪

末tick5907，canonical deg：

| FL通道 | 源N | 实际mapper N | controller | 有效projected residual | 最终target | 实测q |
|---|---:|---:|---:|---:|---:|---:|
| hip | 22.800 | 24.095 | 0 | +0.059 | 24.155 | 23.001 |
| knee | −13.400 | −12.150 | 0 | −0.559 | −12.709 | −12.715 |

这里用的是原同拍`native_drive_target_full12`、原mapper pre-state、原previous_final和物理command读回，不用独立重算zero目标冒充当拍N。末hip中约+1.295°来自mapper tracking compensation，**不是PPO突然给了+1.355°**；knee中+1.25°也来自该mapper状态，PPO有效项为约−0.559°。

后段不是明显驱动失灵：knee target−actual绝对误差最大 **0.01362°**、均值 **0.00254°**。hip反馈有周期性摆动：mapper补偿−1.2048～+1.2952°，实际target21.565～24.201°、actual22.534～23.130°；target−actual绝对误差均值0.676°、最大1.362°。必须保留这项实际跟踪动态，不能只挑几个0.12°误差的时刻说完全静止；但没有发现一个持续下探命令被长期严重跟踪不足所阻塞。

因此最直接可确认的链是：**有限N完成→实际mapper/残差形成固定附近的全身目标→FL关节基本跟随→轮底仍AIR**。尚不能从单轨迹证明调整FL某个关节残差就一定触地；关节符号不是世界竖直方向，全身机身/其他支撑同样参与。

## policy / HISTORY 与四轮真实动作

此评估使用manifest绑定的确定性conditional mean，不抽innovation。降低采样温度不会直接改变这条确定性均值轨迹。

3000–5907 FL raw hip范围 **−0.001690～+0.005856**，knee **−0.025055～−0.021112**；有效残差分别 **−0.0304～+0.1054°**、**−0.6012～−0.5066°**。没有“FL自身产生巨大抬腿残差”的证据。末FR knee有效修正−5.389°、RR knee−3.717°等全身残差也存在，不能把悬空净空单独归给FL policy。

依据原manifest `mu=0.1*base+0.9*previous_raw`、已发上一步raw作代数分解（不是新forward）：末FL hip/knee raw为 `[+.003295,−.023306]`，HISTORY carry `[+.003026,−.021016]`，反推base mean `[+.002694,−.022902]`。基础均值自身也接近该平衡，不能简单说“只有旧HISTORY不放手”。原观察向量未另存于这个视频log，故此项明确是记录合同下的代数分解，不冒称独立观测读回。

末四轮canonical rad/s，source N均0：

| wheel | PPO有效修正 / final target | 实测角速度 | 当前接触 |
|---|---:|---:|---|
| FL | +0.158663 | +0.158582 | AIR，无牵引证据 |
| FR | −0.025209 | −0.074058 | 已验证支撑 |
| RL | +0.212498 | +0.129269 | 已验证支撑 |
| RR | +0.003065 | +0.052539 | 已验证支撑 |

其他轮没有被输出mask关闭；也不是“仅RR在转”。角速度、轮端位移和对地牵引是不同事实。source stop后非零的是保留的PPO wheel修正；若要归因其引起机身/净空变化，需要独立有边界的干预，不能凭当前相关性断言。

## 当前可执行判断

第一优先未完成任务是**真实FL捕获**，不是RR或终端停车。现证据不支持放宽TOP/placed判据、延长时间冒称成功、重置HISTORY或永久屏蔽wheel residual。也没有确认必须修复的底层mask/索引/执行错误。

下一步应让策略形成能真实触地的当前全身状态，并在P05捕获样本上检查均值/任务信号；若另评估实时nominal捕获建议，应独立版本验证且保住成功zero，不把本次端点几何改成历史姿态硬门。所有因果修复都仍需真实训练和重新从P01评估，不把本报告的计算分解称为已修复。

完整证据：[formal_P01_CP178432_failure.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/formal_P01_CP178432_failure.json)。脚本 `analyze_formal_p01_178432.py`；仅新增outputs文件，未改生产/配置/物理状态。
