# Block13 自然 P01 首回合：RL knee 实测硬限终止

固定来源：`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1451368221058Z_g7db0d17f398d_19bb6607a9384f08b00d341ba21060b7`，仅 `residual_and_projection_audit.jsonl` 前449条（字节1–35,083,699，SHA256 `9f921dc51fd010b56319248183d1ca686295edf345bcbffbc047f6c30434231b`）及 `completed_episodes.jsonl` 第1条。2026-09-10T15:01:30.785Z捕获；后续查询只使用内存，不读取第二回合、checkpoint或物理大流。

## 真实终态及信用边界

episode_index=0、seed1001，自然P01、无教师前缀；global149889–150337，共449策略决策、3592物理tick、29.933333333s，P09 **HARD_JOINT_LIMIT**，task_success=false、terminal_bootstrap_allowed=false。completed记录的terminal_info与第449条applied_audit完整一致。源阶段计数为P01=2、P02=122、P03=4、P04=1、P05=200、P06=93、P07=1、P08=1、P09=25。

精确越限关节是 **rear_left_knee（RL膝）**，不是RR/FR：现行guard按实测角检查闭区间[-60,210]°，不按目标检查。终态372维policy/critic相同、finite_fallback=false；schema第38–45零基槽乘90后，RL膝为−60.061542392°。同tick角色实测关节余量回解为−60.061542968°（negative_margin=−0.061542968°），两种持久化证据一致；前者经过float32编码，后者也是已存测量派生值，不冒充未读取的原始物理流。

最终RL膝target=−6.758894898°、nominal=0°，整回合决策末态RL膝target范围[-23.786017,4.853823]°，全部8个servo末态target均未越物理硬限。这不是命令舍入越界。最后实测RL膝随3536→3560→3584→3592 tick由−36.983727→−43.236000→−55.839318→−60.061543°，相应target约−4.282→−7.301→−4.328→−6.759°，实测与命令有明显跟踪差距；但仅凭此不能把接触动力学、执行能力或策略全身动作中的任一项定为唯一原因。终态RL地面反力48.596817N是瞬时测量，不是静态稳定或因果证明。

终态gravity_z=−0.989771128、body线速度[-0.104244,0.007345,0.012598]m/s，独立物理评价明确reason=`hard joint limit: rear_left_knee`、证据VERIFIED/VALID；这是实际安全中止，不是数据无效、阶段超时或完整越障成功。

## FL放置、RR提前抬升与连续交接

FL历史首次Q1095不是最终尝试的唯一Q：随后有5次回地撤销，最后一轮Q2137→C2178→P2631；P05→P06在2632/21.933333s，FL当前TOP、16.943575N、有效载荷份额0.506422，故当时确有放置和承载。

| 交接 | tick / 秒 | RR当前状态 | FL当前承载 |
|---|---:|---|---|
| P06→P07 | 3376 / 28.133333 | AIR、Q/current有效，抬升14.809mm，前缘−344.910mm | TOP、1.277213N，份额0.046373 |
| P07→P08 | 3384 / 28.2 | 同一Q继续，AIR、26.231mm，前缘−369.311mm | AIR、0N |
| P08→P09 | 3392 / 28.266667 | 同一Q继续，AIR、37.441mm，前缘−388.924mm | AIR、0N |

RR另有早期I1090（P05、3.018mm）但无Q；本次有效尝试I3367→Q3371都在 **P06**。I记录全身关节响应12.152°，Q记录RR自身响应6.204°、抬升8.509mm；全身运动与自身关节都参与，不能仅据事件证明某一腿动作是唯一机制。Q随真实运动连续进入P07/P08/P09，全回合8个决策末态current_valid=true，3432仍AIR/current有效；不是进入P09后重新强制抬一遍。

3441/28.675s才发生真实回地撤销（qualification及current两条事件）；3560仍GROUND、前缘−428.576mm。**最终3592又为AIR，但只有0.037693mm地面相对净空，I/Q/current/C/P均false，前缘−426.961mm，不能算重新有效抬起。** RR从未C/P。终态FL AIR/0N；当前支撑为FR TOP20.643202N、RL GROUND48.596817N（载荷份额有效），不是历史FL P在承载。

## 动作、保存状态与结论

3592/3592 native ticks连续、verified；own_phase_request_effect=3584。8次普通阶段切换均无done且允许跨阶段bootstrap，只有最终安全终态不bootstrap。P06→07→08→09的previous residual与carried residual逐项相等、禁止通道丢弃全0、phase-scale裁剪为空；实际wheel和非零residual连续更新。`handoff_hold_used=true`在该bridge实现中记录的是保留非零residual（`retained_nonzero`），不是把身体停稳或清空动作。root pose/velocity、force/impulse、gravity四类in-episode写入计数全部0。没有新增reset、下发丢失或伪终态的明确证据。

主控早先384条已优化时，449−384=**65**条是当时未完成rollout的历史余量，并非64。主控随后确认update1140/global150400已完整保存，该65条现已纳入完整rollout；此报告只分析449条，不将后继回合63条加入本回合样本，也不冒充独立读取过checkpoint。

第一未完成任务是P09：RR需重新建立可用抬升并完成越沿/放置，随后才有后续任务机会。此为自然P01随机训练轨迹，中途有PPO更新，不是固定策略评估或相对FSM/视频的改善证明。保留本次真实硬限失败，不放宽硬限、不据普通探索失败关闭续训；生产/配置/官方CP均未修改。
