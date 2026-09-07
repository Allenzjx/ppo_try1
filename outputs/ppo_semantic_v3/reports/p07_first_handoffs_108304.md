# Run 31：首个 P07 prefix 与前 16 个信用决策

固定范围：[run 20260907T0313083674401Z_gf1a9bbf650b1_98167177c0ff4cd6b15dd57e491fa8c2](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0313083674401Z_gf1a9bbf650b1_98167177c0ff4cd6b15dd57e491fa8c2) 的首个 prefix，读至第一个 `policy_credit_start` 即停止；策略流只读 **g108289–108304**。源108288、同 HEAD f1a9bbf650b1b8f80d5169e09173fcfc68797d99、P07/offset0/frozen FSM/N1/seed1001。本文不重复权重/Adam/哈希审计。

**结论：实测从仍在地面、尚无 RR 硬 Q/C/P 的 P07 接管，连续经过 P08 到 P09；未发现阶段切换清零残差或重置 mapper 的证据。** 这仅是首 16 条已采集信用记录，不是完整128 rollout、optimizer update 或108304保存 checkpoint 的声明；不预填1024计划完成或任务成功。

## 1. Prefix 信用与真实接管

首个 reset 从 P01/t0 开始，实际 **744 个 reset-only decisions / 5952 physics ticks**。prefix source-phase P01–P06 为 `[1,207,4,1,235,296]`；全部 `policy_credit=false`、raw与projected residual全零，5952/5952 ticks 的 native summary verified，no-state-write 验证均 true。actual native PPO effect/own effect 都为0，含义是零策略与同tick零策略反事实相同，**不是机器人没有动作**。

`reset_only_prefix_result`：accepted=true、miss=null；仅一个 prefix start/result，未 fallback。`policy_credit_start` 为 `teacher_initialized_suffix`，requested/actual phase 都是 P07，`requested_phase_still_active_at_credit=true`，`from_P01_current_policy=false`。接管在 **tick5952 /49.6s**，剩余整体任务时钟150.4s，没有重新开200s。

完整 credit-start observation 与 handoff entry observation JSON 相等，均为当前真实 tick5952、all_finite=true；不是历史唯一姿态快照。当前状态：

| 腿 | front / clearance (m) | 当前接触、载荷 | hard Q/C/P 历史 |
|---|---|---|---|
| RR | −.205028680 / −.050919979 | GROUND，AIR=false，load .247819573 | 全 false |
| RL | −.219366219 / −.050144145 | GROUND，AIR=false，load .303504492 | 全 false |
| FL | +.370607260 / +.000256320 | TOP，AIR=false，load .199119297 | Q2461/C3115/P3583 |
| FR | +.470050040 / −.000311152 | TOP，AIR=false，load .249556637 | Q71/C1665/P1695 |

接管 raw contact 独立字段：RR ground pair active/verified，normal_force=7.0948262215N、obstacle pair inactive0N；FL obstacle pair active/verified，normal_force=5.7005860899N、ground pair inactive0N。FL contact point z=.0510792695m，与 evaluator 的 TOP 记录相配。两前腿 Q/C/P 均由本次教师 prefix 获得，不算新增 PPO 成果。

RL 的 `whole_body_initial_clearance` 在教师 tick7、1723 留有审计事件，但从未取得 hard Q；不能把初始轻微离地当作合格抬升。RR 首 prefix 和本次16条内均无 hard Q/C/P。

## 2. 接管时钟与原有控制状态

handoff receipt 的 source_control_tick=5951、command_physics_tick=6131，handoff observation tick=5952；ACK physics_tick=6131、write_count=6132、articulation_writes_this_call=1、motion_start_skew_s=0、feedback sample tick=6131。

教师最后 actual canonical drive 为 `[21.55,−12.15,−.7996284,45.3470521,5.65,.6421600,−1.3499030,.9543499,.3,.3,.3,.3]`。四轮 nominal 仍+.3，controller bias全零；mapper compensation有多个非零量，包括 RR hip/knee `−1.3499030/+.9543499°`。handoff ACK已有当前 tracking-reference/headroom mode，未丢失既有native控制时钟或反馈状态。

后16决策的 compact native ticks 连续为 episode5953–6080、command6132–6259，恒差+179，与最后教师ACK无缺口。末tick详细reference例如 g108289：dispatch6139、previousACK6138、mapper_feedback6139、previous_sample6138；g108304对应6259/6258/6259/6258。接管前后不是把 episode clock 当成 native clock。

## 3. P07→P08→P09 实际连续信用

| Global | 动作 source→end phase | 末 tick /秒 | done / bootstrap | 实际完成值 |
|---|---|---|---|---|
| 108289 | P07→P08 | 5960 /49.666667 | false /true | workspace_RR=1、workspace_RL=1、support_RR=1 |
| 108290 | P08→P09 | 5968 /49.733333 | false /true | workspace_RR=1、load_ready_RR=1 |
| 108291–108304 | P09→P09 | 5976–6080 | 全 false /true | 窗口内无后腿 Q/C/P |

因此样本分配为 **P07=1、P08=1、P09=14**：两段前驱都有实际采样，但各仅1个决策，不能称已充分训练；本报告也未核验完成的optimizer边界。过阶段原因记录为 `current physical goal set satisfied`。P08的load_ready=1也不等于AIR：其末RR仍GROUND、load .154350992。

两次信用内 handoff 的 `previous_projected_residual_full12` 与 `carried_projected_residual_full12` 完整JSON相等，且都不是全零；forbidden/phase-scale-clipped channels为空，hard_safety_modified=false。P07→P08/P08→P09 action jump 最大仅0…4.44e−16°和2.78e−17 rad/s量级，各有一个handoff hold tick。第一次教师P06→P07的carry确为零，是教师原本raw/projected均零，不是把已有策略残差清掉。

| Global | RR nominal hip/knee (°) | RR projected request (°) | RR mapper-native (°) | RR actual canonical drive (°) |
|---|---|---|---|---|
| 108289 | 0 /0 | −2.575696 /−4.0 | −1.349903 /+.954350 | −3.925599 /−3.045650 |
| 108290 | 0 /0 | +.280322 /−3.065379 | −1.349903 /+.954350 | −1.069581 /−2.111029 |
| 108291 | +1.6 /0 | −3.219678 /+.434621 | +2.85 /+.954350 | −.369678 /+1.388971 |

尤其前两行 nominal 为0但 mapper-native仍保留教师非零补偿，不能解释为 mapper 已清零。策略raw、请求、nominal、native和actual不能混用。

前三条轮 nominal 都为 `[.3,.3,.3,.3]`；filtered residual 分别为 `[-.12,-.12,.009414,.12]`、`[-.225,-.015,-.067479,.118687]`、`[-.12,-.12,.037521,.150707]`，对应actual canonical为 `[.18,.18,.309414,.42]`、`[.075,.285,.232521,.418687]`、`[.18,.18,.337521,.450707]`，并未因标签转换全停。后续真实nominal仍可按有限动作推进；本报告不要求固定姿态或维持这一轮速。

## 4. 审计与有限证据边界

16条总128 ticks，全部native summary verified、末tick setter/dispatch与actual mapping核验通过、previous-ACK独立验证通过。native PPO effect=128，own-phase request effect=126；差2恰对应两次handoff hold tick，不把carry旧残差的tick说成当次raw的新作用。

信用流的四类 in-episode root pose / root velocity / force-or-impulse / gravity 写入计数全部为0，no-state-write=true；所有16条physical valid=true、nonterminal、time_outs=false、允许正常bootstrap。`prefix_teacher_data_in_ppo_storage=false`，scope始终`teacher_initialized_suffix`，prefix_attempt_index=0；末行 credited decision_count=16、physical_core_decision_count_including_prefix=760，分账等于744+16。没有把教师5952 ticks计入PPO样本。

prefix compact行只保存no-state-write总验证标志，不逐项导出四个计数；本报告不能从这份prefix文件独立重算每tick的四类计数。128个信用tick也主要是compact审计，完整float32目标/mapper/reference证据只在每个decision末tick；没有独立重建全部120Hz filter状态、全部native buffer或物理动作反事实。carry receipt、时钟连续性和非零实际目标支持“没有无条件清零”，但不等于所有内部数值/轨迹已被本报告重新执行证明。

固定末tick6080 /50.666667s：RR仍GROUND、AIR=false、initial_clearance=false、clearance−.047794684m/load .092489650；RL仍GROUND、无hard Q/C/P。FL在tick5968已变AIR/load0，故接管时的真实TOP和历史placed不代表后续一直承载。首16没有新RR资格或全任务成功，剩余任务时钟149.333333s。

仅新增本报告；不写master，不读108304之后tail或后续checkpoint，不重复hash，不运行Python/PT/GPU/Isaac，不改production/config/tests。固定证据核验完成后停止。
