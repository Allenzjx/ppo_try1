# #32 P10 首次接管与前 16 条信用记录：固定只读核验

范围严格限定为首个 `policy_credit_start` 之前的完整 prefix，以及 global **108801–108816**。未读取其后记录、checkpoint、hash 或 optimizer；以下是已记录的信用窗口，不据此声明已完成一次优化或保存。未修改生产、配置、测试或 master，未运行 Python/PT/GPU/Isaac。

来源为 run `20260907T0351509044004Z_gf1a9bbf650b1_f18d36a5a1bd43de83a723a754e0bc78` 的 [prefix_evidence.jsonl](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0351509044004Z_gf1a9bbf650b1_f18d36a5a1bd43de83a723a754e0bc78/prefix_evidence.jsonl) 与 [residual_and_projection_audit.jsonl](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0351509044004Z_gf1a9bbf650b1_f18d36a5a1bd43de83a723a754e0bc78/residual_and_projection_audit.jsonl)。

## 结论

这是实际 A roll-in 后、RL 尚在地面且没有硬 Q/C/P 的 P10 接管，不是已抬起 RL 的历史姿态重置。前 16 条样本覆盖 P10/P11/P12＝**1/1/14**，两个普通阶段切换均 nonterminal、允许 bootstrap，并携带非零残差。RR 在接管时既有教师完成的 Q/C/P，又有当前真实 TOP 载荷；但窗口末 RR 已 AIR、载荷为零，历史 placed 不能替代当前支持判断。此窗口没有新的 RL 初始离地或硬 Q/C/P，也没有任务成功。

## 1. 实际 prefix 与接管起点

请求为 P10 / offset 0 / seed 1001，teacher 为 `unmodified_SensorFsmController_reset_only`。首个 prefix 实际 **948 decisions / 7584 physics ticks**，accepted=true、miss=null，无 fallback。来源阶段计数为 P01=1、P02=207、P03=4、P04=1、P05=235、P06=296、P07=1、P08=16、P09=187。

全部 948 条 prefix decision 均 `policy_credit=false`、raw/projected policy residual 全零；7584/7584 原生 tick 验证通过、policy-effect=0。这是零政策相对于同历史基线没有额外效果，不是机器人或教师没有运动。prefix 的 no-state-write 汇总标志均为真；compact prefix 未逐条输出四个数值写计数，不能将该汇总说成四计数的独立全 tick 重建。

首个信用起点为 episode tick **7584 / 63.2 s**，requested/actual phase 均 P10，`requested_phase_still_active_at_credit=true`，remaining task time=136.8 s；整任务 200 s 时钟包含 prefix。`from_P01_current_policy=false`，scope=`teacher_initialized_suffix`。handoff 的当前 entry observation 与 credit-start observation 的完整 JSON 相等。

| 腿 | 接管当前接触 | front / clearance（mm） | 当前 load fraction | 硬 Q/C/P tick（均在教师段） |
|---|---|---:|---:|---|
| RL | GROUND=true，AIR=false，TOP=false | −87.234 / −47.864 | 0.090290 | 全无，当前 initial=false |
| RR | TOP=true，GROUND=false，AIR=false | +91.978 / −1.638 | 0.484429 | 6938 / 7109 / 7579 |
| FL | TOP=true，AIR=false | +529.827 / +2.121 | 0.425281 | 2461 / 3115 / 3583 |
| FR | AIR=true，TOP=false | +616.071 / +3.689 | 0 | 71 / 1665 / 1695 |

接管原始接触进一步确认：RL ground pair active/verified，法向力 **2.587906 N**，obstacle pair verified/inactive；RR obstacle pair active/verified，法向力 **13.884838 N**，接触点 world `[0.6096068621, −0.3374710083, 0.0510176010] m`，ground inactive。RR 当时的 TOP 与实际承载并非只由 placed 历史推断。

RL 历史 `whole_body_initial_clearance` 曾出现在 tick 7、1723，但不是硬 qualified；当前 RL initial=false、Q/C/P=false。RR 的 initial tick 6901 与硬 Q tick 6938 同样是不同事件，不能混称。

## 2. 连续时钟、信用与阶段边界

handoff `source_control_tick=7583`、`command_physics_tick=7763`，随后实际观察为 tick 7584；ACK `write_count=7764`、单次 articulation write=1、skew=0，反馈采样 tick=7763。前 16 条信用实际 episode ticks **7585–7712**、native command ticks **7764–7891**：128 个 tick 连续，command−episode 始终为 179，没有重置时钟。

信用 decision_count 连续 1–16，而 `physical_core_decision_count_including_prefix` 为 949–964；prefix_attempt_index=0，`prefix_teacher_data_in_ppo_storage=false` 全部一致。教师 948 条不混入这 16 条政策信用。

| 信用记录 | 实际阶段变化 | tick / time | 当时完成依据 |
|---|---|---|---|
| 108801 | P10 → P11 | 7592 / 63.266667 s | workspace_RL=1，support_RL=1 |
| 108802 | P11 → P12 | 7600 / 63.333333 s | workspace_RL=1，load_ready_RL=1 |
| 108803–108816 | P12 → P12 | 至 7712 / 64.266667 s | 尚无 RL 硬 Q/C/P |

全部 16 条 terminal=false、time_outs=false、terminal_bootstrap_allowed=true、physical evaluator valid=true、task_success=false。P11 的软 load_ready 满足不是硬 AIR/Q 证明：tick 7600 的 RL 仍 GROUND、load=0.194833，随后进入 P12。

## 3. 没有无条件清零动作或 mapper 历史

handoff 原 mapper compensation 为 `[0, −2.5, −2.5, −0.1116725, −2.5, +2.5, −1.25, +1.6450381]°`，八个 servo tracking channel 均保留；RL/RR 名义关节分别 `[15.4,19.4]` / `[−6.9,−37.8]°`，实际 canonical drive 分别 `[12.9,21.9]` / `[−8.15,−36.1549619]°`，FL wheel=−1.07 rad/s。它不是将 mapper 或控制目标复位为零。

首条信用的 P09→P10 carried residual 为零，与教师零政策相符；随后实际 16/16 条 residual 都非零。P10→P11、P11→P12 的 previous/carried residual 12 维逐值相等且全部非零；forbidden/scale-clipped channel 列表为空、hard_safety_modified=false。两个边界各使用一次 handoff hold，servo action jump=0、wheel jump 仅约 1.39e−17 / 2.78e−17 rad/s 的浮点尾差。

| global | RL nominal hip/knee（°） | RL filtered requested residual（°） | RL actual canonical drive（°） | actual canonical wheels FL/FR/RL/RR（rad/s） |
|---|---|---|---|---|
| 108801 | 15.4 / 19.4 | −0.868246 / +2.681624 | 9.531754 / 27.081624 | −1.19 / −0.12 / +0.12 / +0.12 |
| 108802 | 15.4 / 19.4 | −4.368246 / −0.818376 | 6.415215 / 23.581624 | −1.282250 / −0.225 / +0.015 / +0.225 |
| 108803 | 15.4 / 22.6 | −7.868246 / −4.318376 | 2.915215 / 17.031624 | −1.387250 / −0.141485 / −0.09 / +0.294968 |
| 108816 | 0.5 / 35.3 | −9.890000 / −14.885670 | −10.640000 / 19.164330 | −0.862037 / −0.601272 / −0.661373 / +0.083129 |

这些是 canonical 逻辑 double 命令，不是 measured joint q，也不是带机械轴符号的 float32 native readback。mapper native、requested residual、经过 headroom/最终 slew 的 actual drive 不是同一个量，不能将表中 nominal+request 当作全部 servo 的实际目标公式。

16 条详细末 tick 审计均验证 setter/dispatch targets 相等、实际映射重建匹配，dtype=`torch.float32`；上一 ACK tracking-reference 独立校验也为 16/16，previous ACK tick=当前 command tick−1，mapper feedback tick 与当前 command tick 相等。128/128 compact tick verified 且 actual-native-policy-effect=true；own-phase-request-effect=126/128，剩余两 tick 正是有记录的交接 hold，不是丢失下发。

四类 episode state-write 计数（root pose、root velocity、force/impulse、gravity）在全部 16 条均为零，no-state-write verified=true。

## 4. 窗口内真实后腿状态，而非历史支持推断

在 16 个 decision 末快照中，RL ground=11、obstacle pair active=5、TOP=0、AIR=0，load 范围 0.096579–0.491907；首个 obstacle 非 TOP 快照为 g108803。没有新增 lift-attempt 事件，RL 当前 initial/Q/C/P 在所有末快照均 false。窗口末 RL front=−62.974 mm、clearance=−48.428 mm，GROUND、load=0.242308。因此本窗口的首个未完成后腿环节仍是 RL 硬 qualified lift，不能把进入 P12 或障碍接触称为越沿/放置。

RR 末快照 TOP=9、AIR=7、GROUND=0，g108807 首次出现 AIR/load0；窗口末 front=+86.642 mm、clearance=−0.388 mm、AIR/load0，但 Q/C/P 历史仍为 6938/7109/7579。FL 末快照 TOP=13、AIR=3；末尾恢复当前 TOP/load0.365468。此为 16 个末快照计数，不是各接触状态的全 120 Hz 持续时间。

有限 compact 记录与详细末 tick receipt 支持本次接管时钟、当前支持、非零 carry 和原生验证结论；没有导出并独立重放全部 filter/mapper 中间量或四个物理缓冲区的每 tick 完整内容，不能声称完成那种更强的全链重算。未延伸到 global 108816 之后，也未把 suffix 的教师 RR 成果或阶段进入当作 P01 全任务成功。
