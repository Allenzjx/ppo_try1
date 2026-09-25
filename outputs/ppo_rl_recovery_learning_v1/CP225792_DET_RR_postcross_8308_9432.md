# CP225792 DET：RR越沿后未捕获的有限窗口

**结论：这里有真实的nominal强转移组等待，但没有“RR未bearing便锁死所有policy准备”的证据。** RR已持续前送，控制却基本维持原关节目标；RR knee负请求141次均裁到−58°，其他准备关节也未产生有意义的状态改变。既不能称源轮速丢失，也不能用这一窗口证明单一关节方向足以捕获。

只读原录像source `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\video_eval\validation\20260924T1447490554064Z_g892385cba8a7_9356dc07a70e4bff9ae154deafb3c87b\source`，RRcross8308后至9432=78.6s，141个完整decision（8312..9432），全P09。未读后续；无Torch/PXR/仿真/生产修改。

## 物理响应和动作

141/141：RR AIR、current qualified、within_top_xy；TOP/ground/placed均0。gap范围 **56.307–59.318mm**，没有有效下降趋势；前缘从+0.400推进到+96.245mm。CoM Δxyz=95.209 / -1.813 / 0.029mm：明显前送、竖直变化很小，不是完全停住。

| tick / s | RR gap / 前缘mm | RR hip/knee FINAL° | RR hip/knee actual° | RL前缘mm |
| --- | ---: | ---: | ---: | ---: |
| 8312 / 69.267 | 58.652 / 0.400 | 7.590 / -58.000 | 7.330 / -57.767 | -146.623 |
| 8792 / 73.267 | 57.418 / 41.190 | 7.688 / -58.000 | 7.422 / -57.757 | -106.208 |
| 9432 / 78.600 | 58.386 / 96.245 | 7.772 / -58.000 | 7.508 / -57.766 | -51.750 |

末拍9432同拍链如下，未以独立重算nominal之差冒充policy：

| 通道 | source N | mapped N | filtered REQUEST | headroom有效修正 | FINAL | actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FL knee | -13.400 | -12.150 | -32.887 | -32.887 | -45.037 | -45.368 |
| FR knee | 31.100 | 31.290 | -51.473 | -51.473 | -20.183 | -20.475 |
| RL hip | 28.200 | 28.200 | -12.424 | -12.424 | 15.776 | 16.822 |
| RL knee | 0.000 | 0.000 | 0.781 | 0.781 | 0.781 | 0.416 |
| RR hip | -6.900 | -8.150 | 15.922 | 15.922 | 7.772 | 7.508 |
| RR knee | -37.800 | -39.050 | -19.481 | -18.950 | -58.000 | -57.766 |

- RR hip正残差约+15.9°把原负向N抵消为+7.6～+7.8°；此窗口没有继续负向调整。RR hip actual与FINAL差约−.26°，不是很大的跟踪失效。
- RR knee **141个不同raw请求**对应REQUEST −19.610..−19.437°，全部headroom裁到相同FINAL−58°，actual约−57.76°；继续在这一侧扰动不会产生新的FINAL。此证据不证明正向knee或负向hip单轴即可成功，需真实采样检验耦合效果。
- FL knee FINAL范围−45.039..−44.971°，actual约−45.37°，没有正向恢复行程；其准备proxy credit约.35、现有负向可执行余量约12.63°，不是已经到硬限。
- FR N knee本为+31.1°，policy REQUEST −51.47°使FINAL约−20.18°，actual已跟随。FR接收候选并非被nominal写成负值，而是当前合成策略未形成用户建议的准备构型；“+30°一定有效”仍未实测证明。
- RL hip/knee FINAL约15.78/.78°几乎不变；RL仍GROUND，距前缘由−146.623缩到−51.750mm；轮端顶部走廊竖直缺口约70.4mm，rl_space_credit=0。该proxy只测wheel，不证明完整linkage已安全。

四轮canonical forward-positive，顺序FL/FR/RL/RR，rad/s：

| tick | N | REQUEST | FINAL | actual |
| --- | --- | --- | --- | --- |
| 8312 | 0.300 / 0.300 / 0.300 / 0.300 | -1.013 / -0.059 / -0.029 / -0.105 | -0.713 / 0.241 / 0.271 / 0.195 | -0.783 / 0.193 / 0.317 / 0.194 |
| 9432 | 0.300 / 0.300 / 0.300 / 0.300 | -1.016 / -0.065 / -0.018 / -0.106 | -0.716 / 0.235 / 0.282 / 0.194 | -0.921 / 0.187 / 0.251 / 0.193 |

N四轮均+.3，FR/RL/RR目标仍正；FL反向由policy抵消产生，**不是尚未发出的late FL−1.07源脉冲**。FL本窗口实际bearing1.535..3.879N，末拍2.014N，确有负转且有接触；但净前送仍发生，不能仅由这些值认定它是58mm悬空的唯一原因。AIR RR自转不计地面牵引。

## 哪个门确实在等，哪些动作仍开放

**确认的source门：** [semantic_supervisor.py:2440](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:2440) 对P09 late group要求 `rear_dependency(...).support_transfer_permitted`；[semantic_rear_policy_timing.py:76](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_rear_policy_timing.py:76) 要真实RR TOP bearing＋前腿bridge。日志141/141 sourceclock=648、wait_reason=`current_RR_bearing_before_FL_RL_transfer`，没有发出这组。

该source在5.333s四轮stop之后，5.4s changed channels为FL hip、FL knee、RL hip、RL knee、FL wheel；目标分别−18.5、−31.4、15.4、19.4°和−1.07rad/s。FR与RR在这个full12组里是保持值，并没有新的FR knee或RR下降命令。[组识别校验:2102](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:2102)。这些FL/RL source目标整体仍被同一承载条件挡住；**代码没有把此source组内某些通道再拆成“无害准备”。** 但不能据此宣称所有全身准备自由度也被锁死，或无条件释放整组安全。

**当前policy实际未被该门覆盖：**

- 141/141 `cooperative_prep_allowed=true`、`rr_top_reachable=true`、`rear_owner17=全0`、12 mask全1。准备kernel明确 `prep=(rr_carry_capture && rr_top_reachable) || rl_prep_transfer`，[semantic_rear_cooperative_prep_sigma.py:26](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_rear_cooperative_prep_sigma.py:26)，不要求RR承载。
- 实际各cap：FL knee36°、FR knee112°、RL hip24°/knee36°、RR hip24°/knee36°；FRk/FLk/RLhk可双向修正。当前准备sigma乘数FLk×2、FRk×16、RLh×2、RLk×8、RRh×4、FLwheel×2已公开生效；**确定性执行只取均值，sigma打开本身不会改变这个均值轨迹**。
- 141条所有12维FINAL与同拍headroom candidate **完全相等（max差0）**；除了RRk7的headroom clip，未发现后续owner或最终slew把这些准备修正再覆盖。原始native endpoint owner receipt缺失，标N/A，而非凭缺失宣布inactive；当前input17全0与实际候选→FINAL一致为独立证据。
- [RearOwnerRecovery:66](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_rear_owner_recovery.py:66) 只在已赢得late ownership时锚定FL/RL，且仍允许相对request双向变化；此pending未发组不满足winning owner。
- 日志 `rr_capture_recovery_allowed=false` 不是policy禁止下降。它由是否有活跃RR assist DESCEND/确认保持等决定，[semantic_rr_capture_context.py:110](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_rr_capture_context.py:110)，供local warning/短接触交接使用；当前后腿assist关闭，不能拿该false解释成四腿action mask。
- nominal诊断明确 `rr_carry_continuation.added_rolling_suggestion=true`、`source_joint_owners_continued=true`；等待late组时仍有现有+.3前送建议，wheel没有被一起清零。

## 对下个连续学生课程的直接含义

P07前驱→RR抬升/越沿→首次捕获→RL的真实on-policy数据**可以通过现有开放通道改变目标**，不被上述pending strong-source门一概截断。刚完成的512 P12+8采样，其RR首次qualified/cross/placed全由前缀产生，未覆盖“本窗口首次RR捕获如何形成”的学生动作。这支持补齐该物理窗口的数据覆盖，而不是承诺增加步数必然学会。

当前可证实的缺口是：均值残差保持RR正hip/负边界knee，同时FL/FR/RL准备目标几乎不变；有效后继接触没有形成。需保留已改善前段，并在正式兼容采样中观察这些自由度能否形成下降/支撑。**不据此新增姿态脚本、不重开RR自动下降、不调reward/门禁、不把P09 source组全部放行。** 未做物理反事实，单通道因果和全身可行下降路径仍未知。

## 封存追加：自然P01正式确定性回合

同一CP225792已重载且校验通过；892385控制版本、seed4001、原FL capture assist开启，后腿任务辅助关闭。最终1392条issued/completed决策、11132物理ticks、92.766667s；最后一次动作执行4ticks，视频1392帧/15fps=92.8s，多出的0.033333s仅末帧显示量化，不是额外物理时间。本次评估PPO/AUX更新均0。

终止为P09 `INCOMPLETE_CONTROLLER_BLOCKED`，来源 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`：阶段年龄34.9s，原局部30s加当前进展许可4.9s用完；不是外部budget truncation，terminal bootstrap=false。物理评估仍为valid/VERIFIED，physical termination_reason/source均null；记录不支持将其归类为机身碰撞、纯轮爬升、跌倒、NaN或硬限安全中止。进程exit1/run lifecycle=DIAGNOSTIC_FAILURE来自未完成共同物理任务的source acceptance，而非录像故障；viewport manifest为PASS/valid=true。

| 腿 | qualified / crossed / placed tick（s） | 末拍接触与载荷 | 末拍gap / 前缘距离 |
| --- | --- | --- | --- |
| FR | 24(.200) / 2831(23.592) / 2841(23.675) | 合法TOP，13.538N | −.054 / +512.901mm |
| FL | 2866(23.883) / 4438(36.983) / 5606(46.717) | 合法TOP，2.812N | −.670 / +500.974mm |
| RR | 7198(59.983) / 8308(69.233) / 无 | 合法当前AIR，XY内，0N | +58.382 / +100.034mm |
| RL | 无 / 无 / 无 | GROUND，12.420N，未qualified | −50.002 / −49.290mm |

**首次未完成任务仍是RR真实顶部捕获/放置。** RR越沿后直到终态仍未TOP接触或承载；不能把XY内AIR或全机body几何进入平台当成功。RL未形成有效卸载、越沿或放置，未执行P10–P13完整终态；受控停车/home未完成。FR/FL实际放置已恢复，但FL曾有短暂掉载，不能用历史placed声明连续承载。

质量仅为本回合92.766667s已观测窗口：roll RMS 7.986°、pitch RMS 5.437°；roll/pitch绝对p95 10.756°/9.850°；roll/pitch rate RMS 0.056640/0.053038rad/s。all_phases_sampled=false、fixed_quality_score=null；这些不是同任务完成条件下优于N的证据，不对缺失后腿阶段评分。

封存摘要已加到配套JSON `sealed_result`。该追加只读末条及封存manifest，未扫描新训练、未改生产；后续P07连续学生采样由主任务单独负责。
