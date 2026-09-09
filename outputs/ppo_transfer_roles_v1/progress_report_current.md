# 检查点138496进度报告（本批训练与C8已结束，任务目标未完成）

本报告汇总本轮实际执行的19个训练块、8次重载自然P01评估及已验证的累计摘要。进程均已结束，但自然P01完整任务/视频/改善目标未完成。报告汇总没有另跑仿真或重扫大原始流；这不否定本轮已执行的真实Isaac训练、评估和既有测试。

## 结果先行与计数边界

- 从123136/927 PPO updates/18540 optimizer steps继续，已完成Run1–19，实际新增 **15360决策/120更新/2400优化步**。
- 最新固定checkpoint为 **138496/1047 PPO updates/20940 optimizer steps**；主线程确认roundtrip=true。Run18最终512/4/80、Run19最终256/2/40，不能把Run18原请求768全部记入。
- 全版本累计不等于当前配置：**d747已完成4096决策/32更新/640优化步**；其余来自此前明确版本。本文阶段表分别列示。
- 最新正式自然P01评估 **C8已结束：227决策/15.133333秒、P02未完成，累计0/8**。exit0是执行结束，不是任务成功；尚无完整成功checkpoint、成功视频或配对改善结论。

主线程提供的checkpoint138496 SHA256为`e568ccd4573035fb3de734a09ac6afe1a7e212fb0c4c055d2d80e78f8e84279a`；本稿未独立重算。最新表示保存时间/计数，不表示best或任务表现最好。

## 真实生产修订与三版本迁移

| 版本/提交 | 实施范围与限制 |
|---|---|
| `2f27c6f5065a` | 四腿角色、质量加权CoM短窗响应、接收侧空间代理、连续动作及角色奖励。角色语义改变走显式迁移，延续已有HISTORY策略，不清空网络或用旧入口姿态补成功。 |
| `db991aa68103` | 显式追加48维角色状态，由324变372；相对该次直接源保留原324列，复制旧输入权重、新列置零后继续训练。新增列后续已有真实PPO更新；不声称完整deque或完整Markov状态可重建。 |
| `d7479d9fc41c` | 仅P06–P13 FL hip nominal建议限速150→60°/s、residual cap24→32°；其他11通道、硬物理限制、奖励、阶段判定、372维保持。134400→新版本采用严格same372身份迁移。 |

实际生产位置：[角色与短窗](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_transfer_roles.py)、[监督器/连续建议](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py)、[奖励](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py)、[观测编码](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_observation.py)。

已改运行与迁移代码：[训练](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py)、[迁移](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_migration.py)、[HISTORY输入适配](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_history_actor.py)、[CLI](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_cli.py)、[启动器](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/scripts/run_semantic_ppo.ps1)，另有既有prefix/policy-distribution路径适配。配置：[阶段规范](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/stage_task_spec.yaml)、[执行范围](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/execution_profile.yaml)、[奖励配置](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/reward_config.yaml)、[372维schema](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/observation_schema.json)。底层物理backend/机器人USD及环境物理参数未修改。

首次角色版本将FR knee两列残差历史尺度4→6，对actor/critic首层输入列210/222乘1.5补偿；只在旧未clip域主张同物理输入的函数等价，不声称所有参数字节不变。“保留324列”特指随后追加48列相对直接源，不是所有历史版本尺度从未改变。d747迁移原收据记录actor（含已学新增列/std头）、critic、identity normalizer和训练RNG保留；先验源Adam，再重建Adam lr=3e-5，保留origin10112及累计预算，旧未完成rollout不继承。

四角色为 **FR→RL、FL→RR、RR→FL、RL→FR**；后两者分别优先观察FR/RL与FL/RR桥接。优先线索不是唯一固定支撑组合，AIR接收腿当前承载为零，历史placed不代替当前接触。全身运动可提供真实初始离地证据，不强制摆腿自身先动；initial不等于硬Q/C/P。

旧“前缘x/横向ROI就是receiver workspace”“数另两接触就是转移”“低载就是主动卸载”的准备门已从阶段完成列表替换；当前真实支撑/短时响应检查仍保留，不是删除所有物理条件。B/C不再用冻结A的精确关节/回弹入口、膝角方向或前缘到达才dispatch的nominal门、唯一home终姿作为成功标准；普通阶段不清wheel/residual/mapper/filter/history，硬Q/C/P、安全与最终物理停稳标准不放宽。冻结A本身未改。

**P05/P06确认过动作依赖缺口**：已对齐的Recording窗口中P05末FL AIR/0N，P06连续wheel推进后才TOP/5.510N；旧逻辑先要placed才开放P06建议。现P05 `pending_capture`在已合格、未placed、非GROUND及合法当前近沿/真实其它支撑条件下提供四轮 **+.3rad/s建议**，连杆层继续；第一个TOP不撤建议，仍需第二个真实TOP才能placed。不跳P06、不伪造AIR接触、不清残差，也不保证每条轨迹成功。

Recording JSON是 **COMMAND_OBSERVED**：按持久12通道、重叠batch与显式stop解析；FR knee例45.9→31.1为−14.8°变化，FR wheel −.63rad/s约1秒，不能写成历史绝对−40°。有对齐物理日志才是 **MEASURED_RESPONSE**；未测的精确力矩、角动量或因果效应仍是机制假设。−40°是合法探索候选，不是入口/奖励门。详见[既有证据与替换说明](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/logic_conflict_and_replacement.md)及[Recording证据表](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/recording_angle_and_overlap_evidence.csv)；其中旧324叙述是首次版本快照，不代表当前372。

旧FL hip建议22.8→49.2的26.4°变化，存在相对旧24°cap和60°/s残差slew的局部指令表达缺口；d747扩大该通道的表达能力。它是软件建议/权限改变，不是PPO已经学会抵消、稳定或避免碰撞的证明。

奖励保持最多五个dense family，不新增“对角线奖励”：角色准备/转移替换旧几何准备进展，使用同一PBRS；transfer时body成本按`1-.8*motion_fraction`减弱，捕获时逐步恢复；平滑成本取实际applied变化，不重复惩罚nominal/residual/applied。family权重为task1/body.4/contact.2/smooth.1/regularization0；没有模仿、非零残差或普通phase切换奖金。详见[奖励差异与反例](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/reward_diff_and_counterexamples.md)。

既有两组相关CPU回归 **393+247=640项通过，0失败/错误/跳过**，覆盖RSL迁移/保存恢复及CPU mapper/headroom/float32目标缓冲。此处沿用已有收据，未重跑；CPU正确不等于真实物理成功。

## 实际训练配置与恢复语义

N1/CUDA、训练seed1001；物理120Hz、policy15Hz，每次完整决策8 ticks；rollout128、每次PPO更新5 epochs×4 minibatches=20 optimizer steps。当前观测372、动作12，HISTORY rho=.9、状态相关异方差std、identity normalizer；gamma=.9985、lambda=.99。普通phase切换不是done，连续PBRS使用匹配gamma，真终止吸收态Phi=0；不发重复阶段完成奖金。

当前HEAD为`d7479d9fc41c`。同HEAD继续采用普通resume，恢复actor/critic/已学std、Adam、normalizer、训练RNG与计数预算；不重复迁移补零或重建Adam。合法物理reset重新开始真实roll-in/episode，不恢复未完成rollout或声称跨进程PhysX内部状态逐tick续接。只由主线程在现有进程结束后的边界协调恢复，本稿不提供并行启动第二Isaac的命令。

全版本15360决策不能标为当前d747样本：**d747最终4096决策/32更新/640优化步**。本次Run18才为该配置补入P10/P11/P12=6/6/500；P13仍0，不能把更早版本的478个P13样本移归当前配置。

## 已完成Run1–19的phase表（累计聚合已验证）

| phase | 全版本已优化决策 | 当前d747 |
|---|---:|---:|
| P01 | 62 | 22 |
| P02 | 3499 | 1128 |
| P03 | 125 | 56 |
| P04 | 132 | 60 |
| P05 | 3762 | 1436 |
| P06 | 4770 | 720 |
| P07 | 33 | 10 |
| P08 | 47 | 22 |
| P09 | 1010 | 130 |
| P10 | 12 | 6 |
| P11 | 38 | 6 |
| P12 | 1392 | 500 |
| P13 | 478 | 0 |
| 合计 | **15360** | **4096** |

表中只计实际source-phase PPO信用，排除教师/冻结checkpoint策略前缀。P07/P08、P10/P11样本稀少；跨版本后腿局部Q/C/P或到P13不能替代自然P01完整成功，也不能由实时下一阶段位置推算完整块覆盖。

截至Run19：**81个终止episode**（FALL38、BODY_COLLISION31、INCOMPLETE5、HARD_JOINT_LIMIT7），另19个非终止更新边界尾段、3053决策；尾段已优化但不计成功/失败。前缀42205决策/337640 ticks排除信用，76次尝试、71次接受。前缀累计**38952行**缺紧凑native/state-write细项，不能沿用旧33264或把缺失当零作用/独立验证通过。

19块聚合记录122588物理/native作用ticks、122352 own-phase作用ticks、230次非终止handoff，已记录native、state-write、prefix入storage及bootstrap合同异常0；这些不证明平滑或稳定。历史计划17792、实际15360、未消费2432（原2176+Run18未用256）、rounding0，均已由既有聚合器验证。覆盖CSV247×13已导出并经主线程QA看图，本次未重复图审。

Run17请求P06 offset256但前缀在P02未完成，真实fallback后512个PPO样本全在P01–P05、整段非终止；不能把课程请求当实际覆盖。Run18真实退出0、`STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`，原768只消费512/4/80；5终止为HARD3（49、13、15决策）/BODY2（153、177），另105决策P12尾段。6×948=5688前缀决策排除；RL policy initial18/Q1/Q撤销1、C0/P0，初离地不是成功。

Run19真实退出0、训练流程SUCCEEDED，自然P01实际256/2/40，P01–P05=4/213/8/1/30；201决策P05 FALL，另55决策P02尾段，无前缀。两块停止/执行完成均非任务成功。最新checkpoint路径为[checkpoint138496](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000138496.pt)。

## C8最终与全部八次正式C结果；B1另列

C8 run `20260908T0953095796695Z_gd7479d9fc41c_708d4bd483f4419382addb1c487cc75e`：checkpoint138496、自然P01/seed2001/N1、确定性mean、无prefix/optimizer，227决策/1816 ticks/15.133333秒。P02年龄15秒，`INCOMPLETE_CONTROLLER_BLOCKED`；lifted_FR=clear_FR=1、approach_FR=.9964065878801205，仍没有越沿/放置，不能将接近1改称完成。

FR硬Q在tick56，无C/P；终末AIR、support=false、load0、front−.005898353m、gap+.024393461m。P01/P02=2/225，其余0。物理valid=true、raw有限、无collision或独立安全中止、window=false；末native/状态写入验证true、bootstrap=false。roll RMS=.160575514rad、pitch=.148165145rad、角加速度=5.498268253rad/s²；未采阶段及完整quality均null。

| 评估 | checkpoint / runtime | 决策 / 秒 | 首未完成与结果 |
|---|---|---|---|
| C1 | 127232 / 2f27 | 227 / 15.133333 | P02，INCOMPLETE |
| C2 | 130304 / 2f27 | 607 / 40.466667 | P05，INCOMPLETE |
| C3 | 133248 / db991 | 685 / 45.666667 | P05，INCOMPLETE |
| C4 | 135424 / d747 | 227 / 15.133333 | P02，INCOMPLETE |
| C5 | 135680 / d747 | 592 / 39.466667 | P06，FALL安全中止 |
| C6 | 136192 / d747 | 895 / 59.666667 | P06，INCOMPLETE |
| C7 | 137728 / d747 | 683 / 45.533333 | P05，INCOMPLETE |
| C8 | 138496 / d747 | 227 / 15.133333 | P02，INCOMPLETE |

八次均task=false、非外部截窗，正式全版本0/8；不同runtime分别保留，不能统称当前版本八次。C6到P06且终末前腿有实际TOP支撑，不代表可命名best；C7 FR历史placed但已GROUND_AND_OBSTACLE，也证明历史事件不等于当前台面支撑。覆盖、时长和轨迹不匹配的全程RMS不能作改善证明。

B1同d747新版零残差、自然P01、无checkpoint/optimizer，已结束：758决策/6057 ticks/50.475秒、P09 `WHEEL_ONLY_CLIMB`。RR initial5993被接受、own运动不足2°也未被漏记，但最高AIR底部仍低于top约.886mm；从未硬Q，6057当前TOP接触不能补历史Q/C/P。这是现行任务失败，不证明轮驱动是唯一因果。B1不属于PPO学习结果，不加入C分母，不是重跑A或成功对照。

## A具体结果、局限与待交付

A原记录65.3667秒/P10 WAIT_ENTRY未完成：RR knee约−45.8583对旧入口−50.3976差4.5393°，回弹速度2.6107对23.5853差−20.9746，旧精确入口条件未过，原记录未给出物理硬失败。A为seed4001/v2/180settle+64preaction，C为seed2001/v3/180+0；不配对，未重跑A，历史Recording不替代本轮结果。A原视频保留，既有容器duration异常不改变任务标签。

C5局部奖励诊断只证实接收空间proxy可与平台/前送几何暂时抵消；该七决策折扣PBRS仍为负，未证明倒退获得正总收益或停退最优。任务失败率、全程RMS、接口native验证均不足以给唯一动力学原因或配对改善结论。

**未交付：自然P01完整成功checkpoint、≤200秒正常速度成功视频、有效A/C比较视频及配对稳定性改善证明。** 最新不等于best，训练执行SUCCEEDED不等于任务成功，局部放置/前缀接受也不是完整成功。

最终依据：[19块累计摘要](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/cumulative_through_block19/cumulative_training_summary.json)、[覆盖CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/cumulative_through_block19/phase_and_transition_coverage.csv)、[C8诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/evaluation_138496_diagnosis.md)、[保存状态与恢复命令](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/RECOVERY.md)及前七次/B1报告。原始A/Recording和旧证据未覆盖；**检查点进度报告定稿，不代表任务目标已完成**。表格技能用于CSV导出、单元格一致性校验与渲染QA，没有插补未采样阶段的质量值。
