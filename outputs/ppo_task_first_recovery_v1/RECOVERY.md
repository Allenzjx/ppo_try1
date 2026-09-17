# Task-first recovery：CP177152 已从自然P01完成真实越障与受控完成

最新状态：CP177152已正式保存后重载，从自然P01真实连续完成6116 ticks /50.966667s，四腿抬升、越沿和放置、全身进入平台及固定1s窗口后的当前受控完成均通过，结果SUCCEEDED。此checkpoint seed4001仅试验1次，M1/M2/M3=1/1，M4未取得，不宣称泛化或稳定性优于zero。

当前最新训练累计177280 decisions /1350 PPO updates /27000 optimizer steps。本轮实际新增2688/21/420；mean-head分支自己新增512/4/80，独立quarter分支新增128/1/20。完整评估不增加训练信用，P07–P13本轮训练样本仍0。最新训练checkpoint为CP177280，但正式best-task仍保留[CP177152](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000177152.pt)；新候选没有正式任务评估，不能覆盖成功best。祖先、旧失败候选及成功zero均保留。

此前已封存的CP177024回合为6177 ticks /51.475s，四腿均真实主动抬升、越沿和放置，机身进入平台；但P13固定观察窗口结束时没有完成受控停车，结果INCOMPLETE_CONTROLLER_BLOCKED / POST_COMPLETION_LOSS，M3失败。其封存数据中没有机身碰撞、非有限或关节硬限违规。这条旧development partial证据继续保留，未改判为成功。

此前CP176768自然P01回合在52.591667s真实取得FR/FL越沿接触与前送，随后P09机身碰撞；M1/M2历史阶段进展保留，M3失败。它仅是旧执行版本的development partial候选，不是安全完整成功或新修复版本的评估结果；CP176640的P02硬限失败同样保留。

成功CP177152保留于00050a2b14521f41f3194ae7d4513ac1d32a1f8b/runtime 775d959e4bfcd431529d9eca374626d3fcf6ed56ba06f52f7e2136d3a117f74e。当前独立采样候选生产锁定fc14a68c037abd69e24f49b1524457fd6854543a/runtime bfa1e1471cc0e645ae33e873439a242142dde5adceb70a87bcc419152815c48c。CP177024正式回合使用前版ad0c1328f177/runtime 4fbe469a…；更早回合使用b0438f66ec63/runtime e9b6bf32…。root独占一个Isaac资源；修改只在保存并确认无Isaac进程的合法边界实施，运行时未热改。

独立p06_quarter_temperature_v1首块已实际完成128 decisions /1 update /20 optimizer steps，全部P06、1024个policy物理ticks；此前成功N前缀335decisions/2680ticks全部排除。全回合末尾tick3704/30.866667s，PPO后缀8.533333s、非终态并保留bootstrap，无完整episode，更不是从P01的PPO任务成功。FR/FL捕获事件属于teacher历史；RL在PPO段有初始Q3230但未越沿/放置，不授予后腿成功。128/128 epsilon0/full12、1024/1024 native核验且policy真实有效。update1350 KL=.038766、clip=.368750、value loss=.285358、有效LR1e−5，actor确实更新至df018f65…并保存/重载核验通过。[CP177280](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000177280.pt) SHA8c93f9b5…；[实际receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_quarter_P06_177280_actual.json) · [reward/GAE信号](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_quarter_P06_177280_signal.json)。

本分支只将真实随机innovation temperature从.5变.25，保留mean/sigma权重、rho=.9、全12容量、Adam有效LR、Identity、RNG、成功N、六配置与epsilon0，不同时增加质量reward。保持相同权重与相同observation时deterministic mean不变，不把温度迁移本身当成新的任务改进。后续同设置512条已请求，未完成前不计信用。CP177152正式best不动，新候选不能仅因更新更多、reward更高或后缀进展替换它；[明确迁移计划](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoint177152_quarter_temperature_migration.json)绑定实际source checkpoint和新提交。本轮累计PPO阶段量P01=14/P02=991/P03=21/P04=18/P05=1388/P06=256/P07–P13=0。

P13已确认是名义stop所有权的取得条件形成闭环依赖：真实post-completion窗口已开始，但先前条件仍等待名义P13/已停车状态，未及时取得stop所有权。新版本只改NominalMotionProvider._observe_final_stop_owner，在真实窗口触发且当前几何/支撑/物理证据满足时取得现有stop；不延长窗口、不放宽评价器，不要求先前已经停车。其余成功N、residual组合、六配置、epsilon0/HISTORY均未改。

[独立final-stop迁移计划](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoint177024_final_stop_handoff_migration.json)已绑定CP177024与实际新提交。它不拓宽先前composition factor：只许可supervisor/migration/training三个文件，并以完整文件hash及唯一helper外源码一致性双重约束，确保评价器及其它方法未改。保留兼容actor/critic/完整Adam及有效LR/Identity/RNG/计数，清空旧rollout，合法自然P01重新采集；承认执行transition语义改变，不宣称physical MDP等价。

迁移/旧composition/task-first共58项测试通过，包含官方CPU保存→加载→真实update→再保存/重载及拒绝评价器、其它名义方法、签名/模块常量变化。core的70项定向测试及成功zero已封存P13实际2130输入/2129名义输出离线重放通过，目标0差、stop owner均tick8737；离线保护不是新增物理成功。

新128条自然P01正式训练已自然完成：P01=2/P02=126，1024真实ticks，128/128 epsilon0/full12、1024/1024 native核验和policy有效作用，teacher0。末尾P02/tick1024非终态，FR Q23、没有越沿/放置，保留bootstrap；这块前缀本身不是P02成功，也没有采到P13。update1349 KL=.017249、clip fraction=.228125、value loss=.163378、有效LR1e−5；actor从e036d051…真实变为a163f2c9…，保存重载核验通过。后续完整正式回合才提供M3证据；修复/初始化与学习作用不可混为因果证明，尤其不能声称此128条前段样本学会了后腿或P13停车。

最新[CP177152](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000177152.pt) SHA8ffb446a…；[本块实际receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_final_stop_fix_177152_actual.json)与[epsilon0/GAE/轮目标信号](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_final_stop_fix_177152_signal.json)。本轮实际训练阶段量P01=14、P02=991、P03=21、P04=18、P05=1388、P06=128、P07–P13=0；670个teacher prefix决策仍排除，正式评估覆盖不计训练样本。

## CP177152 正式成功：实际链路、终态与边界

[完整成功视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/videos/ppo_task_recovery_cp177152_full_success.mp4) · [逐帧实际四轮角速度视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/videos/ppo_task_recovery_cp177152_full_success_wheel_qd.mp4) · [zero/PPO对比](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/videos/zero_vs_ppo_task_recovery_cp177152_full_success.mp4)。PPO视频51.0s/765帧、正常速度，实际物理回合50.966667s；对比73.8667s/1108帧，右侧在完成后明确冻结343帧，不冒充额外PPO稳定运行。旧zero不是本版本新录制B。

[封存实际审计](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/evaluation_CP177152_actual.json)：official checkpoint load、actor SHA a163f2c9…与保存一致，自然P01/tick0、teacher0、optimizer0、pre-action0。6116/6116实际物理ticks full12许可、native派发核验及policy有效作用，零raw替换0、非法状态写入0；765次策略决策，764次正常环境返回的reward均epsilon0，末次4tick触发成功，不补造reward返回。

| 腿 | 首次抬升资格Q | 越沿前最近有效Q | 越沿C | 放置P | P时真实顶面载荷 |
|---|---:|---:|---:|---:|---:|
| FR | 22 | 22 | 1374 | 1397 | 6.552161 N |
| FL | 1538 | 1538 | 1842 | 2648 | 12.041788 N |
| RR | 4633 | 4940 | 5428 | 5428 | 13.475083 N |
| RL | 3277 | 5517 | 5878 | 5994 | 11.066641 N |

这些是同一真实连续回合；RR在4731落地撤销早期资格，4940重新取得，RL早期也多次落地撤销后于5517重新取得。历史首个Q不表示它们从该tick起一直悬空。最终FR/RL/RR实际顶面承载10.320/13.785/4.355N；FL当前AIR、净空+2.199mm，不虚构其承载。

stop owner在观察tick6000/50.000s取得，真实固定窗口起于49.966667s；入场时final_controlled=false，修复使它可以基于真实post-window证据取得现有stop，而不循环等待此前已停车。首次实际P13 nominal四轮零命令在tick6001生效；raw residual/mapper/evaluator未重置。最终PPO仍全12开放：wheel target（FL/FR/RL/RR）[.145650,-.029877,.097414,-.017324]rad/s，实测[.145103,-.181083,.131986,-.060915]rad/s；机身线速.025004m/s、角速.062091rad/s。固定1s窗口完整，final_controlled/task_completed_controlled/traversal_task_complete=true，post-loss=false。

这是原任务评价器的当前受控完成，不是轮速严格为零或历史home恢复：strict_recovery_quality.passed/controlled_stop仍false，home最大误差35.094149°，诊断原样保留，未修改评价条件。全程6117条观测（含tick0）机身碰撞0、exact body/obstacle active0、非有限0、实测关节越硬限0；最小关节余量7.487758°。第一个未完成任务为无；部署安全和扰动泛化未验证。

物理评估覆盖P01=16、P02=1344、P03=40、P04=80、P05=1168、P06=1704、P07=32、P08=8、P09=1040、P10=8、P11=8、P12=552、P13=116 ticks；这些绝不计入训练阶段样本量。[正式best-task指针](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoint_best_task.json)按完整真实任务而非reward选定CP177152，1/1仅指此checkpoint这一次试验。

质量数据仅作不同build的实测描述：[quality JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/zero_vs_CP177152_descriptive_quality.json)。统计覆盖真实全回合zero73.808333s/PPO50.966667s，不剪去抬腿/放置段：roll RMS .119031/.125496rad、pitch RMS .081318/.105251rad、roll rate p95 .048101/.229873rad/s、pitch rate p95 .055115/.149133rad/s、angular acceleration p95 7.696893/10.441751rad/s²、wheel-target rate RMS2.016369/3.115184rad/s²。两者都成功，但执行源码包含已审核修复差异；不计算稳定性改善百分比，不授予M4，也不把更短完成时间当成更稳定。

## 保留的基线与真实恢复点

成功 zero 保持 8857 ticks / 73.808333s，RR/RL 越障及受控停车已完成，源码、配置、原始数据与视频未改。复用 [zero 视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_final_stop_latest.mp4)，不为同一成果重录。

祖先 CP174592 的174592 decisions /1329 updates /26580 optimizer steps全部保留。它在正式 P02/tick812 RR knee 硬限失败，不是成功 PPO 起点。

reward-only 分支保存：[CP176640](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000176640.pt) · [manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000176640_manifest.json)。

本轮 preserved_mean_reward_only 分支实际新增 **2048 decisions /16 updates /320 optimizer steps**，累计176640/1345/26900。save/load round trip记录通过；不是 PhysX逐tick续接，物理重新合法 reset。Adam有效LR全块1e−5；Identity normalizer保留。旧未完成rollout丢弃，actor/critic/Adam/RNG兼容继承，critic只用新目标数据继续拟合。

## 首块实际样本与任务结果

| 请求阶段 | PPO decisions |
|---|---:|
| P01 | 8 |
| P02 | 613 |
| P03 | 21 |
| P04 | 18 |
| P05 | 1388 |
| P06–P13 | 0 |

共16374真实物理ticks；2048/2048实际记录均 epsilon=0 task-only、full12 mask开放；16374/16374 native派发核验通过且有有效学习目标差异。teacher决策0、teacher ticks0，课程参数中的prefix_source默认字符串不是实际teacher前缀。

三条完整随机训练回合均真实FR捕获后停于P05 local deadline：627/617/666 decisions，41.758/41.133/44.358s，FR placed ticks1251/1285/1488；没有FL捕获。末尾138 decisions处于P02/tick1104，非终态、bootstrap保留，不补造失败或成功。13次普通阶段交接没有done，3次真实终态不bootstrap。

P02原GAE均值+.375838（543正/70负），实际标准化advantage均值−.019314（349正/264负）；old value均值−16.477760、return均值−16.101922。它们是原记录，不用新critic重算。P05终态所在update value loss125.37/129.00/145.80，说明新目标下critic仍有明显终态误差，不能以梯度/actor变化代替能力恢复。

在P02 nominal>+.05的601个决策中，FL最终反向566、FR接近零440、RL接近零226，RR601次增强；最终目标均值FL−.167736/FR+.017835/RL+.064245/RR+.775818 rad/s。这是随机闭环样本描述，阈值不是新门槛，也不是牵引力或正式均值结果；尚不能说错误差速已消除。

## 正式评估与下一边界

[CP176640正式评估](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_task_first_recovery_v1/video_eval/validation/20260916T0417200243144Z_gb0438f66ec63_288f6005a8cf41019a7cd2420445b800/run_manifest.json)：official checkpoint load通过，seed4001、natural P01 tick0、单episode、optimizer0、teacher0；817/817实际native tick full12 mask均开放，派发核验与学习目标差异有效，无零raw偷换或episode内状态写入。真实物理判定valid，但任务失败，非录像/基础设施故障。

FR抬升tick23，front-edge/placed均未发生。最终RR knee target -3.729630°、actual -60.024436°，误差56.294806°；不是命令直接要求-60°。最终wheel targets（FL/FR/RL/RR，canonical rad/s）为[-.224238,+.002450,+.029583,+.797650]，仍是FL反向、FR/RL接近抵消、RR增强。比祖先多5tick不构成任务恢复，quality与完整任务改善均不宣称。

这满足用户授权的“一块新目标续训后同样错误均值仍持续”的mean-head恢复触发条件。仅计划一次明确命名候选，保留sigma/特征层/旧模型与不相关Adam状态；不是反复同样小更新或随机搜索。

offline initializer最初文件因lazy CUDA seed callback覆盖恢复的Philox offset而被拒绝，原文件保留、不得训练。已在同一策略/同一分支修正初始化顺序：先materialize CUDA RNG再恢复，所有RNG精确一致，save/load验证通过；不是第二次搜索或新的mean reset策略。

已验证初始化：[corrected checkpoint](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_initial_mean_head_recovery_v1_from_000176640_rng_corrected.pt) · [receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/mean_head_recovery_v1_initialization_rng_corrected.json)。只将共享输出层mean的前12行归零并清对应exp_avg/exp_avg_sq；sigma后12行、特征层、critic/其他Adam/Identity均保留；共享Adam step3820保留并明确不冒称重新年轻的Adam参数。初始化0 policy decisions/0 updates/0物理步，不是PPO学习成功。

mean_head_recovery_v1首个自然P01真实块已完成：[run](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_task_first_recovery_v1/train/20260916T0426012963516Z_gb0438f66ec63_0e70fe7e93a64b768d439966436da7b7)。实际新增 **128 decisions /1 update /20 optimizer steps**，分支计数起点176640/1345/26900，最新累计176768/1346/26920。本轮两分支合计2176/17/340；初始化和评估均不计训练信用。

最新保存：[CP176768](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000176768.pt) · [manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000176768_manifest.json)。本块P01=2/P02=126，1024真实物理ticks，0完整episode；末尾P02/tick1024非终态、FR Q20无C/P，保持bootstrap。128/128 epsilon0/full12，1024/1024实际native派发核验与残差作用；teacher0。真实update1346 KL=.019741、clip fraction=.310938、value loss=.238826、LR1e−5，actor从67ca847d…变为36b32507…，save/load记录通过。

必须区分初始化与学习：这128条动作全部是在本块唯一update之前，从mean头归零的起点采集，因此其较协调的四轮目标均值[+.214507,+.308216,+.280743,+.321434]来自初始化起点及真实随机闭环，不能归功于更新后的mean。随后确实完成优化并保存了不同actor；**CP176768正式保存后重载已证实这个更新后actor保留了M1/M2能力**，但没有证明reward或单次update独立创造了这些能力。

## CP176768 自然 P01 正式回合：已完成与第一个未完成任务

[最终实际证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/evaluation_CP176768_actual.json)：seed4001，official reload的actor SHA为36b32507…，与训练保存一致。6311/6311 ticks full12许可、native派发核验及非零policy实际作用，teacher0、pre-action0、非法状态写入0。789次实际决策；788次环境正常返回的reward记录全为epsilon0，最后一次在7个物理tick后碰撞中断，不补造其reward返回。

| 物理事件 | tick / 时间 | 当时实测证据 |
|---|---:|---|
| FR主动抬升 / 越沿 / 捕获 | 23 /1461 /1484；捕获12.3667s | FR轮心越前沿+.032210m；顶面触点z=.051611m，真实载荷7.8220N；机身未碰撞 |
| FL主动抬升 / 越沿 / 放置 | 1651 /2350 /2351；放置19.5917s | FL轮心越前沿+.000344m；顶面触点z=.050452m，接触力.249877N；当时FR顶面载荷12.2395N |
| 前部继续推进 | P06 tick2352→5984 | base x从.279599→.633397m，前进.353798m；并非仅凭P06标签授予M2 |
| RR抬升资格 | 6272 /52.2667s | RR有实际初始离地，但没有越沿/放置；不能授予后腿越障成功 |
| 最终失败 | 6311 /52.591667s，P09 | exact base_link/obstacle pair持续接触；机身碰撞几何穿入.00000777m |

FL的首次放置载荷较小且并非持续承载：tick2512时FL已AIR、顶面净空+.013337m；tick6000又实测顶面接触4.739803N。最终四腿当前均AIR、载荷0，RR current_lift_valid=false。历史FR/FL placed不能覆盖这些当前状态。第一个未完成任务为P09 RR保持受控抬升并前送/越沿/顶面放置；之后RL及P13停车未完成。

正式物理阶段覆盖ticks：P01=16、P02=1432、P03=40、P04=104、P05=760、P06=3632、P07=8、P08=8、P09=311，P10–P13=0。这些是评估覆盖，不计入PPO训练样本。此checkpoint仅1次正式试验，M1=1/1、M2=1/1、M3=0/1；不能混合不同checkpoint计算一个模型成功率。

[完整真实尝试视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/videos/ppo_task_recovery_full_attempt.mp4)。[best-task development指针](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoint_best_task.json)绑定CP176768及其实际评估，明确DEVELOPMENT_PARTIAL_M1_M2、full_task_success=false、safety_pass=false、deployable=false；完整安全成功best仍为空。

## 质量只作描述，不宣称优于成功 zero

两次manifest核对seed、冻结物理/A、成功N/mapper/metrics、非reward配置一致；但zero完成任务73.8083s、PPO失败52.5917s，任务能力和统计时窗不等。故不计算改善百分比、不宣称稳定性优越。

| 全回合实测指标 | 成功zero | CP176768失败回合 |
|---|---:|---:|
| roll RMS rad | .119031 | .125278 |
| pitch RMS rad | .081318 | .098248 |
| roll rate p95 rad/s | .048101 | .139387 |
| pitch rate p95 rad/s | .055115 | .123146 |
| angular acceleration p95 rad/s² | 7.696893 | 8.621565 |
| applied wheel target rate RMS rad/s² | 2.016369 | 1.540545 |

最后一项更低不能把未完成任务转为稳定性改善。endpoint机身collider最小z为zero .172999m、PPO .049992m，不冒称全轨迹最小机身高度。接触冲击和slip代理的有效样本及语义限制保存在[描述性质量JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/zero_vs_CP176768_descriptive_quality.json)。

## P06 前驱课程、合法停止与执行修复后的真实续训

原计划512条的P06课程已在首个完整update安全停止，实际128 decisions /1 update /20 optimizer steps，不能计未消费384条。全部128条属于P06，1021个policy物理ticks；前80条在完整episode tick3317机身碰撞，后48条为第二回合P06/tick3064非终态尾段。两次从自然P01运行成功N前缀到P06，各335决策/2680ticks，共670决策/5360ticks全部排除PPO信用；其中FR/FL事件是teacher历史，不是新增PPO M1/M2。已保存CP176896/1347/26940，KL=.022182、value loss=87.513603、有效LR1e−5，128/128 epsilon0/full12、1021/1021 native作用真实。详见[实际receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_mean_head_P06_176896_actual.json)。

确认的实现错误不是新reward假设：nominal_geometry使用已含policy的上一拍最终目标，再由adapter叠加本拍residual，形成累积偏离。修复在adapter中维护8维名义servo命令历史，geometry仅读这个不含policy的命令历史；真实最终slew仍读实际上一拍final。zero快速路径也同步名义历史，普通phase切换不清零。这个内部记忆不增加372维observation；六份配置、成功N源时序、reward epsilon0、残差范围、HISTORY/采样温度均未改。

[明确execution-composition迁移](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoint176896_execution_composition_migration.json)绑定实际最新CP176896 SHA a246fa7d…与新提交，绝不是reward-only迁移。动作执行/有效transition语义确实改变，不能说physical MDP不变；物理场景与actuator能力未改。actor/critic/完整Adam有效LR/Identity/RNG/累计计数保留，不再mean-head reset；旧未完成rollout丢弃，合法自然P01重新采样。

新迁移与既有task-first/timing/height/temperature回归144项通过，含官方CPU学参、Adam、Identity、RNG、计数及迁移后更新/重载测试。旧body-only历史fixture因从带新epsilon0 helper的当前reward反推旧AST而失败；用b043旧validator只读复现同样失败，未放宽旧validator。成功zero的已封存P09=717/P12=431条geometry-active记录离线重放：目标数值、状态、float32 actuator目标均0差；这是保护证据，不是新的物理成功。详见[zero重放receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/retained_zero_geometry_replay_receipt.json)。

修复版本首块从CP176896、自然P01实际新增128 decisions /1 update /20 optimizer steps。P01=2、P02=126，1024真实ticks、teacher0，128/128 epsilon0/full12且1024/1024 native有效。末尾P02/tick1024非终态，FR Q28而未越沿/捕获，保留bootstrap；不能把这8.5333s训练前缀称为P02成功。update1348 KL=.023844、clip fraction=.253125、value loss=.154627、LR1e−5，actor确实变化且保存重载核验通过。

当前最新：[CP177024](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000177024.pt) · [manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000177024_manifest.json)，SHA c4deb9c4…。mean-head分支自己新增384/3/60，连同reward-only分支，本轮实际新增2432/19/380。PPO实际样本分布P01=12、P02=865、P03=21、P04=18、P05=1388、P06=128，P07–P13仍0；正式评估覆盖不冒充训练样本。

[修复后训练receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_execution_fix_177024_actual.json) · [实际reward与动作信号](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_execution_fix_177024_signal.json)。

## CP177024 正式封存结果：四腿链已取得，停止条件未取得

[实际封存审计](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/evaluation_CP177024_actual.json)：official saved/reload、naturalP01、seed4001、teacher0、optimizer0。6177/6177物理ticks均full12/native核验且有真实policy作用，零raw偷换0、非法状态写入0。773次实际决策，772次正常环境返回的reward记录均epsilon0；最后1tick触发观察终止，不补造reward返回。

| 腿 | 主动抬升Q | 越沿C | 放置P | P时实际顶面接触力 |
|---|---:|---:|---:|---:|
| FR | 22 | 1341 | 1363 | 22.770834 N |
| FL | 1516 | 1854 | 2624 | 14.985716 N |
| RR | 4833 | 5490 | 5490 | 12.958039 N |
| RL | 5576 | 5986 | 6057 | 10.525973 N |

上述是当前修复版本同一个自然P01回合的真实事件，不是teacher前缀或拼接后缀。评估物理阶段ticks：P01=16、P02=1312、P03=40、P04=88、P05=1168、P06=1944、P07=16、P08=8、P09=904、P10=8、P11=8、P12=552、P13=113；没有追加到训练样本计数。

全轨迹含tick0的6178条观测中：机身碰撞检出0、exact body/obstacle active0、非有限0、实际关节越硬限0；最小实测关节余量19.105179°。这表示此回合未触发这些硬安全失败，不等于受控停止或部署安全认证。

终态body全在平台范围，FR/RL/RR当前顶面承载约11.097/13.897/3.198N；FL当前AIR、顶面净空+4.839mm，不能用历史FL placed声称它现在承载。最终wheel target（FL/FR/RL/RR）[.330426,.274363,.409621,.300712]rad/s，实测[.329145,.069858,.491733,.258064]rad/s；home最大误差32.084412°、final_controlled=false、strict controlled_stop=false。

物理遍历事件在50.475s记下，固定1s观察于51.475s结束，终止源标签POST_COMPLETION_LOSS；其中post_completion_loss_observed=false，不能把标签曲解成发生了跌落/碰撞。第一个未完成任务是P13当前受控停车。核心诊断正在检查观察窗口触发与stop所有权，本报告不猜测根因或擅自降低成功条件。

此checkpoint正式试验1次：M1=1/1、M2=1/1、完整M3=0/1。四腿放置和body入平台不等于M3。best指针已升级为CP177024的DEVELOPMENT_PARTIAL_ALL_FOUR_PLACED_P13_STOP_INCOMPLETE；完整成功best仍为空，deployable=false。

新[质量描述JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/zero_vs_CP177024_descriptive_quality.json)保留两条不同长度的真实回合：zero73.808s成功，CP17702451.475s未完成停车。roll RMS为.119031/.134389rad、pitch RMS .081318/.095573rad，roll rate p95 .048101/.170187rad/s。新旧adapter/geometry有已审核执行修复差异，并不是新录制、严格代码一致的B/C配对；虽然六配置和冻结物理一致、旧zero离线geometry重放0差，仍不计算成功同等条件的改善百分比，也不宣称稳定性优于zero。

[本块receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_mean_head_176768_actual.json) · [动作与reward信号](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_mean_head_176768_signal.json)。

详细证据：[训练receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_preserved_mean_176640_actual.json)、[reward/full12/GAE信号](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/training_preserved_mean_176640_signal.json)、[逐候选评估](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/evaluation_results.json)、[分支manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/recovery_branch_and_training_manifest.json)。方法与完整旧轨迹回报核算见reward_task_priority_changes.md。
