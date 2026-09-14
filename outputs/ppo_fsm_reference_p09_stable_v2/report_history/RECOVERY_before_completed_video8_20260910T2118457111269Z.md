# Residual PPO：当前状态与恢复说明

## 固定状态：block19 已完成保存，video8 正在自然P01评估

最新独立核验 checkpoint 为 **157568 policy decisions /1196 PPO updates /23920 optimizer steps**。生产 HEAD `d4e46006b382093b960f2c428081d8a1c595bb6a`，runtime `ecee8d3abba1981b95d462a70c9ea39deb3e949878caea9d6589c95cdc043458`；HISTORY372/12维动作、120Hz物理/15Hz策略不变。

[主控完成核验凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/completed_block19_checkpoint_verification.json)记录 block19 于 **2026-09-10 20:59:11.0529200Z** 正常 exit0/SUCCEEDED；主控 **20:59:44.4444588Z** 核验 checkpoint/manifest 双字节 SHA 与官方 roundtrip：

- [Checkpoint157568](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000157568.pt)：`82723c554365ee9f37d6e7fe4b6db6134edeec427dfb892f70bfdd2a40f42493`
- [官方 immutable manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000157568_manifest.json)：`dd6bc6cadda7b1b75163da228fa0331eecffde99a1bb163f2ab4263cfa3a7cbf`
- update1196 后 actor：`1430dc42b722be0a92a29b1bfef0c92301087479161232ccde330a438b7b90c2`；固定更新日志 finite/nonzero gradient=true、actor 实际改变。
- Adam 有效 LR 末值1e-5，仍为原 adaptive 调度；spent budgets 为 full_episode77184/phase_suffix70272/smoke0。

| 范围 | 已保存决策 / PPO更新 / optimizer steps |
|---|---:|
| 旧14完成块，至旧7db来源152192 | **11648 /91 /1820** |
| 新d4e的block15–19，起点152192 | **5376 /42 /840** |
| 本轮基线以来19完成块累计 | **17024 /133 /2660** |

block19 的2048条全部优化保存，未完成 rollout=0；先前74条仅为历史截点，已计入本块一次。自然P01真实产生 **RR I7/Q5/C1/P0**；RR先在4615越沿，但仍AIR、未放置，4640出现RL越沿几何并触发既有RR_FIRST_ORDER。不能写成“RL比RR先越沿”，不能改称完整成功。

video8 已从157568启动自然P01/seed4001条件均值评估。主控固定 **21:02:24.4189779Z** 观察94次决策/P02/t752、未终止；最终 source manifest/checkpoint_load_provenance 尚未生成，不能提前声称独立评估加载哈希核验或完整结果。最近完成的模型全程起点评估仍是 **video7/154240/P06 RR膝硬限**；157568不是best或成功checkpoint。

## Block15：新 nominal 下的真实自然P01训练

Run `20260910T1631133451770Z_gd4e46006b382_079314d0fcb0460bbd585da6715d4cb9` 于 **17:10:12.767Z 正常 exit0**。配置为 full_episode/P01、2048决策、seed1001、N1、cuda:0、每次更新保存；无教师信用。[固定完成快照](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T171034151720Z.json)记录阶段样本：

**P01=19 /P02=886 /P03=70 /P04=15 /P05=740 /P06=318 /P07–P13=0。**

因此，不能把旧实验中的P09数据称作新 nominal 已采样P09。新版本本块尚未发出P07–P13策略动作。

16355个策略物理 tick 全部 verified/actual effect，其中16317为 own-phase、38为普通阶段继承；禁止的 episode 状态写入全0。8个真实终态为 **5 FALL、1 HARD_JOINT_LIMIT、2 BODY_COLLISION**。最后7条已经学习/保存，是 **P02/tick56/0.466667s 的非终止采集尾**，terminal=false/reason=null；不是第9个失败或成功回合，也不是被丢弃的未完成rollout。

本块 RR 新 **I2/Q1/回地撤销1/C0/P0**，current-valid决策末态4；AIR36/GROUND2012。AIR不等于有效抬升。载荷有效性为1989/2048，未知值不能写成0。尾部RR仍GROUND，无I/Q，front−843.286481mm。后腿越沿、放置和完整越障均未证明。

[ep2的固定P05诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block15_episode2_p05_failure_receipt.md)仅分析199条实际样本：FL三次Q1076/1227/1292分别在1199/1268/1473回地撤销，没有C/P；首次未完成的是保持有效抬升并前送越过前缘，不是“已经越沿只差落置”。RR没有I/Q/C/P。终态float32观测回解 base_z≈14.807813mm，支持低高度FALL，但不冒充原始双精度物理流。FR当时真实TOP/10.949826N，FL在GROUND/1.668851N。1590 native effects均有效、普通交接保留residual，没有新增明确下发或判定缺陷。该例没有FL P或P05→P06，因此不声称它验证了晚捕获交接分支；原诊断的历史范围保持不变。

## Video7：真实 RR 膝硬限安全中止，不是完整成功

Run `20260910T1711073721697Z_gd4e46006b382_c8a157341180454faab7e89333997585` 于 **2026-09-10 17:25:18.2845990Z** 完成，exit1/`DIAGNOSTIC_FAILURE` 对应实际 `SAFETY_ABORT/HARD_JOINT_LIMIT`，不是录像失败或时间预算到期。[主控最终加载与评估凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video7_154240_parent_evaluation_receipt.json)核实 checkpoint154240、d4e/ecee、HISTORY372、conditional_mean、seed4001、无迁移、0更新；初始 actor/critic/Adam/normalizer 哈希匹配。自然重置为 P01/tick0/decision0/done=false、reset1/options{}、prime/pre-action/extra ticks全0。**失败先于末端 check_model，末端模型哈希检查未执行**；初始哈希通过和0更新不能替代该检查。

本次发出 **732次策略决策**，其中731次完整返回8个物理tick，最后一次只执行5tick即被安全中止：**731×8+5=5853tick，48.775秒**。不要用正常返回的旧5848末态替代真实5853安全终态。发出阶段计数为 P01=2/P02=182/P03=6/P04=25/P05=149/P06=368，P07–P13未到。

[物理与P05桥接诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video7_rr_hard_limit_and_p05_bridge.md)记录 RR 膝实际 **−60.001520559649°**，下限−60°，当时目标 **−5.938500352311°** 在限内；全5853 native ticks真实执行验证通过。目标与实际不一致不是新dispatch错误或某一控制机制的唯一因果证明。P06双后腿接近尚未完成：rear_approach=min(RL,RR)，最终RL仍在前缘后334.366348mm、RR后50.363805mm，RR I/Q/C/P全部0。

**FL没有掉出台面。** FL I1782/Q1786/C1968/P2910，2912进入P06；从2912至5848的368个放置后正常决策末态均TOP、有效支撑/承载，真实终态5853仍TOP/withinXY、GROUND=false、bearing verified、8.446005N。该范围是决策末态加独立终态，不冒充全时刻接触扫描。368个末态包含返回2912的P05→P06交接，不是368次完整正常P06动作。全局 `CONTACT_BEARING_UNVERIFIED` 从5672起来自RR的混合/歧义障碍接触，不是FL失去TOP；未知载荷比例不能写成0或虚构已验证载荷分配。最终物理run validity为VALID，但任务未成功。

P05晚捕获v2桥本次未触发：旧wheel停止tick2808；在P2910之前，front已于2832超过+.10m且nominal已为0。晚桥不能重启早已停止的建议。之后P06作者发出+.3是新阶段命令，residual连续保留；没有证据将此例写成捕获截断或新软件缺陷。

[可播放诊断MP4](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video7_154240_p01_safety_abort.mp4)为独立无损stream-copy，见[remux凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video7_154240_p01_safety_abort.remux_receipt.json)与[主控首末帧视觉核验](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video7_154240_visual_review.md)：732 unique、1280×720/15fps、black0/fullDecodePASS，包内容/PTS/DTS/duration/key flags一致；48.8s与物理48.775s相差既有15Hz末帧量化0.025s，没有增帧、裁剪、拼接、变速或重编码。原MP4和source manifest保持不变；原容器duration异常不能称原容器验收通过。此文件不是成功发布、同条件FSM配对结果或稳定性改善证明。

## Block16：P06课程实际640条、5次更新后完整边界停止

Run `20260910T1726535008627Z_gd4e46006b382_45112e1f0f1d48088e3832475c43dc6e` 请求P06/phase_suffix1024、seed1001、N1/cuda:0、每次更新保存、TeacherOffset0/frozen_fsm，从154240同HEAD普通续接，无迁移。实际 **640决策/5 PPO updates/100 optimizer steps** 全部保存到154880，lifecycle为STOPPED_AT_VERIFIED_UPDATE_BOUNDARY；正常exit0不代表物理任务成功。剩余请求384没有被采样或计信用，也不是新增失败。

[固定完成快照](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T183855828983Z.json)由现有stdlib-only summarize_live.py针对这个明确run一次生成：**P06=540/P07=7/P08=20/P09=73，其余0**。5108个策略physics/native effect均verified，其中5100 own-phase；640条raw/applied residual均非零，四类episode状态写入全0，teacher误计0、warnings/errors为空。快照中CLI默认arguments.mode仍写semantic_prior_eval，但run kind=train且有5次真实optimizer更新，不得误标为N0/零信用评估。

6个真实终态为 **3 BODY、3 FALL**：依次P08 BODY、P09 FALL、P06 FALL、P06 BODY、P06 FALL、P09 BODY。没有新的hard limit或incomplete终态。最后 **71条（global154810–154880）** 为P06/t4152/34.6s的已学习非终止采集尾，reason=null、RR GROUND/无当前有效抬升；不是第7个失败或成功回合。

RR新 **I3/Q2/Q回地撤销1/current-valid70/C0/P0**；另记录current_lift_revoked_ground事件1，不能把两个撤销标签直接相加成两次Q撤销。AIR88/GROUND552、load-fraction valid620/640；未知值不写成0。决策末态最大抬升189.574mm、最靠近前缘仍−135.043mm，未有真实越沿/放置。最后P09 BODY时虽然RR历史I/Q与AIR成立，物理中止使current-valid=false；不能因离地把任务标成功。后腿课程现在确有新d4e P07–P09样本，但这是后缀训练，不是完整自然P01成功。

[block16实际加载凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block16_actual_training_load_receipt.json)保留17:34:58.556028Z P06/t3584/READY真实加载和每更新保存参数。17:37:06的123条旧采样记录保留为历史，当前完成快照替代其未保存状态。汇总脚本只扫描已完成policy/update/terminal流，并读取小型pointer/sidecar声明；未加载或重哈希tensor、未读大physical流或媒体。独立CP字节验证来自主控凭据，不来自汇总器。

## Block17：首128条完整更新，RR有两次Q但均回地

Run `20260910T1839014552998Z_gd4e46006b382_3bf26c40dc774ea38185c4b77938fded` 请求P07/phase_suffix1024、seed1001、N1/cuda:0、cadence1、offset0/frozen_fsm，从154880同HEAD普通续接，无warm start/migration。实际在首完整128 rollout优化/保存后正常停止：**128决策/1 PPO update/20 optimizer steps**，剩余896未采样、不计信用；这是课程分配的完整更新边界，不是任务停止或成功门槛。

[固定完成快照](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T192028878355Z.json)：**P07=3/P08=12/P09=113，其余0**。全1019 native ticks/effects验证有效，1013为own-phase、6为普通阶段继承；128条raw/applied均非零、4类episode状态写入全0、teacher误计0，warnings/errors与partial-tail-bytes均0。

两个真实终态为首9条的 **P09/t5980 FALL（global154889）** 和接下10条的 **P09/t5991 BODY（global154899）**。剩余 **109条（global154900–155008）** 是已学习非终止采集尾，最后P09/t6784/56.533333s、reason=null、RR GROUND/current-valid=false/C0/P0，不是第三个终态或被丢弃rollout。

**全块RR新I2/Q2，两次真实回地撤Q，current-valid25，C/P0。** 另有current_lift_revoked_ground2标签，不能重复加作另两次Q撤销。AIR37/GROUND91，bearing与load-fraction valid128/128；决策末态最大抬升104.645mm、最靠近前缘仍−63.390mm。因此，不能把末态Q=false写成“本块从未获得Q”，也不能把曾经离地写成已经越沿/放置。此前首9/19条都尚未优化的事实已由完整128保存解决，历史快照仍保留。

[实际加载凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block17_actual_training_load_receipt.json)保留18:52:12.0094720Z P07/t5912/READY、首次154881/teacher_miscredit=false；早期18:42教师前缀与18:52的9条采样均为历史时点。此次仅用现有stdlib脚本扫描已完成run一次，未读大physical流、媒体或CSV；保存字节核验来自主控凭据，不来自汇总器。更细的尾段物理诊断由独立任务完成，本报告不重复扫描或推断原因。

## Block18：P10课程完成512条，RL真实越沿仍未放置

Run `20260910T1921276505520Z_gd4e46006b382_2f570e3174ba4fe3a1035ca1c6162b8b` 请求P10/phase_suffix512、seed1001、N1/cuda:0、cadence1、offset0/frozen_fsm，从155008同HEAD普通恢复，无warm start/migration。实际 **512决策/4 PPO updates/80 optimizer steps** 全部保存至155520；SUCCEEDED只指训练请求正常完成，非物理成功。

[固定完成快照](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T202117929903Z.json)：**P10=3/P11=17/P12=492/P13=0**。全4090 native/actual-effect ticks verified，4084 own-phase；512条raw/applied residual均非零，四类episode状态写入0、teacher误计0、warnings/errors为空。

两个真实终态是 **334条P12/t10256 BODY（global155342）**，再 **109条P12/t8450 RR膝HARD（global155451）**。最后 **69条（global155452–155520）** 是已学习非终止尾，末P12/t8136/67.8s、reason=null，未作为第三个终态。334+109+69=512，没有丢弃尾部rollout。

[完整RL事件补充](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block18_completed_rl_event_summary.md)仅在已完成512行做一次RL标量/事件核对：**新I9/Q6/回地撤Q4/C1/P0**，教师继承RL仅I6。分回合为4/4/3/1/0、4/2/1/0/0、1/0/0/0/0（I/Q/撤Q/C/P）。

RL新C发生在 **tick9256/global155217**，在教师P10信用起点7584之后，不是教师继承或仅AIR标签。全块RL任务TOP端点和history placed均0；最向前front+16.005400mm后发生回退，首回合BODY时front−47.603127mm且AIR/无support。第二回合HARD来自RR膝，RL当时AIR/0载荷，不能虚构其支撑。69条非终止尾末RL已GROUND、真实承载8.147947N，front−50.151762mm、current initial/active_attempt=false、Q/C/P=false；最终无Q不抹去前面真实Q/C。RL没有RR专用current_lift_valid字段，缺席不填false。

RR全512行的历史C/P为true，但**新RR I/Q/C/P全0；教师继承I/Q/C/P各3**，不能算三次新PPO放置。RR current-valid端点158，2次current_lift_revoked_ground不等于新Q撤销事件；首终態RR混合接触/承载不确定、第二终態GROUND、尾末真实TOP/current-valid。历史放置不等于持续顶面承载。

保留原短报告：[前95连续性](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block18_first95_p10_p12_continuity.md)、[RL越沿至276](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block18_new_rl_edge_crossing_through276.md)、[episode1硬限](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block18_episode1_hard_limit.md)。原95采样和443采样/384优化的截点已由512完整保存解决，但原文件不改写、不重复计信用。新RL越沿不是完整后缀成功、自然P01成功或FSM稳定性改善证明。

## Block19：自然P01完成2048条，RR先越沿但放置未完成

Run `20260910T2021542525340Z_gd4e46006b382_6707fa943a204563ab1699572bde18a9`，full_episode/P01/2048、seed1001、N1/cuda:0、每次更新保存，从155520普通同HEAD恢复，无NewMdpWarmStart/ResumeMigration。[实际加载凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block19_actual_training_load_receipt.json)保留20:22:20.2892870Z真实P01/tick0/prefix=null；自然路径的curriculum_start与teacher字段缺省null不硬填false。20:23:37的74条未保存样本是历史时点，现在已包含在完整2048中。

[固定完成快照](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T210052945289Z.json)核实 **2048决策/16 PPO updates/320 optimizer steps** 全部保存至157568；SUCCEEDED只是训练请求完成，不是物理任务成功。阶段样本：

**P01=15/P02=685/P03=29/P04=8/P05=793/P06=150/P07=1/P08=1/P09=366/P10–P13=0。**

16357个native/actual-effect ticks全部verified，其中16326 own-phase、31普通阶段继承；2048条raw/applied residual均非零，四类episode状态写入及teacher误计均0。CLI默认mode字段不是此train-kind请求的策略类型证明，不得把16次真实更新误写为N0。

6个真实终态为 **2 INCOMPLETE_CONTROLLER_BLOCKED、3 FALL、1 BODY_COLLISION**。最后 **40条（global157529–157568）** 是P02/t320/2.666667s、reason=null的已学习非终止采集尾；不是第7个失败、成功回合或丢弃的rollout。六个已终止回合580+150+259+164+276+579=2008，另40尾部，共2048。

[首580条顺序诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block19_episode0_rr_rl_order_terminal.md)保留固定范围：FL I910/Q915/C1222/P1256；RR Q1691在P06建立，1696→1704→1712连续接入P09，普通交接不清零residual、不设done。RR全块新 **I7/Q5/真实回地撤Q4/C1/P0** 均见首回合；current-valid末态62。四次Q撤销与current-revoked是同tick的两个标签，不能重复计数。

RR **C4615** 已记录，4640仍AIR/0N、gap+127.178mm、P=false；此时RL front+0.484mm、真实TOP/verified承载8.789N，但现行顺序分支不登记RL C/P。因此准确说法是“RR越沿后尚未放置，RL出现越沿几何导致顺序中止”，不是“RL先于RR越沿”，也不能伪造RL C=4640。

主控已核对正式规范与现有正反例并[保留规则](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/completed_block19_checkpoint_verification.json)：`INCOMPLETE_CONTROLLER_BLOCKED / RR_FIRST_ORDER` 是顺序未完成终态，不是第三个BODY/WHEEL_ONLY任务失败，也不是安全中止或执行链损坏。它确实done、无bootstrap并获通用终态成本，不能淡化为只诊断；文档未单独指定立即终止或局部恢复，同tick首次RR P+RL C覆盖边界未建立，但本例RR仍AIR未P，不是该边界的证据。没有因此热改生产或新增训练门禁。

[固定四条终态报告](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block19_completed_terminals_fixed.md)仍只覆盖20:43:17.6670342Z的episode0–3，不能当成全块六条。ep4 BODY与ep5 P05 INCOMPLETE来自最终2048快照；原四条报告不覆盖、不改写。训练自然P01的RR新越沿也不是固定策略评估或FSM稳定性改善证明。

## Video8：实际运行，最终加载凭据与结果待完成

Run `20260910T2100033647207Z_gd4e46006b382_e98a9290c0f44825aa705e61f7a57377`，PID22376/session17826，currentd4/full_episode/P01、seed4001、conditional_mean、immutable157568；无migration/warm-start/teacher，0训练信用。

主控 **21:02:24.4189779Z** 已读到decision94/P02/t752非终态。这里只记录实际视频策略活动；最终source manifest/checkpoint_load_provenance未落盘，末端模型哈希未核验，质量/物理结果未完成。本报告不读活跃日志或媒体，不预测video8成功，也不把评估结果当optimizer成功门禁。

## Block14与旧版本来源保留

[block14边界凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p05_handoff_revision/boundary_receipt.json)及[完成快照](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T162522921146Z.json)保留旧7db事实：P06-origin请求1024，实际256/2/40，在完整更新边界停止，未用预算768；16:23:48.8767181Z exit0、PID48164已退出。

[旧checkpoint152192](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000152192.pt)及[sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000152192_manifest.json)于16:24:47.5582756Z由主控双字节哈希和roundtrip核验。它仍属于 **7db/5eb旧 nominal**，没有改标d4e，作为历史恢复/迁移来源保留。

本块阶段249/1/1/5对应P06–P09；2038 native/2035 own/statewrites0；2 BODY终态加9条已学习非终止尾；RR I2/Q1/current7/C0/P0。[前247条诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p06_block14_first_two_body_terminals.md)保留当时事实：RR Q3824在P06形成，经3824→3832→3840连续进入P09，交接时FL真实TOP承载，终态则FL AIR。此前119尚未更新、9尚需新增采样的状态后来在256条完成块中解决，不重复计信用。

BODY是被执行链消费的权威任务失败，不是“执行链损坏”，也不是P05 nominal缺口的因果证明。原始持续接触/穿透细项没有保存，不猜测具体触发分支。

## d4e名义动作修订与一次性MDP迁移

主控在确认旧进程停止后修复了P05捕获轮建议在下一策略边界前过早关闭的问题：仅延续到下一个策略边界；新作者wheel事件、危险状态或当前捕获无效会取消延续。它是已确认的小范围连续性修复，**不是video6退回地面或BODY失败的已证实原因**。

变更范围为五个生产文件和两个测试文件：

- `semantic_supervisor.py` 的 NominalMotionProvider 类新增33行，类外字节通过实际迁移凭据核对不变。
- 当前 task spec 的 `nominal.p05_pending_capture` 从 `current_FL_capture_wheel_continuation_v1` 改为 `current_FL_capture_wheel_continuation_to_handoff_v2`。
- `semantic_cli.py`、`semantic_migration.py`、`semantic_training.py` 增加严格限定的same372新MDP路由和加载校验。

主控最终测试为 **190+34+113=337 passed、0 skipped**。首次缺少project PYTHONPATH的collection错误和两次fixture尝试失败保留，不隐藏，也不称为生产测试失败。

这是物理 nominal MDP变化，**不是 exact-MDP resume**。完整372 schema/layout、学习策略kernel、动作范围、reward、任务验收和物理执行器不变。actor全部参数/buffer（含learned std）、critic、训练RNG、identity normalizer、累计计数和spent budgets保留。仅在验证source恢复后清空Adam moments，并保留有效LR1e-5与原Adam options；不继承旧rollout或物理状态。

[实际加载凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p05_handoff_revision/actual_training_load_receipt.json)记录16:31:41.29976Z自然P01/tick0/prefixnull，正确env Isaac Python、项目cwd、新head/runtime、loggercontrol0。[现存新runtime初始checkpoint](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_initial_v3_from_000152192_s979514bfae6a_gd4e46006b382_ecee8d3abba1981b95d462a70c9ea39deb3e949878caea9d6589c95cdc043458.pt)及[初始sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_initial_v3_from_000152192_s979514bfae6a_gd4e46006b382_ecee8d3abba1981b95d462a70c9ea39deb3e949878caea9d6589c95cdc043458_manifest.json)保留152192/1154/23080，官方roundtrip与source actor一致；**其初始tensor字节没有由主控独立重哈希**。该初始保存产生0训练信用。此限制与后续正常154240已经独立双字节核验是不同证据层次。

本次nominal修订局部计数起点为152192；官方warm-start记录仍保留10112历史实验MDP起点。完成block15–18后，新d4e训练量是3328/26/520，不是整个实验的14976/117/2340。没有修改累计计数来源或重做warm start。

## Video8完成后按实际最新checkpoint恢复

活动video8结束前，不启动第二个Isaac进程。实际退出后检查当时HEAD和官方pointer；如有更新且有效checkpoint，使用真实最新版本，不固守这里固定报告的157568。本报告不替代运行中的保存状态。

当前d4e普通checkpoint恢复 **不带 `NewMdpWarmStart`，也不带 `ResumeMigration`**。以下命令仅在无活动Isaac且157568仍是当前HEAD下最新有效版本时使用：

```powershell
.\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead d4e46006b382093b960f2c428081d8a1c595bb6a -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage full_episode -FromPhase P01 -Decisions 2048 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000157568.pt
```

现存d4e初始checkpoint仅为更旧的fallback，恢复它也不带NewMdpWarmStart。不要再次对同一immutable initial路径重放旧152192→d4e迁移命令，以免重复创建初始文件/重置Adam；实际历史命令保留在归档RECOVERY与训练provenance中。旧152192的精确行为恢复需要对应旧7db运行时，不能静默当作d4e加载。本报告不切换checkout或启动进程。

从选定checkpoint恢复权重、Adam、RNG、identity normalizer和计数，物理世界仍须合法reset；不能恢复虚构机器人姿态。按真实证据和资源继续，不新增A5/5、N0成功、固定姿态或非零探针全程成功门禁；单块预算不是整个任务停止条件。

## 历史已完成评估：旧video6与N0，当前模型评估以video7为准

[video6/151936诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video6_151936_p06_diagnosis.md)记录旧7db下1176决策/9408ticks/78.4s、0更新，P06 local deadline恰40s、rear_approach0。FL C4486/**P4604**，4608进入P06时真实TOP/5.405339N；4880失TOP、5168回GROUND，RR无I/Q/C/P。首次未完成是当前FL顶面支撑退化时的后腿准备，历史P不代表当前承载。

最终run validity=VALID，physical evidence=**CONTACT_BEARING_UNVERIFIED**，success=false。FL数值12.369541N/load0.377640不是已验证载荷分配。native执行通过不能改变任务未完成结论。失败分支在末端check_model前退出，**未执行末端模型不变哈希检查**；初始加载验证加0更新不能替代该检查。

主控已看首末帧并展示[可播放无损诊断MP4](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video6_151936_p01_incomplete.mp4)。[修复凭据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video6_151936_p01_incomplete.remux_receipt.json)记录1176 unique、1280×720/15fps、black0/fullDecodePASS、78.4s有效容器，原video/manifest未改，无重编码/裁剪/拼接/变速/插帧。它与历史video3/4/5都不是成功发布、配对FSM稳定性改善或d4e结果证明。

N0仍是独立无actor诊断：1108决策/8861ticks/P09未完成，RR Q5509在6243撤销、C/P0、训练信用0；不是A、当前策略评估或optimizer门禁。

## 固定奖励分析与CSV证据

[block15唯一RR Q的固定诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block15_unique_rr_q_in_p05.md)解释“有Q但无P07”而不修改阶段或门禁：ep6/P05的RR I1805/Q1813在1848回地撤销；FL随后P2150、2152才进入P06，此时RR已经GROUND/current-valid=false。1816/1824/1832/1840四个current-valid末态不能当作后续仍抬起，历史Q时间戳也不等于当前状态。其56个P06末态无新RR Q、rear_approach=0；未发现阶段历史丢失。本块仍是I2/Q1/撤销1/current4/C0/P0，未新增训练信用或新nominal P07样本。

[block13 reward/执行消费者诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/block13_reward_execution_learning_diagnosis.md)仅分析已完成旧7db的2048条，不代表d4e分布。PBRS/family权重/tick成本与raw消费者检查一致；42次普通切换保留非终止bootstrap与势能连续性。它量化局部进展与body/smoothness成本的权衡，没有证明reward符号或消费者错误，也没有因此改超参。该审查未再次加载rollout tensor；限速/headroom与raw大探索不同，不能称通道关闭。旧block13的adaptive LR历史仍为14次末值1e-5，1147=5.0625e-5、1149=2.25e-5，不是“整个块固定1e-5”。

19个完成块累计阶段：

**P01=104/P02=5279/P03=238/P04=145/P05=6155/P06=2868/P07=47/P08=59/P09=977/P10=6/P11=20/P12=1071/P13=55**，总17024。策略physics135934；73真实终态为21 BODY、43 FALL、6 HARD、3 incomplete；18个非终止尾不另计失败/成功回合。RR current-valid端点778包含教师建立后的当前状态，不是778次新抬升；累计新C2/P0。block19增加RR新C1，不把block18教师继承C/P加入新事件，也不抹掉block18的RL新C1/P0。

新增[blocks14–15 CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks14_15_000154240.csv)为 **2304×263**，globals151937–154240：256旧block14＋2048新block15，全部credited/optimized/saved=true、teacher=false。主控已完成scalar grid/recalculation、独立ImportCsv和PNG视觉核验，见[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks14_15_000154240_authoring_qa/receipt.json)及[visual review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks14_15_000154240_authoring_qa/visual_review.md)。两个source分别19,473,953B与146,295,300B；2304行缺失body姿态/线速度保持空白，不能由命令或CoM填补。本报告只读QA凭据，没有读取CSV/raw或新建CSV。

新增[blocks16–17 CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks16_17_000155008.csv)为 **768×263**，global154241–155008：block16=640＋block17=128，P06/P07/P08/P09=540/10/32/186。全部credited/optimized/saved=true、teacher=false；body未知768行保持blank，没有读取大physical流填补。主控已读完整[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks16_17_000155008_authoring_qa/receipt.json)、独立Import-Csv并看PNG，见[visual review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks16_17_000155008_authoring_qa/visual_review.md)。Builder回传发生serialization error，exit code没有交付；CSV和完整QA已生成、Node已退出，主控独立核验通过。没有重跑marker或builder，也不虚构exit0。本报告只读QA文档，不运行CSV工具或重扫源数据。

新增[blocks18–19 CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks18_19_000157568.csv)为 **2560×263**，global155009–157568，即512+2048；全部credited/optimized/saved=true、teacher=false，8条真实terminal。主控已确认现有builder exit0、scalarGridVerified/engineRecalculated、独立Import-Csv与PNG视觉核验，见[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks18_19_000157568_authoring_qa/receipt.json)和[visual review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_blocks18_19_000157568_authoring_qa/visual_review.md)。只读取两份完成策略audit，源50,056,873B与157,447,313B；没有教师、大physical或active video数据。缺测body字段仍空白，margin fallback不冒充直接物理测量。本报告仅阅读这两份小型QA文档，没有读取、重建或渲染CSV。

**不重叠union=19384＝17024已保存训练＋1252早期非视频评估＋1108 N0。** 在原16824上只加本2560一次，现覆盖completed19；不包含video8或其他videoeval、旧重叠CSV1/2或重复部分采样。原CSV与QA全部保留，数据发布不等于任务成功。

## 归档和限制

本次更新前已归档四份block18版报告，并逐份核对归档内容与原报告一致：[training](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/training_and_evaluation_before_completed_block19_20260910T210052945289Z.json)、[identity](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/runtime_identity_before_completed_block19_20260910T210052945289Z.json)、[publication index](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/checkpoint_publication_index_before_completed_block19_20260910T210052945289Z.json)、[RECOVERY](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/RECOVERY_before_completed_block19_20260910T210052945289Z.md)。当前顶部替换为completed19与video8固定运行时点；保留旧块、video7/N0、原CSV及下方historical tail，不无限追加旧top。

index仍为 `READ_ONLY_PUBLICATION_INDEX_NOT_OFFICIAL_RESUME_METADATA`。本报告作者只改四主报告和一次归档，仅阅读给定完成快照、小型核验/诊断/QA凭据；没有修改官方pointer/sidecar/resume state、immutable checkpoint、生产、进程、媒体或CSV，没有重复模型hash、运行仿真或额外汇总。原四条终态诊断不改为六条。

从P01完整越障的成功checkpoint、≤200秒成功PPO视频、同条件FSM/对比视频，以及稳定性优于FSM的证据，仍未完成。报告和单块预算不是整个任务的停止条件。

## Historical recovery state — block8 saved and video3 launch-only snapshot

The following prior text is retained as historical evidence. Its request-only video3 and no-active-training statements are superseded by the completed evaluation and active block9 snapshot above.

### Historical current-state heading — recorded before video3 completion

Checkpoint accounting is fixed at **2026-09-10T11:36:41.705167Z**. Block8 finished
SUCCEEDED/exit0 and Isaac exited at11:35:19Z per parent. This is completed training,
not physical task success. The new video3 started at11:38:47.227154Z; its actual
policy load/actions/outcome have not been reviewed here. Root owns that process.

### Saved checkpoint source differs from current production

Latest immutable saved checkpoint:
[checkpoint_step_000145920.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000145920.pt).
Checkpoint SHA256:
e5a62122d091c9d826880f8c0df2338846d8d7dab7ad0cb08944b8db8a14fbfc.
Official immutable manifest SHA256:
8de12f0d7d8e24cd7bcd0c43c632fa178ac6088bd9df1ac959e50ee48da55b1d.
Parent recomputed both hashes and verified the official roundtrip; this report
read immutable metadata and the fixed completed snapshot, not tensor data.

Saved lifetime: **145920 decisions /1105 PPO updates /22100 optimizer steps**.
New since140544/1063/21260: **5376 /42 /840**, all sampled/optimized/saved,
zero unfinished rollout. Effective saved LR1e-5, identity normalizer retained.
Saved source is **69aeaca777dcc8653e60f19da56ae1cd002e5271** /
runtime21ad0d667c84e5c53e5e8aa37239421f8dcc4f717f5bcf6d1c817c2dd95293a4.

Current production is separately **7db0d17f398d393ce026b6990bd2566d53366407** /
runtime5eb1536f17f3037639015e714f17ae8e805489c35ca3cf5c4318546ef3a4ac93.
There is no claimed7db checkpoint.145920 is latest_not_best, not a success policy.
The publication index remains a read-only reference, never official resume metadata.

### Completed P10 block8

Run20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541
actually saved **512 decisions /4 PPO updates /80 optimizer steps**.
Saved phase samples P10=2/P11=2/P12=508, P13=0. All4093 physical policy ticks
have verified native effects:4089 own-phase ticks and four ordinary handoff-hold
ticks, with no forbidden state writes. Prefix events have zero policy credit.

Episode0 had456 decisions/3647 policy ticks and ended P12
INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED:
RL I11/Q4, all four Q attempts subsequently ground-revoked, C/P0.
Episode1 had56 decisions/446 policy ticks and ended P12 HARD_JOINT_LIMIT:
actual rear_right_knee−60.025815543° despite target−41.584803440°.
RL had no new I/Q/C/P in that second suffix. Unknown load stays unknown, not zero.
See [episode0 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block8_episode0_terminal.md),
[episode1 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block8_episode1_hard_limit.md),
and [completed block8 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T113641705167Z.json).
Their older384-saved/128-pending timing statements remain historical; this final
snapshot now confirms all512 saved. Neither episode completed the RL task or P13.

The old collector ran an unused third63.2s teacher prefix before the last update
saved. It contributes **zero** policy decisions, episodes, events or update credit.
RR C/P present in suffix observations was teacher history, not new policy success.
Actual block8 load identity was10:52:49.789459Z/PID19228 atP10/tick7584/READY
after real reset and source145408 loading; it does not identify the new video process.

Cumulative saved phase counts:
**P01=33/P02=1690/P03=59/P04=42/P05=2139/P06=548/P07=24/P08=17/
P09=184/P10=3/P11=3/P12=579/P13=55**.
There are **26 real task/safety/incomplete terminals**:
8 BODY_COLLISION task failures,17 independent safety aborts (FALL15/HARD2),
and1 finite incomplete-task terminal. Seven nonterminal collection tails remain.
No collision and no safety abort still do not imply task completion.

### Terminal-tail scheduling revision and exact recovery migration

Only semantic_training.py changed in7db: for supported N1, identity-normalized,
nonrecurrent/no-RND collection, a true terminal on the last rollout tick is
persisted, stored and updated before reset; a verified checkpoint is forced,
and reset occurs only when another rollout begins. Mid-rollout terminals and
ordinary nonterminal phase transitions retain their prior behavior.
The six configs, physics, task/reward, nominal, action mapper, HISTORY372/12
policy kernel and hyperparameters are unchanged. This is not a new MDP.

[Regression XML](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/rollout_tail_reset_regression_v1.xml)
records376 CPU tests passed,0 failures/errors/skips,51.315s. This includes official
PPO terminal-return/save seams; it does not prove real Isaac reset RNG or future
physical trajectories are bitwise identical.

Resuming saved69aeaca checkpoint145920 undercurrent7db requires the exact
[terminal-tail migration](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/migrations/terminal_tail_reset_from_000145920_to_7db0d17.json),
SHA d38f63f0c88a581bfae8bccd2e641e9c7be546dbcca425f5c26212e2793d190e.
It preserves actor/critic/std, Adam and actual LR, normalizer, checkpoint RNG,
counters and spent budgets; discard old rollout and start legal P01 or a declared
zero-credit prefix. No NewMdpWarmStart, new network or unconditional RNG reset.
Do not omit this plan merely because the observation dimension remains372.

### Video3 is ACTIVE_REQUEST_ONLY, not an evaluation result

Run20260910T1138465345417Z_g7db0d17f398d_9bd793c700bd4040b4c517a0cd625ebb
was launched with parent PID43176, source145920 and the exact migration above.
[Started manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1138465345417Z_g7db0d17f398d_9bd793c700bd4040b4c517a0cd625ebb/run_manifest.started.json):
semantic_residual_eval/C, naturalP01/full_episode, seed4001/N1, visible capture,
MaxDecisions3000, no short Decisions argument, no teacher prefix or warm start.
Setup had0actions at the launch report; later sampled actions and exact load proof
are unreviewed, not asserted zero forever. Video evaluation has no optimizer
updates or checkpoint writes. Do not launch a concurrent recovery process.

Latest **completed naturalP01 evaluation still uses142848**, seed2001:
617 decisions/41.075s, P05 LOCAL_TASK_DEADLINE, first unfinished FL crossing/
controlled placement, no BODY/FALL, task_success=false.145920 roundtrip and the
new video request do not replace that missing latest-checkpoint full evaluation.

After the active process ends, any recovery command must use actual current HEAD,
the immutable145920 source and its7db migration unless a newer verified checkpoint
is deliberately selected. A same69aeaca resume or old142848 migration is not the
current recovery route. Neither successful recording nor task completion alone
would establish FSM-relative stability improvement.

### Timeline publications and preserved history

The unchanged3556-row consolidated CSV plus2560-row block7 delta remain available.
The new [block8 delta](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block8_000145920.csv)
adds512×263 withteacher0, P10=2/P11=2/P12=508, all512 saved/optimized.
Its [QA review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block8_000145920_authoring_qa/visual_review.md)
records scalar/PowerShell/visual verification and no error literals; missing
raw body fields remain blank for512 rows. The three disjoint files total
**6628 rows =5376 saved train+1252 completed nonvideo eval**, excluding video3.
No old CSV was overwritten or extra sheet generated here.

All four previous main reports are archived before this update with suffix
before_completed_block8_20260910T113641705167Z; historical diagnostic and
checkpoint timing claims remain unchanged in those files and below.

## Historical recovery state — prior completed-block7 snapshot retained verbatim

The following older current-status labels are historical, superseded by the
completed-block8 accounting and distinct current7db video request above.

## Current recovery state — block7 complete; P10 curriculum request active

Saved accounting is fixed at **2026-09-10T10:35:47.134148Z**. The separate
block8 started-manifest status was read at10:38:03Z. No active pointer or
ongoing policy/prefix stream was polled for this report. Root owns the active
simulation; do not launch a concurrent Isaac process or repeat completed work.

### Verified completed block7 and current checkpoint

The naturalP01 run
20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e
is **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**. It planned4096 decisions but actually
collected, optimized and saved **2560 /20 PPO updates /400 optimizer steps**.
The final manifest records requested_policy_decisions=2560,
planned_requested_policy_decisions=4096, unconsumed_requested_policy_decisions=1536,
rounding_overrun=0. The unused1536 are not samples, updates, missing data or a
required automatic rerun. This was a curriculum allocation at a complete update
boundary, not a task-success declaration or optimizer gate.

Latest immutable checkpoint at this boundary:
[checkpoint_step_000145408.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000145408.pt).
Checkpoint SHA256:
c5d66b2aea266a138dea607b8880a8bfab3526a3b22e985cd289c6f348565d61.
Official immutable manifest SHA256:
142d5d5298de1e309890b16450891b0913684402e0f93087c34706d9dc94bbc8.
Parent computed both byte hashes and verified the official roundtrip; the report
read the immutable sidecar and completed snapshot without tensor loading.
Alias/resume-state hashes and later active pointers were not reverified here.

Saved lifetime: **145,408 decisions /1,101 PPO updates /22,020 optimizer steps**.
New since140544/1063/21260: **4864 /38 /760**, all sampled, optimized and saved.
There is no unfinished rollout at this completed boundary. Source HEAD remains
69aeaca777dcc8653e60f19da56ae1cd002e5271, runtime
21ad0d667c84e5c53e5e8aa37239421f8dcc4f717f5bcf6d1c817c2dd95293a4.
Saved adaptive LR1e-5; identity normalizer hashc230b0db is retained.
The publication index is read-only, not official loader metadata.
145408 is **latest_not_best**, not a success checkpoint and not yet fully evaluated.

Block7 saved phase samples:
**P01=24/P02=1265/P03=42/P04=19/P05=989/P06=178/P07=3/P08=3/P09=37**,
P10–P13=0. Cumulative seven-run saved phase samples:
**P01=33/P02=1690/P03=59/P04=42/P05=2139/P06=548/P07=24/P08=17/
P09=184/P10=1/P11=1/P12=71/P13=55**.
These replace the earlier partial saved128/unknown-phase account. Its historical
snapshot is preserved, not retroactively relabeled.

Block7 has10 true safety terminals: **FALL9/HARD_JOINT_LIMIT1**, no full-task
success. The hard-limit record identifies front_right_knee; it is an independent
safety stop, not one of the two task-failure categories. The last completed
episode ended P09/FALL at21.116667s. The final episode's **57 decisions at
P02/tick456/3.8s are nonterminal**, ending only when the verified update completed.
They are saved data, not another failure or success. Across seven finalized runs:
24 real task/safety terminals: eight BODY_COLLISION task failures and16 independent
safety aborts (FALL15/HARD_JOINT_LIMIT1), plus seven nonterminal collection tails.

RR recorded I9/Q6 events and30 current-valid decision ends during block7,
but **no new crossing or placement**. Maximum measured lift200.227mm occurs at
a FALL terminal with current_lift_valid=false and load_fraction_valid=false;
it is not stable carry, and unknown normalized load is not zero load.
The actual2560 actions were nonzero with20442 verified native-effect ticks,
20386 own-phase request-effect ticks, and no forbidden root/velocity/force/gravity
writes in retained evidence. These counters do not identify a unique fall cause.
See [completed block7 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T103547134148Z.json)
and [final training manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e/training_manifest.json).

The [bounded later-episode diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/natural_p01_block7_later_episodes.md)
confirms episode7's actual FR knee−60.002540312° despite target−58°.
For the five FALL records in episodes4–9, terminal372 float32 observations plus
locked obstacle planes imply base height below.015m; these are derived values,
not raw double-precision base measurements. Within that bounded evidence, no new
explicit execution/evaluation defect was found; neither lift differences nor
later phase visits establish learning or stability improvement.

### Latest completed full naturalP01 evaluation remains142848

Latest completed deterministic naturalP01/seed2001 evaluation still loaded142848:
**617 decisions /4929 physics ticks /41.075s**, zero training or optimizer credit.
It ended **P05 INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_TASK_DEADLINE**, first
unfinished FL front-edge crossing and controlled placement, no BODY/FALL.
It was not an external short-window cutoff or full traversal success.
145408's save/load roundtrip does not replace this missing latest-checkpoint
evaluation. Both older completed evaluations, failed-video initialization evidence,
and the [eval2 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/seed2001_142848_p05_diagnosis.md)
remain intact. Quality RMS describes incomplete trajectories only; no FSM-relative
stability superiority or successful video is claimed.

### Active block8: P10-origin512, request only

Run20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541
started at10:36:04.060860Z, requested **phase_suffix/P10/512 decisions**, seed1001/
N1/CUDA0, checkpoint interval1, teacher offset0 and frozen_fsm prefix, starting
from immutable145408. It uses the same69aeaca runtime and ordinary resume:
no ResumeMigration, NewMdpWarmStart or policy-distribution migration.
See [started manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541/run_manifest.started.json).

The parent reported teacher prefix without policy credit at dispatch. This report
has not reviewed new actual identity, credit start, samples, updates or saves.
Those counts remain **unknown**, not an asserted zero for all later time;512 is
only a request. Prefix physical events never enter policy credit. Earlier block7
09:50:14Z/PID36972 identity remains proof of that completed run, not proof of
this new P10 run's loaded objects. Root will provide its actual receipt separately.

After the active process ends, inspect the actual latest pointer and completed
update ledger before choosing recovery or remaining requests. Do not use this
fixed145408 index to overwrite a later checkpoint. Ordinary same-HEAD resume
from145408 requires no migration; if deliberately falling back to source142848
under69aeaca, its exact
[video-only plan](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_natural_reset_from_000142848_to_69aeaca.json)
is still required. That plan preserves weights/std/Adam/LR/normalizer/RNG/counters
and spent budgets, discards rollout storage, and is not a new MDP.
The six configs/control/nominal/reward/observation kernel remain unchanged.

Conditional non-video full evaluation reference after normal process exit and
same-HEAD verification; do not run it concurrently:

~~~powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 69aeaca777dcc8653e60f19da56ae1cd002e5271 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000145408.pt
~~~

### Publication and history

Latest consolidated CSV remains
[p09_evidence_timeline_000142848_eval2.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000142848_eval2.csv):
**3556×263 =2304 saved train+635+617 eval**, eight completed runs through142848.
It excludes all of now-completed block7, active block8 and the zero-action video.
The consolidated CSV and prior CSVs/QA are untouched; no full-task-success video
is published by this report.

An additional disjoint [block7 delta CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block7_000145408.csv)
contains2560×263, all2560 optimized/saved, teacher0 and10 terminal rows.
Its [QA review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block7_000145408_authoring_qa/visual_review.md)
records parent visual inspection and independent PowerShell readback. Together
the unchanged3556-row consolidated CSV and this delta cover6116 rows:
4864 saved training+1252 evaluation, without overlap or any active block8 rows.

Separate [block8 episode0 diagnostic](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block8_episode0_terminal.md),
fixed at11:03:20Z:456 sampled/384 optimized/72 not yet optimized; P12 task
incomplete via LOCAL_BOUNDED_RECOVERY_EXHAUSTED. RL I11/Q4, all4 Q revoked
on ground return, C/P0. Its145792 checkpoint is only a snapshot claim without
parent byte rehash; this supplement does not replace the main10:35:47 verified
145408 checkpoint or its accounting, and does not establish full-task success.

All four previous main reports were archived before updating under
[report_history](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history),
with suffix before_completed_block7_20260910T103547134148Z. Older timestamps,
requests and evidence remain historical, not current instructions.

## Historical recovery state — prior09:54:44 snapshot retained verbatim

The following sections preserve earlier states. The completed10:35:47 accounting
and new P10 request above supersede their current-status labels.

## Current recovery state — fixed09:54:44; natural P01 training running

This authoritative section is a **fixed 2026-09-10T09:54:44.201940Z snapshot**,
not a live pointer. Later updates are intentionally excluded. Root owns the
active process/session54301; do not start another Isaac process, stop it, or
repeat the4096 request from this document.

Production is69aeaca777dcc8653e60f19da56ae1cd002e5271, runtime
21ad0d667c84e5c53e5e8aa37239421f8dcc4f717f5bcf6d1c817c2dd95293a4.
Control/MDP semantics and all six configs remain the28609010 version.
Only semantic_video.py changed at this revision; exact SAME372/12 policy
compatibility is retained. See the
[applied reset-proof receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_natural_reset_applied_69aeaca.md).
Its CPU regression XML records316 passed/0 failed/0 errors/0 skipped in20.092s.
The first173-test receipt with one launcher-environment assertion failure is
preserved. CPU success is not a live repaired-video or physical-success result.

### Saved, sampled and active accounting

Latest checkpoint **at this fixed boundary**:
[checkpoint_step_000142976.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142976.pt).
SHA256: 29349ad04cdf557e6e45d6050a247807ab8c45d0825cedbf0860efaecd189563.
Official immutable sidecar SHA256:
01081c0a9f030f4854791f8c33187db3b4a44b72bf620329fd6b866ccde6e810.
Root computed both hashes and verified the official save/load roundtrip; this
report read the immutable metadata, not checkpoint tensors or active rawstreams.
Alias/resume-state byte hashes were not independently rechecked for this update.

Lifetime saved: **142,976 decisions /1,082 PPO updates /21,640 optimizer steps**.
New saved since140544/1063/21260: **2,432 /19 /380**. Saved source/runtime are
69aeaca/21ad0d66 above; adaptive LR1e-5 and normalizer hashc230b0db remain recorded.
It is latest_not_best, not a full-task-success checkpoint, and has not yet had
its own naturalP01 full evaluation.

The new naturalP01 training run
20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e
requested4096 decisions, seed1001/N1/CUDA0, checkpoint interval1, unchanged
128-decision PPO rollouts. At the specified snapshot it had **245 sampled /
128 optimized and saved /117 unfinished-rollout decisions**, one PPO update
and20 optimizer steps. Cumulative sampled is2549, distinct from2432 saved.
Sampled phase counts for this active run only:
**P01=6/P02=179/P03=3/P04=1/P05=56**, P06–P13=0.
The saved128 phase split was not separately read; these245 phase counts must
not be relabeled saved. The report retains the six completed runs' exact2304
saved phase counts separately, plus128 saved with phase allocation unreviewed.

Its first episode ended in a real P05 FALL after166 decisions/11.008333s.
The terminal atglobal143014 is beyond saved142976, so this is sampled terminal
evidence, not yet a saved terminal at the fixed boundary. It is not a completed
4096 request or a task success. The117 unfinished rollout contains the last38
decisions of that episode and79 decisions of the next episode; none receives
extra optimizer/checkpoint credit here. RR I was seen, but no qualified Q/C/P
in this245-row snapshot.
See [fixed active snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T095444201940Z.json).

Actual loading is now proven, not merely requested:
[live identity](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e/live_runtime_identity.json)
records09:50:14.775516Z/PID36972, training_after_actual_reset_and_checkpoint_load,
P01/tick0 and initialized true. Actual controller/nominal/config paths match
69aeaca/current project; logger control/import calls are0/0. This is not
source-FSM trajectory-equivalence or USD resolver proof.

### Latest completed full naturalP01 evaluation:142848, still P05 incomplete

The completed deterministic seed2001 evaluation loaded142848 under64c0324:
**617 decisions /4929 physics ticks /41.075s**, zero optimizer and training credit.
Phases: P01=2/P02=158/P03=4/P04=2/P05=451, P06–P13=0.
Task result: **INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_TASK_DEADLINE**.
The job's SUCCEEDED lifecycle only means it ran to completion.
There was no BODY_COLLISION/FALL, no external short-window cutoff and no
traversal success. First unfinished task remains **FL front-edge crossing and
controlled placement inP05**. Local age30.008333s exceeded the current30s limit;
current progress and allowance were0;158.925s remained in the global task budget.

FL produced fresh I1374/Q1378, then ground-return revoke2805, with C/P still0.
Nearest decision-end front distance was−21.437mm with+26.887mm top clearance;
maximum clearance+56.245mm occurred at a different tick. Actual base x showed
P05 net retreat .233751m; this is body position, not CoM. One local reward
interval gained transfer-evidence credit while geometry slightly regressed,
but this does not prove deliberate reward exploitation or a unique cause.
See [eval2 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/seed2001_142848_p05_diagnosis.md)
and [eval2 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T094646793551Z.json).
Roll/pitch RMS .0922318/.1283252rad describe this incomplete trajectory only;
higher lift, nearer edge or lower RMS does not establish FSM superiority.
The older141568/635-decision evaluation remains preserved. The separate09:30
video attempt is still a zero-action initialization diagnostic before policy
load, not this completed evaluation and not a physical task failure.

### Migration and recovery after the active process finishes

The actual initial142848→69aeaca launch used
[migration plan](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_natural_reset_from_000142848_to_69aeaca.json),
SHA42d4eea45cd9d5ee39246477fd09fa265065108e8565c2586f8073d5f01ce51e.
It preserves actor/critic/std, Adam and actual LR, normalizer, RNG, counters and
spent budgets; old rollout is discarded and fresh legal P01 data collected.
It is video instrumentation only, **not NewMdpWarmStart**. No six-config,
physics, nominal, reward, observation or policy-kernel change is authorized.

A real69aeaca checkpoint142976 now exists, so ordinary same-HEAD continuation
from it does not reuse the source142848 migration. If recovery must instead
use old142848, its exact plan above remains required. After normal process exit,
inspect the actual latest pointer/sidecar and completed updates before choosing
the remaining training request; do not repeat4096 or assume this fixed128 is
the final saved total. Later pointers may already exist.

A conditional, non-video full evaluation entry using this fixed69aeaca
checkpoint is below. It must not run concurrently with root's active process:

~~~powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 69aeaca777dcc8653e60f19da56ae1cd002e5271 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142976.pt
~~~

### Fixed CSV scope and remaining deliverables

Newest completed export:
[p09_evidence_timeline_000142848_eval2.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000142848_eval2.csv),
**3556 rows ×263 columns =2304 saved train +635+617 eval**, eight completed runs.
It excludes the active seventh training run, including saved142976, and the
zero-action video diagnostic. Teacher credit0. All1252 evaluation rows use
exact physical joins;2304 missing training body attitude/linear rows remain
blank, not replaced by command or CoM. The parent verified every scalar and
reviewed the QA image; see
[visual review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000142848_eval2_authoring_qa/visual_review.md).
Earlier CSVs/QA and report snapshots are preserved.
No fullP01 success, successful ≤200s PPO video, paired FSM/PPO video or
full-task stability superiority is claimed.

## Historical recovery state — block6/eval1 snapshot retained verbatim

The following sections describe earlier timestamps and requests. They are
preserved evidence, superseded by the fixed09:54:44 accounting above.

## Current recovery state — block6 saved; video C initialization diagnostic

Fixed evidence snapshot: **2026-09-10T09:32:56.5152480Z**. The latest saved
checkpoint uses source HEAD64c03243ac05e43e7a5843eaee252eef1136c925 and runtime
hash99ea40879c1ba6e2618bfba901d2be77edafacb039114f641725f3e325249dee.
Its control/MDP semantics remain the28609010 revision;64c0324 added video-only
runtime changes, not a new MDP or a new372-dimensional policy kernel.

Latest immutable checkpoint:
[checkpoint_step_000142848.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142848.pt).
SHA256: 7fb0a5a1607125d477c7e49c05422a40e3cb33d72dcad83c1fb1aeaf5e59eaa1.
Official sidecar SHA256: fb89869010f6fd100e4d0c37f3e5c2ba43e79df4312ec805f92a5b86bca4344e.
Saved lifetime counters: **142,848 decisions /1,081 PPO updates /21,620 optimizer steps**.
Relative to the preserved140544/1063/21260 baseline: **2,304 /18 /360** newly
sampled, optimized and saved; teacher prefixes and the video diagnostic add no
credit. There is no unsaved training tail in this six-block accounting.
The final saved adaptive LR is1e-5. Official save/load roundtrip is true;
immutable/alias checkpoint hashes, official/resume sidecar hashes and the
unchanged pointer were verified without Torch or tensor loading.
[checkpoint_last_manifest.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last_manifest.json)
is a read-only publication index, never replacement loader metadata.
This is **latest_not_best**, not a full-traversal-success checkpoint.

### Completed P04 block6 and bounded physical evidence

Run20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd
completed its1024-decision request: **1024 decisions /8 PPO updates /160 optimizer
steps**, using eight unchanged128-decision rollouts, source141824 and no migration.
Actual phase samples: **P04=5/P05=761/P06=220/P07=2/P08=6/P09=30**; others0.
There were three real safety terminals: FALL atP06/t26.166667s, FALL atP05/
t23.241667s, and BODY_COLLISION atP09/t40.075s. The fourth episode's316-decision
tail ended atP05/t35.2s because the collection budget ended: **no terminal,
no FL placement, neither success nor another failure**. Its retained
CONTACT_BEARING_UNVERIFIED and RR load_fraction_valid=false mean load is unknown,
not zero.

The actual P04 identity was recorded **08:57:20.957634Z, PID23900**, after real
reset and checkpoint141824 loading, atP04/tick1696/READY. Active semantic
controller, nominal provider, source MotionExecutor, mapper, sensing and config
paths match this project/runtime;14 module paths were checked on disk. This is
real process evidence, not a started-manifest assumption or offline reconstruction.
It does not prove source-FSM trajectory equivalence or USD dependency resolution.
See [actual identity](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd/live_runtime_identity.json).

The [first-two-episode diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p04_first_two_completed_episodes.md)
covers318 decisions, including one fresh FL placement and two real FALL terminals.
The retained evidence lacks exact base_z/projected_gravity_z, so it cannot
independently choose the violated FALL subpredicate. Episodes span existing PPO
updates and are not matched frozen-actor trials; the existing transient adaptive
LR1.5e-5 was not a new hyperparameter setting.
The [third-episode diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p04_third_episode_to_p09.md)
covers390 decisions: fresh FL Q2618/C2828/P2864, then P07/P08 preparation and
P09 BODY_COLLISION, but no qualified RR lift/crossing/placement. At P08→P09,
CoM moved toward FL while FL was AIR; this is not FL bearing. RR's earlier
incidental I inP06 was not qualified Q. The first unfinished RR task was usable
lift/carry, then crossing and placement—not holding an already-qualified lift
for a fixed duration. Exact body-contact manifold/gap/force is not retained and
is not inferred. Neither bounded diagnosis proves a unique causal action chain.
See [completed block6 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T092950976322Z.json).

Cumulative saved phase samples are:
P01=9/P02=425/P03=17/P04=23/P05=1150/P06=370/P07=21/P08=14/
P09=147/P10=1/P11=1/P12=71/P13=55.
The six blocks have14 safety terminals (FALL6/BODY_COLLISION8) and six
nonterminal collection tails. Prefix RR events remain excluded.

### Evaluation boundary and zero-action video diagnostic

The latest **completed naturalP01 evaluation still loaded141568**, seed2001:
635 decisions/42.283333s, P05 LOCAL_TASK_DEADLINE, first unfinished FL
crossing/placement; no BODY_COLLISION/FALL.142848 has a save/load roundtrip,
but no completed naturalP01 evaluation yet.

The new C/seed4001/P01/no-teacher attempt
20260910T0930063395586Z_g64c03243ac05_4bb994da17ed47bea5c9e0b04c3a2a41
ended at09:30:41.953417Z with lifecycle DIAGNOSTIC_FAILURE:
SemanticVideoError: video must own the first natural reset of a fresh process.
The final manifest records reset_count1/options{}, training_phase_snapshot="P01",
checkpoint_load_provenance=null, **0 issued actions /0 environment steps /
0 optimizer updates**, physical_episode=null and physical_task_success=null.
This is an initialization-guard/P01-marker mismatch **before policy loading**,
not a PPO task failure or another completed naturalP01 evaluation.
See [final diagnostic manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T0930063395586Z_g64c03243ac05_4bb994da17ed47bea5c9e0b04c3a2a41/run_manifest.json).
No new checkpoint or successful video resulted. Root owns a minimal video-only
guard repair after the process exited; this snapshot has not verified a repaired
HEAD, test result or rerun. Do not blindly repeat the failed64c0324 video command.

### Recovery entry and remaining work

Do not rerun the already-completed P04 request1024. Before any new process, read
the actual pointer and current committed runtime, and coordinate with root.
On **unchanged clean64c0324 only**, after confirming no active simulation, the
existing non-video naturalP01 evaluation entry remains:

~~~powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142848.pt
~~~

This is a conditional recovery reference, not authorization to launch a second
process or a substitute for root's video repair. If the actual committed
runtime changes, use its explicit reviewed video/instrumentation-only SAME372
migration, preserving actor/critic/std, Adam moments and actual LR, normalizer,
RNG, lifetime counters and spent budgets. Such a runtime-only migration is
**not NewMdpWarmStart**; do not reset compatible states or bypass the runtime
identity check. No future HEAD or migration path is invented here.
The remaining workflow includes naturalP01 training/evaluation and genuinely
paired FSM/PPO video evidence; no suffix success substitutes for a full task.

The newest fixed CSV remains
[p09_evidence_timeline_000141696.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696.csv),
**1787 rows =1152 saved training+635 earlier evaluation**, cutoff141696.
It excludes completed blocks5/6 and the zero-action video diagnostic. Its QA
and the original1659-row CSV/QA are preserved. The
[applied-video report](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_applied_64c0324.md)
and231-pass CPU receipt remain limited to their stated test scope; the real
initialization diagnostic is additional evidence, not hidden by those tests.
No full-task stability superiority, successful PPO video or paired comparison
video is claimed.

## Historical recovery state — prior block5 snapshot retained verbatim

The following prior sections describe their own older timestamps and plans;
they are preserved evidence, not current commands or current counters.

## Current recovery state — block5 saved; P04 request1024 active

Publication verified **2026-09-10T08:57:47.0058826Z**. Production and the latest
saved checkpoint both use `64c03243ac05e43e7a5843eaee252eef1136c925`; control/MDP
semantics remain28609010. No new migration, normalizer reset, Adam reset, policy
kernel change or new-MDP warm start is required for the current same-HEAD resume.

Latest immutable checkpoint:
`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141824.pt`.
SHA256: `73fb969d5af9b233172e161c7b2ef0aa8a8011aac04115cc05aca054606ba000`.
Official sidecar SHA256: `5cd29a7107453cbdc20d4b9790ed5607be4e88d84e13bfdbf62c1d249628e13a`.
Saved lifetime counters: **141,824 / 1,073 PPO / 21,460 optimizer**.
New saved totals since140544/1063/21260: **1,280 / 10 / 200**. Effective LR remains
`1e-5`. The official save/load roundtrip passed; immutable and alias checkpoint
hashes, official/resume-state hashes and the unchanged pointer were verified.
`checkpoint_last_manifest.json` is still only a read-only publication index.

### Completed P10 block and first unfinished task

Block5 actually collected128 policy decisions and completed1 PPO update/20
optimizer steps. Credit by source phase: **P10=1, P11=1, P12=71, P13=55**.
There were no task terminals. Its128-decision tail ended at P13/tick8608/
71.733333s; on-policy time was8.533333s after the teacher prefix.

This is now real loading/credit evidence, not merely a started manifest:
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0828133987091Z_g64c03243ac05_8f4efe6ab2af4675b6e3f42469a8acfa/live_runtime_identity.json`
records **08:47:15.728149Z, PID22760, P10/tick7584, READY**, after actual reset and
checkpoint141696 loading. Runtime64c0324/hash99ea4087 matches; actual classes,
configuration paths and source project agree. Policy credit began at63.2s while
the requested P10 was still active. Teacher RR Q6912/C7109/P7579 were already
present and receive **no new policy event credit**.

The policy suffix produced fresh **RL I7746/Q7750/C8089/P8164**. At the end,
all placement histories and the final region were valid, but **final_controlled
was false** and the existing final observation had not started. All55 P13 sampled
ends passed region/support/evidence; wheel speed failed the existing limit in
55/55, body speed in46/55 and angular speed in28/55. Final body speed .115041m/s
and FL/RL/RR wheel speeds .892218/.383814/.504495rad/s remained too high; angular
speed .284159rad/s passed. RL being currently AIR is not an extra failure gate:
FL/FR/RR provide actual TOP support and the region test already passes.

P13 was sampled for only3.666667s. Its source nominal wheels were still +.3rad/s
throughout these55 rows, within the source0–16.733333s rolling segment and before
the finite17.8s nominal home/zero tail. This is an observed nominal/finish-goal
tension, not a dispatch-failure verdict or proof that a future stop will succeed.
Residuals remained continuous at transitions; new RL events occurred before the
single update, so they do not prove update1073 improved the policy.
See [completed P10/P13 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block128_rl_and_p13.md)
and [block5 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T085154904868Z.json).
Neither this suffix nor its nonterminal tail is complete naturalP01 success.

### Active P04 request and current recovery command

The new started manifest is
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd`.
It requests **P04-origin1024 decisions**, seed1001/N1/CUDA0, offset0,
`frozen_fsm` prefix and checkpoint interval1. The request was expanded from the
earlier512 plan to allow **eight unchanged128-decision rollouts** and a real
P05 task outcome; this does not change PPO hyperparameters or MDP semantics.
The started timestamp is08:52:46.550594Z. For this new active P04 run, this report
has not inspected actual loading/credit-start or policy samples; requested1024
is not sampled or saved credit. Root owns session3185. Do not launch another
Isaac, touch its process, or interrupt its unfinished rollout.

After the active process exits normally, inspect the actual latest pointer and
completed-update accounting first. If it still points to141824 and this request
has not saved any update, the same-HEAD restart entry is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P04 -Decisions 1024 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

No `-ResumeMigration`, `-NewMdpWarmStart`, or policy-distribution migration flag.
If some P04 updates have already been saved, use the actual latest checkpoint
and determine the remaining authorized request; do not repeat1024 blindly.
The follow-up plan remains naturalP01 training and full deterministic evaluation
from the actual latest saved checkpoint, not an exclusive suffix curriculum.

**Latest saved141824 is not the latest completed naturalP01 evaluation.** The
latest completed full-episode evaluation still loaded **141568**, seed2001,
and ended after635 decisions/42.283333s at P05 `LOCAL_TASK_DEADLINE`, with FL
crossing/placement unfinished and no BODY_COLLISION/FALL.141824 has a verified
save/load roundtrip but has not yet completed its own naturalP01 evaluation.
The unchanged same-HEAD evaluation command after training/process exit is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

The newest CSV remains
[p09_evidence_timeline_000141696.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696.csv):
**1787 rows =1152 saved training+635 earlier evaluation**, through141696 only.
It excludes completed block5 and the active P04 run until another final export.
Its QA and the older1659-row CSV/QA remain unchanged. Applied video64c0324 and
the231-pass CPU receipt remain valid within their stated test scope; no current
success video, fullP01 checkpoint success or FSM stability improvement is claimed.

## Earlier recovery snapshots — retained, superseded for current accounting

The following sections retain their original timestamps, old planned512 request,
then-pending P10 load, saved counts and commands. They are historical records,
not the latest operating instruction. The block5/P04 section above supersedes
them; old partial samples and teacher events are never counted again.

## Current recovery state — video-only revision 64c0324

Status recorded **2026-09-10T08:30:24.047381Z**. Production HEAD is
`64c03243ac05e43e7a5843eaee252eef1136c925`; the control/MDP semantics remain
`28609010db4e57c5b34304a4ae2563c69f9d00b9`. The latest saved checkpoint is from
the intervening `06716a88` runtime. These three boundaries must not be conflated.

The fourth block, P07-origin 128 decisions, completed one real PPO update and
20 optimizer steps, with an actual save/load roundtrip. Current immutable source:
`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141696.pt`.
Checkpoint SHA256: `70ae4fedc9210b7e48d4ec33a825aa9b5e47f6acf7d80e064914f03920ff4395`.
Official sidecar SHA256: `852500f344b85b4ee398d79e49abf93dff34f38261f2489079473519b6a19d06`.
Saved lifetime counts are **141,696 / 1,072 PPO / 21,440 optimizer**; relative to
the unchanged 140,544 / 1,063 / 21,260 baseline, **1,152 / 9 / 180** are saved.
The actual pointer, immutable/alias checkpoint hashes and official/resume-state
hashes were checked at the timestamp above. The publication index is only an
index; `resume_state.json` and the immutable official sidecar remain authoritative.

Block4 phase credit is P07=7, P08=7, P09=114. Its six completed episodes remain
BODY_COLLISION failures. Episode6 has a 57-decision nonterminal tail ending
P09/tick6368/53.066667s, with RR not placed. The first lift attempt was revoked
on ground return at6055; the new edge attempt qualified at6249, continued through
actual contact and AIR, and crossed at6328. The final RR position had retreated
outside the top-placement region. Crossing history is not current placement or
full traversal. All these samples preceded the block's one update; their events
do not establish that the updated policy has improved.
See [block4 evidence](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p07_edge_crossing_block128.md)
and [completed snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T082308229118Z.json).

### Active P10 request and safe continuation

The started manifest records a new real P10-origin request, **128 decisions,
seed1001, N1, CUDA0**, teacher offset0 and `frozen_fsm` prefix:
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0828133987091Z_g64c03243ac05_8f4efe6ab2af4675b6e3f42469a8acfa`.
Its started timestamp is 2026-09-10T08:28:14.072875Z. This report has verified
the startup configuration only, **not actual policy loading, live identity or
policy credit-start**. No requested or prefix decisions are added to saved counts.
Do not start another Isaac or interrupt the root-owned active process.

The exact video-only migration is
`outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_from_000141696_to_64c0324.json`.
It binds source141696/06716a8 to64c0324 and five reviewed production paths only.
All six configs, physical control, nominal, reward, preprocessing, N1 topology
and HISTORY372 kernel are unchanged. It preserves weights, learned std, Adam
moments, actual LR `1e-5`, normalizers, RNG, counters and spent budgets; old
rollout storage is discarded and reset/prefix physics is recollected legally.
**Do not use `-NewMdpWarmStart` or the older141568 instrumentation plan.**

Only after normal process exit and checking the actual latest pointer, if the
source still is141696, the approved restart command for this request is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P10 -Decisions 128 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141696.pt -ResumeMigration outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_from_000141696_to_64c0324.json
```

After a new64c0324 checkpoint is actually saved, ordinary same-HEAD continuation
uses its latest pointer with no migration or warm-start flag. The root's next
sampling plan is **P04 continuous512, then naturalP01 training and evaluation**;
this is a plan, not completed credit or an exclusive P10 curriculum. The planned
P04 command, only once a same-HEAD checkpoint exists and the active run has exited:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P04 -Decisions 512 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

The completed naturalP01 evaluation still refers to141568, not141696:
635 decisions/42.283333s, P05 `INCOMPLETE_CONTROLLER_BLOCKED` from
`LOCAL_TASK_DEADLINE`, no BODY_COLLISION/FALL, FL crossing/placement unfinished.
The new saved policy requires its own naturalP01 evaluation; no success or FSM
stability superiority has been established. Once a latest same-HEAD checkpoint
exists, the unchanged deterministic evaluation entry is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

### Current report artifacts

The [new fixed CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696.csv)
contains1787 rows/263 columns:1152 saved training decisions from four blocks plus
the635-decision141568 evaluation. Active P10 is excluded. Parent scalar and
[visual QA](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696_authoring_qa/visual_review.md)
passed; missing measurements remain blank with valid flags. The prior1659-row
CSV and QA are retained unchanged.

[Applied video record](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_applied_64c0324.md):
the video-only patch is now applied, with **231 passed,0 failed,0 errors,0 skipped**
in `video_current_experiment_regression_v2.xml` (17.411s). This includes a real
CPU HISTORY372 update/save/reload and preserved Adam/RNG/normalizer/kernel checks.
The first failed v1 receipt remains intact. CPU success is not live video or
full-task success. No PPO success or comparison video is claimed.

## Preserved prior recovery snapshots — not current commands

Everything below is retained history. Its earlier headings saying "current",
12/50/61 sample fragments, old saved counters, and deferred/unapplied video
status describe their own timestamps. The64c0324 section above and actual latest
pointer supersede them; never add historical samples twice.

## Current recovery state — logging-only revision 06716a8

Production HEAD: `06716a88bcc9ee2fff615bcf2681dd59d97142e5`.
Project: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1`.

The current run is
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0647306226372Z_g06716a88bcc9_c6cfcadd350b418a9969f7dfc35406a5`.
Its actual started manifest requests **128 P07-origin policy decisions**, N1,
seed1001, CUDA0, teacher offset0, `frozen_fsm` prefix, checkpoint interval1.
Do not start a second Isaac process. The teacher prefix and any pending rollout
are not saved policy credit; use a subsequently published checkpoint pointer,
not the requested128 count, to determine whether this block has completed.

### In-progress snapshot — 2026-09-10T07:56:44.591846Z

One `summarize_live.py --kind train --write` inspection recorded **61 sampled
policy decisions**, phase counts **P07=5, P08=5, P09=51**, and470 on-policy
physics ticks. The five completed episodes contain12/12/12/14/11 policy
decisions; all terminated in **P09 BODY_COLLISION**, with the physical evaluator
reporting VALID/VERIFIED and BODY_CONTACT. The first uncompleted phase is P09;
RR has no crossing/placement. Recorded RR initial-lift evidence appears in2
decision samples, but lift-established/current-valid/crossing/placement counts
are all0. Diagnostic AIR duration is not a qualification or success gate.

This run has **0 new PPO updates, 0 optimizer steps and 0 saved decisions**.
The latest logged global decision141629 is a sampled counter, not a checkpoint.
The active process remains owned by the main task and continues collecting the
requested128-decision rollout; this snapshot neither stops nor completes it.
Teacher prefixes are excluded. Individual files were read at separate byte
boundaries, so this is a timestamped lower bound rather than an atomic live state.
No repeated polling, Torch load, checkpoint tensor rehash or Isaac launch was
performed by this report update.

Evidence: [07:56:44 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T075644591846Z.json).
The original12-decision first-episode fragment remains in the report, and the
[07:41:59 snapshot with50 sampled decisions](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T074159118501Z.json)
is retained unchanged. These fragments are not summed or credited a second time.

The completed [CSV evidence snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline.csv)
contains1659 rows/263 columns (1024 completed training+635 deterministic evaluation),
with [visual QA](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_timeline_authoring_qa/visual_review.md).
It excludes all current06716a8 P07 samples. The
[deferred-video parent review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/deferred_video/parent_review_06716a88.md)
records passed Python/PowerShell syntax checks, but **pytest and checkpoint
roundtrip tests have not run and the production video patch is not applied**.
The reviewed draft must not be imported into this active process and is not an
optimizer gate or evidence of successful traversal/video publication.

The latest verified checkpoint before this run is
`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141568.pt`.
SHA256: `1fa8de42754f04c5931b28f775f286c03a7791afeaf789fed0b51d4479607055`.
Sidecar SHA256: `8a24472f7f9dd7589522ca7cd08328342811702e6ac2efcfba00ce1ec169bf4b`.
Actual saved lifetime counts: **141,568 / 1,071 PPO / 21,420 optimizer**.
The experiment has added **1,024 saved policy decisions / 8 PPO updates /
160 optimizer steps** relative to140544. Source effective Adam LR is `1e-5`.
The save/load roundtrip passed. Latest is not best or complete-traversal success.

The new revision changes actual-object/path logging and exact SAME372 resume
compatibility only. It does **not** change the control/MDP semantics, six selected
configurations, actor kernel, nominal, reward, observation preprocessing, or
residual ranges from28609010. This is **not another new-MDP warm start**:
actor, critic, learned std, Adam moments and effective LR, identity normalizers,
RNG, lifetime counters and spent budgets are preserved. Old rollout storage is
discarded, and reset/prefix physics is recollected legally. The explicit plan is
`outputs/ppo_fsm_reference_p09_stable_v2/migrations/runtime_identity_from_000141568_to_06716a8.json`.
Its scope is a reviewed logging-only claim, not a claim that two physical
trajectories are identical.

The first saved-policy deterministic naturalP01 evaluation is **complete**:
`runs/ppo_fsm_reference_p09_stable_v2/validation/20260910T0624345531274Z_g28609010db4e_2d880bd470694bcea9a6028c0902774e`.
It loaded141568, seed2001, no teacher prefix, and made zero optimizer updates.
It ran **635 decisions / 5,074 physics ticks / 42.283333s**, completing P01–P04
but not FL crossing/placement in P05. Recorded terminal:
`INCOMPLETE_CONTROLLER_BLOCKED`, with semantic source `LOCAL_TASK_DEADLINE`.
P05 elapsed30.016667s against effective30s, with progress allowance0 and global
time still157.716667s. This was a finite task terminal, **not** an external
diagnostic cutoff, BODY_COLLISION, or FALL. FL was back on ground, crossing and
placement false. Full traversal remains incomplete. Evaluation samples are not
training credit. Window quality and complete evidence are in
`training_and_evaluation_manifest.json`; no full-task superiority over FSM or
success video is claimed.

### Recovery commands

Run commands only after the active Isaac run has exited or saved and stopped at
a complete update boundary. Check `checkpoints/checkpoint_last_pointer.json`
first. If it still points to141568 under28609010, the current approved startup is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 06716a88bcc9ee2fff615bcf2681dd59d97142e5 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P07 -Decisions 128 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141568.pt -ResumeMigration outputs/ppo_fsm_reference_p09_stable_v2/migrations/runtime_identity_from_000141568_to_06716a8.json
```

Do not pass `-NewMdpWarmStart`; that would reset Adam again. The migration plan is
bound to exactly141568 and must not be reused for a later checkpoint. Once the
current run has published a verified06716a8 checkpoint, ordinary continuation
uses its actual latest pointer with **no migration flag**:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 06716a88bcc9ee2fff615bcf2681dd59d97142e5 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P07 -Decisions 128 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

That128 block is an example continuation, not authorization to assume a completed
block or an exclusive long-term P07 curriculum. Retain naturalP01 and later-stage
sampling according to the actual unfinished tasks. After a new same-HEAD saved
checkpoint exists, deterministic naturalP01 evaluation uses:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 06716a88bcc9ee2fff615bcf2681dd59d97142e5 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

`live_runtime_identity.json` is emitted only after the actual reset/prefix and
checkpoint restore. The current run's real file appeared at
**2026-09-10T07:00:48.739846Z**, PID28624, P07/tick5912, prefix `READY`:
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0647306226372Z_g06716a88bcc9_c6cfcadd350b418a9969f7dfc35406a5/live_runtime_identity.json`.
Its06716a8 revision and runtime-content digest match the started manifest. Actual
Python is the expected `env_isaaclab` executable, cwd is this PPO project, and all
14 recorded `wlr50_clean` modules resolve within its `src`. Active objects are
`SemanticControllerAdapter`, `NominalMotionProvider`, source `MotionExecutor`,
`RobotAdapter`, `ServoTargetMapper`, and `SemanticSensorReader` with exact-pair
contacts and semantic collider geometry. Actual backend task/profile paths point
to the new namespace; loaded FSM/contract paths point to this project's frozen
`configs/fsm_states.yaml` and `configs/recording_motion_contract.json`.
The active robot's stored asset config is
`C:\robotics_sim\wlr_robot\usd\wlr_robot_drive_test.usd`.

Recovery is explicitly absent on semanticN: a loaded `RecoveryPlanner` class is
not an active selected recovery object, just as a loaded `SensorFsmController`
class does not make the active controller originalA. The logger performed zero
control calls and zero imports, with no historical-run backfill. This actual
object/path evidence does not certify USD-resolved dependencies, fullA controller
equivalence, policy updates, task success, or matching physical trajectories.

## Historical notes — preserved snapshots, not current commands

The text below is retained for chronology. Its old HEADs, provisional run status
and commands are superseded by the current section and actual latest pointer.
In particular, references below to pending evaluation or unapplied logging
drafts describe earlier snapshots, not the present06716a8 runtime.

Historical production revision: `28609010db4e57c5b34304a4ae2563c69f9d00b9`.
Project: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1`.

The initial source is the immutable all-stage checkpoint at
`outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000140544.pt`.
It contains **140,544 saved policy decisions / 1,063 PPO updates / 21,260 optimizer steps**.
SHA256: `c5ab9153671cff596fb7dc1059b8064adf7322c3abd42bc442c580be13ead49c`.

The previous run logged an additional update at 140,672 / 1,064 but did not save it.
That update and its remaining rollout are not recovered or counted as this experiment's learning.
The previous run remains unfinalized, with no process present at this revision's startup.
Its exact interruption cause is not established.

The new-MDP boundary preserves actor, critic, learned distribution, normalizers,
training RNG, lifetime counters and spent stage budgets. Adam moments are cleared;
the verified source effective learning rate is retained (`1e-5`, configured initial
default remains `3e-5`). Observation shape remains 372; the existing RR lift bit
now expresses current validity. Source-derived nominal and event semantics changed.
Every new rollout is collected under this committed revision.

## Verified continuation checkpoint and current run

**First fixed-version check completed:** immutable
`checkpoints/history/checkpoint_step_000141568.pt`, SHA256
`1fa8de42754f04c5931b28f775f286c03a7791afeaf789fed0b51d4479607055`.
Save/load round trip passed. Lifetime counts are **141,568 policy decisions /
1,071 PPO updates / 21,420 optimizer steps**; this revision has added
**1,024 / 8 / 160**, all saved, without crediting teacher prefixes or old unsaved
data. The final 384-decision P01 block
`20260910T0616115212299Z_g28609010db4e_f2de0b3725a5425e9cbd9bff946c1dd9`
ended at a complete update boundary, with its one episode nonterminal at 25.6 s,
P05; FL placement was still unfinished. This collection boundary is neither
task success nor a task-timeout label.

Aggregated policy phase credit across the three completed blocks:
P01=9, P02=425, P03=17, P04=18, P05=389, P06=150, P07=12, P08=1,
P09=3, P10=P11=P12=P13=0. Later-stage opportunities remain required.
The seed-2001 deterministic natural-P01 full evaluation command below has now
been launched from this checkpoint with no teacher prefix. Its real final result
is not yet known at this update. It must run to a real terminal or the 200 s task
limit; do not turn a diagnostic window cutoff into a physical failure.
The 1,024-step first check is not the task's total budget or a stopping condition.
After evaluation, continue predecessor and later-stage training from the actual
latest checkpoint, retaining natural P01 learning and validation.

**Latest completed boundary (06:16 UTC):** the P06 block stopped normally at a
verified update boundary, with **128** actual credited decisions, **1** PPO update,
**20** optimizer steps; 896 of its originally requested decisions were not used.
The latest checkpoint is `checkpoints/history/checkpoint_step_000141184.pt`, SHA256
`33083ddb33dc5f7e1cb1092bc25787d9a18c5fd67ef5b49043a6337d8175b7b0`.
Round-trip verification passed; lifetime **141,184 / 1,068 / 21,360**.
New-version saved learning totals are **640 / 5 / 100**.
P06-block phase credit is P06=112, P07=12, P08=1, P09=3, all others=0.
It had one BODY_COLLISION (P09) and two FALL terminals (P07, P06), no success.
The last three samples were a nonterminal P06 tail, not another completed episode.

A same-version, ordinary-checkpoint-resume natural P01 training block of 384
decisions was launched next. No new-MDP flag, weight reset, or production change
is used. If it completes, the first fixed-version check will reach 1,024 saved
decisions; use actual manifests rather than assuming completion. Then immediately
perform the natural P01 deterministic reload evaluation below, followed by further
training according to actual unfinished tasks. No resource boundary is claimed.

Snapshot: 2026-09-10 05:54 UTC (later pointer/manifest takes precedence).
The completed natural-P01 training block
`20260910T0530598363441Z_g28609010db4e_4c4cee3253cf41af9c521d5cb576ccbd`
added **512 saved decisions / 4 PPO updates / 80 optimizer steps**.
Its immutable checkpoint is
`checkpoints/history/checkpoint_step_000141056.pt`, SHA256
`b8079ebad977391395b762cce38fd6d187d8a11206960d61a5934eef0e06f7cc`.
The saved manifest reports round-trip verification and lifetime counters
**141,056 / 1,067 / 21,340**; adaptive optimizer LR is `1.5e-5` at this checkpoint.
The completed episodes fell at P06 (17.9583 s) and P02 (5.45 s); a final P05
episode was unfinished at the collection boundary. None is full success.

The next run
`20260910T0542011958077Z_g28609010db4e_b88e6f85c2e1480ea33d2fbcf28d6139`
is collecting 1,024 P06-predecessor decisions from that verified checkpoint.
Its first 82 credited samples reached P09, recorded RR initial and established
lift (five current-valid decision-end samples), then terminated with verified
BODY_COLLISION at 35.3333 s, before RR crossing or placement. These 82 samples
were persisted before the next teacher reset but were not yet a complete update
at this snapshot; do not count them as saved learning. Teacher-prefix steps are
not policy samples. No deterministic natural-P01 reload evaluation has completed
under this revision yet. The planned evaluation below must still run.

The live process must finish or stop safely before any production revision.
The deferred runtime-identity drafts under `deferred_runtime_identity/` are
not applied, not tested, and must not be added to PYTHONPATH.

At 06:02 UTC the P06 run had 102 logged decisions, no complete new update,
and two real terminals: P09 BODY_COLLISION after 82 decisions, then P07 FALL
after 20. A pinned `stop_after_update.request.json` was added to this exact run
to save and stop at its next complete update (not immediately, no kill).
The 1,024-decision request will therefore not all be consumed by this run.
After its verified completion, use the actual latest pointer and collect a
natural P01 block sufficient to bring this revision to at least 1,024 new saved
decisions, then run the deterministic P01 evaluation below. Continue learning
with P06/P07 predecessor and later-stage opportunities afterward. This is a
curriculum allocation decision, not a task stop or a fabricated resource limit.

Before starting, verify no project Isaac process is already running and check
`checkpoints/checkpoint_last_pointer.json` in this directory. When it exists,
its immutable checkpoint and manifest, not this initial source number, are authoritative.
Do not run a second Isaac process or hot-edit production during collection.

Initial migration command (only before a new-namespace checkpoint exists):

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 28609010db4e57c5b34304a4ae2563c69f9d00b9 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage full_episode -Decisions 512 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -NewMdpWarmStart -Checkpoint outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000140544.pt
```

After the first verified update, continue from the new namespace (no migration flag):

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 28609010db4e57c5b34304a4ae2563c69f9d00b9 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P06 -Decisions 1024 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

Natural P01 deterministic evaluation, with normalizer/optimizer updates disabled:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 28609010db4e57c5b34304a4ae2563c69f9d00b9 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

Teacher-prefix physics and zero-residual checks are excluded from on-policy counts.
Latest is not best, and local progress is not full-task success. No full-success
checkpoint or success video is claimed by this recovery document.
