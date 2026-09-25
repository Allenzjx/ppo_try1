# CP225280 前段保护与后腿续训：CP227328 已重载评估并交付视频

状态：2026-09-24，CP227328自然P01确定性评估已封存并完成三视频QA：92.433333s，FR/FL真实放置、43.6s FL→P06保留，RR有效抬升/越沿但捕获未完成；RL尚未有效卸载。最新checkpoint227328/1729/34580；保护分支新增2048/4PPO/80Adam。当前完整精简交付见 `CP227328_URGENT_STEER_DELIVERY.md`，视频在 `video_review/CP227328_deterministic_front_preserved_v1_review/`。下面的评估进行中描述与CP226304/CP225792账本均为历史快照；没有完整越障成功声明，也没有后台运行。
此报告对应独立分支 `cp225280_front_preserved_v1`，不替换旧分支 CP231680 的记录。

## 新封存训练：CP227328（全程物理评估进行中）

P10课程实际1024学生决策、2PPO、40Adam；五条真实nominal前缀共3835决策/30680 ticks，全部0 PPO credit。学生轨迹59/111/269/452/133；前四分别机身碰障、FL knee实测硬限、FL knee实测硬限、局部任务未完成；最后133是正确bootstrap的非terminal预算尾。没有用phase超时伪装数据截断。实际阶段P10=5、P11=5、P12=1014。

非互斥物理窗口：RR可达AIR准备179、真实TOP并前侧准备126、固定FR方向投影与RL载荷份额下降代理69（非已证明受控右侧转移）。学生新RL资格2事件/3决策端点，无继承资格、无RL越沿或放置。第三回合RR掉地后重新有效AIR并合法TOP加载，单列为支撑恢复；原首次cross/placed仍是教师事件，不冒称学生新越沿。明细：`892385_P10_CP226304_course_coverage.md/json`。

保护分支累计2048真实学生决策/4PPO/80Adam；前段Gaussian KL回放2560次、80个原Adam小批次，新增PPO样本/独立AUX step均0。最后heldout mean/max KL=.011718149/.033050478，不能替代物理前段保持验证。旧分支在紧急Steer后的正常收尾另512/1PPO/20Adam，保存为CP231680；它不属于新学生谱系。

最新checkpoint为本分支`checkpoints/history/checkpoint_step_000227328.pt`，SHA `5b871af2df8d6b9c0180f7a51863b18b2153148d8dc2a94876f21f5bd73791b4`，sidecarSHA `0200daa9a22652e1f1c64e5c5ff82a0c977f0418c508158455b3a5ce85bac308`；227328/1729/34580，save/load=true，现有自适应优化器的实际LR=3.375e-5。新DET从自然P01运行中，不把下面CP226304视频当成CP227328结果。

## 最新正式视频与物理结果：CP226304

同一保存重载学生、自然P01、单条完整回合，无教师前缀或模型拼接。FR placed2793=23.275s；FL qualified2805、cross4319=35.991667s、placed5256=43.8s并进入P06。FRplaced后约6秒FL gap58.739mm、CoMz164.394mm/pitch−.401°，与接受的CP225280对应58.108mm/164.020mm/−.343°接近；未重现CP230144的−6.808mm/134.829mm/+7.114°。没有固定高度或自动FR设角新规则。

原FL辅助仍ON：4320开始DESCEND，5256进入HOLD，hip实际辅助修正−26.245°；这项捕获贡献不归为纯网络。后腿实时capture/geometry/强制前向辅助OFF，已声明owner投影ON。RR qualified7108=59.233333s、cross8152=67.933333s，但始终未placed；末RR合法XY/AIR/0N，gap55.335mm。RL未qualified/cross/placed。第一处未完成是RR顶部捕获和可用承载，不是FL。

终止P09 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；无机身碰障/NaN/硬限终止。末RL为GROUND_AND_OBSTACLE、OBSTACLE_AMBIGUOUS、bearing_verified=false，保留CONTACT_BEARING_UNVERIFIED，不虚构承载。源拒收因真实任务未完成，不是录像故障。1381决策/11044物理ticks/92.033333s；末决策4ticks，15fps编码1381帧/92.066667s，仅末帧显示量化。

视频目录 `video_review/CP226304_deterministic_front_preserved_v1_review/`：

- `CP226304_DET_full_RL_completion_attempt_INCOMPLETE.mp4`：自然P01完整尾段。
- `CP226304_DET_FL_to_FR_RL_recovery_detail_INCOMPLETE.mp4`：同次RR捕获窗口，标签RR CAPTURE INCOMPLETE / RL NOT REACHED，不伪称RL已开始。
- `N_vs_CP226304_DET_same_camera.mp4`：历史成功N，结束后明示定格，非新同构B。

三片全量decode、连续单调PTS、black-like帧0，主任务已目视full尾帧和detail起帧。绑定见同目录export_receipt.json。完整物理及动作表见 `CP226304_DET_SEALED.md/json`、`CP226304_DET_FRplaced_front_space_6s.md/json`、`CP226304_DET_FLplaced_P06_evidence.md/json`。保留CP225792和旧分支全部产物。

## 最新完整更新账本

新增合计 **1024真实学生决策 / 2 PPO / 40 Adam**，CP226304 / PPO1727 / Adam34540。实际request phase P07/P08/P09/P10/P11/P12为6/6/430/2/26/554；其余阶段本次新增on-policy样本为0。5424个真实教师前缀决策（43392物理tick）全部排除。前段保留合计1280次离线Gaussian KL曝光，嵌入40个原Adam小批次，额外PPO样本/AUX optimizer step均为0，不将其冒充前段真实新采样。

第二块学生取得两次RR真实placed，随后均失去TOP支撑；RL新qualification/cross/placed为0。两块合计RR可达AIR准备115个端点、实际TOP并前侧准备76、固定FR方向投影与RL载荷份额下降代理40（非已证明右侧转移）。RL新资格1个事件/27端点，另12端点继承前缀资格；RL越沿/放置仍为0。窗口非互斥，不将阶段标签等同物理能力。

第二块6条学生轨迹长度74/113/48/37/136/104；前5条分别因RR knee硬限、FR knee硬限、机身碰障、机身碰障、FR knee硬限终止，第6条为正确bootstrap的非terminal预算尾。没有放宽任何安全限制。正式新学生表现须以下方当前评估封存后为准，这些随机训练轨迹采集于更新前。

最新checkpoint：`outputs/ppo_rr_rl_timing_policy_learning_v1/branches/cp225280_front_preserved_v1/checkpoints/history/checkpoint_step_000226304.pt`；SHA256 `c0f4727bcc4e1963b5e9d079158dde1f5f662390397b61049ff893c40e1910f6`，sidecar `4d1e103507e7083f9d1ecb6d5ff230bd3d106054da2af0ef85e0ce81b9c8569b`。实际LR 1e-5，save/load roundtrip验证通过。明细见 `892385_P07_CP225792_512_coverage.md/json`。

## 当前已经验证的结果

- CP230144 的前部降低由事件对齐的真实刚体安装点、CoM、关节响应及轮端空间共同确认，不仅是画面推测；它在 FL 越沿前失去空间，不是顶部捕获下探预算不足。
- 显式恢复兼容 f6d439 CP225280 的网络、critic、Adam、实际 LR、Identity normalizer 与 RNG，保留当前正确的 N、mapper、HISTORY、FL 辅助和已声明的后腿 owner 接口。原模型、旧最新模型与成功 N_ref 均未覆盖。
- 生产版本 `892385cba8a7089b52567bb7f558c018fac82a77`：前段 Gaussian KL 保留在原 PPO Adam minibatch 中合并，不增加独立 optimizer step；没有新增前段 reward、强制抬高机身或指定关节角。
- 真实续训新增 **512 个 P12 学生决策 / 1 次 PPO update / 20 个 Adam step**。保存并重载的新学生为 **CP225792 / PPO1726 / Adam34520**。
- 新学生自然 P01 确定性录像：FR placed tick2841（23.675s）；FL crossed tick4438（36.983333s）；FL placed tick5606（46.716667s），随后进入 P06。此后 FL 可按后腿准备再次减载，历史 placed 不冒充当前承载。
- 前段执行没有教师换模，仍声明原 FL assist ON；后腿实时 capture/geometry/强制前向 assist OFF，已有可观察 owner projection ON。FL assist 从 tick4440 进入 DESCEND，捕获时 hip correction 约−28.014°、knee +.154°，是本次真实捕获的重要控制贡献，不归为纯网络学习。

## 实际进入 optimizer 的后腿数据

512 条都是学生在真实连续 nominal 前缀交接后的 on-policy 数据。两条前缀共 1554 decisions，PPO credit 为零；前缀 RL qualified6212 先于 handoff6216，不归为学生发现。

非互斥物理端点窗口：RR 合法顶部承载并前侧准备 57；固定 FR 方向 CoM/body 投影和 RL 载荷份额下降代理 22（不是已证实侧向转移）；学生新 RL 资格 27 个端点；合法资格的边缘恢复 2 个端点，但两者继承前缀资格。RL crossed/placed 均为 0。

学生新 RL 抬升最高 gap 61.475mm，但仍在前缘前 159.412mm，随后 RR 支撑和 RL 资格丢失。不得将这个随机训练片段称为新学生自然 P01 成功或完整 RL 越障。

前段保留是独立离线约束：100 个训练状态、99 个 heldout，20×32=640 次拟合曝光，新增物理/PPO 样本和额外 AUX step 均为零。P01/P02/P03/P04/P05/P06 曝光分别 7/224/14/7/196/192。更新后 heldout mean KL .007886615；有限分布约束本身不保证物理能力，以上真实 P01 的 FL→P06 结果才是此次前段保持证据。

## Checkpoint 与证据

- checkpoint：`outputs/ppo_rr_rl_timing_policy_learning_v1/branches/cp225280_front_preserved_v1/checkpoints/history/checkpoint_step_000225792.pt`
- SHA256：`17a7e849afc242856f035350d814349be1e101b49385c42ab9c90966a84fc5a7`
- sidecar SHA256：`d42847f6c90612366bdb16ca574e11403430ac368ed487a69a0d904b73d8cc6d`
- 训练与账本：`892385_P12_front_preserved_512_coverage.md/json`
- 旧模型退化对齐：`CP225280_vs_CP230144_FRplaced_front_descent_event_aligned.md/json`
- 新模型 FR 放置后空间：`CP225792_DET_FRplaced_front_space_6s.md/json`
- 同次 FL 捕获及 P06：`CP225792_DET_FLplaced_P06_evidence.md/json`
- 自然 P01 source：`runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1447490554064Z_g892385cba8a7_9356dc07a70e4bff9ae154deafb3c87b/source`

同一学生的后续：RR qualified7198（59.983333s）、cross8308（69.233333s）；到92.766667s仍合法XY/AIR/0N、gap58.382mm，未placed；RL尚无有效抬升、cross或placed。**第一项未完成任务是RR顶部捕获与承载，不是FL越沿。** 1392次决策、11132物理tick，末决策4tick；实际编码1392帧/92.8s（末帧显示量化差.033333s），没有剪尾或接其他回合。source的DIAGNOSTIC_FAILURE只因未满足共同物理任务，未改成成功，非摄像回调故障。

RR越沿后窄窗中N hip/knee为−6.9/−37.8°，mapped N为−8.15/−39.05°；策略hip正残差抵消源负向，knee最终持续裁到−58°。普通准备许可为true、owner17全0、mask全1，不是四腿准备全锁。保留待RR实测支撑后才释放的FL/RL强发力组，不能以解除全部门禁作为捕获替代。

正式视频位于 `video_review/CP225792_deterministic_front_preserved_v1_review_v2/`（旧导出原样保留；v2只把细节标题收紧为真实RR捕获未完成，不改变物理结果）：

- `CP225792_DET_full_RL_completion_attempt_INCOMPLETE.mp4`：完整1392帧、92.8s、自然P01单回合、正常速度、真实尾段。
- `CP225792_DET_FL_to_FR_RL_recovery_detail_INCOMPLETE.mp4`：同次[865,1392)帧，35.133334s，标签明确RR CAPTURE INCOMPLETE / RL NOT REACHED；它是RR捕获尝试细节，不是RL越障进展。
- `N_vs_CP225792_DET_same_camera.mp4`：历史成功N对照，不是本版新跑同构B；N结束后标注定格，不计额外稳定物理时间。

各导出文件已通过全量decode、连续单调PTS和无black-like帧检查；主任务已目视完整片及N对照末帧。绑定信息见同目录 `export_receipt.json`。全段roll/pitch RMS7.986°/5.437°，缺P10–P13，因此没有声称稳定性优于完整N。

## 修改与测试边界

生产代码修改集中在 `semantic_front_preservation.py`（独立分支和完整状态兼容迁移）、`semantic_front_replay.py`（有限前段条件分布 KL）及 `semantic_training.py`（原 PPO 优化小批次中的显式合并与计账）；`semantic_front_retention439.py`、`semantic_rear_policy_timing_migration.py` 只接通新身份的受验证加载路径。没有重写 FSM、清空网络、放宽物理能力或改前段 nominal。

14 个新增 CPU 单元/真实优化集成测试通过，覆盖 raw/old-current log probability、512 buffer、每样本5次官方曝光、20个Adam step、640个独立replay曝光、heldout排除、hook恢复；41个既有定向测试通过。另有14个标准库身份测试、5个数据集测试、13个旧导出及6个新导出测试通过。它们不计入真实机器人决策或物理成功证据。
