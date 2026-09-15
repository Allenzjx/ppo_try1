# 视频优先诊断与有限 PPO 续训

本文件为本轮可读入口。首批三条视频已经在开始新训练前交付；任务失败与媒体有效性分开保存，未将失败改成成功。两块真实续训已保存，共新增640个决策/5次PPO更新/100个optimizer steps。最新模型自然P01重载评估也已完成：仍在P02、7.325秒RR knee硬限安全中止，未改善本次任务结果，无成功或稳定性提升声明。

## 先看真实视频

- 零残差 B0：`videos/zero_residual_before.mp4`，73.867秒。
- 当前冻结 PPO C0：`videos/ppo_before.mp4`，8.200秒，checkpoint166784。
- 同实际经过时间对比：`videos/zero_vs_ppo_before.mp4`，73.867秒。右侧结束后明确标记冻结终态；没有额外物理步进、倍速或阶段对齐拉伸。
- 补充 RR 通道关闭干预：`videos/rr_channels_off_diagnostic.mp4`，8.333秒；不是正式 PPO、修复后表现或训练样本。
- 实际续训后：`videos/ppo_after_learning.mp4`，7.333秒，checkpoint167424，自然P01、无教师、确定性、0次评估optimizer更新。未改生产控制/reward/物理，不能标成“控制器修复后”。

三条主视频均为H.264/yuv420p/faststart，完整解码及起末帧检查通过。原片保留。首次编码在tick8；有tick0物理记录但没有编码的控制前pre-roll，未伪造补帧。两次干净启动的初始纹理加载画面有区别，记录的物理初态完全相等。

## 主要结论

| 问题 | 实际结果与证据边界 |
| --- | --- |
| B0到哪里 | P09，73.833333秒，TASK_INCOMPLETE。RR仍在地面且尚未越沿/放置；当前冻结版局部恢复时限耗尽，非录像故障、非成功 |
| C0到哪里 | P02，8.191667秒，RR knee硬限安全中止。最终目标−5.654127°、实际−60.007388°，第一未完成任务是FR持续抬起前送、越沿及放置 |
| 学习后完整评估 | checkpoint167424重新加载、自然P01至真实终态，P02/7.325秒/879ticks，仍是RR knee HARD_JOINT_LIMIT；目标−6.358188°、实际−60.016800°。110个issued decisions，其中109完整返回、最后7ticks中止；不是从教师后缀开始。首个未完成任务是P02 FR当前净空与接近，尚未越沿/放置 |
| 最早RR异常 | C0 RR knee实际−最终目标的绝对误差，连续12个120Hz样本≥1°的起点tick35/0.291667秒，≥5°起点tick178/1.483333秒。只是诊断窗口，不是新增门禁 |
| 直接目标还是跟踪 | Policy从首tick改变全身目标，但RR最终大误差不是命令RR到−60°。B0早期RR误差峰值仅0.865149°；同983tick内两路N逻辑目标一致 |
| 是否RR先失去地面接触 | 已有tick0/35/50/178/976十行对照均显示RR为GROUND且法向力为正。是仍有接触力时出现跟踪偏离，随后body/CoM下沉；不能描述为已证实RR先失去接触。导出的normalized承载比例/validity为空，不能伪造载荷比例 |
| RR关闭干预 | 从首动作关闭raw6/7并保留真实masked HISTORY，仍于998tick/8.316667秒RR knee硬限中止，目标0°、实际−60.018801°。直接RR残差不是必要条件；其他通道的全身作用尚需载荷/力矩时序分解，不能推断网络动机 |
| A与N的具体差异 | A的P01持续13.333秒，N在0.133333秒进入P02但继续未完成RL hip/wheel动作owner；N增加基于实际FR抬起证据的条件wheel approach。不是完全等时重放，也没有发现本轮新丢失的旧P03修正/交接修复 |
| Reward是否证实偏好失败 | 相同976tick下C0姿态成本较小，但总回报0.076726低于B0的0.297919，同时机身/CoM更低、角速度更大、FR净空更差。单项存在张力，尚不能证明组合reward偏好失败或网络追求水平度是根因 |
| 本轮修了什么 | 生产控制器、reward、动作范围、HISTORY与物理参数均未改。仅新增输出目录诊断导出/同步提取/干预工具；修复比较视频补帧后的PTS，旧无效导出尝试保留。未确认的控制或reward假设未硬改 |
| 原A结果 | 本轮未重新运行A，也未因历史Trial043物理成功而改写任何旧A失败。成功源作为有来源的动作参考，不冒称当前同条件配对A |
| 完整成功/稳定性优于FSM | 本轮首批B0/C0均未完成越障，无合格全程PPO最佳checkpoint，不声称稳定性优于FSM |

精确角度坐标、各reward有符号分量、有效驱动参数与证据缺口见`reward_and_tracking_diagnosis.md`。瞬时关节驱动力矩未测得；已记录接触力，但本次尚未完成各腿承载与关节力矩的时序因果分解。未提高力限/增益、放宽硬限或关闭安全。

## 本轮实际续训

原始可恢复checkpoint166784：166784个决策、1268次PPO更新、25360个optimizer steps。原checkpoint及历史视频保留。

生产版本HEAD `4b2c038887c4109c639eea720f9e60de2c6d8d93`，runtime内容SHA `ecee8d3abba1981b95d462a70c9ea39deb3e949878caea9d6589c95cdc043458`。从旧d4 provenance到当前4b无生产文件差异，采用显式provenance-only迁移，保留actor/critic/Adam/normalizer/RNG/累计预算，丢弃旧未完成rollout并重新合法初始化。不重新建网络，不清空Adam。

实际配置：现有`fsm_reference_p09_stable_v2`，372观测、12残差、HISTORY rho0.9；N=1，cuda:0，物理120Hz/策略15Hz，seed1001；rollout128，5epochs×4minibatches，gamma0.9985、lambda0.99、clip0.2。Entropy沿用现有全局计数线性调度`0.005−0.004*min(global_step/210000,1)`，不是固定0.005。保留原adaptive LR，恢复时实际1e−5，首块四次更新实际为1e−5/1e−5/1.5e−5/1e−5（配置名义初值3e−5不能冒称实际恢复值）。无新的reward、探索方差或动作mask。

| 块 | 请求与实际采样 | 新决策 | 新PPO更新 | optimizer steps | 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | 自然P01，实际P01–P05 | 512 | 4 | 80 | 完成，checkpoint167296重载校验通过 |
| 2 | P01冻结FSM实际运行至P06再交接，offset0；前缀无学习credit | 128 | 1 | 20 | 正常保存/停止，checkpoint167424重载校验通过 |
| 合计 | 仅真实on-policy数据，不含896个教师决策 | 640 | 5 | 100 | 累计167424 /1273 /25460；不代表任务成功 |

运行中调整：首个P06前缀耗448教师决策/3584ticks/29.866667秒，在同tick的P06开始policy credit，没有takeover跨阶段等待。之后32个学习决策（P06=19/P07=1/P08=1/P09=11）在P09/FALL结束并重跑前缀。前驱阶段确实进入数据，不能称课程丢失准备动作；成本来自短后缀后的长物理初始化。为保持本轮视频优先、有界续训，提交原生`stop_after_update.request.json`，最终在完整128决策更新边界正常保存，lifecycle=STOPPED_AT_VERIFIED_UPDATE_BOUNDARY。没有kill进程或丢弃已优化数据；原请求剩余384没有运行，也未伪报成1024个新学习决策。

第一块实际每阶段policy decisions：P01=2，P02=101，P03=8，P04=1，P05=400，P06–P13=0。训练完成仅指更新块完成，不是episode成功；首块未完成episode。

第二块实际每阶段policy decisions：P06=115，P07=1，P08=1，P09=11，其他阶段为0。共1017个学习物理ticks；两次教师前缀各448decisions/3584ticks，总896decisions/7168ticks全部排除。第一回合FALL，第二回合最后96个决策仍在P06，末tick4352/36.266667秒非终止；两个前缀都被接受，无fallback、无后缀成功。所有128行，包括第一回合32行，均已完成实际PPO更新。

单环境每个训练块使用明确课程；配置中sampling_target百分比不是本次实际随机采样器，不报告虚构的阶段比例。普通阶段切换不是done；教师前缀、视频帧与RR干预均不计入PPO决策。

`PrefixSource=frozen_fsm`是现有物理前缀初始化模式，不是原A Trial043的完整等时回放：原SensorFsmController提供教师命令，当前TaskStageSupervisor按120Hz真实观测另行判断任务阶段，所以前缀日志的P02/P05不是教师本身阶段时刻。到semantic P06的决策边界后，接管当前监督器及最后实际dispatch的nominal/tracking/bias，保留物理状态和连续动作，不恢复固定入口姿态。

最新模型为`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000167424.pt`；SHA256=`9eb4bbdb2cb93afcc4fc04ab1295b92bde06167a72af9febe09804e9e5603a44`。保存/重载校验通过，保持identity normalizer，Adam继续更新。自然P01、无教师、确定性的保存模型重载视频评估run为`runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260914T0540369315910Z_g4b2c038887c4_85756e634ebf4aa2999aa1273ba6077b`，于2026-09-14 05:43:17 UTC完成并保留DIAGNOSTIC_FAILURE。原封装只接受任务成功而返回exit1，物理终态是RR硬限，不是录像损坏；新视频已完整解码、检查原帧/PTS一致、查看首末帧并在会话展示。7.325秒早于before的8.191667秒，只说明这次同配置重评没有改善，不能推断所有后续训练必然更差。

学习后官方reload记录的actor/critic/Adam/normalizer哈希与最新保存checkpoint一致。首物理row的关节q/qd、wheel速度、root pose/速度、raw世界CoM位置/速度九组数组相对B0的最大差值均为0，四wheel接触记录一致；本次小核验没有另行比较native pre-step q与standing offsets。终点RR仍GROUND/support=true，承载力15.839683N，末态监督器load fraction0.510683且valid=true；这与此前稀疏CSV未提供相应normalized字段是不同证据来源，不能补填此前未知值。FR有Q历史事件tick26，但当前净空−16.112mm、前缘距离−147.151mm，无C/P事件。精确边界见`after_learning_eval_receipt.json`。

## 验证、保存与恢复

32项选定已有CPU回归通过，另有独立reward-only/frozen-actor不变性测试通过；它们不替代真实物理评估。RR干预验证125个masked-history决策且最终actor/critic/Adam/normalizer未变。原C0初始官方重载哈希匹配，但失败分支未执行生产recorder末尾模型哈希断言，此缺口如实保留，不虚构检查。

`reference_and_runtime_identity.json`保存首批模型/版本/初態绑定；`video_manifest.json`将physical_result与media_validity分开。`rr_first_divergence.csv`是306条真实120Hz采样、明确稀疏窗口的科学诊断导出，单位/坐标分列，未知留空，不是训练集或新学习样本。原数据和历史run均未覆盖。

当前恢复与运行进程状态见`RECOVERY.md`。最新不等于最佳；本轮没有提升或覆盖合格best。未创建后台自动续训、推送或提交，也未启动并行Isaac实例。
