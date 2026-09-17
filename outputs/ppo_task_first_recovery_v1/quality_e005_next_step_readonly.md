# M3之后的小质量候选：只读建议，尚未实施

当前512条quarter训练仍活动；本建议不扫描其大日志、不计未封存样本、不改生产或启动Isaac。已保护的完整成功依据是CP177152、seed4001、自然P01、50.966667s、M3=1/1；不是后缀或teacher成功。

## 是否已有依据尝试质量

足以启动一个独立、可撤回的小质量候选，但不足以宣称所有坏策略都已消除或质量一定有益。已封存的[四轮证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/four_wheel_P02_checkpoint177152.md)表明：P02正向N窗口1320 ticks，四轮N均+.3rad/s；最终target均值FL/FR/RL/RR为.314101/.346469/.377189/.290260，四轮反向、接近零、低于各自N一半的target样本均0。全部12许可、实际写入与qd映射核验；旧“FL反向、FR/RL抵消、RR增强”没有在这个正式回合延续。四轮不是等速，FR当时悬空也不等于提供地面牵引。RL仍有瞬时实测反转，不能隐藏或误称mask。

[同回合完整审计](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/evaluation_CP177152_actual.json)随后确认四腿真正放置、当前受控完成、零机身碰撞/非有限/越硬限。但只有一个seed，恢复混合了明确mean-head初始化、真实更新及两项执行修复，不能独立归功于reward或一次update。当前quarter探索失败不取消CP177152已取得的M3，也不证明它已具有稳定的后腿训练分布。

确有次级优化余地：[质量描述](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_first_recovery_v1/zero_vs_CP177152_descriptive_quality.json)中zero/PPO的roll RMS=.119031/.125496rad、pitch RMS=.081318/.105251rad、roll-rate p95=.048101/.229873rad/s；不同build、不同长度，不能用它们声称严格配对改善。M4仍未取得。

## 最小reward改动

沿用现有五family计算、carry/transfer allowance、实际执行差分与接触诊断；无需新reward结构、新姿态模板或动作mask。在同一task_first namespace中新增独立objective/revision/branch_id，quality_epsilon=.05，family权重为task=1、body=.020、contact=.010、smooth=.005、regularization=0（原.4/.2/.1乘.05）。epsilon只是与权重一致的版本声明，不再额外乘一次。

六配置中仅reward_config的profile/revision/epsilon/上述family权重改变；另外五份逐字节保留。reward中的+40/-40事件、势能系数5、时间成本.02/s、gamma=.9985、lambda=.99、200s时域、真实终态/普通阶段/rollout尾语义均不变。原始质量单family每秒有界，新增质量成本总上界.035/s；这不是任务保留证明，仍须真实更新后完整重载评估。

## 独立迁移而非混改温度

当前验证器只允许epsilon0，不能仅编辑YAML后继续，更不能放宽旧task_first_reward_factor。新增明确quality_reward_factor，校验实际源checkpoint/manifest/hash、source与target完整policy_contract相等、同372 observation、同12容量、相同rho与温度、N/场景/评价器不变。若quarter最新正式M3成功，可从它继承.25；若失败，从受保护CP177152显式分支并继承它的.5，不把两种温度和quality同时改变。

保留actor/learned sigma/features/critic、完整Adam及实际LR/计数、Identity/RNG；不重置mean头。旧unfinished rollout清空，合法reset，在新目标下重采on-policy数据。critic以新数据拟合，不把旧值当作新目标真值。只改semantic_reward.py的一致性校验、semantic_migration.py的新factor和semantic_training.py的加载/分支元数据支持；semantic_return_profile和CLI/视频路由原则上无需改动。CPU官方load→update→save/reload与拒绝非reward差异测试不可省，但不建立重复全程成功门禁。

源码边界须如实处理：CP177152来自00050a，而当前fc14已经加入quarter actor/contract代码。若从CP177152分支，不能假称这两个源码版本字节相同，或把.5→.25的温度迁移冒充同policy迁移。可从其已验证实现版本组织质量分支；如保留当前额外quarter代码，则需独立绑定既有quarter receipt的受保护AST相同证据、证明选用的.5 actor/contract路径未变，且不启用quarter kernel。不能无依据扩大quality allowlist吞掉这些差异。

## 从best分支的输出隔离

现状：train_semantic按global step写checkpoints/history/checkpoint_step_N.pt，save_semantic_checkpoint遇同名文件明确拒绝。从CP177152再采128会产生177280，已与quarter历史文件冲突；不能跳高计数、覆盖或移动旧文件。

最小方案：经新factor审核的artifact branch元数据记录branch_id、source SHA和真实counter_origin=177152/1349/26980。仅对这个显式分支，将不可变文件写到`checkpoints/history/<branch_id>/checkpoint_step_N.pt`，mutable last/resume/pointer写到`checkpoints/branches/<branch_id>/`。命名只隔离产物，不改任何学习预算或终态。新的分支计数是本分支累计减源计数；总工作量按各分支实际增量求和，不能用最大global step相减。

现有CLI的_resolved_checkpoint已接受checkpoints之内的显式嵌套CP，并要求checkpoint_last指向checkpoints/history之内；上述布局符合现有限定，正式eval可直接使用不可变绝对路径，无需新增namespace白名单或工程。只需training保存/发布函数与迁移绑定元数据的窄改动；测试拒绝`..`/绝对分支名、同名覆盖、跨分支pointer和错误source，验证nested官方load/update/reload及branch续训。不要把新artifact branch覆盖已有mean-head ancestry；旧来源分支继续留在parent记录。

先等待当前quarter块封存与其正式P01评估，再选择实际成功源。质量候选完成有界新目标更新并保存后，必须自然P01/teacher0/full12整回合复验；任务退化则保留CP177152 best，不能因累计更新数、奖励或平滑分数更高而升级。质量实验与P06采样/温度实验保持独立；本文件没有批准或执行新配置。
