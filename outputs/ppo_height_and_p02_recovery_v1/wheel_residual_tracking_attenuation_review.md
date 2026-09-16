# 后备审查：承载腿跟踪误差驱动的 wheel residual 衰减

结论：可以作为透明的控制结构候选，但当前证据不足以称其安全、确定因果，或指定已验证阈值。先完成既定 sampling-only。以下未实施、未运行；没有读取 live B 或新增大日志扫描。

## 证据边界

已完成 wheels4 诊断在原 nominal 滚动继续时避免 RR 大幅下坠并完成 FR 放置，支持“policy wheel 分量可能参与全身载荷路径”的假设。但它同时移除共同推进和差速分量，且 tick9 起 servo 输出随闭环/HISTORY 改变，不能证明 RR 自己的 wheel、某个误差阈值或差速是唯一原因。C170240 的目标更接近零但 actual 更差、仍 verified GROUND；94 个端点 PD 估算裁到2.7Nm不等于94个独立实测 PhysX 力矩，更不证明停止 residual 就能恢复承载。诊断中也曾有 PD 估计裁剪而没有失控，因此不能用“估计饱和”单独触发。

动作统计亦不支持统一缩 sigma：P02 各通道条件 sigma 均值0.06585–0.24258，存在局部高 sigma/tanh 尾部；正式 C170240 全部 early raw 的最大绝对值1.4764，小于2。当前均值轨迹本身也必须接受物理检查。来源：p02_wheels4_completed_comparison.md、training_2048_action_distribution.json、p02_policy_recovery_checkpoint170240.json。新4×256后的实际 checkpoint 是后备对照对象，不能以170240代替将来的最新状态。

## 最小候选语义（不是安全定理）

- 使用每个120Hz **当前已观察帧**：逐腿真实 GROUND/TOP 接触、独立 bearing_verified 与支撑力；不能把未知载荷当零，也不能让 FRONT_WALL 法向反力或悬空腿冒充承载。只改变 policy wheel residual，不改 N、controller bias、servo residual、物理、任务判据或12维 raw。
- 跟踪误差使用同一 canonical 单位/时钟的 **最近已实际 dispatch 的 final servo target 与当前 actual q**，不是新一步尚未执行的 nominal waypoint、历史入口膝角或目标相对零的大小。对各腿 hip/knee 的误差幅值生成单调、连续的严重度；没有 P02 专用角度、停稳门或固定支撑组合。阈值、过渡宽度及最小增益均须单独声明候选参数，本文不推荐数值为“已验证安全”。
- 例示形式：有效承载置信度乘以误差的平滑 ramp，得到各腿 severity；共同增益 `g = 1 - (1-g_min) * max(severity_leg)`，`0 < g_min <= g <= 1`。只把四轮的期望 residual 乘 g，保留 residual 方向/相对比例，随后继续现有 slew/caps/hard limits。不是强制四轮同速，也不把 N 乘 g。无需新增持久滤波状态；接触资格可能离散变化，最终动作连续性由现有120Hz slew保证。若另加低通/hysteresis，则其状态、reset/observation/迁移必须明确，不能暗藏在372观察不变的声明里。
- raw、old logprob、likelihood 和 previous_raw HISTORY 始终记录真正采样的原12维动作；环境执行确定性的状态相关变换。不能对过滤后的值重新算旧概率，不能把执行映射伪装成 actor 输出变化。所有12 mask仍为1；正增益不等于每一 tick 都有非零 native效应，需继续实测验证。
- 任一不完整/不合法传感器上下文不能被标为“过滤已保护”：保持原错误/安全处理并给出 unavailable。若选择诊断失败开放为g=1，必须明确记录；不能悄悄把 unknown 解释成无危险。普通阶段切换不清 g/残差/mapper/HISTORY、不设done；真实 episode reset 才清既有 episode状态。

## 每腿局部衰减还是四轮共同增益

每腿仅减其对应 wheel 的参数较直观，但会主动改变差速比例，可能增大偏航/横摆，并忽略“其他轮驱动使此腿受载”的交叉作用。现有四轮移除试验没有识别这种局部因果，所以不优先采用 RR-only 或每腿独立增益。

若后续证据支持试验，**四轮共同增益是更窄的首候选**：由所有实际承载腿的严重度共同决定，仅一个控制强度，保留 residual 的向量方向。代价是可能抑制本来用于 CoM 转移/减载/越沿的有益推进，造成停滞；某腿合法快速抬升产生暂时跟踪误差也可能误触发。N仍可继续滚动，故这不是刹车或消除力矩的保证。max严重度是连续但非处处可导，PPO并不需要对环境过滤器反传；不能据此宣称闭环稳定。

## 可放置的位置及容易漏掉的交接错误

首选在 `SemanticEpisodeEnv.step` 的 source帧采样计算严重度，沿 `PhaseTransitionBridge.project_tick` 传入显式、默认identity的 residual gain，在 `ActionProjector.project` **tanh/尺度之后、已有residual slew之前**施加四轮增益。这样有效 residual 会进入同一 bridge存储、reward样本和 previous_residual HISTORY，N保持原值。现有接口没有这个参数，不能把非二值增益塞进 runtime mask。

必须定向处理 bridge 的 handoff：当前 `_raw_that_holds(carried)` 是为既有有效 residual 生成反变换。若再无条件乘 g，会在换阶段时二次衰减，即使物理状态和g完全不变也跳变。应显式保留“handoff持有的是已有效 residual”的语义；g改变只能通过同一个slew推进，不能把 phase transition 变成额外限幅。也不能通过除以g再atanh来恢复不可达旧值。

不建议只在 `apply_semantic_residual` 最终 wheel setter 前加一行乘法：会绕过 wheel residual slew，使 bridge/reward所记残差与实际执行分离，并可能破坏 same-tick zero反事实/native-effect审计。若确因最新 mapper ACK 必须落在 dispatch，则需完整记录 requested/effective residual、按实际有效值维护历史及唯一slew、重用相同prestate上下文验证反事实；这已不再是较小候选。上述首选用前一已执行目标，不需要额外 advance mapper或额外physics。

代码定位：semantic_env.py:148–183；phase_action_masks_v2.py:296–462；action_projection.py:515–569；semantic_residual_adapter.py:27–177；semantic_backend.py:254。具体行号以当前HEAD为准。

## 最少反证与验证，不成为训练门禁

现有旧 CP/旧候选的 wheels4 结果不足以为新 checkpoint 选择通用增益或 tracking阈值。若 sampling-only 后正式均值仍失败，至少先做一个 **同最新CP、同HEAD/N、同seed、自然P01、冻结actor、无更新** 的预声明有界 wheel小增益干预，对比真实承载腿误差是否在减弱后恢复，以及 FR crossing/placement 是否仍能发生；必须保留所有12原raw及执行变换记录，不能称其正式未过滤PPO成功。不再要求5/5或整个探针必成功。

如要进一步声称“差速”而不是整体推进是原因，应补一个保留 wheel residual 共同分量、仅减差分分量的诊断，或相反的互补对照；现有 wheel4 mask不能分离二者。若想采用每腿方案，则还缺对应轮与其他轮的交叉对照。一次有界干预可以筛选，不足以确立泛化安全；不需要为当前optimizer添加这些门禁。

CPU最低测试：identity/zero路径与原N一致；未承载/未知/墙接触不虚构支撑；所有腿/方向对称且无phase专用入口；增益上下界与恢复；没有额外mapper/write/physics；持续严重度和接触变化下仍满足wheel slew；跨阶段恒定g无二次衰减；raw/likelihood/HISTORY与实际native审计一致；真终端不改，P01及后腿样例分别验证推进/减载未被永久关闭。物理试验还须记录每腿误差/资格/g、变换前后残差/四轮差分、actual q/载荷/CoM/首次未完成任务，不能只看姿态变平。

## 迁移和收益归属

这改变 action execution/MDP，即使输入372、raw12、权重形状和N完全不变，也不是现有 height-only 或 reward-only迁移。当前 `_height_recovery_factor` 只允许 geometry advisory/height配置及严格文件集，不允许 env/bridge/projector和新residual控制键；应另建明确的 narrow execution-filter版本边界，锁定真实源CP/新HEAD/精确文件与配置，保留 actor/critic/Adam/normalizer/RNG/预算，清旧rollout并重新采集；不偷放进height白名单，不泛化372豁免。

用同CP的 filter-off/on 配对只报告**控制结构收益**；随后训练前/后的CP在相同filter-on条件下比较才是该结构下的**学习增量**，再保留 finalCP filter-off 诊断说明策略是否依赖结构。B zero残差应数值不变，冻结A不改，所有正式train/eval使用同一过滤语义，视频明确标示过滤版本/活动比例。只有真实P01任务完成才能称联合系统成功；不能把滤波挽救称为原policy已经学会，也不能把更平/未碰撞当完成越障。
