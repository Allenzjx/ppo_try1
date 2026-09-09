# 实施与训练检查点：135424

这是截至第12个已完成训练块及C4评估的固定快照，不代表任务已完成。第13块P07连续课程已启动，但尚未保存的新数据不计入下列训练量。实际最新状态以 training_manifest.json 和 checkpoint_last_pointer.json 为准。

## 结果先行

- 从本轮实际恢复点123136继续，已新增 **12288 policy decisions / 96 PPO updates / 1920 optimizer steps**。
- 最新完整保存：**135424 decisions / 1023 PPO updates / 20460 optimizer steps**；保存重载验证通过。
- 最新自然P01、确定性、无前缀评估：**未成功，首个未完成P02**。FR有效抬升，但未越前缘/放置；15.133333秒、227决策后任务截止。正式完整评估累计 **0/4**，全部尝试保留。
- 无完整P01成功checkpoint，无成功视频，无配对稳定性改善证据。没有创建 first_full_success 或 improved 文件。

## 已修改的生产路径

本轮工作在原工程、原分支继续；未clone、未重建网络、未覆盖历史FSM或输出。主要提交：2f27c6f5065a（四腿角色与动作连续性）、db991aa68103（显式追加角色观测并迁移）、d7479d9fc41c（后段FL建议速度与residual控制范围）。最新提交未push。

- semantic_transfer_roles.py、semantic_supervisor.py：四腿角色、质量加权CoM的固定短窗方向、接收侧空间代理、当前真实支撑与历史事件分离；任务切换不恢复旧入口。
- semantic_backend.py、连续投影/观测路径：保留wheel、residual、mapper/filter与合法Q/C/P历史；阶段切换不是done，不切断GAE。全身驱动造成的RR真实离地可获得初始进展，不要求RR自身关节先明显变化；越沿前GROUND撤销当前抬升资格。
- semantic_reward.py及reward配置：至多五类dense family；有实际角色响应时允许必要机身运动，执行平滑成本以applied为准，连续PBRS为gamma*Phi(next)-Phi(current)，不发阶段切换大奖。
- semantic_observation.py、policy/HISTORY及迁移/CLI路径：原324维顺序与尺度保留，显式追加48维可辨识的短窗角色状态至372。追加时旧列复制、新列置零；随后这些新列已被真实PPO更新。rho=.9、状态相关std、gamma=.9985、lambda=.99保持。
- 最新 d7479d9 只在P06–P13将FL hip nominal建议限速150→60°/s、residual cap24→32°；其他11通道、奖励、阶段条件、物理限制和372维schema不改。semantic_migration/training/cli新增严格same372版本迁移。

两次完整相关CPU回归393+247项通过（0失败/错误/跳过）。包含真实RSL权重/std/Adam/normalizer/RNG、保存重载与继续更新，及真实CPU mapper/headroom/float32目标写入路径；CPU目标缓冲测试不冒称真实动力学成功。

## 四腿交替的语义

| 目标摆腿 | 对角接收侧 | 优先分析的当前支撑/动作线索 |
|---|---|---|
| FR | RL | RL空间及支撑调整，其他轮腿参与卸载与前送 |
| FL | RR | FR新台面支撑须实时可用；RL回归与RR调整，不要求精确home |
| RR | FL | FL腾空间；FR+RL优先桥接，wheel/全身运动协同 |
| RL | FR | RR实时台面支撑；FR腾空间，FL+RR优先桥接 |

偏好不是唯一硬支撑组合。AIR接收腿当前承载为零，历史placed不提供当前支撑。两点接触不是静态稳定证明；工作空间只使用带有效性标记的关节/相对几何/局部响应代理，未伪造完整FK可行性证明、角动量或切向冲量。

## 已确认的动作依赖与最新窄修依据

已有Recording合同/传感窗口中，FL在P05末仍AIR（约13.075mm净空），P06连续wheel推进后才出现TOP接触（约5.510N）。P05已允许pending_capture时提供闭环连续wheel建议，但仍要求真实接触；没有将AIR标placed、降低力阈值到零或跳阶段补成功。

旧P07 nominal把FL hip从22.8推进至49.2，变化26.4°，建议150°/s，而residual限速60°/s且cap24°。有限时间和最终幅度都存在不能完整抵消该建议的指令表达缺口。新配置将这一通道建议速度降至60、cap扩至32；不扩大std噪声、不改变150°/s物理最终slew。有限latent可以表达取消/反向，不等于保证native物理抵消或避免碰撞；mapper反馈、headroom和最终slew仍单独核验。这是B/C建议调度修改，不能归功于PPO学习。

v010等JSON已按成熟Fast Replay parser解析持久12通道、重叠batch和显式wheel stop。FR knee示例是45.9→31.1，delta=-14.8，并有FR wheel -0.63rad/s约1s，不能称为历史绝对-40°。-40°仅为合法范围内可探索候选，不是任务入口或奖励角度。JSON是COMMAND_OBSERVED；有对齐日志的接触响应才是MEASURED_RESPONSE；缺失的精确CoM/力矩仍为机制假设。本轮未重跑Recording或A。

## 最新配置迁移与真实训练

134400→d7479d9采用同372身份映射：实际CUDA初始保存的actor、critic、normalizer SHA及完整training RNG与源一致；包含已学48列与std头，不重复补零/缩放。先验证源Adam，再重建Adam lr=3e-5；origin10112和已用full61696/suffix62592预算保留。旧未完成rollout不继承，使用128×1×12新rollout和合法物理reset，不宣称PhysX接触逐tick续接。

第12块：自然P01、N1/CUDA、seed1001、120/15Hz、rollout128、5epochs×4minibatches、1024决策/8更新；637.317秒训练计时。采样P01–P06为8/476/22/6/417/95，P07–P13为0。3个终止episode均FALL（P05两次、P06一次），另328决策未完成尾段。8180物理tick均记录真实native作用，177次累计非终止handoff中的本块17次均无bootstrap异常；本块native/状态写入审计异常均0。

| 阶段 | 本轮累计已优化决策（12块，跨明确版本） |
|---|---:|
| P01 / P02 / P03 | 48 / 2847 / 91 |
| P04 / P05 / P06 | 78 / 2743 / 4145 |
| P07 / P08 / P09 | 23 / 25 / 880 |
| P10 / P11 / P12 / P13 | 6 / 32 / 892 / 478 |

这些是source-phase PPO样本数，不是成功次数。已终止训练episode58个：FALL31、BODY_COLLISION20、INCOMPLETE4、HARD_JOINT_LIMIT3；未完成尾段另列，不自动当成功或失败。计划量中仍有2048决策未消费，不算训练成果。第13块请求128个P07连续准备/接管样本，真实前缀单独排除，不能替代自然P01。

## C4完整评估与A原结果

C4：checkpoint135424、seed2001、N1、确定性、无教师、无optimizer，P01自然开始。P01=2/P02=225，后续阶段未评价。FR在tick53形成有效抬升；最终AIR、载荷0、净空64.311mm、前缘距离-50.068mm，未C/P。P02的15秒任务截止产生INCOMPLETE_CONTROLLER_BLOCKED；物理有效、无body碰撞、无独立安全中止。全程roll RMS=.174334rad、pitch RMS=.162984rad、角加速度RMS=5.392786rad/s²；缺失阶段/完整质量分数为null，不填零。

A原运行在65.3667秒、P10 WAIT_ENTRY未完成：RR knee约-45.8583对旧入口-50.3976差4.5393°，回弹速度2.6107对23.5853差-20.9746。属于旧精确入口条件未通过，不是本次成功；真实物理判定未记录硬失败。A使用seed4001/v2/180settle+64preaction，C使用seed2001/v3/180+0，条件不匹配。A视频原始字节保留，既有审计发现容器duration字段异常，不能据此把任务标签改成功。没有重新运行A、做5/5门禁或用Trial043代替本次结果。

## 交付位置

- checkpoint：outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000135424.pt，SHA256=1ad778b5936d0a2521110d419de701e8078eb415c0794cf767bce41fcb3179e3；同名_manifest.json及last pointer保留。
- 机制/奖励：logic_conflict_and_replacement.md、reward_diff_and_counterexamples.md、transfer_role_mapping.yaml、recording_angle_and_overlap_evidence.csv。
- 当前生产阶段规范以configs/ppo_semantic_v3/stage_task_spec.yaml为准；输出目录new_stage_goal_spec.yaml是首次角色版本快照，未覆盖。
- 累计覆盖CSV：cumulative_through_block12/phase_and_transition_coverage.csv，156×13数据网格逐值验证并渲染查看，原始4096及其他累计表保留。
- C4详情：evaluation_135424_diagnosis.json/.md；training_manifest.json保存实际状态和恢复规则；video_manifest.json明确成功视频/改善尚缺。

尚缺：自然P01完整成功、≤200秒正常速度成功视频、有效对比视频及配对改善结论。此快照不把继续训练或局部完成写成这些交付已完成。
