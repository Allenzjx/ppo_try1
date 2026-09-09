# 实际进度快照：checkpoint133248

本报告截止第9个训练段和第3次自然P01评估。第10段P04前驱准备课程已在同一版本继续运行；其未完成数据不计入下列数字。最新指针可能继续前进，请以 `checkpoints/checkpoint_last_pointer.json` 为准。

## 已实施的生产修改

已提交 `2f27c6f5065aee6fe7b17f165a876e6dc5307841`：四腿角色监督、连续阶段衔接、真实全身抬升、P05接触捕获建议、合理扩大residual控制范围和五类reward内的冲突修正。FR→RL、FL→RR、RR→FL、RL→FR；当前承载与历史放置分开，接收侧悬空不虚构载荷。普通阶段不done、不重置动作/滤波/mapper历史；已离地后腿连续接管，落地失效的资格按实测撤销。原FSM/Recording保持不变。

主要实现位于 `src/wlr50_clean/ppo/semantic_transfer_roles.py`、`semantic_supervisor.py`、nominal/backend/adapter及reward路径；完整配置与机制证据见同目录 `role_integration_design.md`、`reward_diff_and_counterexamples.md`、`residual_authority.md`、`recording_mechanism_evidence.md`。

已提交 `db991aa68103642c179e130fe4cce6ee892c89ab`：显式372维角色布局，涉及 observation schema/encoder、policy contract/history actor、checkpoint prefix、migration、training与CLI。原324列不变，追加四腿各12项已被监督器/reward消费但不可由原聚合量唯一恢复的角色摘要。不是从零训练，也不声称完整重建所有滑动窗口内部状态。

角色版前置CPU501项通过；追加版组合CPU618项通过、零失败/错误/跳过。两组有重叠，不能相加成独立测试总数。未修改USD、质量、摩擦、出生位置、关节执行器或物理频率。两个提交均留在现有分支，没有push或清理用户工作树。

## 恢复、训练与配置

实际源为123136 decisions /927 PPO updates /18540 optimizer steps，未回退到历史10112/44。当前已验证保存重载：

`C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000133248.pt`

累计133248 /1006 /20120；本轮新增 **10112 decisions /79 updates /1580 optimizer steps**。这里10112是新增量，恰好与旧恢复点数字相同，含义不同。教师前缀、评估、未完成rollout均不充作新增训练。

仍为N1、12动作、120Hz物理/15Hz决策、128步rollout、5epochs×4minibatches、HISTORY异方差policy、rho=.9、gamma=.9985、lambda=.99。前期扩大FR膝与wheel等residual，并验证实际native控制；硬限/速率保护保持。详细逐通道范围见 `residual_authority.md`。

324→372边界保留actor/critic原列及其他权重、learned std、identity normalizer、全部RNG及累计预算；新增48列零初始化。源Adam核验后因输入形状变化新建Adam3e-5，清空旧rollout。之后正常恢复保留Adam状态，未重复迁移。真实迁移初始mean差7.45e-9、std/value差0、投影动作差0。到132864时，两模型各12288个追加列参数均已非零，mean/log-std头均变化；这证明参数更新，不证明任务贡献或收敛。

## 实际阶段样本量

以下仅九段已优化数据；第9段原计划1536，实际在完整更新边界停止384，余1152保留为未执行，不算完成。

| 阶段 | policy decisions |
|---|---:|
| P01 | 40 |
| P02 | 2371 |
| P03 | 69 |
| P04 | 51 |
| P05 | 1757 |
| P06 | 3673 |
| P07 | 13 |
| P08 | 13 |
| P09 | 717 |
| P10 | 6 |
| P11 | 32 |
| P12 | 892 |
| P13 | 478 |

累计终止回合：FALL19、BODY_COLLISION14、INCOMPLETE4、HARD_JOINT_LIMIT3；非终止update边界尾段另列，不当作失败/成功。完整训练成功仍为0。第8段RR课程出现3次策略有效离地，越沿/放置各0；它不是后腿成功。第9段三个硬限中止均保存，最终checkpoint及停止请求可追溯。

两例安全复核未证明目标、符号、下发链错误：FR例目标−58°仍在保护范围，实测随后越界；RR例目标约−3°、离下界很远。静态margin无法保证实测q，增加2°余量也不能解释第二例。终止compact缺精确post-step q/qd/torque，保留量化包络与动力学越限的不确定性；没有降低判定阈值、改物理或据此盲改控制。见 `run9_first_two_joint_limit_review.md`。

## 自然P01重载评估：0/3

三次均确定性、seed2001、无教师前缀、optimizer0、非外部截断；所有尝试计入分母。

| checkpoint | 决策 / 物理秒 | 第一个未完成任务 |
|---|---:|---|
|127232|227 /15.1333|P02：FR尚未足够前送|
|130304|607 /40.4667|P05：FL未放置，AIR约43.8mm|
|133248|685 /45.6667|P05：FL未放置，AIR约14.1mm|

C3 FL已有Q/C，没有P；front+116.1mm、AIR、负载0、障碍物接触inactive、连续TOP0。`.85`只是资格/越沿/几何部分进度，不是承载或真实放置，也不是PBRS plateau。其P05到30秒期限仍未接触，属于合法未完成，无判定矛盾。C3物理有效、无碰撞/安全终止；后续阶段均未访问。

C3全局roll/pitch RMS=.130458/.096691rad，角加速度RMS=5.238936rad/s²；P05分别.030261/.023906/4.660277。实际触地速度指标未知，未访问阶段质量与全任务综合分为null，不能填零或称稳定性优于FSM。

## A与视频状态

保留A原结果：P10 WAIT_ENTRY未完成，RR膝角及带符号回弹速度未满足历史精确入口；不是本次物理碰撞或仅靠wheel爬墙失败。历史Trial043成功不替代它。本轮没有重跑A，也未建立A5/5或人工非零全程成功门槛。

A为v2/seed4001/180+64预动作ticks；当前C为v3/seed2001/180+0，故**未配对**。A原始视频保留；历史解码PTS记录通过，但container duration不合法，不能称当前严格视频交付已验证。完整PPO成功checkpoint、PPO成功视频及对比视频尚未产生，没有“improved/success”重命名。

## 当前续作

第10段已从133248正常恢复，P04 reset-only前缀、offset0、请求1024新决策；从FL准备学习真实落脚，后续阶段可继续推进。RR前驱准备与自然P01仍需补采，RL未执行配额保留。没有因诊断重置网络或Adam。N8与渲染检查无可直接安全切换的收益，继续现有N1路径，不等待重写vector reset。

证据入口：`training_manifest.json`、`video_manifest.json`、`evaluation_133248_diagnosis.json/.md`、`cumulative_through_block9/cumulative_training_summary.json`、`cumulative_through_block9/phase_and_transition_coverage.csv`（117×13，逐格校验及QA预览完成；原52行first4096 CSV保持不变）。上述均为实际快照，不把当前运行中的计划数字当结果。
