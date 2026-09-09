# 四腿交替载荷转移：首个 4096-decision 检查块实报

这是已完成检查块的报告，不是最终成功交付。生产版本 `2f27c6f5065aee6fe7b17f165a876e6dc5307841`；范围是实际旧 checkpoint **123136 decisions / 927 PPO updates** 到新 checkpoint **127232 / 959**。未重新运行 Recording 或冻结 A。随后已开始同版本第二轮混合训练，实时恢复信息见 `training_manifest.json` 和 checkpoint pointer。

## 已实施的生产修改

- `semantic_transfer_roles.py`：真实质量加权 CoM/速度、固定短窗口参考方向、接收轮端位移、当前载荷和几何／关节余量 workspace proxy；不将移动接收脚冒充 CoM 平移。
- `semantic_supervisor.py`、`stage_task_spec.yaml`：角色准备替换错误支撑／低载门，旧前缘 x 区间隔离为 `edge_proximity`。当前接触和历史 placed 独立；不要求接收腿先承载，不恢复固定姿态／膝角／回弹速度或 FSM 相似度门。
- 同一 supervisor 的 nominal 调度：P05 内真实 `pending_capture` 可连续使用既有 +0.3 rad/s wheel 建议，解除先要求 FL placed 才开放后续轮行的依赖；AIR 仍不能算 placed。跨阶段保留 wheel、residual、mapper/filter、接触／Q/C/P 历史及回报链。
- `semantic_reward.py`、reward 配置：在原五类 dense family 内替换错误准备代理；按当前实测 transfer 放松必要姿态／角运动成本，capture/settle 恢复原权重。无第六类对角线奖励、非零 residual 奖励或重复阶段大奖。
- execution／observation 配置：P01–P05 wheel cap ±0.6；P06–P13 FR knee cap ±112；所有 12 通道保持开放。仅两列历史观测尺度迁移，物理硬限、执行率和 slew 不变。
- `semantic_training.py`、`semantic_migration.py`、`semantic_cli.py`、两种 prefix adapter 和 PS 入口：新实验目录、显式兼容迁移、前驱课程扩展、旧 rollout 清空及前缀独立计账。

保护情况：冻结 FSM、机器人／场景物理参数和历史文件保留；工作树原有无关未跟踪文件未动；没有新建训练工程、清空网络或 push。集成回归 **501 passed，0 failed/error/skipped**，之后才开始真实 PPO 更新。详见 `logic_conflict_and_replacement.md`、`reward_diff_and_counterexamples.md` 和 `residual_authority.md`。

## 四腿角色：偏好，不是唯一动作脚本

| 目标摆腿 | 对角接收侧 | 中间支撑偏好 |
|---|---|---|
| FR | RL | FL / RR |
| FL | RR | FR / RL |
| RR | FL | FR / RL |
| RL | FR | FL / RR |

接收腿可以先收缩／开空间，也可以短时 AIR；这不等于它正在承载。两个真实接触既不因缺第三腿自动否决，也不被宣布静态稳定。全身动作造成的初始离地可被记录，仍须随后满足真实净空、越沿、放置；越沿前 GROUND 会撤销当次资格。阶段转换非 done，合格的下游在途动作可连续接管。

精确角动量、完整接触 wrench/冲量及笛卡尔可达性未证明，均不伪造。另发现一个不参与硬谓词、reward 或 324 编码的诊断字段限制：`target_contact_reaction_possible` 目前只确认 bearing GROUND/TOP 接触，可能漏报侧壁反作用。旧记录的 false 不能解释为任何接触反作用都不存在；该诊断问题不是本次训练失败的已证实原因，未在固定训练块中热改生产代码。

## Recording 和 A 证据

成熟 parser 离线解析九个现有版本；v010 为 142 个 segment、12 通道持久命令。`recording_angle_and_overlap_evidence.csv` 有 **1704 行、24 列**，逐字段核对角度起终／增量、轮命令／积分和重叠时间；命令、实测和机制假说分别标记。

v010 rear preparation 的 FR knee 命令是 45.9→31.1°（−14.8°），同时存在 wheel 动作，不是绝对 −40°模板。P05 合同末 FL AIR/0 N，P06 连续轮行后出现真实 TOP/5.510 N：确认原动作依赖风险，但不将其称为所有 P05 失败的唯一原因。

原 A `20260905T0902190128552Z_g00b94fbb276a_e0ddaa746bc84bddb8e4f80506e09fac` 在 **65.3667 s、P10 WAIT_ENTRY** 未完成；RR knee 和速度不满足旧精确入口，RL 尚未完成。保留原结果，不由历史 Trial043 改成成功。视频有既有连续解码证据，但容器 duration 元数据无效，不能宣称严格发布合格。A 的 seed4001／180+64 pre-action ticks／旧评价版本与本 C 不匹配，全部标为非配对参考。

## 实际训练配置与迁移

共享 actor/critic、324 观测／12 latent actions、原 RSL、history-conditioned heteroscedastic policy、rho0.9、gamma0.9985、lambda0.99、120 Hz physics／15 Hz decisions、N1 CUDA。Rollout128，5 epochs × 4 minibatches，每4更新保存，并在最终完整更新保存。

首次新 MDP 迁移保留兼容网络、std 头、identity normalizer、RNG 和终生计数。FR knee requested-history 列210/222尺度4→6，actor/critic首层对应输入列×1.5，补偿旧未裁剪输入域的函数；不声称新动作域／任务特征下行为相同。旧 Adam 经核验后重建，初始 lr3e−5；后续正常 resume 保留 Adam，实际自适应 lr 已降为 **1e−5**。没有旧未完成 rollout 混入本版本。

四段实际新增分别1024；P04 checkpoint-policy 前缀四次均在 P02 未完成，全部改为现有合法 fresh-P01 fallback，不算 P04 接管成功。P06 四次／P10 两次最短真实 FSM 训练前缀成功接管。总计前缀 **4596 decisions / 36768 ticks** 全排除。

## 真实覆盖和结果

**新增4096 decisions / 32 PPO updates / 640 optimizer steps**。32次更新均记录参数变化及有限非零梯度，块内 actor 链连续；政策物理 ticks 共32728。记录的 native 下发、四类在回合状态写入、跨阶段 bootstrap 异常均为零；旧前缀缺少的逐项 native 明细仍标 unavailable，不补零。

| 阶段 | 政策 decisions |
|---|---:|
| P01 | 16 |
| P02 | 870 |
| P03 | 23 |
| P04 | 28 |
| P05 | 843 |
| P06 | 1224 |
| P07 | 2 |
| P08 | 2 |
| P09 | 64 |
| P10 | 2 |
| P11 | 28 |
| P12 | 516 |
| P13 | 478 |

完整分运行覆盖表 `phase_and_transition_coverage.csv` 为52行／13列，已经表格数值和预览核对。以上是标签覆盖，不代表每条任务链充分探索；准备中的 Q 已发生可快速连续接管，不能为了补齐标签数强制停留。

9个实际终止回合：**FALL5、BODY_COLLISION1、INCOMPLETE3**；另有4个非终结 update-boundary 尾段、980 decisions，不能当4次失败或成功。完整任务／后缀任务成功均为0。

- 自然 P01 采集中，两次出现 PPO 自己完成 FL 首次放置；后续全任务仍失败。教师继承的 FL placed 不计新增技能。
- P06 课程中 RR 有真实初始离地和合格 Q，但资格撤销，没有政策新增 RR 越沿或放置。P10 中 RR Q/C/P 全由前缀继承。
- P10 第二回合接管 tick7584 后，RL Q7901、C8082、P8324 均属于 PPO；进入 P13 后最终 RL 又退至前缘后方并 AIR，不能沿用历史 placed 作为当前 support。尾段 final_region/controlled 均 false，stable0 s。最大实测 wheel speed0.963672 rad/s、命令0.885082 rad/s，body线速0.096830 m/s／角速0.333735 rad/s，均未满足原收尾条件。
- 第三段 P06 实际 FR knee 最终 canonical target 约−57.50～122.97°，四轮均出现净反向命令，扩大后的范围确被使用。它们是驱动目标快照而非 measured q，不是任务成功证明。

## 重新加载自然 P01 完整评估

运行 `20260908T0029370739322Z_g2f27c6f5065a_1c6693d2963848b088c02e561fccdbad`：checkpoint127232，seed2001，无前缀、确定性、零 optimizer 更新，进程正常退出。**227 decisions / 1816 ticks / 15.133333 s，P02 INCOMPLETE_CONTROLLER_BLOCKED，成功0/1。** 不是外部3000-step窗口截断。

首未完成条件 `approach_FR`：FR已有合格抬升和净空（两项1），净空为台面上80.7078 mm；距前缘42.9243 mm、距现有−5 mm接近阈值还差37.9243 mm，approach进展0.848302635。没有 FR 越沿／放置，未访问 P03–P13。后段质量未评价，不填零或与 A 比较优劣。详细速度／动作和稳定性见 `evaluation_127232_diagnosis.json/.md`（完成后单独发布）。

## 保存、视频和继续训练

本检查块不可变 checkpoint：`outputs/ppo_transfer_roles_v1/checkpoints/history/checkpoint_step_000127232.pt`，同名 `_manifest.json`，累计959 updates／19180 optimizer steps，save/load round-trip=true。当前最新值应以 `checkpoint_last_pointer.json` 为准；后续训练会继续发布新版本，不能被本报告的旧数字覆盖。

没有创建 `checkpoint_first_full_success`、`checkpoint_improved`、`ppo_success_clean.mp4` 或 `fsm_vs_ppo_success.mp4`。旧 A 原视频完整保留，路径及技术／配对限制见 `video_manifest.json`。本轮尚未具备真实成功视频的条件，不以失败或拼接后缀冒充。

已开始同生产版本第二轮4096检查块：1536 P01 + 1536 P06 + 1024 P10。保持现有判据、reward、动作域和截止时间，正常恢复优化器；优先补 FR 接近和 RR 净空／前送经验，并保留 RL/收尾关注。最终再次从实际最新 checkpoint 自然 P01 评估，保留全部尝试分母。

中断后使用 pointer 指向的最新已发布 checkpoint、相同 commit/config 正常 resume，不再传 `NewMdpWarmStart`。PhysX接触状态不完整序列化，须合法reset；不声称逐 tick 无缝恢复。具体正在运行的ID、已完成边界、课程及恢复说明见 `training_manifest.json`。
