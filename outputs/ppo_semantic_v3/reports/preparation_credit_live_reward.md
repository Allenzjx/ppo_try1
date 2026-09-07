# Preparation credit：固定1024决策边界的P06真实reward核验

## 本次追加边界：update336 / global47488

先确认 `optimizer_updates.jsonl` 已存在唯一的 ppo_update336/global47488，随后只读取本轮前1024条完整决策（global46465–47488）。本轮已完成8个updates、合计160 optimizer steps；第336次记录actor_parameters_changed=true、finite_nonzero_gradient_observed=true。没有读取raw尾部来充当优化计数，也没有把计划50560写为已完成。

以下为同一真实P01回合的截至边界数据：episode tick8192/68.266667s，尚无终止；后续训练保持不变。全部操作仍为PowerShell及本报告，未运行Python或修改生产。原首128决策记录完整保留于文末附录。

### 实际phase覆盖与P06完成

| 来源phase | 已优化决策数 |
|---|---:|
| P01 / P02 / P03 / P04 / P05 | 1 / 176 / 4 / 1 / 147 |
| P06 | **367** |
| P07 / P08 / P09 | 3 / 1 / 324 |
| 合计 | **1024** |

真实跨阶段记录：P01→P02 tick8；P02→P03 1416；P03→P04 1448；P04→P05 1456；P05→P06 **2632/21.933333s**；P06→P07 **5568/46.4s**；P07→P08 5592/46.6s；P08→P09 5600/46.666667s。

P06自己的367个决策是global46794–47160，首决策末tick2640，最后tick5568。最后一条真实 `stage_transition_evidence` 明确 `reason=current physical goal set satisfied`、`completion_values.rear_approach=1`；该条terminal=false、time_outs=false。它不是阶段被当作episode结束，也不是给一次phase标签就算成功。

在tick5568，RL/RR front分别 **−.219808009 / −.193440225m**，两者满足既有workspace；RL自身Q/C/P仍false。此前实际FR/FL placed为1447/2630，后腿approach使用的是实测状态。这个事实只说明本条训练轨迹通过了P06，不归因于新增信用，更不称全任务成功。

### 对1024样本完整重建；P06差值只有RL workspace

继续使用实际每条physical_evaluator的几何、support/load、initial与hard history，独立重建旧ordered公式和新增blocked-leg workspace，不通过“new减extra”冒充重建旧式。1024条均非终止、RR/RL均未placed，因此无需虚构allplaced finish状态。

| 核验 | 最大误差 |
|---|---:|
| 保存新Φ −（独立旧orderedΦ + 唯一新增preparation项） | **1.21e−16** |
| 保存F − `5*(.995*Φ_after−Φ_before)` | **0** |
| 五family和与stored float32 reward | **1.38e−8** |

P06期间FR/FL已placed，RR前置集合已满足但自身未placed，RL前置集合仍缺RR。故在这些真实样本中精确为：

`Φ_new − Φ_old_ordered = .02125 * workspace_RL`。

没有新增RR的重复workspace，也没有提前加RL unload/initial/lift/carry/capture。P06有11个决策末RL initial=true，但RL qualified/cross/placed全部0；即使出现已有initial诊断，新分支仍只取workspace，其余ordered项继续排除。

| 样本 | RL front m | workspace_RL | 旧Φ | 新增Φ | 新Φ |
|---|---:|---:|---:|---:|---:|
| tick2640 / global46794，P06首决策末 | −.56473450 | 0 | .44602174 | 0 | .44602174 |
| tick3480 / global46899，首个正preparation决策末 | −.46872633 | .00509468 | .44759009 | .00010826 | .44769835 |
| tick5568 / global47160，P06→P07 | −.21980801 | 1 | .4675 | .02125 | .48875 |

远处workspace为0来自未修改的既有.25m渐变范围，不是新造的可达性阈值。全1024 native audit验证8192/8192 ticks；P06验证2936/2936，其中own-phase-request-effect2935（仅排除该阶段首handoff tick），没有把日志信号与控制请求完全脱节的证据。

### 真实接近与退回：旧Φ不变的隔离对照

P06内部366对相邻决策末：新增Φ上升208对、下降54对、持平104对。这里“上升/下降”先指实际几何potential差，不强求含折扣的F每次同号。

| ticks / 后样本global | RL front mm | 新增Φ变化 | 新增F=`5*(.995E后−E前)` | 旧Φ变化 | 实际总F |
|---|---|---:|---:|---:|---:|
| 5552→5560 / 47159，接近 | −222.380→−220.156 | +.000189012 | +.000414142 | **0** | −.011273358 |
| 5488→5496 / 47151，退回 | −229.735→−229.893 | −.000013380 | −.000577130 | **0** | −.012264630 |
| 5424→5432 / 47143，小幅接近 | −225.489→−224.850 | +.000054289 | −.000249498 | **0** | −.011936998 |
| 5240→5248 / 47120，较大退回 | −231.930→−242.923 | −.000934418 | −.005154628 | −.018825328 | −.110399331 |

前三对旧orderedΦ两端都为.4675。RR位置和RL载荷仍实际变化，但RR仍处workspace/旧unload饱和状态，所有旧项合计不变，所以可以直接辨认新增RL依赖：原来被排除的RL准备进展现在改变global Φ，而不仅改变P06 completion显示。

第二对RR front从−.206967→−.217856m，仍在workspace；RL非AIR、load .53723→.48211。这不是用历史placed或把AIR自动罚分制造差值。第四对旧项也下降，不能把该次全部惩罚归因于RL preparation。

第三对小幅接近虽然E提高，但`.995`折扣使新增F仍负；相对于保持同E，其reward变化仍为 `5*.995*ΔE>0`。这不是反号bug，也不应通过去掉折扣来强求“每次前进reward都正”。同理第一对总F仍受旧Φ的普通折扣影响为负，新增项却提高了该步的task信号。

5552→5560后样本的weighted body/contact/smooth分别−.00058881/−.00000106/−.00281324，实际task family为−.01260669；5240→5248后样本分别−.00053580/−.00000884/−.00328880，task为−.11173266。质量成本和时间项没有被prep项替换，也没有将prep再追加成第二个reward family。

### 当前真实结果与归因边界

在固定47488边界，当前stage=P09；RR实际qualified事件6080/50.666667s，之后在6342/52.85s记录 `qualification_revoked_ground_before_cross`。边界末RR front−.11929765m、clear−.05053438m、active=false；RL front−.23243493m；RR/RL仍无cross/placed，episode尚未终止。

因此能确认的是“新RL workspace依赖已进入真实P06 reward并完成优化，且这一训练回合已经真实通过P06”；不能确认“这次通过由新增reward单独造成”，更不能把进入P09或曾有RR qualified称为任务成功。与C46464相比，当前是训练中的随机策略、不同seed且多次在线更新，不能当配对因果实验。

既有hard任务/history代码未变，Q后GROUND仍实际撤销未完成资格；P06→P07不done、不bootstrap特殊化。此次只报告已完成optimizer边界，不为继续训练增加门禁或修改当前运行。

## 附录：首个128决策边界原始核验

2026-09-06。当前Isaac训练保持固定运行；本次仅PowerShell读取与报告写入，未运行Python/Isaac、未改生产/参数、未提交。未修改 `eval_46464_diagnosis.md`。

### 固定证据边界

run：`runs/ppo_semantic_v3/train/20260906T1133239688401Z_g9d70aae58243_1e3e34207a9442c387f5123a55a5e121`；runtime `9d70aae58243b9fb2248d64561959e6a67901b44`。来源checkpoint global46464，N1、seed1001、真实P01 reset。

本报告只纳入第一条已完成 `optimizer_updates.jsonl`：**ppo_update329 / global46592**，即128个实际credit决策（global46465–46592，episode tick8–1024）。该update记录20 optimizer steps、actor_parameters_changed=true、finite_nonzero_gradient_observed=true；gradient norm范围1.00445–1.41421，实际学习率1e−5。

128条phase来源为P01:1、P02:127；这个选定边界**尚无P06样本**，以下是实际早期FL preparation，而非提前声称后腿修复。后续已生成但不在这个边界内的数据均未纳入。4096是计划块，不把50560写作已完成global，也不等待全任务成功才认可真实optimizer更新。

### 版本和公式确切绑定

`new_mdp_warm_start.json` 记录：source_global=46464，v3原始origin仍10112，保留既有v3预算和lifetime；old_rollout_buffer_inherited=false、physical_state_inherited=false。本轮有明确任务potential版本变更，不冒充旧MDP exact resume。

配置记录中，action_schema、execution_profile、observation_schema、quality_score、reward_config旧新SHA一致；stage_task_spec从 `edc5e593a2c14b0b61e9c1422769c864e891e5405386bc38d40bece83c34e057` 变为 `094bdadf3f1954c40bdff70fbe34ac754003081ff5872f4d300bb9155fa04044`。本次单独计算当前stage文件SHA，与target记录精确匹配。

显式新标志为 `preparation_credit_semantics: current_workspace_before_predecessor_placement`，revision为 `continuous_whole_body_v3_current_workspace_preparation_credit`。

git 49749→9d70aae 的supervisor改动仅增加这一模式的校验和 predecessor-blocked 分支：若某腿尚未placed且其前置placed集合尚未满足，原来的0改为 `.1*workspace(leg)`，随后立即continue。**该分支没有增加unload、initial、lift、carry或capture；硬initial/Q/C/P、普通阶段完成和nominal执行代码未变。** reward、observation、backend与execution_profile的此区间diff为空。

因此真实非终止状态应满足：

`Φ_new = Φ_old_ordered + .85/4 * .1 * Σ workspace(leg)`

求和仅遍历“未placed且predecessor集合尚未满足”的腿，既有workspace横向条件、区间[−.22,+.06]m及区间外.25m比例不变。它不是额外奖励family，也不是提前发放历史完成位。

### 对128条真实metadata独立重建

用每条 `applied_audit.semantic_task.physical_evaluator` 的当前几何、support/load和history，在PowerShell重建完整旧ordered公式（含既有current-gap lift、carry、capture）及上述唯一新增项，再与保存的 `reward_breakdown` 比较。没有用 `new_phi−extra` 自称独立计算旧式。

| 核验 | 实际最大误差 |
|---|---:|
| `Φ_new_logged − (Φ_old_reconstructed + preparation)` | 3.12e−17 |
| `potential_shaping − 5*(.995*Φ_after−Φ_before)` | 0 |
| 五个已加权family之和与stored float32 reward | 3.85e−9 |

128条中FR均尚未placed；FL因此仍被predecessor阻挡。RL/RR离workspace过远，workspace=0，没有新增信用。**全部新增信用均来自FL自身当前workspace**：

| 样本 | FL front m | FL workspace | 重建旧Φ | 新增Φ | 保存的新Φ |
|---|---:|---:|---:|---:|---:|
| global46465 / tick8 | −.27219435 | .79122259 | .03792521 | .01681348 | .05473869 |
| global46592 / tick1024 | −.16100800 | 1 | .15078222 | .02125 | .17203222 |

这一边界FL initial/Q/C/P均未取得；FL末尾load约.5061，不是已经unloaded的腿。记录的唯一qualified历史是FR tick44；cross/placed历史为空。新增FL preparation并未把它伪装成抬升或放置成功，也未给予其现有load水平另一份unload信用。

### 实际接近与退回：增量只计一次

对同一回合127对相邻决策末样本，新增preparation potential升高53对、降低9对、持平65对。以下用同一对实际状态计算新增项在reward里的独立贡献：

`F_preparation = 5*(.995*preparation_after−preparation_before)`。

这是总potential-based shaping的代数拆分，不是追加到日志reward的第二项。第一条reset→decision的新增项拆分没有另造reset状态；127对比较从已有两条真实状态开始。

| ticks / 后样本global | FL front mm变化 | 新增Φ变化 | 新增F | 总F / 实际task family |
|---|---|---:|---:|---:|
| 432→440 / 46519，接近 | −228.565→−225.963 | +.00022123 | +.00058759 | +.00520983 / +.00387649 |
| 320→328 / 46505，接近 | −249.568→−247.063 | +.00021289 | +.00059072 | −.00302774 / −.00436108 |
| 16→24 / 46467，退回 | −272.846→−273.936 | −.00009259 | −.00087958 | +.09649777 / +.09516444 |
| 56→64 / 46472，退回 | −273.428→−274.463 | −.00008795 | −.00085525 | −.00898131 / −.01031464 |

新增项方向正确：这些真实接近给正的独立preparation贡献，退回给负贡献。但不能要求总reward与FL单独运动始终同号：例如16→24旧ordered Φ另增.01976370，其他前腿任务进度盖过FL退回的损失；320→328旧Φ略降且还有折扣。没有把总reward正号错误归因于preparation奖励退回。

到workspace饱和后，恒定新增Φ=.02125只产生既有折扣项 `5*(.995−1)*.02125=−.00053125`，不每帧重复发正的准备奖励。

几个对应的实际weighted质量成本：

| 后样本tick | body | contact | smooth |
|---|---:|---:|---:|
| 440 | −.00041471 | −8.11e−8 | −.00249442 |
| 328 | −.00073439 | −.00000104 | −.00242284 |
| 24 | −.00003273 | 0 | −.00247218 |
| 64 | −.00035702 | 0 | −.00259741 |

这些成本未被prep项替换，实际task family还包含时间项。潜在旧项、prep项、质量项及总reward均应分开理解；本次没有改family权重。

### 可确认与不能确认

已确认：新增的唯一workspace项已经进入真实rollout reward，并被一次完整官方PPO update消费；真实当前位置的接近/退回具有相反的该项差分；没有新增提前lift/unload/carry或硬历史完成信用。

尚不能确认：P06及后续后腿准备的实际样本效果、固定mean下的物理成功、相对C46464的性能改善。这个首128边界只证明信号正确接线与真实优化发生，不能证明算法已经学会利用它。后续可在另一个完整update边界追加P06证据，不需将任务成功设为继续训练的门禁。
