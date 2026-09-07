# C38272 自然P01确定性评估：RR先触障碍，未获hard资格即跨线

保留正式结果：**P09 /WHEEL_ONLY_CLIMB，741决策、5923物理ticks、49.358333333s，task_success=false。**独立physical evaluator的`valid=true`表示输入/判定有效，不是物理任务成功。RR在早期摆动期间没有获得顶面抬升资格；随后先接触障碍，才出现弱initial迹象，最后AIR且略高于顶面时跨过前沿，但仍未qualified。该次没有RR合格cross/placed，也未进入RL课程。报告不重分类原结果、不修改判定或控制路径。

## 来源、完整物理终止和审计

- Run：`runs/ppo_semantic_v3/validation/20260906T0910569133558Z_g64abc5357d00_cfe4fd1de5334206935a80ff3bba87da`。
- HEAD=`64abc5357d00763419fe51c8126db8454fcf7697`；checkpoint38272 SHA256=`f8b6e19233283c0678f336072d5febffe94329c097b5662bcb1b400e5b0ddde3`，本报告读取manifest声明、不重复hash大文件。seed2001，natural P01、deterministic_policy=true，无teacher prefix，evaluation optimizer updates=0。
- `controller_task_success=false`、`task_success=false`；独立结果`termination_reason=TASK_FAILURE_WHEEL_ONLY_CLIMB`，理由为`RR crossed front without uninterrupted above-top active lift and current AIR/TOP geometry`。`window_ended_before_task_terminal=false`；末次`time_outs=false`、`terminal_bootstrap_allowed=false`，不是3000决策窗口提前截断或bootstrap成功。
- 741条policy audit推进5923ticks：前740次各8ticks，末次3ticks真实终止。全部5923 ticks native verified且记录实际target effect；四类in-episode root pose/velocity/force/gravity写入均0。5924条120Hz物理观测包含tick0，all_finite全部true、tick连续。`physical_failure_is_not_interface_failure=true`。

| policy source phase | 实际决策 |
|---|---:|
| P01 | 1 |
| P02 | 189 |
| P03 | 4 |
| P04 | 1 |
| P05 | 148 |
| P06 | 315 |
| P07 | 1 |
| P08 | 1 |
| P09 | 81 |
| P10–P13 | 0 |
| 合计 | 741 |

source P09第一决策从tick5281开始（44.008333s），其末tick5288；普通阶段接续没有reset。FR已有Q51/C1529/P1549，FL已有Q1643/C2575/P2742；这些历史字段不代表后续FL一直承载。

## RR真实时序与FL实时载荷

以下为120Hz原始`physical_observations.jsonl`与15Hz判定历史对齐。front=RR wheel bottom.x−obstacle.front_x；clearance=bottom.z−top_z，单位mm。AIR按真实ground/obstacle pair均不active；力为实际接触传感器normal_force_n，未由placed历史推算。“首wall”特指P09中首次RR障碍pair激活，当时bottom仍在前沿后方、接近地面；不是凭标签猜测顶面接触。

| 事件 | tick /时间s | RR front mm | RR clearance mm | RR接触；FL实际障碍载荷 |
|---|---|---:|---:|---|
| P09首物理tick | 5281 /44.008333 | −205.179420 | −50.712685 | GROUND 2.646016N；FL无接触0N |
| P09首AIR | 5297 /44.141667 | −200.243231 | −51.596458 | AIR；FL无接触0N |
| **首wall之前最高AIR净空** | **5357 /44.641667** | **−249.032315** | **−48.698115** | AIR；FL无接触0N |
| 最后一次GROUND | 5520 /46.000000 | −51.681724 | −50.313427 | GROUND 0.441583N；FL无接触0N |
| **P09首wall** | **5523 /46.025000** | **−49.751934** | **−50.310622** | 障碍8.268302N；FL无接触0N |
| **全回合唯一RR initial** | **5810 /48.416667** | **−15.249010** | **−4.642374** | AIR；FL障碍0.803382N |
| 最后一次障碍active | 5816 /48.466667 | −14.187750 | −4.309556 | 障碍0.399409N；FL障碍1.476249N |
| 首AIR且净空≥0 | 5896 /49.133333 | −3.476974 | +0.147970 | AIR；FL障碍1.772045N |
| **全回合最高AIR净空** | **5920 /49.333333** | **−0.500080** | **+1.133033** | AIR；FL障碍1.079576N |
| 跨线前最后tick | 5922 /49.350000 | −0.171896 | +0.973107 | AIR；FL障碍2.021398N |
| **跨线触发正式失败** | **5923 /49.358333** | **+0.001792** | **+0.846834** | AIR/load0；FL障碍1.458184N |

关键区别是先后顺序：P09早期已有真实关节摆动、短暂无接触，但没有抬到顶面；最高无接触bottom仅约1.302mm高于地面，离50mm顶面仍差48.698mm。之后GROUND→低位障碍接触→靠近顶面的AIR→跨线。RR的first initial在wall之后287ticks才出现，不是先有qualified再接触越沿。

### 为什么AIR和正净空仍不能报告hard qualification

- 全回合RR `qualified_measured_upward_lift`事件0，`active_lift.RR=false`，无资格撤销事件——不是“曾合格而日志末尾隐藏了它”。RR hard cross/placed账本也均无记录。几何上实际过线与合格的`front_edge_crossed`事件是两件事。
- 唯一initial tick5810记录upward excursion4.765339mm、own joint motion0.645251deg、whole-body joint motion8.031805deg、command motion32.172557deg。它是全身actuation关联的弱initial证据，不能替代hard资格。
- 当前共享v3规则同时要求正向lift gain至少8mm（0.5s窗口）及AIR/顶面/运动证据等；whole-body模式不是只要求RR自身关节超过2deg。正净空的5个决策末样本（5896/5904/5912/5920/5923）recent gain仅3.033705/2.926357/2.879120/2.683088/2.421781mm，虽然whole-body joint motion约7deg、AIR且顶面净空为正，仍不能将它们改记为hard lift。这里列的是决策末采样诊断，完整120Hz事件账本明确无Q。
- 最后front仅+0.001792mm，独立evaluator在此正式终止；不把该微小越线幅度扩写成“已经成功通过”，也不把当前AIR误称为终止瞬间仍在墙上承载。原`WHEEL_ONLY_CLIMB`是整段资格/接触历史规则的输出，不是“此刻wheel contact必须active”的标签。
- `soft_air_actuation_earned`在5896/5920决策末为true，终止时false；软奖励证据不是hard Q/C/P。此处不凭其变化另造资格或成功声明。

## 支撑、COM与实际命令：诊断不等于因果归属

P09入口到首wall的上述关键时刻，FL ground/obstacle pair都不active，载荷0；不能用此前FL placed2742宣称它在RR摆动期间持续承载。首wall时COM为x0.641075851/z0.153803896m，实测support margin约−3.821mm；最高AIR时COM x0.695782449/z0.154544238m、margin+26.461mm。这些只是该轨迹的实际支撑/姿态差异，不证明某一个关节命令或支撑缺失是唯一原因。

终止时共享evaluator的当前状态：RR AIR/load0；FL TOP/load fraction0.051523（1.458184N）；FR TOP/load0.454876；RL GROUND/load0.493601，RL front−170.477476mm/clearance−47.143383mm。RR虽当前top_geometry=true但不是TOP contact，且无hard历史；final_region=false、final_controlled=false。本次首个未完成环节是RR合格抬升/合格越沿，不应诊断为已执行但失败的RL课程或P13停止。

P09 RR projected residual绝对峰值hip1.196164deg/knee1.870345deg，远低于24/36deg名义残差幅度；未显示该两通道整体饱和。相同轨迹的实际测量关节在首wall前最高AIR时约hip50.145215/knee−1.383501deg，说明“无hard lift”不等于RR关节没有动作。

| 决策末tick | RR逻辑nominal hip/knee deg | projected residual | 实际native drive target |
|---|---|---|---|
| 5288 | 1.6 /0 | −0.991258 /−0.705337 | 1.858742 /−0.705337 |
| 5352 | 52.4 /0 | +0.185303 /−1.146247 | 53.835303 /−1.146247 |
| 5528 | −6.9 /−37.8 | −1.003351 /−0.068080 | −9.153351 /−39.118080 |
| 5920 | −6.9 /−37.8 | −0.960095 /+0.530322 | −9.110095 /−38.519678 |
| 5923 | −6.9 /−37.8 | −0.911578 /+0.614591 | −9.061578 /−38.435409 |

终止真实测量RR关节约−9.298422/−38.228360deg，与drive target也不相同。最后轮nominal四通道均0.225rad/s，RR wheel residual+0.137598、实际target0.362598rad/s、测量0.361554rad/s。native target含mapper、原反馈修正、硬限/速率限和独立残差组合；不是简单nominal+residual，也不是已达到的物理角度。报告不据小残差断言扩大动作必然有效、不要求历史唯一姿态。

## 与C28032：只比较观测结果，不夸大因果

两次均为HEAD64abc、seed2001、natural P01确定性评估；下表是各自独立实际轨迹，不是同状态回放。

| 项目 | C28032 | C38272 |
|---|---|---|
| 结果 | P12 INCOMPLETE_CONTROLLER_BLOCKED | P09 WHEEL_ONLY_CLIMB |
| 决策/ticks/秒 | 1203 /9617 /80.141667 | 741 /5923 /49.358333 |
| P09首tick RR front /FL load | 5281；−182.518mm /0N | 5281；−205.179mm /0N |
| RR hard链 | Q5820→C5998→P5999 | 无Q/C/P；initial5810 |
| P09最高无接触顶面净空 | +5.966mm，tick5928 | +1.133mm，tick5920（此前已触wall） |
| 后续实际限制 | RR placed后退回地面；RL Q6387后GROUND撤销6445，无RL C/P | RR未获Q即跨线终止，未到P10/P12 |
| P09 RR residual绝对峰值hip/knee | 1.931665 /9.405639deg | 1.196164 /1.870345deg |
| full task success | false | false |

能确认的是C38272这一次没有复现C28032的RR合格事件链；不能据一次结果把原因定为“新增训练损坏策略mean”或“只是探索噪声”。两者评估虽都关闭随机动作采样，进入P09时RR前沿几何已经差约22.662mm，后续全身状态也不同；训练包含在线随机rollout与更新，当前没有相同状态/同动作反事实来分解权重、入口分布、动力学累积偏差的贡献。也不能拿此前随机训练回合的RR成功子事件替代当前确定性结果。

本报告仅PowerShell轻量/流式读取已完成C38272数据并写入此独占文件；未运行Python/Isaac、未改生产/master、未提交。正在进行的current B评估没有预填结果。
