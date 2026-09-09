# 实施与训练检查点：136192

固定范围：15个已完成训练块及C6正式评估。依据既有小型摘要整理，未重扫原始流、加载Torch/checkpoint、运行测试或仿真；旧报告、顶层manifest和CSV均未修改。

## 当前结果

- 恢复点123136/927 PPO updates/18540 optimizer steps；实际新增 **13056决策 / 102 PPO updates / 2040 optimizer steps**。
- 最新固定保存：**checkpoint136192 / 1029 PPO updates / 20580 optimizer steps**。块15列出四次不可变保存及sidecar；本次文档整理不声称重新核验权重或保存重载。
- C6自然P01、确定性N1、seed2001、无前缀/optimizer：**895决策、7160 ticks、59.666667秒，P06满40秒仍未完成**。正式完整C评估 **0/6**。
- **仍无完整任务成功、成功视频或配对改善证据。** C6没有重复C5的退台FALL，但不能据此把未完成改称改善或成功。

## 生产修订、角色与迁移

| 主要提交 | 实际改变与保留边界 |
|---|---|
| `2f27c6f5065a` | 四腿角色、质量加权CoM短窗响应/接收侧空间代理、连续动作与角色奖励；当前支撑不由历史Q/C/P替代，不恢复唯一旧入口。 |
| `db991aa68103` | 相对该次直接源保持前324列，显式追加48维角色状态至372；复制旧列、新列置零后继续训练，新增列随后已有真实更新。 |
| `d7479d9fc41c` | 仅P06–P13 FL hip nominal限速150→60°/s、residual cap24→32°；其他11通道、硬物理限制、奖励、阶段判定及372维不改。 |

四角色是FR→RL、FL→RR、RR→FL（优先观察FR/RL桥接）、RL→FR（优先观察FL/RR桥接）。它们是协同线索，不是唯一固定支撑组合；AIR接收腿不提供当前承载。空间/响应代理不等于完整FK或完整Markov状态证明。

版本改变走显式迁移；134400→d7479d9同372身份映射的原始收据记录actor（含已学新增列/std头）、critic、identity normalizer及训练RNG保留，源Adam先核验再重建为lr=3e-5。origin与已用预算保留，旧未完成rollout不继承。HISTORY rho=.9、异方差std、gamma=.9985、lambda=.99保持。此后Run15为同HEAD普通续训，保留Adam等恢复状态，不再次补零或重置网络。

沿用既有 **393+247=640项相关CPU回归通过** 的收据（0失败/错误/跳过），涵盖RSL迁移/恢复及CPU mapper/headroom/目标缓冲路径；本次未重跑。指令表达能力修正与CPU正确性不能归作PPO已学会协调或真实动力学成功。

## 累计实际覆盖与分账

| phase | 已优化policy决策 |
|---|---:|
| P01 | 48 |
| P02 | 2847 |
| P03 | 91 |
| P04 | 78 |
| P05 | 2743 |
| P06 | 4751 |
| P07 | 33 |
| P08 | 47 |
| P09 | 1010 |
| P10 | 6 |
| P11 | 32 |
| P12 | 892 |
| P13 | 478 |
| 合计 | **13056** |

这些是source-phase PPO样本，不是阶段成功次数；P07/P08、P10/P11信用仍稀少。跨多个明确版本汇总，不能当成同分布对照试验。前缀历史不能计作PPO创造的后腿Q/C/P，局部放置也不能替代自然P01全任务成功。

累计 **72个终止episode**：FALL35、BODY_COLLISION29、INCOMPLETE4、HARD_JOINT_LIMIT4。范围分为全程PPO17、FSM初始化后缀50、checkpoint策略初始化后缀5。另有 **15个非终止更新边界尾段、2088决策**；它们已优化，但不计入成功/失败分母。

前缀累计 **36290决策/290320 ticks** 排除信用；69次尝试、65次接受。旧摘要中33264行缺紧凑native/state-write细项，不能以缺字段或显示0推定未作用或已独立验证。正式PPO摘要记录104184物理/native作用ticks、198次非终止handoff；native、状态写入、prefix入storage及handoff/bootstrap合同异常记录均0，但这些不是平滑运动或稳定支撑证明。

历史计划总15232、实际13056、未消费2176、rounding0。与上一快照衔接：12544+512=13056，98+4=102，1025+4=1029；未发现计数矛盾。

## Run15：当前checkpoint策略P06前缀后的普通续训

源135680→136192，P06 offset0，N1/seed1001；实际 **512/4/80**，wall1278.041041秒，执行状态SUCCEEDED仅表示训练流程完成。四次更新均记录actor变化、有限梯度及连续actor链。

6次从P01开始的冻结源checkpoint策略前缀，各353决策/2824 ticks，共 **2118决策/16944 ticks**，全部实际接受并排除PPO；不是持续调用随optimizer变化的actor来补教师信用。接管保留该真实roll-in状态，不称跨训练进程恢复PhysX内部状态。

本块512个信用样本全部P06：终止episode依次78/HARD_JOINT_LIMIT、129/BODY_COLLISION、70/FALL、79/FALL、14/FALL；另142决策非终止尾段。合计370+142=512，未采到P07–P13。当前策略前缀被接受不表示后腿课程已成功。

四次保存为135808、135936、136064、136192，各有同名manifest。最后checkpoint位于`checkpoints/history/checkpoint_step_000136192.pt`；本报告不重复读取PT或计算SHA。

## C6：实际障碍接触支持仍在，后腿approach未完成

C6 run `20260908T0747015601267Z_gd7479d9fc41c_2eaa941373dd4eeaa572d4805d297880`；阶段决策为 **2/189/7/14/83/600/0/0/0/0/0/0/0**。首未完成P06/EXECUTION，age40秒，rear_approach=.3353188817911007；正式原因`INCOMPLETE_CONTROLLER_BLOCKED`，不是外部MaxDecisions截断或独立安全中止。

| 腿 | 历史Q/C/P tick | 终末真实接触/支撑/载荷 | front/clearance(m) |
|---|---|---|---|
| FL | 1793/1969/2357 | OBSTACLE（评价TOP）/true/.380411 | +.287853/+.000525 |
| FR | 54/1536/1580 | OBSTACLE（评价TOP）/true/.085782 | +.300837/−.000569 |
| RL | 无Q/C/P记录 | GROUND/true/.126969 | −.352965/−.050015 |
| RR | 无Q/C/P记录 | GROUND/true/.406838 | −.386170/−.050069 |

两前轮真实障碍pair均active且verified；终末竖直力FL=10.656651N、FR=2.403052N，接触点z分别.050633/.050475m，当前support=true。此证据独立于历史placed；不同于C5终末FL回地面、FR AIR，但仍不能说明全程稳定或归因于某次更新。两后腿仍GROUND，前缘距离未达到要求。

物理观测valid=true、终止raw有限、base/obstacle碰撞未检出、physical termination=null。final_support_available=true，但final_region=false、controlled=false、stable_for=0；有支撑不是最终成功。末决策native映射/写入与状态写入验证为true，bootstrap=false、无有限观测fallback；只沿用原报告首尾/manifest证据，不声称全流重新审计。

全程roll RMS=.104372rad、pitch RMS=.108379rad、角加速度RMS=5.693531rad/s²。未采P07–P13及完整quality分数均null，不能填0。更长运行时间、较低全程RMS或C5→C6失败类型变化，均不能在不匹配轨迹/时长下称稳定改善。

## 奖励局限、A对照与未完成交付

`c5_p06_reward_direction_readonly.md`确认rear距离进度方向正确，前腿退台也会损失当前保持信用；局部接收空间proxy上涨可压过部分几何退步。但该固定七决策区间折扣PBRS合计−.00845113、其它family非正，**未证明倒退获得正总收益或奖励使停退最优**，也未证明它是FALL唯一原因。本次不据此改变奖励或物理标准。

A原记录仍是65.3667秒/P10 WAIT_ENTRY未完成：RR knee约−45.8583对旧入口−50.3976，回弹速度2.6107对23.5853，旧精确入口未过。A为seed4001/v2/180settle+64preaction，C为seed2001/v3/180+0，不配对；未重新运行A或用历史Recording代替当前成功。A原视频保留，既有容器duration异常不改变任务结果。

仍缺自然P01完整成功checkpoint、≤200秒正常速度成功视频、有效比较视频及配对改善结论；不能把已保存参数、前缀接受、局部Q/C/P或训练执行SUCCEEDED当成这些交付。

来源：`cumulative_through_block15/cumulative_training_summary.json`及既有CSV、`block15_summary`、`evaluation_136192_diagnosis.json/.md`、上一快照135680；均保留不改。

## 快照后补充：同新版零残差B1

B1 `20260908T0800275970113Z_gd7479d9fc41c_e857a8e4844d4ea984ff8f4e385d3d19` 已完成：自然P01、seed2001、零residual、无checkpoint/optimizer，758决策/6057ticks/50.475秒，P09 **WHEEL_ONLY_CLIMB**，不是成功，不计C的0/6分母，也不是重跑原A。阶段决策2/189/4/1/146/323/1/1/91/0/0/0/0。

`B1_task_failure_diagnosis.md/.json`的固定P07–P09窗口确认：RR全身初始离地在5993被接受，但从未取得硬Q、也无GROUND撤销；最高AIR在6016，底部仍低于top约.886mm。6057首次center越沿时真实TOP接触成立、load=.503990，但不能补足此前缺失的Q/C/P。现行规则命中不证明轮驱动是唯一物理原因；未找到需要改判或修传感链的证据，未放宽标准。

B1全程roll RMS=.135179rad、pitch RMS=.108117rad、角加速度RMS=4.350216rad/s²。B1与C6虽采用同新版评价入口，覆盖与时长不同且两者均未完成，不能用这些全程数值声称PPO更稳。B1比C6走到更后阶段也不能归作PPO收益。

Run16已从136192普通续训，计划1024自然P01决策，用于保持完整起点学习；本固定快照不累计其未完成预算。运行记录与最近已验证checkpoint以顶层training_manifest及checkpoint pointer为准。
