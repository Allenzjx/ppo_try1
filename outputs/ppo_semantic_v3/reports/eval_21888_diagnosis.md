# C21888 自然 P01 确定性评估：RR 真实失败链

结论：保留原始 **WHEEL_ONLY_CLIMB** 结果。本次RR在障碍接触之后出现初始离地诊断，始终没有合格抬升资格；最后在有障碍载荷时首次越过前沿。没有RR合法cross/place或任务成功。全程只读PowerShell，未修改原始run、判定器、生产代码或checkpoint。

## 来源和实际计数

- Run：`runs/ppo_semantic_v3/validation/20260906T0610532492401Z_gcae1d6e7cfdd_c3726cd7caa64cf1ba0036ccbf1a8edc`。
- HEAD：`cae1d6e7cfdd8383fec67e8f919cc823f9e65448`；checkpoint21888，seed2001，natural P01，deterministic policy；evaluation optimizer updates=0。
- 实际 **751 policy decisions /6001 physics ticks /50.008333s**：750次完整8-tick决策，加最后1-tick终止决策。`physical_observations.jsonl` 有6002条（含tick0初始观测），不是6002个推进tick。
- 阶段决策：P01=1、P02=187、P03=4、P04=1、P05=147、P06=324、P07=1、P08=1、P09=85，P10–P13=0。P09物理窗口tick5329–6001，共673ticks。
- `task_success=false`、termination_reason=`WHEEL_ONLY_CLIMB`。之前随机训练第四回合曾合格抬升，不等于这次不同seed的最终确定性策略也能复现；两类证据不混合。

## RR initial / qualified / 撤销 / 跨线

全部6002条原始观测finite。原始front距离按实际轮bottom.x减障碍front_x=0.5213121737735307m；clearance为bottom.z减top_z=0.05m。

| 事件 | tick / 时间s | RR front mm | clearance mm | 实际接触证据 |
|---|---|---:|---:|---|
| P09首物理tick | 5329 /44.408333 | −176.190 | −50.928 | GROUND，3.528N |
| P09首AIR | 5333 /44.441667 | −174.925 | −51.098 | ground/obstacle均无；下tick即再GROUND |
| 首wall之前最高无接触净空 | 5502 /45.850000 | −103.583 | −48.942 | AIR，只有约1.058mm离地高度，未达到顶面净空 |
| 首次障碍接触 | 5551 /46.258333 | −49.683 | −49.662 | GROUND 8.964N + obstacle 14.528N |
| wall后首次GROUND解除 | 5571 /46.425000 | −49.674 | −49.816 | ground=0，obstacle仍18.357N；不是自由AIR |
| obstacle载荷峰值 | 5613 /46.775000 | −48.977 | −44.768 | obstacle 19.949N |
| 唯一initial事件 | 5815 /48.458333 | −28.186 | −10.192 | 此tick AIR；事件发生在前述wall接触之后 |
| 全P09最高无接触净空 | 5972 /49.766667 | −6.104 | −1.785 | AIR、ground/obstacle均无，但仍低于顶面 |
| 首次几何跨线/失败 | 6001 /50.008333 | +0.127 | −1.587 | obstacle 13.703N，RR load fraction=0.421888 |

- 唯一RR `whole_body_initial_clearance` 事件在tick5815：upward excursion=10.566510mm、own joint motion=0.933222deg、whole-body motion=7.018738deg。**initial不是qualified**，也不能倒推wall之前已有合法抬升。
- 全程无RR `qualified_measured_upward_lift`，因此也无RR资格撤销事件。终止history的qualified/cross/placed仅包含FR/FL，不含RR。
- 全P09包括有接触样本的最高clearance也只有−1.552856mm（tick5988，有obstacle接触）。本次没有测得RR bottom真正高于top的样本。
- tick6001当前几何记录top_geometry/top_contact=true、连续TOP仅1sample，但active_lift=false；这不是“已有合格AIR资格、仅微小负净空待完成”的分支。原始结果不重新归类为pending或success。
- 原始接触证据证明轮—障碍接触先于后来的抬升/几何跨线；仅凭这些body-pair力不额外宣称精确接触法向或全部动力学因果。首个未完成物理目标仍是RR有效抬升→无违规carry越沿→稳定落脚。

## FL 真实承载与CoM窗口

- P09开始时FL已经AIR，不能用历史placed_FL代替当前承载。tick5329–5824连续**496ticks /4.133333s**没有FL ground/obstacle接触力，覆盖RR首次wall接触及initial事件。
- 全P09673ticks中，FL有obstacle接触139ticks、无ground接触、无接触534ticks；raw contact_class使用OBSTACLE而非TOP，不能把其标签单独当作顶面资格判据。
- FL首次重新出现障碍载荷在tick5825 /48.541667s，3.007625N，clearance=+2.272022mm。终止时FL top_contact=true、load fraction=0.402397、obstacle力13.070232N；这是**终止时**的真实承载，不覆盖前面长时间卸载窗口。

| 窗口点 | CoM (x,y,z) m | base z m | FL obstacle N | 支持投影记录 |
|---|---|---:|---:|---|
| P09开始5329 | (0.597260,−0.117985,0.168268) | 0.096176 | 0 | 3点，inside，margin18.045mm |
| 首wall5551 | (0.633662,−0.131264,0.158367) | 0.085371 | 0 | 3点，inside，margin12.155mm |
| initial5815 | (0.656179,−0.114299,0.145853) | 0.049765 | 0 | 2点，polygon margin unavailable |
| 最高无接触净空5972 | (0.678546,−0.111931,0.147704) | 0.049473 | 0.585081 | 3点，inside，margin9.217mm |
| 终止6001 | (0.690727,−0.116062,0.163770) | 0.082141 | 13.070232 | 4点，inside，margin196.193mm |

P09 CoM z范围[0.145343,0.168389]m，x范围[0.597260,0.690727]m。支持投影673条中：444条有效inside、101条有效outside、128条unavailable；有效margin最低−8.822875mm。**unavailable不是outside**，终止时inside也不能掩盖之前承载变化。这里不添加新的支持/姿态成功门槛。

## 实际 nominal / mapper-native / residual 幅度

下表RR hip/knee的nominal、mapper-native、residual单位均为canonical degree；最后一列为实际写入的float32机械target radians，含原有零位/符号映射。它们与实测关节位置不是同一个量。

| tick | RR nominal | mapper-native | PPO projected residual | 实际机械target rad |
|---|---|---|---|---|
| 5329 | [0,0] | [0,0] | [−1.954346,−9.665371] | [+0.028473,+0.172449] |
| 5502 | [23.8,−37.8] | [23.8,−39.05] | [−1.733414,−8.895513] | [−0.390772,+0.840564] |
| 5551首wall | [−4.8,−37.8] | [−6.05,−39.05] | [−1.845879,−8.838946] | [+0.132172,+0.839576] |
| 5815初始离地 | [−6.9,−37.8] | [−8.15,−39.05] | [−1.828291,−8.952000] | [+0.168517,+0.841550] |
| 6001终止 | [−6.9,−37.8] | [−6.9,−30.3] | [−0.333620,−5.989539] | [+0.120613,+0.645855] |

- 上述RR controller feedback bias均为0。不能把mapper-native或最终mechanical目标简单等同于nominal+residual；原mapper跟踪及最终hard/slew仍存在。终止实际canonical drive RR为[−7.233620,−36.789539]deg；原始实测关节为[−6.564803,−41.427272]deg。
- 全P09673物理ticks的RR projected residual：hip[−2.245301,−0.333620]deg，knee[−9.829559,−5.989539]deg；最大cap使用率9.355%/27.304%。全12通道在该窗口均未达到配置cap，不能描述为actor饱和。
- P09轮残差范围FL[−0.014666,+0.001440]、FR[+0.015953,+0.035605]、RL[−0.006955,+0.022831]、RR[+0.062503,+0.076284]rad/s。nominal轮命令在5329约[.19932]×4，5502/5551全0，5815为[.3]×4，5972为[.2]×4，终止为[−.525,0,0,0]。这里只记录实际连续nominal，**不把所有后段非零轮命令自动归为已退火的P06旧prior**。
- 全6001 native audit均verified，setter/dispatch一致、实际重建一致、same-tick counterfactual一致；所有在回合写入计数0。P09全部673ticks有至少一个真实native target变化；不把某单通道因slew当tick无delta误写为完全没执行PPO。

## 边界

本报告没有改hard evaluator、重新标注历史成功、修改动作范围或加入新准入门；不把训练流程成功、历史FL placed、短AIR或单样本TOP当作整个任务完成。后续reward调整或新评估须用新的显式版本证据；本run原始WHEEL_ONLY_CLIMB保持不变。

证据：本run的 `evaluation_manifest.json`、`run_manifest.json`、`physical_observations.jsonl`、`native_tick_audit.jsonl`、`residual_and_projection_audit.jsonl`。唯一写入为本报告。
