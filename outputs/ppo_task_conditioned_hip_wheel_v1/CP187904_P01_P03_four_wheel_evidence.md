# CP187904：P01–P03 四轮完整通路证据

结论：这次真实确定性评估的 **P02 不是“只有 RR 有轮指令/转动”**。194 个前送端点中，四轮目标与实测 canonical qd 全部非零（均 |值|>0.01 rad/s）。RR 的残差反而平均 **−0.126913 rad/s**，其他三轮没有被取消。不能由这次新录像反推用户更早录像的具体视觉成因，但“RR 轮端更明显移动”不能作为 RR 单轮驱动的证据。

来源：CP187904 det 自然 P01，run `91d8d06aaced49ebb0ad9d5a5c067b31`；独立 eval，optimizer updates=0。读取封存 P01–P03 的 1624 个 120 Hz 派发和 203 个决策端点，不读取正在运行的 B，不作新仿真或生产修改。正式全通道 PPO 与旧独立 mask 干预不混称。

## 参考与当前 shared N

冻结 motion contract 的 P01 原始原子组是 `wheel all 0.300`（源时刻 .333333 s），直到 13.266667 s 的四轮显式 stop。源 P02 只改 FR knee，不等于当前连续调度器应把其余通道归零。当前 **同次执行记录**：

- 关联 episode post-step tick 1–40：N=[0,0,0,0]；
- tick **41–1592**：N=[+.300,+.300,+.300,+.300]，覆盖当前 P02；
- tick **1593 起 P03**：N=[−.790,0,+.51911,0]。

P03 原始组 RL=.610，经既有冻结 FSM logical-command 修正 −.149 得 .51911；这是 shared N 的既有源调节，不是 PPO 残差或相似度验收门槛。FR/RR 的 N=0 来自合法 stop/owner 交接，不能指认为 mask。上文 N 是此 PPO 轨迹上的共享源建议，不是另一条 zero 物理轨迹；新同版本 B 尚未封存时不声称已做配对。

## 同拍四轮表

轮速均为 rad/s，u 是无单位 policy raw（含现有 HISTORY）；记录的四轮 phase residual mask 全为 **1**，作用于 additive residual，不乘掉 N。Δ 为同次派发记录的 effective residual，并与该拍限速/投影后的 REQUEST 相等；不是独立重算 N 后相减。P01/P02 四轮 residual cap=.6，P03 FL/RL cap=1、FR/RR=.6。

tick 为 episode **步进后**时钟：16=.133333 s；400=3.333333 s；1200=10 s；1600=13.333333 s。原生 dispatch tick 分别195/579/1379/1779（另有 settle/reset 偏移 +179，不混用两个时钟）。

| tick / 阶段 | 轮 | N | u | Δ | final canonical | final native buffer | measured native / canonical qd | 接触 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 / P01 | FL | +0.00000 | +0.00728 | +0.00437 | +0.00437 | −0.00437 | +0.02039 / −0.02039 | GROUND |
| 16 / P01 | FR | +0.00000 | +0.01377 | +0.00826 | +0.00826 | +0.00826 | +0.00831 / +0.00831 | AIR |
| 16 / P01 | RL | +0.00000 | +0.00636 | +0.00381 | +0.00381 | −0.00381 | −0.00367 / +0.00367 | AIR |
| 16 / P01 | RR | +0.00000 | −0.03659 | −0.02195 | −0.02195 | −0.02195 | +0.00249 / +0.00249 | GROUND |
| 400 / P02 | FL | +0.30000 | +0.03499 | +0.02098 | +0.32098 | −0.32098 | −0.19899 / +0.19899 | GROUND |
| 400 / P02 | FR | +0.30000 | +0.09744 | +0.05828 | +0.35828 | +0.35828 | +0.35917 / +0.35917 | AIR |
| 400 / P02 | RL | +0.30000 | +0.01028 | +0.00617 | +0.30617 | −0.30617 | −0.38982 / +0.38982 | GROUND |
| 400 / P02 | RR | +0.30000 | −0.22184 | −0.13096 | +0.16904 | +0.16904 | +0.25471 / +0.25471 | GROUND |
| 1200 / P02 | FL | +0.30000 | +0.03585 | +0.02150 | +0.32150 | −0.32150 | −0.19678 / +0.19678 | GROUND |
| 1200 / P02 | FR | +0.30000 | +0.09521 | +0.05696 | +0.35696 | +0.35696 | +0.35797 / +0.35797 | AIR |
| 1200 / P02 | RL | +0.30000 | +0.01441 | +0.00864 | +0.30864 | −0.30864 | −0.44850 / +0.44850 | GROUND |
| 1200 / P02 | RR | +0.30000 | −0.22186 | −0.13097 | +0.16903 | +0.16903 | +0.25699 / +0.25699 | GROUND |
| 1600 / P03 | FL | −0.79000 | +0.02421 | +0.02420 | −0.76580 | +0.76580 | +0.84318 / −0.84318 | GROUND |
| 1600 / P03 | FR | +0.00000 | +0.09319 | +0.05576 | +0.05576 | +0.05576 | +0.05619 / +0.05619 | AIR |
| 1600 / P03 | RL | +0.51911 | +0.01299 | +0.01299 | +0.53210 | −0.53210 | −0.44150 / +0.44150 | GROUND |
| 1600 / P03 | RR | +0.00000 | −0.22375 | −0.13205 | −0.13205 | −0.13205 | −0.11592 / −0.11592 | GROUND |

表中 native buffer 只是目标，不冒充物理角速度；native qd 由同 tick `robot.data.joint_vel` 独立记录，canonical qd 是 sensor 的轴向映射读回。FL/FR/RL/RR 的原生方向符号为 **−/+/−/+**。startup 的 joint_names 原生 DOF 顺序位置为8/9/10/11；**数值 actuator joint_ids 未在本组逐拍 audit 落盘，JSON 留 null**，不把名称索引假称已记录的 setter IDs。

最后可见写入路径为 `apply_semantic_residual → set_joint_velocity_target → write_data_to_sim`，随后读取 `robot._joint_vel_target_sim`。1624/1624 audit 的 setter/dispatch 映射一致；203 个端点同次 baseline=N、REQUEST=effective、baseline+effective=final 的最大误差均为0；native target 符号映射最大误差2.93e−8（float32）。完整源事件 owner ledger 和独立 runtime/safety mask 向量未保存，明确缺测。以上结合真实 qd 排除本窗口的固定漏轮/nominal 被 mask，不只依据“mask 全1”；不外推所有旧 run/reset。

## P02 前送全窗口：目标存在，但跟踪不等于完美

194 个决策端点 tick48–1592：

| 轮 | mean Δ | mean final | mean actual [min,max] | target–actual RMS |
| --- | ---: | ---: | --- | ---: |
| FL | +0.02076 | +0.32076 | +0.19503 [+0.05166,+0.31343] | 0.12886 |
| FR | +0.05572 | +0.35572 | +0.35630 [+0.32220,+0.36103] | 0.00091 |
| RL | +0.00710 | +0.30710 | +0.37020 [+0.27046,+0.48343] | 0.07496 |
| RR | −0.12691 | +0.17309 | +0.25644 [+0.03535,+0.37281] | 0.08750 |

第一份决策端点 tick8 已有非零策略 wheel 修正；P02 第一端点 tick24 时源 N 尚未开始前送，四轮小命令来自 residual。首个前送端点 tick48 已同时包含四轮 +.300 N 与策略修正。因此第一处分歧在**policy residual 层**，不是后层把 FL/FR/RL 丢失。P03 的源变化时间由120 Hz日志精确定位为关联 tick1593。

FL 的实测速度明显低于目标；RL/RR 有时高于目标，存在闭环/接触耦合跟踪误差，尚不能单凭这些数值把原因唯一归为负载、驱动参数或 slip。FR 在所列 P02 前送拍为 AIR，轮轴确实旋转，但无 ground/obstacle 接触力，不能算成提供地面牵引。轮中心位置/线速度独立保存在 JSON；轮端平移、轮轴自转、接触牵引分别记账。

tick16→400：base z 96.52→73.61 mm，RR knee actual −.908→−3.422°，FR 底部相对障碍顶净空 −46.44→+98.87 mm。机身下降与 FR 抬起、全身构型变化共存，不证明“只有 RR 轮旋转导致下沉”；base z 也不是机身 collider 净空。

复用既有 [P06 正式训练诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/block02_P06_wheel_window.md)：该窗口也有四轮 +.300 N 与真实转动，FL 的8个反向目标来自策略抵消、RR被减速。它是更新中的 stochastic rollout，**不是本次固定CP的B对照**。本次未确认需修复的 mask/派发错误，未改 reward、HISTORY、温度或共享控制；全部逐通道原始单位、6拍同拍证据、配置/代码哈希及缺测边界见 [JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/CP187904_P01_P03_four_wheel_evidence.json)。
