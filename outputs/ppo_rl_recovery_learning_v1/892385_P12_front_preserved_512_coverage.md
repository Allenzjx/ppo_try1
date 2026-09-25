# 892385：前段保留分支首个 512 PPO block（已封存）

分支 `cp225280_front_preserved_v1`，source CP225280；run `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260924T1412116439544Z_g892385cba8a7_791e04b05a7a4df9a493a606cd958a57`。训练生命周期 SUCCEEDED 仅表示采集、更新和保存完成，**不是越障成功**。以下轨迹全部是此次更新前源 actor 的真实随机采样，不是更新后 CP225792 的物理收益证明。

## 实际训练与保存

- **512 新 on-policy decisions / 1 PPO update / 20 official Adam minibatches**；global 225281–225792，累计 CP225792 / PPO1726 / Adam34520。
- 512 collection，gamma .9985、lambda .99；实际 update LR 1e−5。全部请求阶段 **P12=512**，其余 P01–P13=0（前段覆盖来自排除信用的真实前缀及独立保留样本，不能混入此表）。
- 2 条真实 successful_nominal P12+8 前缀，共 **1554 decisions / 12432 physics ticks**，交接均 tick6216；PPO credit=0。没有快照 teleport 或前缀样本混入 storage。
- 同拍 512 条 raw/mean/sigma/old logp 与 request 审计相等；12 通道 mask 全开、后腿任务 assist OFF、每决策一次抽样。独立 stdlib Gaussian logp 最大差 2.6142e−6。正式 likelihood **每条 raw 恰好 5 次曝光，共20 minibatches**。
- checkpoint保存重载 roundtrip=true；只读 SHA 核验与 sidecar 一致。路径：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\cp225280_front_preserved_v1\checkpoints\history\checkpoint_step_000225792.pt`。
- checkpoint SHA `17a7e849afc242856f035350d814349be1e101b49385c42ab9c90966a84fc5a7`；sidecar SHA `d42847f6c90612366bdb16ca574e11403430ac368ed487a69a0d904b73d8cc6d`。

## 两条连续学生后缀与真实任务覆盖

| 后缀 | 学生数 | 终态 | 新 RL 资格 / 继承端点 | RR 当前合法 TOP 端点 | RL 越沿/放置 |
| --- | ---: | --- | --- | ---: | --- |
| 1 | 442 | 81.266667s，P12 INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_TASK_DEADLINE | 6437新资格27端点；6212前缀资格继承8端点 | 21；GROUND后恢复0 | 0 / 0 |
| 2 | 70 | 56.466667s，非terminal预算尾 | 无新资格；6212前缀资格继承4端点，6249落地撤销 | 36；无GROUND | 0 / 0 |

两条前缀都已完成 RR qualified5427、cross/placed6133，不能归给学生。全512 RL当前有效资格39端点 = **12继承＋27学生新资格**，不是39次新卸载。第二条有2个合法资格下的边缘恢复端点（6240/6248），仍继承6212资格；是 OBSTACLE_AMBIGUOUS 接触，没有cross或TOP放置。

| 物理端点窗口（可重叠） | 全512 | 首442 | 后70 |
| --- | ---: | ---: | ---: |
| RR可达AIR准备 | 27 | 2 | 25 |
| RR实际合法顶部承载＋前侧准备 | 57 | 21 | 36 |
| 固定FR轴CoM/body正投影＋RL载荷份额下降代理 | 22 | 8 | 14 |
| RL有效资格边缘恢复 | 2 | 0 | 2 |
| RL有效AIR可捕获区域 | 0 | 0 | 0 |
| RL真实TOP承载 / placed | 0 / 0 | 0 / 0 | 0 / 0 |
| P13 / 完整任务 | 0 / 0 | 0 / 0 | 0 / 0 |

FR投影代理不是已验证的侧向转移或绝对载荷下降。输入端点对齐窗口为27/56/22/2，其余0；两次前缀交接输入缺少前一学生行，不补造。通用实测支撑和合法TOP分别统计：GROUND有力不等于已回台面。

首回合新 RL 6437抬升→6504净空峰值61.475mm，却距前缘159.412mm；6496 RR离合法XY，6641 RR GROUND，6655 RL落地。新资格首端点6440→6504累计reward −.060766。详见 `892385_first442_RL_window.md`。

第二尾端 RR合法TOP **13.484N**、前缘+104.508mm；FL/FR合法TOP 11.582/1.477N。RL当前资格false、OBSTACLE_AMBIGUOUS，gap−48.488mm、前缘−50.949mm，接触反力18.222N但未核实为bearing（bearing_force=0、bearing_verified=false）；不能说RL已抬升或承载。

首回合真实terminal包含在同一个512 rollout中，terminal rawGAE −31.729610、官方normalized advantage −3.452530；第二非terminal尾按官方compute_returns bootstrap，不伪装成终止。尾端normalized advantage +.580701只是相对整批标准化值，不代表任务成功。

## 前段 replay 单独计账

- **100 个固定训练 observation / 99 个 heldout**，索引无交集；本次 **20×32=640次训练样本曝光**，100个训练源点全部参与；heldout未参与拟合。
- 保留曝光阶段：P01 7、P02 224、P03 14、P04 7、P05 196、P06 192。这些是重复离线约束曝光，**不是640个新物理决策或PPO样本**。
- 梯度为官方PPO actor梯度＋reference KL梯度，在原norm clip之前合并、同一Adam步消费一次；额外on-policy=0、独立AUX/optimizer steps=0，部署无实时teacher。
- 额外actor forward 24 = 20拟合＋4训练/heldout前后统计；额外随机抽样0。
- 更新后训练mean KL=.007811164、heldout mean KL=**.007886615**（max .013667703）。这只约束有限保存状态上的分布漂移，**不能证明真实前段已保留或自然P01成功**。更新后视频由主任务另行评估，本报告没有读取活动eval。

检测脚本本次显式读取 likelihood JSON 的顶层 `minibatches`，避免旧流式 helper 误匹配新增 replay 内层同名字段；只做内存读取适配，没有改生产或 helper 文件。checkpoint及历史分支原样保留；本次新分支实计不覆盖旧CP231680分支档案。

