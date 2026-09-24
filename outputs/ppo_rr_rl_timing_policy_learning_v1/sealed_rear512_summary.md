# d7 已封存后腿 512 决策摘要

只读范围：课程 `p02_progress1536_gd7e97ee7b7e4` 最后 P07/P10 各256块；源码 `d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd`。未读取新课程、未启动 Torch/Isaac、未修改原始数据。配套 JSON 保留精确端点、源时钟及误差。

## 真实学习计数与物理结果

两块均是 **训练预算正常封存**，不是越障成功；合计新增 **512 policy decisions / 4 PPO updates / 80 optimizer steps**。两块均无任务终止端点、无完整成功，普通跨阶段未 done，四个完整128 rollout尾均保留非终止 bootstrap。

| 已封存块 | 有效决策 / 更新 | 实际阶段样本 | N+0 连续前缀，零学习信用 | 末端状态 |
|---|---|---|---|---|
| P07 | 222721–222976; 1706/1707 | P07=1, P08=1, P09=254 | 645 decisions，入口 t5160/43.000s | P09/60.067s，RR gap 41.242mm |
| P10 | 222977–223232; 1708/1709 | P10=1, P11=1, P12=254 | 767 decisions，入口 t6136/51.133s | P12/68.200s，RR gap 46.608mm |

所有接触/资格计数均为 **256个15Hz决策末端传感器样本**，不是完整120Hz接触时长。unloaded仅表示本拍有效归一化载荷≤0.2；不代表已合法抬升。

| 块 / 腿 | current lift valid | AIR | TOP / 当前RR承载 | crossed / placed历史 | unloaded | 末gap |
|---|---:|---:|---:|---:|---:|---:|
| P07/RR | 220 | 235 | 0 / 0 | 197 / 0 | 196 | 41.242mm |
| P07/RL | 0 | 0 | 0 / 不适用 | 0 / 0 | 0 | -50.401mm |
| P10/RR | 247 | 216 | 40 / 40 | 256 / 256 | 239 | 46.608mm |
| P10/RL | 0 | 1 | 0 / 不适用 | 0 / 0 | 9 | -50.032mm |

- P07：RR已形成有效抬升和越沿，但 **TOP与当前承载均0**。合格AIR且进入顶部XY的最小gap为 **24.958mm**，末端仍悬空41.242mm；未完成可接续捕获。全窗口原始最小gap包含起始地面状态，不能当作顶部捕获进展。
- P10：RR的placed在 **N前缀 t6133** 已获得，早于PPO入口t6136；256个placed历史值不能记作PPO新捕获。学习期间40个TOP/承载端点是间歇分布，最后TOP在 **54.267s/t6512**，其后末端AIR gap46.608mm。
- RL：P10块9个低载端点、仅 **51.733s/t6208** 一次AIR，但该拍current_lift_valid=false、未越沿；其余255端点GROUND。两块均未形成合格RL摆动、越沿或TOP。不能把低载或短AIR称为RL越障。

## 首次观测到 RR 掉载

首次端点掉载位于 **(51.933333s,52.000000s] / (t6232,t6240]**；不从15Hz数据捏造精确首次120Hz断触拍。P09 late此前已在t6114启动。

| 端点 tick / s | RR / force / gap | P09源tick | P12 wheel / RL joint tick | 端点依赖状态 |
|---|---|---:|---:|---|
| 6232 / 51.933333 | TOP, 5.671N, 0.004082mm | 767 | 81 / 79 | RL prep；允许当前支撑转移 |
| 6240 / 52.000000 | NONE, 0.000N, 0.036143mm | 775 | 89 / 83 | recapture；P12 holding_RL_joint_lane |
| 6248 / 52.066667 | NONE, 0.000N, 0.002752mm | 783 | 97 / 87 | recapture；P12 holding_RL_joint_lane |

P09/P12主源tick来自当拍diagnostic；RL独立源tick由**下一请求中的同一端点观测**按既有200s/120Hz归一化逆解，没有重算仿真。79→83→87并非全窗冻结：中间物理拍可能瞬时触回，不能仅凭末端AIR判断每拍许可错误，也不能声称已逐拍证实暂停。末端P12明确失去转移许可且RL不是合法ongoing AIR。

该窗口FL/RL已派发source nominal始终为FL[-18.5,-31.4]°、RL[0.5,35.3]°；P12 wheel source为四轮−0.3，不是P09旧FL−1.07的无限延长。P09仍继续源时钟。**源cursor暂停不等于撤销已派发的远端servo目标**，具体限制见 `drafts/cooperative_prep_v4/P09_drop_load_remaining_issue.md`。这些数据不能单独证明所有gap变化的因果。

## 原始采样、共享核与 optimizer/GAE

- 512/512：request与decision的原始raw、old log-probability、mean和sigma逐项完全相等；单forward/单sample、无额外draw，rear-assist14槽全0，rear_task_assists_enabled=false；两个prefix-storage标志均false。
- 仅用标准库从日志的当前网络head、该样本HISTORY中心、已记录parent/rear倍率复核旧共享核：mean=.9×history+.1×network mean；sigma=learned sigma×.25×该观测有效倍率。不是加载模型、不是用新版v4核重解释旧样本。
- 四个更新共80个minibatch，128个样本各被使用5次；保存的old logp与原采样逐项完全相等。标量Normal logp与float32日志最大绝对误差 **2.887e-6**；当前sigma最大误差5.029e-7，mean最大误差1.085e-6；所有head梯度有限，actor参数每次更新均变化。
- 此核验没有读取PT tensor文件，不能替代完整storage重载审计；它直接交叉核验这两块residual、optimizer、advantage和全部四个likelihood文件，没有额外model forward。

| update / 块 | KL mean | clip fraction | LR | raw GAE均值 | 标准化adv正 / 负 |
|---|---:|---:|---:|---:|---:|
| 1706/P07 | 0.016262 | 0.256250 | 0.00001 | 0.545328 | 56 / 72 |
| 1707/P07 | 0.021537 | 0.300000 | 0.00001 | 0.618276 | 87 / 41 |
| 1708/P10 | 0.028912 | 0.315625 | 0.00001 | 0.785581 | 41 / 87 |
| 1709/P10 | 0.028581 | 0.310937 | 0.00001 | 0.428485 | 67 / 61 |

GAE为正只表示相对critic/return的估计，不证明动作成功或已经解决RR捕获/RL转移。

## 封存来源

- P07: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260923T2341477711792Z_gd7e97ee7b7e4_d0bb8df8bc8c4cfb9f9c620a3a4e6890`；CP222976 SHA `53417f0eaf229566b6dcbb14dd22e47258dd73505fbd25209657bf58f607629a`。
- P10: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260923T2359308297105Z_gd7e97ee7b7e4_b9e3f44fc10f468b9792649c280aef23`；CP223232 SHA `2739173651e516ab19a5213e6d3206f7c970e7d76ed0f3adbde1bc95b7dc86c7`。

最终为 **CP223232**：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000223232.pt`。本摘要未迁移或更新它。下一处真实阻碍仍是RR可持续顶部捕获/保留，以及之后合法RL卸载；不是缺少optimizer更新，也不是完整P01成功。
