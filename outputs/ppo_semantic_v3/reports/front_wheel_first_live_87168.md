# Front-wheel cap revision：真实首次 128 更新与 87168 边界

## 结论与固定范围

本次训练 run：

`runs/ppo_semantic_v3/train/20260906T2301404237753Z_g2677995544c9_2288cdc238fb40bf96f60d1b04de349d`

已实际出现首个完整更新 **87040→87168，PPO 645→646，lifetime optimizer steps 12900→12920**，即本次新增 **128 policy decisions / 1 update / 20 optimizer steps**。对应不可变历史 checkpoint 及侧车已存在，侧车 `source_run` 精确指向本 run，`save_load_round_trip=true`；不是把计划 2048 预填为完成。报告只消费本 run 第一条更新和 audit 前128行（87041–87168），不评估后续训练结果。

这是 **P01 / full_episode / N1 / seed1001 / NewMdpWarmStart**。目前是全程 PPO，不是 suffix，也没有实际 teacher/prefix credit。启动参数中的默认 `prefix_source=frozen_fsm` 在 P01 不激活前缀，不能仅看该默认参数误称教师课程。

## 真实初始迁移

源为 `checkpoints/history/checkpoint_step_000087040.pt`，源侧车记录 checkpoint SHA：

`66fdb6b0ca8be83c31566a0aa6701834d5d8159d4614b4f43251b00a51388729`。

新初始发布：

`checkpoints/history/checkpoint_initial_v3_from_000087040_s66fdb6b0ca8b_g2677995544c9_f90ca2e4c21184cbbf014990fcdbeaff59b521785c6a331d7ec746b73620f3da.pt`

初始侧车记录 checkpoint SHA：

`8f749455697b3e379af32cb2aaa8632b4bc5937b9cbcda7b94b7fbb2b9ac12a6`。

这些是本次只读取得的**已有侧车指纹**，没有重新 hash .pt 或加载 tensor。

|项目|源87040与新初始的实际记录核对|
|---|---|
|Actor全部参数|相同：`ded137cc5567b5df36678c25b1d28587d08bb67e23da9c0d017dd001e9fad164`|
|Critic参数|相同：`3ab4ca3d4a901d9f366a44352f82287f1aed07899bb5f120a1401335d7f50e22`|
|Normalizer状态|相同：`c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`；identity RSL|
|Policy contract|完整JSON相同：`heteroscedastic_log_v1`、官方 HeteroscedasticGaussianDistribution、log std、324 obs / 12 raw act|
|RNG|源与新初始的完整 `training_rng_state` JSON精确相同（Python、NumPy、Torch CPU/CUDA状态）；没有输出大状态数组|
|Adam|源 state指纹 `a5f518ca…` → 初始 `ebddf13d…`；初始 LR=3e−5，不继承源 moments|
|Lifetime / v3 origin|87040 / 645 / 12900 保持；origin=10112|
|已消费课程预算|full_episode=33280、phase_suffix=43648、smoke=0 保持，不因迁移清零|
|初始化 save/load|侧车 `save_load_round_trip=true`|

此 policy 的 learned std 在 actor 的异方差半头权重/bias 中；上述 actor **全 named_parameters** 指纹相同覆盖 std 半头，并非重置为某个标量 sigma。没有独立声称本代理做了前向 std 张量等价试验。

`new_mdp_warm_start.json` 明确 `old_rollout_buffer_inherited=false`、`physical_state_inherited=false`。实际生产接线 `semantic_training.py:_load_v3_warm_start` 先要求新 storage=(128,1,12)、step0、transition.actions=None；验证源网络/optimizer/normalizer后创建**新 Adam(lr=3e−5)**，再恢复源 RNG。本次初始指纹和已完成首更新与此路径一致。无旧 rollout 数据作为本次样本，也不将初始保存或此前 probe 计作新 optimizer step。

## 只有 execution profile 改变；当前拓扑不残留 suffix

迁移记录的 `runtime_changed_files` 只有：

`configs/ppo_semantic_v3/execution_profile.yaml`

其 source/target 指纹为 `ad857772…` / `32cdef63…`；observation/action schema、reward、stage task spec、quality配置指纹不变。当前 profile 明确 P06–P13 的 FL/FR wheel cap 为1.2，RL/RR仍.6；P01–P05四轮仍.3，原轮速硬界和 residual slew 不变。

初始以及87168侧车均为：

- `sampling=implemented_reset_sampling=P01_full_task_only_initial_version`；
- `curriculum_epoch.prefix_request=null`；
- 顶层及 `execution_topology.phase_suffix_curriculum_implemented=false`；
- topology N1 / 324 / 12 / `peer_reset=none` / `task_timeout_bootstrap=false`；
- 顶层与当前 curriculum_epoch 都没有非空 `prefix_policy_provenance`。

初始侧车的 `source_run` 仍保留旧源运行路径作为加载来源；不能把这个来源字段当作本次执行拓扑。87168的 `source_run` 则是当前 run。源 checkpoint 原来是 suffix=true，但当前实际 curriculum/topology 已正确切换为 P01。

实际 t0 same-state action comparison 记录 phase=P01、324维观测、零 residual history，所有 old/new bounded/scaled/rate/applied 差均为0。这符合本次只放大 P06 以后前轮范围，而非证明后腿阶段改动无效；它也不是 native dispatch/物理后续轨迹比较。首条实际采样的 old actor mean 与该 t0 deterministic mean 相同，随后 raw 是随机采样，没有拿确定性比较代替 PPO动作。

## 首128样本与实际 raw-storage 接线

一次顺序读取 audit前128行所得：

|事实|结果|
|---|---|
|Global范围|87041–87168，严格连续128行|
|Source phase histogram|P01=1、P02=127|
|首/末物理端点|tick8 / .0666667s → tick1024 / 8.5333333s|
|物理 ticks|1024；terminal=0|
|raw / old mean / old std形状|每行12维；raw、mean、std、old logprob/value/reward全部有限|
|std|每个分量严格正；全128范围 .1015476733–.2255406380|
|同次动作绑定|每行顶层sampled raw12与applied_audit请求raw12逐值相等|
|Native audit|1024/1024 ticks verified；own-phase request effect1023ticks（有一次阶段交接首tickhold）|
|state writes|128行均验证四类禁止in-episode写入为零|
|prefix|无curriculum_start或checkpoint-prefix数据字段；首条即decision1/tick8|
|本窗口 front projected residual峰值|.180619889 rad/s；尚未到P06，不能宣称本窗口已使用>旧.6权限|

首条实际 old std：
`[.1664818972,.1623278856,.1381482184,.1390271634,.1452665180,.1498678476,.1618425399,.1611497402,.2037643641,.2075944096,.1399070919,.1566139013]`。
它是这条观测上的异方差输出，不是全阶段固定sigma。

`rollouts/rollout_000646.pt` 实际存在。本代理未读二进制；raw-storage结论分两层：

1. 上述128 JSON行给出实际sampled raw、old mean/std/logprob/value及同请求绑定；
2. 生产 `semantic_training.py:740` 起，在每次 `process_env_step` 后严格 `torch.equal(storage.actions[tick], sampled_raw)`，随后同样核对storage old mean/std，失配会在compute_returns/update前抛错。实际646更新及保存已发生，说明此128样本通过运行内核对。

这不是本代理额外加载 .pt 后重新逐tensor比较的声明。

## 已完成首更新及保存

第一条 `optimizer_updates.jsonl` 与87168侧车 `last_update` 一致：

- optimizer_steps=20；actor_parameters_changed=true；finite_nonzero_gradient_observed=true；
- actor before=`ded137cc5567b5df36678c25b1d28587d08bb67e23da9c0d017dd001e9fad164`；
- actor after=`d961cb8311698e2ccb3d0d2f6f2b386df028d62134dab6a0817f0ab25464e3e6`；
- recorded gradient norm min/max=1.002126614 / 1.374856799；
- KL=.019298862，clip fraction=.3203125，entropy=−5.437122583；
- value loss=.001766061，surrogate loss=−.042470780；
- 首更新最终 adaptive LR=1e−5。这是更新后记录，不应误报初始Adam未用3e−5创建。

87168侧车 actor after与更新一致，critic指纹变为 `56d29ff4c0fc4905f7002bd6e5b65a292ad912b7e84eeacbeeeb44b9e3f10a47`；normalizer指纹不变。预算为 full_episode33408、phase_suffix43648、smoke0，增加的128只计本次full_episode。

保存路径为 `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000087168.pt` 和同名 `_manifest.json`。这是实际首个不可变保存边界；没有依据后续计划推完整2048已完成，也没有任务成功/稳定性改善结论。此前独立front-authority probe不在这128行或本次新增PPO/optimizer账内。

仅小型JSON侧车、代码及前128 audit行的只读PowerShell检查；未重hash、未tensor load、未运行Python/CUDA/Isaac或修改生产/config/tests。报告写完即停止。

