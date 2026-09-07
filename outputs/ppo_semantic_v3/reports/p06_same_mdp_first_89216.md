# P06 checkpoint-policy prefix：同 MDP 普通 resume 首个89216边界

## 固定实际账本

Run：`20260906T2326102309749Z_g2677995544c9_5826da63aa384e5d8f6e7ea4a025cfaf`。仅核验首个已保存 checkpoint 及 **global89089–89216**，没有读取之后的PPO采样。

|项目|源|首个实际保存|本窗口新增|
|---|---:|---:|---:|
|global policy decisions|89088|89216|128|
|lifetime PPO updates|661|662|1|
|lifetime optimizer steps|13220|13240|20|
|full_episode预算已用|35328|35328|0|
|phase_suffix预算已用|43648|43776|128|

`checkpoint_step_000089216.pt` 与 `checkpoint_step_000089216_manifest.json` 实际存在，侧车 `source_run` 绑定本run、`save_load_round_trip=true`；`rollouts/rollout_000662.pt` 也实际存在。计划4096不计为已完成。

## 普通恢复，而非再一次新MDP初始化

本次参数为 `new_mdp_warm_start=false`、P06、checkpoint_policy、offset0、phase_suffix、N1、seed1001。源/目标 runtime_content 指纹相同：

`f90ca2e4c21184cbbf014990fcdbeaff59b521785c6a331d7ec746b73620f3da`

它们对应 HEAD `2677995544c974c03d8b1e41d77375b45e323c9e`。当前是固定课程分布从P01改为P06前缀后学习，不是改物理MDP；也不是恢复旧物理场景快照。

源89088侧车所记录的参数/状态：

- actor：`733b1c1efaa2fddd8c5754f2ec94cb288f951fc0934503fb02acee51f33312bd`；
- critic：`3f7b6c62e09add07b0a4d194b71faf61ba0c9c6326ed3e6a773af9fb0657111e`；
- Adam：`fc773e2e31d72a8890aa34f34750a14d02cdd1798ea39de58743424de3f7f819`，effective LR=1e−5；
- normalizer：`c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`，identity RSL；
- policy contract：`heteroscedastic_log_v1`，官方异方差log std，324 obs /12 raw act。

**证据分层：**

1. 实际首update的 actor-before 指纹与源完全一致；冻结prefix actor指纹也相同。actor指纹覆盖整个异方差网络，包括 learned std 半头，不是将sigma重置为初值。
2. 普通 `semantic_training.load_semantic_checkpoint` 实际接线要求 fresh storage/无pending transition；官方 `runner.load(strict=True)` 恢复actor、critic、Adam、normalizers，并将实际状态指纹逐项与源侧车严格比较，不一致会在训练前抛错。其后恢复源 `training_rng_state`，没有 NewMdp 分支里的新Adam创建。底层 loader 同步源 effective LR。
3. 本窗口成功到达首optimizer和保存，与上述fail-closed路径相符。但**本代理没有读取.pt、没有捕获恢复后/更新前的critic或Adam张量、没有重新比较RNG数组**；不能把89216更新后的状态变化当成恢复前不一致。这里是源侧车、实际运行receipt与已读代码结合的证据，不是另一次tensor round-trip试验。

89216的 actor-after为 `ad8f469c6b4ac71c499c564e1fedbf0e522e7253d937f95c678503aac6898013`，critic/Adam指纹亦已更新，normalizer仍同源。首update与源LR都为1e−5。

## 冻结C真实前缀与信用隔离

只读首attempt到 `policy_credit_start`，共354条初始化/决策/结果记录，其中：

- **350个真实冻结C动作、2800个physics ticks**，所有记录 `policy_credit=false`；
- 首动作source=P01，末动作source=P05；第350个decision-end实际进入P06；
- `accepted=true`、`miss=null`、`target_first_observed_decision=350`，没有fallback；
- 开放信用时 actual/requested phase=P06，tick2800、23.333333 s，剩余任务时间176.666667 s；
- scope=`checkpoint_policy_initialized_suffix`、`from_P01_current_policy=false`；前缀不是FULL/SUFFIX PPO成功样本。

第一条PPO是 **g89089，credited decision1，physical core decision351，tick2808**；第128条是 **g89216，credited128，core478，tick3824**。所以350前缀+128PPO=478物理动作，2800+1024=3824ticks，时钟连续到31.866667 s；不存在将前缀重复加入global或当前预算的差额。

同一个core的accepted路径不调用reset/zero-action接管，仅重置信用计数和回报基准；mapper、nominal、bridge、reward/contact/观察历史延续。这个连续性有实际tick/core-count和代码依据，**并未在此报告额外重建每个native目标的跨接管差分**。

当前 `curriculum_epoch.prefix_policy_provenance` 精确绑定源89088（checkpoint SHA由侧车记录为 `b5befddddbe2e877b9df76926ba3271e17a4e0eb79a352040aae1e01acd69242`）并记录：

- `frozen_for_entire_training_block=true`；
- `independent_parameter_and_buffer_storage_verified=true`；
- `distribution_cache_copied=false`；
- 源actor与frozen_actor指纹相同；
- deterministic `stochastic_output=False`、无投影的324→12 callback。

callback使用独立冻结actor，不随主PPO optimizer更新。当前规范的provenance位于curriculum_epoch中，顶层同名字段为空并不表示缺失。每条PPO的固定curriculum_start一致；topology N1、suffix=true、peer_reset=none、task_timeout_bootstrap=false，采样标签为 `natural_P01_frozen_checkpoint_policy_prefix_then_semantic_suffix_N1.v1:P06:offset_0`。

## 首128真实PPO采样及前轮权限

一次顺序读取89089–89216所得：

- phase=P06×128；1024physics ticks；terminal0，尚无完整回合结果；
- 每行raw/old mean/old std均12维且有限，old std严格正；
- 顶层sampled raw与同次applied_audit raw请求逐值相等；
- 每条 `prefix_checkpoint_policy_data_in_ppo_storage=false`；
- native verified=1024/1024ticks，128行四类禁止in-episode state writes零计数均验证通过。

front wheel列为canonical FL/FR。raw与std为Gaussian latent无量纲，projected residual为rad/s；下表只统计**决策末**投影残差，不冒充每个120Hz tick最大值。

|前轮|raw最小/最大|old std最小/最大|projected绝对峰值rad/s|决策末abs(projected)>.6|
|---|---|---|---:|---:|
|FL|−.84371209 / +.25387138|.18292251 / .22833797|.59699441|0/128|
|FR|−.17979230 / +.80543733|.11903697 / .17520773|.74886843|17/128|

因此首个真实训练rollout已有FR请求使用超过旧.6上限的空间，不只是静态配置放大或此前probe。它是投影残差控制权限被使用的证据，不等于轮体速度达到该值、不等于接触推进有效、也不是任务改善或成功证明。FL本窗口没有越旧界不能说明其新权限失效。

生产在每步存储后严格检查 `storage.actions==sampled_raw` 和old mean/std；仅完成检查后才能compute_returns、保存rollout、update。首662更新实际发生，表明其运行内128行raw-storage审计已通过。本代理未加载rollout二进制，未做独立tensor重算。

## 实际首更新

第一条 optimizer_updates 与89216侧车 last_update一致：

- optimizer_steps20，actor_parameters_changed=true，finite_nonzero_gradient_observed=true；
- gradient norm min/max=1.005030585 / 1.414213305；
- KL=.0322993744，clip fraction=.378125，entropy=−6.038648176；
- value loss=.00338205745，surrogate loss=−.0343301395；
- effective LR=1e−5。

这只确认首128已真实优化和不可变保存，不预填后续4096、没有prefix/full-task成功混记。读取范围限定首attempt与首128；无新增Python/Torch/CUDA/Isaac、无重新hash、无生产/config/tests/master修改。报告完成后停止。

