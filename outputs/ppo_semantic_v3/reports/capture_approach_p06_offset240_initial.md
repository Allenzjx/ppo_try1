# RUNNING：capture approach / P06 offset240，首128已保存窗口

本报告只核验初始迁移与 globals **72321–72448**，对应第 **531** 次累计 PPO update；不纳入其后的采集或更新，不把计划2048视为完成。读取方式仅 PowerShell、小型JSON清单与共享只读句柄逐行JSONL；未运行Python、加载tensor、启动Isaac、修改生产或重复计算大型checkpoint哈希。

Run：`runs/ppo_semantic_v3/train/20260906T1738509257629Z_gc34262abffc1_04bbf6558046407990f85358d738cc17`。

目标HEAD：`c34262abffc16847ff32d15ecbf790dd60803e0a`。源HEAD：`73e937039708a7306b2b2c941e1f9108b8fa8b3d`。配置为v3、N1、seed1001、P06 offset240、NewMdpWarmStart。首窗口是教师初始化的后缀，不是当前策略从P01完成全任务。

## 1. 初始迁移与不可变计数

直接对照源72320 sidecar与独立initial sidecar，以下字段逐项相等：actor/critic/normalizer参数摘要、完整`training_rng_state` JSON、完整`policy_contract`与`runner_config`、累计global/PPO/optimizer、stage ledger与new-MDP origin。

| 项目 | 源 → 新initial |
|---|---|
| global / PPO / optimizer | 72320 / 530 / 10600 → 不增加 |
| stage full / suffix / smoke | 29184 / 33024 / 0 → 不增加 |
| new-MDP origin | 10112 → 不变 |
| actor SHA | `1587569b6d1510508833bfca0f9d1ce128cc3fe47838ff38510a93dff2744300`，相同 |
| critic SHA | `050d3c403a9b0170d01b181334d7c46ea874dc19289a1fd96487aad7f4b156f3`，相同 |
| normalizer SHA | `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`，相同 |
| policy | 保持`heteroscedastic_log_v1`，324观测 / 12 raw action；actor摘要包含既有learned std分支 |
| Adam SHA | `c37c1dd7…` → `ebddf13d…`，显式重置moments，非exact optimizer resume |
| 已记录初始LR | 源结束1e−5 → 新initial 3e−5 |

迁移record声明`old_rollout_buffer_inherited=false`、`physical_state_inherited=false`：保留源学习权重及RNG后，重新真实reset并收集新MDP样本，不继承旧storage或物理状态。初始保存`stage=initial_v3_warm_start`、`save_load_round_trip=true`，与正常训练checkpoint分开：

`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000072320_s96d26c928dcd_gc34262abffc1_b87dc5107514f1cdbc128e9f8b3ad07d11beacfb83c371ed288651b6b9e124f7.pt`

其sidecar记录checkpoint SHA为`9061fcd2b42c339cc27c770f7b5b09cd9ce08a59e7f76a6a6b3f84ed5db43221`。此处核验保存证据与清单一致性，未独立重新tensor-load或重哈希。

## 2. 配置边界及采样描述修复

迁移源/目标runtime contract中29个`frozen_A_files`摘要映射完全相同；local runtime版本相同。动作schema、观测schema、execution profile、reward config、quality score摘要均未改变。实际声明变更仅：

- `configs/ppo_semantic_v3/stage_task_spec.yaml`
- `src/wlr50_clean/ppo/semantic_supervisor.py`
- `src/wlr50_clean/ppo/semantic_cli.py`

共享物理势能定义改变，因此这是显式新MDP，不因为reward配置文件摘要不变而称exact MDP续训。

initial的`sampling`、修复后的`implemented_reset_sampling`、`curriculum_epoch.reset_sampling`与topology的reset sampling均为：

`natural_P01_A_teacher_prefix_then_semantic_suffix_N1.v2:P06:offset_240`

`curriculum_epoch.prefix_request`也记录P06、offset240、最大prefix1800、最大takeover30、一次fresh-P01 fallback、任务200s包含prefix、不采用精确历史快照。第72448正常训练sidecar继续记录相同target sampling。

一个已报告给主代理的描述残留：initial顶层`phase_suffix_curriculum_implemented=false`仍是源值，但其`execution_topology.phase_suffix_curriculum_implemented=true`且本次真实采样/接管为后缀；第72448正常save顶层已为true。未修改任何live记录，不将这项顶层布尔作为实际reset证据。

`new_mdp_initial_action_comparison.json`在真实P06/t5504给出六类12维旧/新逻辑输出差全0，optimizer更新0、环境及bridge历史未修改。这只证明该同状态逻辑投影对照，不是native轨迹等价证明。

## 3. 真实教师前缀与READY接管

第一条`policy_credit_start`前有688条教师decision，合5504 physics ticks；688条raw均为全零。`reset_only_prefix_result`为accepted=true、miss=null，没有fallback。语义P06首次被观察于t3584，实际offset为(5504−3584)/8=240决策；打开策略信用时P06仍有效。

- mode：`teacher_initialized_suffix`；requested/actual均P06，`from_P01_current_policy=false`。
- credit start：t5504 / 45.8666666667s，剩余任务154.1333333333s，未重置200s任务时钟。
- handoff source_control_tick5503、实际command/ACK tick5683；这是两个记录时钟，不把command tick误当episode tick。后续逐tick关系亦一致。
- controller bias、ACK combined bias、独立policy residual三组12维值均0；tracking集合为空。ACK为一次articulation write、write_count5684、skew0，确认采用刚实际dispatch的接管收据。
- 教师末nominal为`[22.8,-13.4,0,45.9,6.9,0,0,0,.3,.3,.3,.3]`；保留真实mapper进程与历史，而非重置成历史关节姿态。

接管物理状态来自该次实际handoff/credit-start记录，不由旧teacher轨迹推断：

| 腿 | 当前contact / AIR | front距离mm | clearance mm | load fraction | 当前Q / C / P历史 |
|---|---|---:|---:|---:|---|
| RR | GROUND / false；无障碍pair | −259.737071 | −51.134913 | 0.202801524 | false / false / false |
| RL | GROUND / false；无障碍pair | −274.036851 | −49.493952 | 0.353307154 | false / false / false |
| FL | TOP / false；top geometry true | +316.319767 | +0.106265 | 0.148777919 | true / true / true，均教师历史 |

FR教师Q/C/P ticks为71/1665/1695，FL为2461/3115/3583。RR尚无initial lift事件；RL已有教师initial事件7/1723，但没有硬Q/C/P。这不是已RR离地的课程入口；初始台上前腿也不记为PPO新完成事件。

## 4. 首128真实PPO与native审计

只逐行读globals72321–72448，连续128行，全为P06，episode physics ticks5505–6528，共1024，无终止/短tick；末时间54.4s，`termination_reason=null`、`task_success=false`。physical core decision count从689到816，严格等于688教师+1至128策略；128行`prefix_teacher_data_in_ppo_storage=false`，start引用始终t5504/attempt0。

| 检查 | 固定窗口结果 |
|---|---:|
| 新增优化decision / PPO update / optimizer steps | 128 / 1 / 20 |
| teacher仅reset / teacher physics ticks | 688 / 5504；不加global、stage或PPO credit |
| PPO physics ticks / verified ticks | 1024 / 1024 |
| 真实native effect / own phase request effect ticks | 1024 / 1024 |
| 四类回合内state writes | root pose、root velocity、force/impulse、gravity全部0 |
| 原生末tick重建/dispatch/setter/同tickcounterfactual失败行 | 0 / 128 |
| 全tick审计连续性 | 1024条均verified；command tick=episode tick+179 |
| raw / old mean / old std维数 | 全128行均12，finite，std严格正 |
| std实际范围 | 0.083154604–0.251474500；127行与首行向量不同 |
| raw绝对最大 / old mean绝对最大 | 1.301114202 / 1.012804508 |
| Gaussian old LP独立double重算最大绝对误差 | 9.0008713e−7 |

这里的native通过依据独立target-buffer审计；不把`actual_drive_target_full12`的canonical double逻辑向量当作float32 native readback。LP重算使用日志中的同次raw/old mean/std；保存代码另对official storage action及distribution逐值绑定，未在本次PowerShell审计中打开二进制rollout文件。该窗口含128条非零raw采样，但不据此推断新sigma带来物理改善。

首update531记录actor从源`1587569b…`变为`bfd77823673d8ff3faf122e862b15a45c18eb413bfdc5a23bc8ee2cf1321ef28`，`actor_parameters_changed=true`，`finite_nonzero_gradient_observed=true`；gradient norm范围1.004323092–1.414213309，KL均值0.019763741、clip fraction0.35、entropy−4.606111765、value loss0.003102766、surrogate loss−0.029683723。该update结束LR为1e−5；不将其解读成全部20个minibatch恒定LR。128条存储reward合计−2.095899386，尚非完整回合return。

## 5. 第72448保存边界与后腿状态

已读取immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000072448_manifest.json`：累计72448 / 531 / 10620，roundtrip=true，checkpoint记录SHA `09221a4457e3cd0b289ef3d43612bf56043ed427cffe4c90c8d3154a29cf34a8`，actor与首update after摘要一致，normalizer不变，source ancestry仍指向72320。stage ledger为full29184、suffix33152、smoke0，origin10112，恰只加入本窗口128个suffix决策。

截至t6528，RR出现PPO段initial clearance事件5609、5764、5832、6020、6506，但无硬Q/C/P；RL无新硬Q/C/P。RR末仍GROUND：front−121.689316mm、clearance−49.417483mm、load0.389811631；RL仍GROUND：front−234.845575mm、clearance−51.508972mm、load0.351440281。FL历史placed仍true，但当前为AIR、load0、clearance+65.750678mm；不把历史placed替代当前支持。

这个固定窗口仅exercise P06准备，尚未进入后腿carry/capture阶段，不能验证新post-cross capture shaping的实际分支，更不能称后缀成功或全P01成功。计划2048的其余部分不在本报告核验范围；完成本首128有界审计后停止扩展。

证据文件：run内`run_manifest.started.json`、`new_mdp_warm_start.json`、`new_mdp_initial_action_comparison.json`、首个`prefix_evidence.jsonl` credit start之前的记录、`optimizer_updates.jsonl`首行及`residual_and_projection_audit.jsonl`首128行；源、initial与72448三个sidecar。Windows活跃写句柄下目录Length曾显示缓存0；共享FileStream实际可读，此为目录元数据现象，不是审计缺失。
