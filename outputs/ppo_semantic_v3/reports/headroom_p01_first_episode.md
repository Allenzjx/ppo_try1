# e99fde1 同MDP自然P01：首个真实完成回合（固定只读窗口）

运行：`runs/ppo_semantic_v3/train/20260906T2007180722147Z_ge99fde1b3e83_4cec420c803f4493a1cdb29abf662bd0`。来源step78464，N1/seed1001，HEAD仍 `e99fde1b3e8366f0ff1d484140b82877df745f0c`；full_episode请求4096。这里不是4096整块完成报告，也不是确定性评估。

本报告固定首回合 **g78465–79502，1038 decisions / 8304 ticks / 69.2s**。实际终止为 **P09 `INCOMPLETE_CONTROLLER_BLOCKED`**，回报−56.9408466180657；task_success/full_task_success均false。共同物理evaluator valid=true、physical termination_reason=null：这是任务未完成，非接口/传感无效或已记录物理安全failure。

## 1. 普通resume与自然P01起点

`run_manifest.started.json`确认：from_phase=P01、offset0、`new_mdp_warm_start=false`、resume_migration=null、policy_distribution_migration=false，三项内部migration record均null。源step78464和目标运行的**完整runtime contract及content SHA相等**，未更改MDP/执行profile。

源step78464实际sidecar：78464 / 578 PPO updates / 11560 optimizer steps；full_episode29184 / phase_suffix39168 / smoke0；origin10112，roundtriptrue。

| 保留项 | 依据与边界 |
|---|---|
| actor及learned heteroscedastic std | 源全actor参数hash `cc5caa5dfdb7e8dd59ffd7fa4b195607e763dcb4f6baed02bede108bc6fa36ea`，与首update579的实际before hash精确相等；policy仍heteroscedastic_log_v1、324观测/12 raw动作 |
| critic | 源hash `594f6eb0b69325e2f033d933426ff742eec3b1569cc2ccdebca0a7eeb66883d4`；普通loader对实际加载critic hash执行严格核对，之后正常学习，不要求更新后还等于源 |
| Adam | 源state hash `c1ed23e53e0dc0d9c959771fb71292163a0f685f42e76a681bcfd017188408a9`，记录LR1e−5；普通loader加载并验证optimizer state，未走fresh-Adam/new-MDP分支 |
| normalizer | 源 `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`，加载核验，首save78592及固定可见save79488仍相等 |
| RNG | 普通loader实际执行`restore_training_rng_state`，来源为已验证checkpoint infos、seed1001；本报告不另起Python/tensorload，也没有伪造独立“加载瞬间RNG快照” |
| rollout/physical | loader只允许storage.step=0且transition.actions=None；构造新N1环境，旧partial rollout不继承，物理状态不从checkpoint续接 |

证据链代码：`semantic_cli.py`的普通train路径构造runner时`initialize_actor=False`，无migration参数地调用loader；`semantic_training.py:350–419`在resume中验证checkpoint/sidecar/actor/critic/Adam/normalizer并恢复RNG，和new-MDP的显式Adam重置分支分开。首个真实update及新save证明该路径已完成；以上不是本报告重复hash或独立torch.load的声明。

实际首行g78465：source phase P01，decision_count=1，t0→8、0→0.0666667s；之后P02。无curriculum_start、无prefix_teacher字段，运行目录没有prefix_evidence.jsonl或new_mdp_warm_start.json。1038行均无prefix字段，目标新save的sampling为`P01_full_task_only_initial_version`。因此本回合从P01前腿开始全部计入当前PPO，未用A teacher roll-in形成入口；不能把源checkpoint的历史P07课程标签误作本回合teacher证据。

## 2. 首回合真实阶段样本

按每个15Hz决策的source phase计数；tick区间是其所覆盖物理区间，不将下一phase到达行重复计数。

| phase | PPO decisions | 物理tick区间 |
|---|---:|---:|
| P01 | 1 | 0→8 |
| P02 | 178 | 8→1432 |
| P03 | 3 | 1432→1456 |
| P04 | 1 | 1456→1464 |
| P05 | 149 | 1464→2656 |
| P06 | 254 | 2656→4688 |
| P07 | 1 | 4688→4696 |
| P08 | 1 | 4696→4704 |
| P09 | 450 | 4704→8304 |
| 合计 | **1038** | **8304 ticks** |

P10–P13本回合未访问，不将其质量指标填为零或作成功判断。P09 age=30s到期，completed_stage_ids为P01–P08；entry_valid=true、entry_reasons为空。首未完成任务是 **RR真实放置/capture**，不是RR完全未抬起或未越沿。终止decision完整8ticks，无short-tick差额，time_outs=false、terminal_bootstrap_allowed=false。

## 3. 四腿Q/C/P及当前接触，严格区分历史与末态

共同物理evaluator的120Hz事件历史如下（Q=qualified lift、C=front-edge cross、P=placed）：

| 腿 | Q tick | C tick | P tick | t8304实际末态 |
|---|---:|---:|---:|---|
| FR | 58 | 1440 | 1455 | 当前TOP/contact、load0.489816612，front+185.034mm，clear−1.330mm |
| FL | 1542 | 2574 | 2654 | 当前TOP/contact、load0.068578912，front+34.853mm，clear−0.007811mm |
| RR | 5525 | 5610 | 无 | 当前AIR、obstacle inactive、load0，front−275.749mm，clear+19.609mm，无current TOP/XY |
| RL | 无 | 无 | 无 | 当前GROUND、obstacle inactive、load0.441604475，front−482.133mm，clear−50.005mm |

这些前腿事件全发生于本回合PPO信用段，不是teacher事件。RR initial-clearance事件记录t4388、5148、5357，随后Q5525/C5610；历史active_lift/cross保持true，但不能据此说末态仍在台面前沿后方、承载或放置。RL多次initial-clearance尝试存在，但没有硬Q/C/P，不能把initial或soft credit升级成完成。

临近事件的**决策末实测快照**支持而不替代精确事件tick：t5536的RR AIR、front−30.125mm、clear+7.748mm、无接触；t5616 AIR、front+0.338mm、clear+10.271mm、within_top_xy=true、top_contact=false，历史已经记录C5610。到终止时RR退回前沿之前仍AIR，整个回合没有P事件。这里没有把t5616快照冒称为t5610的逐tick原始几何，也不将全部后续失败归因于某一个nominal/residual因素。

## 4. 原生下发、真实更新与持久边界

首回合1038行global严格连续，raw/old-mean/old-std每行均12维有限，std正；没有使用filtered/effective动作替代RSL原始Gaussian sample。

- 8304个物理tick在线native验证通过，actual-native-effect8304，own-phase-request effect8296；8处phase边界不被强行算作新phase本次own action的效果。
- 1038决策末完整审计均verified、same_tick、setter==dispatch、actual reconstruction==dispatch、float32；headroom mode均正确。187个决策末触发裁剪，全部RR knee/index7。
- root pose、root velocity、force/impulse、gravity四项in-episode写入均0，no-state-writes验证全部true。本任务只读已记录在线审计及必要状态，不另做8304tick巨型离线数学重算，也不把target effect当物理成功。

首update579：20 optimizer steps，actor before与源相等，after为`f377b3a751f439f43620d93784e2902f67b2c2b3e82e85f30fbc536af64aa34b`；finite nonzero gradient=true，norm1.006125913–1.414212883；KL0.018256961、clip fraction0.315625、entropy−5.413685083、value loss0.003181218、surrogate−0.043215680。首save78592/579/11580，roundtriptrue，full预算29312、suffix仍39168。

为确认首回合全部已进入完成优化，本报告同时固定检查update579–587：**9个完整updates / 1152 decisions / 180 optimizer steps**，到global79616/587/11740；9次actor hash链连续、9次actorchanged、9次finite-nonzero-gradient。9条记录的update结束LR均1e−5；未逐minibatch记录LR，不断言全程恒定。

其中1038决策属于已完成首回合；余114属于下一回合，**不包含在上面的phase/事件/native表中**。本次读取时可见immutable已保存边界是 **79488/586/11720，roundtriptrue**，full30208/suffix39168/smoke0；`checkpoint_step_000079616_manifest.json`当时不存在。因此“已完成update587”与“已持久保存到79616”不同，本报告没有把后者写成事实。最新运行可以继续，本报告不追随后续边界。

来源：本run started manifest、首条completed_episodes记录、固定g78465–79502 audit、固定update579–587，以及source78464/首save78592/已见save79488 sidecar。只读PowerShell，新建本报告；无Python、GPU、Isaac、生产/config/tests改动或checkpoint hash复扫。报告至此完成并停止。
