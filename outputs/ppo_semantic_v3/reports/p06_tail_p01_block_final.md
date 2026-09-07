# P06 有限源尾段版：P01 4096 决策块最终对账

**执行最终完成：SUCCEEDED；不是任务成功。** 已读取真实 `run_manifest.json` 和完整 4096 条审计、32 条更新、4 条完成回合记录，不计任何未来训练/评估。原请求与实际均为 4096，rounding overrun=0、unconsumed=0、无 stop request。

运行：`runs/ppo_semantic_v3/train/20260906T1650134938238Z_g73e937039708_7681a31f01714528b75509300dfac00f`；HEAD `73e937039708a7306b2b2c941e1f9108b8fa8b3d`；N1 / seed1001 / natural P01 / full_episode。Wall time **1553.931632 s**。

## 1. 保存边界、迁移和预算

| 账目 | 来源 | 本块新增 | 实际最终 |
| --- | ---: | ---: | ---: |
| Lifetime policy decisions | 68224 | 4096 | 72320 |
| Lifetime PPO updates | 498 | 32 | 530 |
| Lifetime optimizer steps | 9960 | 640 | 10600 |
| v3 full_episode spent | 25088 | 4096 | 29184 |
| v3 phase_suffix spent | 33024 | 0 | 33024 |
| v3 smoke spent | 0 | 0 | 0 |

原 v3 origin 仍为10112；full+suffix=62208=72320−10112，没有重新分配旧预算。来源到新初始checkpoint是显式 **NewMdpWarmStart**，不是保留旧Adam的exact-MDP resume。兼容 actor（含state-dependent log-std）、critic、identity normalizer、RNG、计数和预算保留；Adam清空并从3e−5开始，fresh rollout，不继承物理状态。详见已保留的 [首128迁移与更新报告](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p06_tail_p01_initial_update.md)。该报告固定边界未覆盖或改写。

实际最终 immutable checkpoint：`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000072320.pt`，sidecar同stem加`_manifest.json`。最终sidecar和pointer一致：

- checkpoint SHA `96d26c928dcd9d07f95bf4f33cd5fade64220de7bdffbb5b581f0dcadc64c10f`。
- sidecar SHA `67503b620a98eebe12607838f3b1213fbd759992810c80b3ff676dc9e9d554f8`。
- `save_load_round_trip=true`；stage、当前run、72320/530/10600、预算与上表一致。
- normalizer hash仍为`c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`。

根代理已实际核验最终两个文件hash；本报告只读其记录及pointer，不重复大文件hash/tensor load。初始清单的旧P06 `implemented_reset_sampling`描述残留已在首128报告披露；最终该字段和`sampling`均为`P01_full_task_only_initial_version`，`prefix_request=null`、`phase_suffix_curriculum_implemented=false`。

## 2. 全部实际优化阶段样本

| Phase | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 决策 | 5 | 894 | 18 | 5 | 740 | 2422 | 1 | 1 | 10 | 0 | 0 | 0 | 0 | 4096 |

四个终止回合和一个未终止尾回合均自然P01起步。P10–P13无访问/无质量样本；表中的0只表示样本数，不是质量、稳定性或成功分数0。

## 3. 完成回合与非终止尾段严格分账

| Episode | 实际global范围 | 决策 | Physics ticks / duration | 最后阶段 | 原始结果 |
| --- | --- | ---: | --- | --- | --- |
| 0 | 68225–68872 | 648 | 5177 / 43.1416667 s | P09 | BODY_COLLISION |
| 1 | 68873–69806 | 934 | 7472 / 62.2666667 s | P06 | INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 69807–70737 | 931 | 7448 / 62.0666667 s | P06 | INCOMPLETE_CONTROLLER_BLOCKED |
| 3 | 70738–71674 | 937 | 7496 / 62.4666667 s | P06 | INCOMPLETE_CONTROLLER_BLOCKED |
| 4，非终止尾 | 71675–72320 | 646 | 5168 / 43.0666667 s | P06 | terminal=false；无任务终止结论 |

648+934+931+937=3450个终止回合样本，另646个未终止尾样本，总4096；不是第五个失败回合。4个完成回合的task_success/full_task_success均false，manifest success_count=0。

各回合P01…P09样本（其余阶段未访问）：

- ep0：`1,166,3,1,149,316,1,1,10`。
- ep1：`1,180,4,1,148,600,0,0,0`。
- ep2：`1,177,4,1,148,600,0,0,0`。
- ep3：`1,184,3,1,148,600,0,0,0`。
- ep4尾：`1,187,4,1,147,306,0,0,0`。

唯一短物理决策是 **ep0 / global68872 / P09 / terminal tick5177**：实际执行1tick后BODY_COLLISION，不是通常8tick。因此32761 = 4096×8−7；其余4095条各8tick，无其它差额。没有把collision后未执行的7tick计为PPO物理证据。

## 4. 实际腿事件和第一个未完成目标

所有5个回合的历史里，前腿FR、FL都各有真实Q/C/P；这是当前自然P01策略段事件，不是教师提供。该历史不代表它们之后始终TOP或始终承载。

| Episode | RR initial-clearance历史 | RR Q/C/P | RL initial-clearance历史 | RL Q/C/P | 实际首未完成 |
| --- | --- | --- | --- | --- | --- |
| 0 | 无 | 全无 | 9次，首tick10 | 全无 | P09 RR放置链；碰撞终止，RR连硬lift资格也未获得 |
| 1 | 22次，首2941、末7206 | 全无 | 1次，tick11 | 全无 | P06当前双后腿workspace准备 |
| 2 | 16次，首3113、末7241 | 全无 | 2次，tick12/543 | 全无 | P06当前双后腿workspace准备 |
| 3 | 7次，首3186、末6734 | 全无 | 1次，tick13 | 全无 | P06当前双后腿workspace准备 |
| 4尾 | 10次，首3159、末5099 | 全无 | 1次，tick17 | 全无 | 当前仍P06；尚非终止 |

`whole_body_initial_clearance`只是低阈值实测初始抬离事件，不等于above-top硬资格，更不等于越沿/放置。上述RR/RL未出现硬Q、C或P；不能把反复initial事件计为成功重试。

ep0：P06开始tick2560，P07 tick5088，P09 tick5104，最后tick5177碰撞。终止RR current AIR、front −227.468 mm、clearance −45.950 mm、load0；RL current AIR、front −154.351 mm、clearance −51.598 mm、load0。原始终止理由保持BODY_COLLISION，不重分类为wheel-only或任务完成。其P06仅持续2528tick，短于3064tick有限源endpoint，因此这一回合进入P09**不能归因为新的endpoint尾段已激活**。

ep1/2/3最后RR与RL均GROUND、无TOP；当前front距离分别为：

| Episode | RR front / clearance mm | RL front / clearance mm | 记录rear_approach |
| --- | --- | --- | ---: |
| 1 | −380.249 / −51.268 | −461.729 / −49.895 | .0330844855 |
| 2 | −437.673 / −50.653 | −553.447 / −50.191 | 0 |
| 3 | −474.081 / −50.863 | −578.060 / −50.250 | 0 |

这里0是**已访问P06实际记录的当前准备predicate**，不是未访问后续阶段的质量评分。三回合均未进入P07。

ep4尾：RR AIR、front−444.717 mm、clearance−49.665 mm、load0；RL GROUND、front−550.118 mm、clearance−50.494 mm、load .508048；无后腿TOP、无Q/C/P。尾段当前rear_approach=0，不预判后续结果。

## 5. 尾段修正确实被执行，但不能等同物理完成

全4096条decision末的`nominal_provider_diagnostics.p06_wheel_tail.status`：no-P06-layer1657；finite_source_before_endpoint1773；retired11；terminal_no_tail4；live_endpoint_tail651。`finite_source_tail_replaced=true`有**651条decision末快照**，不是651个完整独立物理区间或成功事件。

第一条为ep1/global69589/episode tick5736/P06：source_endpoint_issued=true、wheel_gain=1、原source rolling tuple四轮.3，当前nominal四轮均.3。P06进入2672，加原有限源3064tick恰为5736。诊断明确是**P06 layer贡献、后续owner组合及slew之前，不是actual applied target**。原终止行使用terminal_no_tail。其余原生应用命令及residual仍独立审计；651条激活不证明几何前进、净空、rear lift或任务成功，也不单凭本块结果推断尾段无效/有害。

## 6. 4096条原生审计、无教师和原始PPO密度

- 全部global严格连续68225–72320；4096条各保存12维raw Gaussian action、old mean、old std。
- physics ticks=32761；verified ticks=32761；实际native residual-effect ticks=32761；每条all_ticks_verified=true。
- 4096条末tick均`verified`、`setter_dispatch_targets_equal`、`actual_mapping_matches_dispatch`、`same_tick_counterfactual`为true。
- own-phase-request effect ticks=32733；这与32761实际native effect是两个不同口径，未把phase/request排除计为native未验证。
- 四个in-episode root pose / root velocity / force-or-impulse / gravity write计数，在4096条中全部为0；no-state-writes验证全部true。
- `actual_drive_target_full12`仍是canonical逻辑double向量；native float32下发与counterfactual来自独立audit，不能混称。
- 无prefix evidence文件，当前实际topology/curriculum为P01、prefix_request=null；core decisions=policy decisions=4096。reset/settle无策略信用；教师决策/教师物理tick均0。本块无需扣除任何A roll-in样本。
- old std全部正且finite，范围 .0871047899–.2593973577。按记录raw/mean/std独立重算对角raw Gaussian log probability，最大绝对差`1.5555665e-6`（来源float32，重算double）；不以projected/native action代替PPO采样密度。

## 7. 真实32次优化链

更新499–530每次global+128，32次均actor changed和finite nonzero gradient为true，每次20个optimizer steps，总640；31个相邻before/previous-after hash全部相等。首before=`d4b129487079242acdb524ae4b87e6485edc3e54243a04e47aa92a1b8e4c1c42`，与source68224和新initial相等；末after=`1587569b6d1510508833bfca0f9d1ce128cc3fe47838ff38510a93dff2744300`，与最终sidecar相等。

记录范围：gradient norm 1.00048513–1.41421364；mean KL .01337712–.03232487；clip fraction .18125–.3953125；entropy −5.79519157至−4.38088858；value loss .000620815–106.662969。数值均为真实记录，未把较大的value loss隐藏或宣称已收敛。

**32个已记录update边界/结束LR均1e−5；未逐minibatch记录LR，不作全程常数断言。** 与initial fresh Adam3e−5是不同时间点。

结论仅限本块：模型真实更新并完成保存；P01到前腿目标有实际策略样本，但后腿硬Q/C/P及全任务成功均未获得，P10–P13未访问。后续固定均值评估尚属独立运行，不在此报告预填结果。本报告全程只用PowerShell读取和报告写入，未运行Python/Isaac、修改生产或提交。
