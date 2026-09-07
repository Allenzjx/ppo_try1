# C73088：P01固定mean评估完成，首未完成仍为FL真实capture

只读对象：`runs/ppo_semantic_v3/validation/20260906T1758410256842Z_gc34262abffc1_1c3cc10c00874db8a1b189eb741a4832`。

正式`evaluation_manifest.json`与`run_manifest.json`已生成；以下统计严格截止其终止tick5192。仅PowerShell逐行读取该次实际日志，没有Python/tensor加载、重复checkpoint大文件哈希、生产更改或新仿真。

## 1. 正式结果与边界

- HEAD `c34262abffc16847ff32d15ecbf790dd60803e0a`，checkpoint73088；清单记录SHA `3efd9531d6bf797a2852107c1183ac8be2b96bcf5eaa3ed59358f0cd9d44ffef`。
- seed2001，`semantic_residual_eval`、deterministic=true、自然P01、无teacher，评估optimizer updates=0。
- lifecycle **SUCCEEDED仅表示执行完成**；controller与独立physical task success均false。
- **649 decisions / 5192 physics ticks / 43.2666666667s**；649×8精确相等，无短terminal tick。
- 终局P05，`INCOMPLETE_CONTROLLER_BLOCKED`，stage age30.0s；`window_ended_before_task_terminal=false`。
- 独立physical evaluator `valid=true`、`termination_reason=null`、reason为空、success=false：这是未完成任务期限，不是接口错误、BODY_COLLISION或WHEEL_ONLY_CLIMB。

第一未完成目标是**FL真实放置/capture**：硬Q、C已经成立，却无真实TOP序列与placed事件。P06–P13均未进入，不填写这些阶段的质量分数，也不将后腿未发生事件描述为它们的执行失败。

| phase | 策略decisions | native physics ticks |
|---|---:|---:|
| P01 | 1 | 8 |
| P02 | 194 | 1552 |
| P03 | 3 | 24 |
| P04 | 1 | 8 |
| P05 | 450 | 3600 |
| 合计 | 649 | 5192 |

## 2. 真事件与阶段链

| 腿 | initial lift事件 | 硬Q tick | C tick | P tick |
|---|---:|---:|---:|---:|
| FR | 20 | 48 | 1570 | 1584 |
| FL | 1615 | 1669 | 2708 | 未发生 |
| RR | 未记录 | 未发生 | 未发生 | 未发生 |
| RL | 13，仅initial | 未发生 | 未发生 | 未发生 |

阶段变化P01→P02 t8；P02→P03 t1560；P03→P04 t1584；P04→P05 t1592；P05终止t5192。initial是小幅实际离地/动作证据，不等价硬Q，更不是放置。

FR实际raw正例：t48为AIR、净空+1.618973mm；t1570越沿时front+0.741664mm、净空+58.242797mm，仍AIR；t1582尚AIR、净空+3.138072mm。第一障碍接触为t1583，front+19.208315mm、净空−1.560279mm、法向力22.375881N；t1584第二个接触sample为26.616413N、front+20.014170mm，正式placed成立。事件链来自本次完整传感器日志，不由阶段标签猜测。

FL实际raw证据：t1604为最后GROUND；t1605起至5192连续3588个AIR samples，地面与障碍pair均inactive。t1669硬Q时净空+1.149697mm；越沿前t2707 front−1.805064mm、净空+68.909650mm，t2708 front+0.199739mm、净空+69.536125mm。此时真实离地且越沿，并非沿墙接触爬升。

## 3. 越沿后仍没有真实FL touchdown

完整raw文件有5193行（reset t0加5192步），时钟连续、all_finite全部true。P05含边界t1592的raw区间内，FL为13个GROUND samples、3588个AIR samples、**0个obstacle-active samples，最大障碍法向力0N**。

| FL越沿后实际raw点 | tick / 时间s | front mm | clearance mm | 实际contact |
|---|---|---:|---:|---|
| 首越沿 | 2708 / 22.566667 | +0.199739 | +69.536125 | AIR，无障碍力 |
| 最大净空 | 2729 / 22.741667 | +27.338210 | +82.107266 | AIR，无障碍力 |
| 最大front | 2740 / 22.833333 | +31.436089 | +74.506064 | AIR，无障碍力 |
| 最小净空 | 2791 / 23.258333 | +3.200832 | **+2.763440** | AIR，无障碍力 |
| 终局 | 5192 / 43.266667 | +10.349873 | +5.563301 | AIR，无障碍力 |

最小净空与终局raw的inactive obstacle pair虽仍带有contact-point字段，但force=0且active=false，不能把该字段当作真实触地/支持证据。原测量几何与任务结果保持原样；这里不重新分类。

## 4. 新capture分支实际覆盖，而非仅配置启用

当前实现仅在未placed、硬Q+C均成立且前驱已placed时使用：

`capture = .5 × clip(1 − outside/.25) × .025/(.025+abs(clearance)) + .5 × clip(consecutive_TOP/2)`。

它重分配原有.2 capture份额，不增加成功事件；同条件下已有.1准备卸载份额视为完成，实际load/support与hard predicates不变。下面是本次649个decision-end评估快照的真实分支统计：

| 腿 | Q+C、未placed快照 | 实际tick范围 | capture值范围 | real-contact份额 |
|---|---:|---|---|---|
| FR | 1 | t1576 | 0.214327554 | 0；first TOP t1583处于两个decision快照间 |
| FL | 311（310非终止+1终止快照） | t2712–5192 | 0.117010143–0.450217285 | 全部0 |
| RR / RL | 0 / 0 | 未进入 | 不适用 | 不适用 |

FR完整物理链证明其未placed的post-cross窗口确实经历AIR→首TOP→placed；但不能把decision-end只有一行误写成仅一个物理tick触及。FR placed后452个decision快照走既有current-region retention分支，不再走first-capture approach。FL始终未placed，故后者持续可见：

- t2712：净空72.246139mm、outside0，capture0.128539807，记录phi0.387962942。
- t2792：净空2.764372mm、outside0，capture0.450217285，记录phi0.401634235，仍AIR、TOP次数0。
- t5184：净空5.551708mm，capture0.409142424，phi0.399888553，当次shaping为−0.009913904。
- t5192终止快照：capture0.408987230、诊断phi0.399881957；reward使用absorbing next potential0，不把该诊断phi继续bootstrap。

逐条重算649行`5×(.995×potential_after−potential_before)`与记录的potential_shaping最大误差0。终局potential_before0.399888553、after0、shaping−1.999442765、terminal event−40、总reward−42.001499329、time_outs=false、terminal_bootstrap_allowed=false。所以近表面AIR并未获得放置，静止悬停也没有被简单当作持续正收益。

本次确认**公式分支已实际exercise**，不等于新版本改善了轨迹。相较此前C72320，当前仍是FL无接触capture失败；HEAD与训练权重均不同，不能凭微小几何/时长差给出因果改善结论。

## 5. 当前支持、原生控制与信用

| 终局腿 | 当前接触 | front mm | clearance mm | load fraction | 历史placed |
|---|---|---:|---:|---:|---|
| FL | AIR | +10.349873 | +5.563301 | 0 | false |
| FR | TOP | +109.425369 | −0.121781 | 0.426240811 | true |
| RL | GROUND | −570.038658 | −50.288884 | 0.500309320 | false |
| RR | GROUND | −548.619911 | −50.183917 | 0.073449869 | false |

`placed_FL` completion predicate为.85，实际Q/C成立、当前top geometry=true、TOP序列0；它不是“85%任务成功”。全任务失败与physical measurement valid可同时成立。

独立`native_tick_audit.jsonl`5192行连续；每行verified、actual mapping matches dispatch、setter/dispatch相等、same-tick counterfactual通过，四类in-episode root pose/root velocity/force-or-impulse/gravity写入均0。实际native effect5192ticks；decision聚合own-phase effect5188ticks，未将4个未计own-phase的转换tick混成审计失败。前腿范围未触及后腿nominal geometry路径。

本run没有prefix evidence文件，配置from_phase=P01，只有一个自然reset episode；无teacher信用。所有统计是确定性评估数据，optimizer0，不加训练全局样本/updates/预算。当前未完成环节已由真实raw contact定位为FL capture；未进入后腿窗口，不臆测其能力。

证据：本run正式evaluation/run manifests、649行`residual_and_projection_audit.jsonl`、5193行`physical_observations.jsonl`、5192行`native_tick_audit.jsonl`，以及其中实际stage transition和hard事件历史。固定最终窗口已核验完成，停止扩展本报告。
