# C46464：P06 rear_approach 未完成的实际证据

2026-09-06。评估 run：`runs/ppo_semantic_v3/validation/20260906T1109596110104Z_g49749aa527a4_df39981a0f5c40cd921698578dec5e67`。固定 runtime `49749aa527a4a00901e434104821d7e8241ea8da`，checkpoint `checkpoint_step_000046464.pt`（checkpoint SHA256 `fc1fd6710965d627aff6e6ecc91e02c5ed9c144f7637cba9a50cfa079cf5513d`），seed2001，从真实P01 reset执行确定性mean；评估期间optimizer更新0。

**执行正常结束不等于任务成功。** 939决策、7512物理ticks、62.6s后，实际结果是 P06 `INCOMPLETE_CONTROLLER_BLOCKED`，task_success=false。physical evaluator valid=true、无physical failure/safety termination；不是成功、不是全任务完成，也不是接口崩溃。

本报告读取上述已完成JSON、当时git版本的源码/配置。另在确认无Isaac后，以锁定Python执行一次0.63s只读CPU函数诊断，现已退出；未运行仿真/optimizer、未改生产、未调整nominal/动作/reward。CPU读取的是git固定49749版本，不把root正在实施的后续opt-in混进本次诊断。

## 1. 第一个未完成任务与真实时序

`stage_transition_evidence.jsonl` 中，排除同阶段内部事件后：

| 真正阶段转移 | physics tick | task time s |
|---|---:|---:|
| P01→P02 | 8 | .066667 |
| P02→P03 | 1488 | 12.4 |
| P03→P04 | 1520 | 12.666667 |
| P04→P05 | 1528 | 12.733333 |
| P05→P06 | 2712 | 22.6 |
| P06→下一阶段 | **未发生** | — |

决策来源phase计数：P01=1、P02=185、P03=4、P04=1、P05=148、P06=600。历史事件FR Q53/C1498/P1516，FL Q1610/C2568/P2711；RR和RL均无Q/C/P。这些历史与当前接触分别记录，不能用前腿历史placed推断末尾仍在承载。

49749的任务配置P06要求 `rear_approach=min(workspace_RR, workspace_RL)`；workspace要求横向有效且front在既有 `[−.22,+.06]m`，外侧用既有.25m归一化连续进度。两腿必须同时满足，`active_leg: RR` 并不让RL workspace变成可选。

| 决策末样本 | RL front mm | RR front mm | rear_approach | global Φ |
|---|---:|---:|---:|---:|
| tick2720，P06首决策末 | −542.151 | −531.261 | 尚远离workspace | — |
| tick5656，RR首个已进入workspace的决策末 | −240.726 | −219.684 | .917097 | .4675 |
| tick5784，RL最近的决策末 | **−227.394** | −206.185 | **.970423** | .4675 |
| tick5792，nominal轮已零 | −227.455 | −206.283 | .970180 | .4675 |
| tick7504，非终止末前样本 | −236.834 | −216.826 | .932665 | .4675 |
| tick7512，实际终止 | −236.834 | −216.834 | .932665 | .4675（reward next-Φ=0） |

因此终止时RR已在workspace、RL横向有效但距下界仍差 **16.834mm**；RL最接近的决策末也还差7.394mm。此处第一个缺失的是后腿对的接近工作区，而不是RR抬升/cross之后失败：本回合根本没进入P07–P13。

P06配置 `maximum_task_duration=40s`；从22.6s进入到62.6s恰为40s，真实completion未满足，源码按阶段期限记录 `INCOMPLETE_CONTROLLER_BLOCKED`。`stall_diagnostic=true` 是诊断字段，不是额外提前成功/失败门。全任务200s上限没有被重新定义，但本回合先触及已有P06阶段期限。

终止时：RR/RL均GROUND，clearance分别−50.116/−50.272mm，load分别.07078/.49483；FR当前TOP/load.43438；**FL当前AIR、load0**，尽管FL历史placed=true。physical support_count=3。身体线速度.02619m/s、角速度.06052rad/s、实际最大轮速.19987rad/s均有限；这些数值不补足RL缺少的workspace。

## 2. P06轮建议为何结束：有限原始片段，不是retirement

`nominal_provider_diagnostics.p06_rolling_retirement` 在P06所有600个决策末：peak_fraction最小=最大=0，wheel_gain始终1。终止样本明确 `terminal_no_new_retirement`，保留peak0，不虚构末尾测量。

这个peak在内部每物理tick保留历史最大值；因此“所有决策末peak0”也排除了中间某tick曾把这一P06层真正退火然后恢复的解释。诊断的时序标签是 `current_nominal_suggestion_before_slew_not_applied_target`，不能将gain直接冒充last-applied target。

当时 `configs/recording_motion_contract.json` 的P06有限片段：t=0给四轮+.3rad/s，t=25.5333333333s给 `wheel stop`，end_full12四轮0。连续层实现一直推进该片段；只对它乘retirement gain，而gain本次为1。到达有限末尾不需要task成功，也不会停掉可学习residual。

实际 `native_tick_audit.jsonl` 逐120Hz tick锁定归零时间：

| episode tick / time | 进入backend的nominal四轮（每轮相同） | 实际native四轮target rad/s |
|---|---:|---|
| 5776 / 48.133333s | .3 | [−.31946462,+.23071750,−.27058890,+.40376565] |
| 5777 / 48.141667s | .275 | [−.29451802,+.20567818,−.24556163,+.37872720] |
| 5784 / 48.2s | .1 | [−.11951803,+.03067818,−.07056163,+.20372722] |
| 5788 / 48.233333s | 6.94e−18（浮点尾数） | [−.02009649,−.06871070,+.02968103,+.10295006] |
| 5789 / 48.241667s | **精确0** | 同上一行 |
| 7512 / 62.6s | 0 | [−.01922317,−.06662547,+.02686545,+.10469555] |

P06从tick2712/22.6s开始，25.533333s的有限建议末端对应5776附近；实际dispatch在5777出现第一步下降，以既有 `3rad/s² ÷ 120Hz=.025rad/s/tick` slew衰减。5788浮点近零、5789精确零的差别如实保留。左右轮native符号映射与逻辑full12不同，不能把表中native负号直接称作车辆倒转或sign bug。

本次轮零并非人工phase mask、retirement提前削弱或策略被禁用。tick5792对应确定性raw四轮 `[+.033507,−.115022,−.049509,+.173298]`，逻辑实发轮target `[+.020096,−.068711,−.029681,+.102950]`；终止逻辑target `[+.019223,−.066625,−.026865,+.104696]`。它们与上表native经既有符号映射一致，且不是实际测得的轮速度。

全939决策 native验证7512/7512ticks，own-phase-request-effect7507/7512（五次阶段交接首tick被排除）；所有decision的no-in-episode-state-writes验证为true。没有证据表明剩余14秒被统一强置零，也没有证据单凭这些目标就能判定为何整机没有再前进所需的几毫米。

## 3. RL workspace 在本状态确实缺少独立task shaping

以下引用均以固定49749版为准，避免混入后续修改：[semantic_supervisor.py](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py) 当时501–502行的P06谓词同时读取RL/RR workspace；553–554行的global physical potential却先检查placement predecessors。RL的前置集合为FR、FL、RR，而本回合RR尚未placed，故RL整份potential贡献直接为0，连workspace也未计入。

RR已进入workspace、其unload饱和而尚无initial/Q/C时，本回合变成：

`Φ = .85 * (FR历史1 + FL历史1 + RR的workspace .1 + RR的unload .1 + RL的0) / 4 = .4675`。

实际tick5656–7504的232个非终止决策末，Φ最小/最大精确相同.4675；与此同时RL到workspace的距离真实先缩短、后增大，P06进度也随之变化。并非整个reward为0：该段weighted平均task/body/contact/smooth约 `−.01301946 / −.00010733 / 0 / −.00063354`；task中仍有折扣potential及时间成本。最后terminal next-Φ=0且无bootstrap，不能把最后惩罚删掉再分析。

### 固定源码、真实状态的隔离函数反事实

短CPU检查从git取当时完整 `TaskStageSupervisor` 和YAML，调用其真实 `physical_potential` / `predicate`；对真实非终止tick5656、5784、5792、7504，首先重算Φ并断言与记录精确一致。然后复制同一evaluation，只改一个leg当前front及其对应goal字段；不修改历史、载荷、AIR、phase，也不调用仿真/阶段转移。

四个真实状态均有同样结果：

| 隔离输入变化 | rear_approach | global Φ | 相对原Φ |
|---|---:|---:|---:|
| 仅RL front=−.24m | .92 | .4675 | 0 |
| 仅RL front=−.23m | .96 | .4675 | 0 |
| 仅RL front=−.22m | 1 | .4675 | 0 |
| 仅RL front=−.215m | 1 | .4675 | 0 |
| 负对照：仅RR front=−.23m | 随原RL限制 | .46665 | −.00085 |

这证明的只是实际函数依赖：**P06尚需RL workspace，但在RR尚未placed时RL自身workspace没有global task-potential差分。** counterfactual不是一个保证物理可实现的独立腿平移，不是实际新rollout，也没有证明给这份credit就必定到达后腿区域。

RL当前front/clearance/load本来存在324维actor observation中，缺的是这一状态下的直接reward依赖，不是观测把RL抹掉。RL载荷/支撑仍能间接影响RR unload，身体与实际控制质量也仍有成本；因此不能说“RL完全不影响reward”或“全部失败都是奖励缺失造成”。

## 结论边界

已确认两个不同事实：本回合有限nominal按原片段结束且retirement从未生效；同时P06目标要求的RL workspace在前置RR未placed时被global potential整体排除。它们解释了具体调度与信号接线，不构成单因子的物理因果实验，更不能将本次C46464改称成功。

root后续正在实施的显式opt-in pre-predecessor workspace credit 属于另一个版本；本报告不验证或预告其物理效果。本次没有同时变更nominal、硬任务条件、动作范围、entropy或网络，也不提出新增执行门禁。
