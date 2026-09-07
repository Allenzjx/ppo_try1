# P06 新 lift-credit 课程：有界物理证据

**最终整块更新：6144个策略决策、48次PPO更新、960次optimizer step已全部完成并保存checkpoint28032，roundtrip=true。7个正式回合均未成功；第八回合456决策尾段也已到P13，两后腿当前TOP，但预算结束时未终止、未成功。第六回合P13有限nominal在75.866667s退场，不能误诊为永久滚动。最终对账见文末。**下方首报、24960和27008追加均保留其当时的固定边界，不以其中“仍运行”的历史描述覆盖最终状态。

状态：**运行中的有界首报，不是整块6144决策完成报告。**首个755策略决策回合已真实失败并进入完整PPO更新；soft-air过程信用在真实轨迹中确实获得、保留和清除，但尚未取得RR hard qualification/cross/place。后续采样由根代理继续监控，本报告不纳入固定边界之后的数据。

## 固定优化边界与信用分账

- Run：`runs/ppo_semantic_v3/train/20260906T0636173277212Z_g64abc5357d00_6614a15089014b2ebef51695b9b8cc22`。
- HEAD：`64abc5357d00763419fe51c8126db8454fcf7697`；P06 offset0、seed1001、N1、new-MDP checkpoint21888 warm start。原计划是6144个PPO决策，尚不能写为已全部完成。
- 报告只读取 `optimizer_updates.jsonl` 已完成边界 **global22656 /累计update142**，audit global21889–22656连续无缺号：**768个已优化策略决策、6次新PPO update、120次新optimizer step**。6次更新均报告actor参数改变、有限非零梯度。
- 这768个策略决策推进**6144个physics ticks**；这个物理tick数与原计划6144策略决策恰好同数，单位不同，不能混淆。

| 阶段 | 首回合已优化决策 | 截至22656已优化决策 |
|---|---:|---:|
| P06 | 303 | 316 |
| P07 | 1 | 1 |
| P08 | 1 | 1 |
| P09 | 450 | 450 |
| 其余 | 0 | 0 |
| 合计 | 755 | 768 |

后13个决策属于第二次真实P06接管后的未终结尾部，末行为tick3688 /30.733333s。这里不声明最终checkpoint或整块预算已完成。

## 首次真实前缀与接管

- 教师从P01真实运行到P06：**448个reset-only教师决策 /3584ticks**，接管时钟29.866667s；其raw policy action全零、policy_credit=false、native审计全部verified。
- 接管模式teacher_initialized_suffix，requested/actual=P06，offset0；`from_P01_current_policy=false`。同一200s物理任务时钟没有因接管重置。
- 接管RR仍GROUND：front−494.883957mm、clearance−50.949587mm、load=0.228028；initial/soft/hard/cross/place均未取得。FL真实TOP，load=0.224897，clearance=+0.605491mm。
- 首条策略行global21889，P06、tick3592，policy decision_count=1；physical core含prefix计数449，正好448+1。所有固定边界内行均为teacher_initialized_suffix且明确prefix_teacher_data_in_ppo_storage=false。
- 768行native 8-tick汇总全部verified，无在回合状态写入违规。教师前缀不计入上述768个优化决策。

## 首回合实际结果

- global21889–22643共755个PPO决策；终止tick9624 /80.2s，P09 **INCOMPLETE_CONTROLLER_BLOCKED**，episode return=−54.57763334709749。
- P09阶段450决策/30s用尽；task_success/full_task_success=false，time_outs=false，terminal_bootstrap_allowed=false。合法失败数据已进入global22656的完整更新，不是被门禁排除。
- terminal RR：GROUND=true，front−55.150774mm、clearance−50.188428mm、load=0.459342；soft=false、initial=false、hard-qualified/cross/place均false。
- terminal FL：TOP=true、load=0.540658、clearance−0.162798mm、连续TOP22samples。终止时承载不能倒推整个探索窗口一直承载。

## soft_air_actuation_earned 的实际获得/清除

这是为shaping提供的过程证据，**不是hard qualification或成功条件**。首回合RR的21条120Hz历史事件均为whole_body_initial_clearance，hard-qualified事件为0。15Hz决策末记录中可见**15个不同soft_air_earned_tick、28个earned=true样本**；它们是至少15次实际获得，不冒充所有物理tick可能出现的短暂事件总数。

| 记录的获得tick | 决策末仍true的tick区间 | 首个false决策末tick | 该false样本实际接触 |
|---:|---|---:|---|
| 3968 | 3968–3976 | 3984 | GROUND |
| 6070 | 6072–6080 | 6088 | GROUND |
| 6231 | 6232–6240 | 6248 | GROUND |
| 6689 | 6696–6704 | 6712 | obstacle |
| 6852 | 6856–6872 | 6880 | GROUND |
| 7333 | 7336 | 7344 | obstacle |
| 7370 | 7376 | 7384 | GROUND |
| 7440 | 7440–7456 | 7464 | GROUND |
| 7486 | 7488–7504 | 7512 | GROUND |
| 7548 | 7552–7576 | 7584 | GROUND |
| 7663 | 7664 | 7672 | GROUND |
| 8091 | 8096 | 8104 | GROUND |
| 8362 | 8368 | 8376 | GROUND |
| 8529 | 8536 | 8544 | GROUND |
| 9063 | 9064 | 9072 | obstacle |

获得tick由实际soft_air_earned_tick字段给出。清除时刻没有独立120Hz事件日志；只能定位在“最后true决策末tick、首个false决策末tick”之间的最多8tick区间。例如第一次是 **(3976,3984]**，不能伪写成精确tick3984触发撤销。15段下一采样均清零，其中12次观察到GROUND、3次观察到obstacle；没有把老的true资格永久保留。

- 第一次获得tick3968 /33.066667s：AIR、initial=true、无ground/obstacle，upward excursion=3.055722mm，own joint motion=1.368265deg、whole-body motion=5.954207deg、command motion=28.722839deg。RR front−386.843961mm、clearance−48.236898mm；hard资格仍false。
- tick3976仍保留相同earned_tick3968，clearance升至−46.109508mm；tick3984重新GROUND，initial与soft同时为false。
- 本回合最高无接触**决策末**净空在tick6696/global22277：clearance−37.715043mm、front−52.437779mm，AIR、soft earned_tick6689、initial=true。仍低于顶面37.715mm，不能报作合格抬升。
- 最后一次initial事件在tick9298 /77.483333s；末端soft/initial都已清除。全首回合RR hard-qualified、cross、place样本和事件均0。

## FL实际载荷及证据边界

- 首次RR soft获得时FL没有TOP接触、load=0；新信用不是因为历史FL placed便假设当前FL一定承载。
- 28个soft=true决策末样本中，FL有TOP且非零载荷4个，其余24个load=0；该组最大load fraction=0.152729。
- 首回合P09的450个决策末样本中：FL TOP=255、AIR=195。以上是15Hz采样计数，不伪装成全部3600个P09物理tick的接触统计。
- 新soft状态实际生效、反复获得并清除，是功能证据；它不是足够的学习效果或成功证据。本首报不推断新的确定性评估结果、不重分类任务失败、不修改hard evaluator或动作范围。

证据：固定global边界内的 `optimizer_updates.jsonl`、`residual_and_projection_audit.jsonl`、首条 `completed_episodes.jsonl`、首个 `policy_credit_start` 之前的 `prefix_evidence.jsonl`。唯一写入为本报告；无Python/Isaac启动、生产编辑或提交。

## 追加：第四回合突破，固定到已保存边界24960

此更新采用 `checkpoint_step_000024960_manifest.json` 及完整update24960/160作为边界。相对源21888，本run实际已有**3072个完成优化的策略决策 /24次新update /480次新optimizer step**；不是6144计划已全部完成。边界内累计阶段样本由audit重算：P06=1185、P07=5、P08=4、P09=1426、P10=1、P11=1、P12=450，其余0。

第四回合（episode_index=3）global24090–24855，**766个PPO决策**。该回合开始前仍是448决策/3584ticks的真实A教师前缀，接管为P06、29.866667s；RR仍GROUND，initial/soft/hard/cross/place均未取得，FL TOP load=0.224897。首条当前策略行global24090、tick3592、policy_count=1、core含前缀计数449；from_P01_current_policy=false。这是**教师初始化suffix**，不是自然P01完整策略回合。

### RR真实事件已完成优化，非仅采集尾部

| RR事件 | 实际历史event tick /时间s | 首个携带该事件的策略行 | 观测说明 |
|---|---|---|---|
| P06初始离地 | 3956 /32.966667 | 当前策略阶段内 | excursion3.390051mm；之后另有3983、5526、5698的初始事件 |
| 合格抬升 | 5742 /47.850000 | global24359，末tick5744 | 事件excursion50.639547mm，joint motion35.819680deg，above-top +0.719452mm |
| 合法跨线 | 5998 /49.983333 | global24391，末tick6000 | 末样本front+4.036847mm、clearance+8.123026mm、AIR，qualified/cross=true |
| 放置事件 | 6092 /50.766667 | global24403，末tick6096 | 末样本TOP=true、load0.196189、front+2.398375mm，placed=true |

这些事件最晚global24403，连根代理最初询问时的完整边界24448也已覆盖；不是把采集中的global24520提前计成已优化。随后24960边界覆盖了整个第四回合。

- RR最高决策末clearance=+25.155098mm（tick6064/global24399，跨线后AIR，front+6.376628mm）。这是真实训练轨迹中的RR单腿事件链，不等于全任务成功或最终策略确定性复现。
- RR资格、cross、placed是历史事件。P12期间首个**决策末**RR重新GROUND样本在tick7352 /61.266667s（global24560），front−50.138410mm、clearance−50.441361mm、load0.518881；历史标志仍true，但当前已不在TOP。
- 第四回合终止时RR仍GROUND，front−48.468974mm、clearance−49.493200mm、load0.190564。报告不把历史placed写成后续始终稳定承载。

### 跨阶段连续，不是新reset/重放入口

| 阶段转换 | 实际tick |
|---|---:|
| P06→P07 | 5488 |
| P07→P08 | 5496 |
| P08→P09 | 5504 |
| P09→P10 | 6096 |
| P10→P11 | 6104 |
| P11→P12 | 6112 |

第四回合全766行global及physics tick连续（每策略决策+8ticks），从3584连续推进到9712，没有阶段间reset；全部6128个策略physics ticks的native汇总verified、在回合写入违规0、teacher_credit=false。

P06→P12上述6个政策阶段handoff均记录handoff_hold_used=true、hard_safety_modified=false；最大servo action jump仅1.7764e−15deg，最大wheel action jump2.7756e−17rad/s（浮点舍入量级）。初次教师接管后的首策略步不是这6个阶段handoff：其P05→P06记录0.5deg/0.015rad/s，不能混写成所有首动作都零变化。

第四回合阶段样本：P06=238、P07=1、P08=1、P09=74、P10=1、P11=1、P12=450。P07/P08/P10/P11各只有实际1个策略决策，仍保留真实物理和actuator历史，不宣称长时间驻留训练。

### RL和FL：有进展，但后半任务未完成

- RL历史tick7、1723的initial事件属于教师前缀，不归功于本回合PPO。RR placed之后，当前策略首次RL initial/soft earned在**tick6308 /52.566667s**，首决策末tick6312/global24430，AIR、front−54.604050mm、clearance−43.675309mm。
- 全第四回合RL soft=true有83个决策末样本、23个可见earned_tick；最高无接触决策末clearance=−2.869574mm（tick6496/global24453，front−128.504392mm）。仍无RL hard qualification、cross或place事件/样本。
- FL在RR合格抬升的首末样本tick5744为AIR/load0；在跨线后tick6000为TOP/load0.050057；在placed后tick6096为TOP/load0.151593；进入P12的tick6120为TOP/load0.432133。这些是当前载荷，不从旧placed_FL标志推断。
- RR再GROUND的tick7352，FL尚TOP/load0.481119；但终止时FL已AIR/load0。因此不能把局部放置事件或较早的FL承载当作持续完整稳定性。

### 终止与checkpoint

- 第四回合终止global24855、tick9712 /80.933333s，**P12 INCOMPLETE_CONTROLLER_BLOCKED**，episode return=−57.56525047710726。RL终止时GROUND，front−50.974246mm、clearance−50.643945mm、load0.406424，soft/initial/hard/cross/place均未保留。
- task_success/full_task_success=false，scope=teacher_initialized_suffix，time_outs=false，terminal_bootstrap_allowed=false。此回合有RR真实子目标突破，但没有suffix episode success，更没有P01 full-task success。
- 边界24960另包含第五次P06接管后的105个已优化决策，末tick4424 /36.866667s；不与第四回合混算。
- 可重载文件：`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000024960.pt`。sidecar记录global24960 /累计PPO update160 /optimizer steps3200，save_load_round_trip=true，SHA256=`72d8b1cd29e67ef44bd3c3dd1eea45f3d6637cff43eba84bc42d84d6ab77b442`。
- 此追加只覆盖已完成并保存的24960边界，后续采集或优化由根代理继续；无生产代码修改、Python/Isaac启动或提交。

## 第六回合追加：两后腿真实通过，P13未受控停止（固定checkpoint27008）

本节固定读取global21889–27008，依据已完成`optimizer_updates.jsonl`和不可变checkpoint sidecar，不将更新边界之后的采集计入。该块当前可对账为**5120策略决策 /40次新PPO更新 /800次新optimizer step**；累计**27008 /176 /3520**。原6144请求仍由根代理继续，不能把预计28032写作已完成。

| 阶段 | 第六回合全部已优化样本 | 本run截至27008已优化样本 |
|---|---:|---:|
| P06 | 263 | 1726 |
| P07 | 1 | 7 |
| P08 | 1 | 7 |
| P09 | 83 | 1959 |
| P10 | 1 | 2 |
| P11 | 1 | 2 |
| P12 | 67 | 517 |
| P13 | 900 | 900 |
| 合计 | 1317 | 5120 |

第六回合为`episode_index=5`、global25552–26868。第七回合的140个P06决策亦已纳入27008，末tick4704 /39.2s，尚未终止，不与第六回合混算。

### 真实前缀与连续信用边界

- 第六次接管仍为P06 offset0；448个教师决策/3584个physics ticks后，在29.866667s交接。首个PPO决策末tick3592，最后tick14120；1317策略决策完整推进10536ticks，逐行时钟连续，无遗漏或重复。
- 首个PPO决策末RR仍GROUND：front−488.838386mm、clearance−50.844296mm、load0.081017；RL亦GROUND/load0.475626；FL当前TOP/load0.035282。接管时RR/RL均无hard/cross/place历史。FR/FL完成以及RL早期tick7、1723 initial属于教师历史，不归功于本次PPO。
- 全1317行`prefix_teacher_data_in_ppo_storage=false`、`from_P01_current_policy=false`。每行8tick原生审计全部验证，共10536已验证且有实际原生target effect的ticks；四类in-episode写入计数全零。原始prefix时间未从200s任务时钟扣除。
- 真实阶段切换tick依次为P06→P07 5688、P07→P08 5696、P08→P09 5704、P09→P10 6368、P10→P11 6376、P11→P12 6384、P12→P13 6920。七个后续交接均`handoff_hold_used=true`、`hard_safety_modified=false`，最大servo action jump仅1.332268e−15deg、wheel仅2.775558e−17rad/s；不存在阶段reset或重新加载精准历史姿态。教师P05→P06的首策略渐入不混作这七个零跳变交接。

### 两后腿：事件历史与当时真实接触

| 腿/事件 | 事件tick /任务时间s | 首个记录该事件的global（决策末tick） | 物理证据 |
|---|---|---|---|
| RR本credit首initial | 4536 /37.800000 | — | 实测正向excursion3.031842mm；之后仍有重试 |
| RR hard qualified | 6170 /51.416667 | 25875（6176） | excursion23.146197mm、joint motion15.672851deg、事件时above-top+0.299991mm |
| RR crossed | 6350 /52.916667 | 25897（6352） | 决策末AIR，front约+1.733mm、clearance约+3.741mm、load0 |
| RR placed | 6368 /53.066667 | 25899（6368） | RR当前TOP/load0.511756；FL当前TOP/load0.364789 |
| RL本credit首initial | 6426 /53.550000 | — | excursion3.404252mm；不是教师早期initial |
| RL hard qualified | 6710 /55.916667 | 25942（6712） | excursion30.088857mm、joint motion11.852394deg、事件时above-top+2.011659mm |
| RL crossed | 6852 /57.100000 | 25960（6856） | 决策末AIR，front约+4.931mm、clearance约+51.612mm、load0 |
| RL placed | 6914 /57.616667 | 25968（6920） | 两后腿当时都TOP：RR load0.133704、RL0.421226；FL亦TOP/load0.055133 |

上述两后腿事件最晚在global25968已记录，故在最初观察P13时的完整25984更新里已优化，**不是尚未优化的global26017采集才产生的突破**。RR qualified首记录时FL AIR/load0，RL qualified首记录时FL TOP/load0.546782；不从旧placed_FL推断持续支撑。

### P13按实际名义动作退场分段，不误判永久滚动

P13阶段从tick6920 /57.666667s开始，策略源行global25969起。逐行比较真实`nominal_action_full12`，只有以下两个连续段；第二段之后没有恢复非零nominal。

| P13段（15Hz决策末采样） | global /任务时间s | 样本数 | body速度超0.05m/s | angular超0.30rad/s | 实际wheel速度超0.25rad/s | commanded wheel超0.02rad/s |
|---|---|---:|---:|---:|---:|---:|
| 有限nominal仍在执行 | 25969–26240 /57.733333–75.800000 | 272 | 210 | 79 | 269 | 272 |
| 全12通道nominal持续为零 | 26241–26868 /75.866667–117.666667 | 628 | 383 | 44 | 183 | 626 |

- 首个P13决策nominal wheels为[0.175]*4；较早67s附近的[0.3]*4是有限源尚未退出，不是无限继承。全零段首行tick9104，nominal已全零，实际wheel residual为[0.021856,0.019976,0.143488,−0.033878]rad/s。
- 全零段628个决策末样本`final_region_valid`与`final_support_available`均true；`final_controlled`均false，采样到的`final_stable_for_s`最大0。两后腿同时实际TOP为231/628，不将余下样本的历史placed当作当前双后腿支撑。前段对应区域通过251/272、支持272/272、双后腿实际TOP259/272。
- 全零段只有global26290、26719的最大wheel command≤0.02（分别0.013661、0.013115rad/s），但其body速度分别0.118558、0.130289m/s，仍未受控；不能仅凭某次停轮命令较小宣告稳定。以上分布是15Hz决策末审计，不宣称穷举120Hz内每个短暂状态。
- 全零段真实command绝对峰值0.288996rad/s。这里的命令可直接归于实际projected residual/执行审计，不把此前nominal+residual合成命令都归于actor；也不由这些分布提出新的成功门槛或改变nominal。

### 正式终止与当前物理状态

- global26868、tick14120 /117.666667s，P13 stage age60s，正式结果**INCOMPLETE_CONTROLLER_BLOCKED**，episode return=−75.45826238403649；task_success=false、full_task_success=false、scope=teacher_initialized_suffix。time_outs=false、terminal_bootstrap_allowed=false。这是具有两后腿子目标进展的有效失败rollout，既不是suffix成功，也不是自然P01成功。
- 终止历史active/cross/placed四腿均true，但**当前RR为AIR/load0，RL为TOP/load0.487898**；FL AIR/load0，FR TOP/load0.512102。当前RR front+595.039431mm、clearance+0.271068mm；RL front+631.653491mm、clearance−0.761626mm。区域和至少两个实际支撑均通过，历史placed不等于四腿同时承载。
- 终止`final_controlled=false`：按原表达式顺序，首个未满足项是body linear speed **0.068714>0.05m/s**；同时最大command **0.133108>0.02rad/s**。body angular0.111517≤0.30、实际wheel max0.163151≤0.25均通过。全12 nominal为零，wheel residual/command为[0.017219,−0.133108,0.013788,0.074493]rad/s。home误差14.4632deg仅诊断，不是失败条件。
- 本节可重载文件为`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000027008.pt`；sidecar记录累计27008/176/3520、`save_load_round_trip=true`，checkpoint SHA256=`8ea4b507c3785b41b75814a55018139d78548d0dc205f73a8e67e96dbe2e10f0`。本报告读取sidecar既有hash，不重复运行大文件hash或任何Python。6144原预算仍在执行，后续结果未纳入本节。

## 最终整块对账：6144完成，checkpoint28032（覆盖全部八次真实接管）

本节在`run_manifest.json`正式结束后，以其`result`、checkpoint sidecar、48行`optimizer_updates.jsonl`、完整6144行策略audit、7行completed episodes和8次prefix证据交叉核对。只补最终事实，不改上文历史有界记录，不将随后启动的自然P01评估预填为成功。

### 真实优化、信用与物理审计总账

- 执行lifecycle=**SUCCEEDED**，字段明确`training_success_is_not_task_success=true`。requested=planned=actual=6144，unconsumed=0、rounding_overrun=0；N1/P06 offset0/seed1001、source checkpoint21888和原HEAD64abc均未在本run内改变。
- global21889–28032共6144行连续；48个完整128决策更新，新增960次optimizer step，累计**28032策略决策 /184次PPO更新 /3680次optimizer step**。48次均记录actor参数改变和有限非零梯度，相邻更新actor before/after hash及global+128链无不一致。
- actor参数hash：本run开始`238d76693f7f6533e84b3d862288dacbf30f88b27e52647ce96a90ffc1cb69df`，结束`f5ea5f3f2e4e51f201b7525da4885948a1fe5c588e202ff60fa049d2e9e189de`。这是真实优化结果，不以测试通过代替更新证据。
- 实际策略信用推进**49138 physics ticks**，所有tick原生审计verified且有实际target effect；6144行均无in-episode root pose/velocity/force/gravity写入，教师信用标志全false。不是49152ticks：第二、第三回合最后决策各仅执行1tick（global23364、24089），各少7tick后正常任务终止，逐回合时钟连续。
- **8次教师prefix全部accepted、无fallback**，每次448decisions/3584ticks，共**3584个教师决策 /28672ticks**。独立prefix JSONL逐行确认raw PPO action和projected residual全零、`policy_credit=false`、native audit全部验证、无in-episode写入；这3584个决策没有进入PPO storage/global。
- 含教师的物理core总账为**9728decisions /77810ticks /8episodes**，恰等于6144+3584及49138+28672。原200s时钟始终包括prefix；每次29.866667s后才开始当前PPO信用。
- `result.wall_time_s=3282.2445895`（约54分42.245秒）。telemetry另列reset wall30.4051242s、roll-in wall1240.1457101s；这些是各自记录的时间口径，不再盲目加到result wall上重复计时。

| 策略采样phase | 最终已优化决策 |
|---|---:|
| P06 | 2103 |
| P07 | 9 |
| P08 | 9 |
| P09 | 2482 |
| P10 | 3 |
| P11 | 3 |
| P12 | 585 |
| P13 | 950 |
| 合计 | 6144 |

### 全部八次接管：七次正式失败与一个未终结尾段

表内“有事件”只表示当次历史真实子事件，不保证末态仍有资格或接触支撑；所有事件均在当前PPO信用段内核对，教师FR/FL历史不计为PPO后腿成功。

| episode_index | global区间 | PPO决策 | 末阶段 /任务时间s | 正式结果 | 后腿事件与末态限制 |
|---|---|---:|---|---|---|
| 0 | 21889–22643 | 755 | P09 /80.200000 | INCOMPLETE_CONTROLLER_BLOCKED | RR无hard/cross/place；RR末GROUND，RL末AIR |
| 1 | 22644–23364 | 721 | P09 /77.875000 | INCOMPLETE_CONTROLLER_BLOCKED | RR曾hard qualified tick6230，无cross/place；末非AIR且非TOP，不以旧qualified tick当当前资格 |
| 2 | 23365–24089 | 725 | P09 /78.141667 | INCOMPLETE_CONTROLLER_BLOCKED | RR无hard/cross/place；末RR AIR、RL GROUND |
| 3 | 24090–24855 | 766 | P12 /80.933333 | INCOMPLETE_CONTROLLER_BLOCKED | RR hard/cross/place5742/5998/6092；后来回GROUND；RL未hard/cross/place |
| 4 | 24856–25551 | 696 | P09 /76.266667 | INCOMPLETE_CONTROLLER_BLOCKED | RR无hard/cross/place；末RR AIR、RL GROUND |
| 5 | 25552–26868 | 1317 | P13 /117.666667 | INCOMPLETE_CONTROLLER_BLOCKED | RR6170/6350/6368、RL6710/6852/6914；末RR AIR、RL TOP，未controlled stop |
| 6 | 26869–27576 | 708 | P09 /77.066667 | INCOMPLETE_CONTROLLER_BLOCKED | RR无hard/cross/place；两后腿末GROUND，FL AIR |
| 7（尾） | 27577–28032 | 456 | **P13 /60.266667** | **未终结，termination=null** | RR6066/6263/6267、RL6407/6756/6832；末双后腿当前TOP，但尚未受控停止 |

前七回合共5688决策；尾段456决策亦已完成优化，不是未优化的采集残留。正式completed episodes=7，全部INCOMPLETE_CONTROLLER_BLOCKED；`teacher_initialized_task_success_count=0`、`success_count=0`。历史RR hard出现于4次接管，其中3次cross/place；RL hard/cross/place出现于2次接管。这个描述不是自然P01成功率，也不是可靠成功checkpoint声明。

### 第八回合尾段：没有漏掉第二次双后腿到P13

- 尾段phase采样为P06=261、P07=1、P08=1、P09=73、P10=1、P11=1、P12=68、**P13=50**，总456。P12→P13实际发生于tick6832 /56.933333s；最后tick7232 /60.266667s，P13 stage age3.333333s。预算边界不是物理失败终止，不能虚构第八个completed episode。
- RR信用内first initial tick4384 /36.533333s；hard qualified tick6066 /50.55s，实测excursion21.475226mm、joint motion13.730716deg、above-top+0.164160mm；cross tick6263，placed6267。RL first initial6389 /53.241667s；hard6407 /53.391667s，excursion49.963422mm、joint motion7.296882deg、above-top+0.957767mm；cross6756，placed6832。
- 后续阶段真实连续切换tick为5672、5680、5688、6272、6280、6288、6832（P06→P13）；不是从P12精准历史入口重新reset。
- 最后当前RR TOP/front+78.466002mm/clearance−0.950499mm/load0.135796；RL TOP/front+130.033377mm/clearance−0.632373mm/load0.509111。FR TOP/load0.355094，**FL AIR/load0**。这是末态实际接触，区别于placed历史。
- 最后`final_region_valid=true`、`final_support_available=true`，`final_controlled=false`、stable0、task_success=false。body linear0.081593>0.05m/s、actual max wheel0.408138>0.25rad/s、max commanded wheel0.516135>0.02rad/s；body angular0.131167≤0.30。尾段仍在P13有限nominal运动早期，wheel nominal=[0.3]*4、residual=[−0.008289,0.041259,0.216135,0.008952]；不由此诊断永久滚动，更不预言若继续一定成功。

### 最终交付的真实checkpoint边界

- `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000028032.pt`，sidecar global28032、PPO184、optimizer3680、`save_load_round_trip=true`，SHA256=`9e1cc652ae19cbee87f0612051b62675a6397671b92d5ff8ccb18da344b355cb`（与根代理实测一致）。此前immutable checkpoints和本报告有界历史不覆盖。
- checkpoint明示`physical_env_state_saved=false`、`resume_physics=legal_reset_not_bitwise_continuation`：保存权重、优化器等训练状态，不声称重载可从第八回合当前P13物理状态原地继续。
- 本报告最终仅追加输出文档；未修改生产、未运行Python/Isaac、未提交。自然P01/seed2001的checkpoint28032确定性评估在根代理另一个run进行，其结果尚未写入本报告。
