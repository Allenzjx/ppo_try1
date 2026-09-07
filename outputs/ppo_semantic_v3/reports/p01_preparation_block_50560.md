# FINAL｜P01 preparation-credit课程：实际50560，训练执行完成、任务尚未成功

最终run manifest已SUCCEEDED（训练执行完成），实际immutable checkpoint **50560 /360次PPO更新 /7200个optimizer steps**：来源46464 /328 /6560，本run实际新增**4096决策 /32次更新 /640 optimizer steps**。3个完整回合共3377决策，均P09 INCOMPLETE_CONTROLLER_BLOCKED；第4回合719决策尾已优化但仍未终结P09。4096请求已耗尽，未消费0、舍入超额0。四回合均实际通过P06，但无RR/RL越沿/放置、无P10–P13样本、无任务成功。以下48512有界原文保留为历史，最终整块追加在文末；不预判另行运行的C50560自然P01固定mean评估。

## 来源与真实保存/优化边界

- Run：`runs/ppo_semantic_v3/train/20260906T1133239688401Z_g9d70aae58243_1e3e34207a9442c387f5123a55a5e121`；HEAD=`9d70aae58243b9fb2248d64561959e6a67901b44`。N1、seed1001、from_phase=P01、stage=full_episode、requested4096、semantic_version=v3。
- 这是显式new-MDP warm start，不是旧MDP exact resume。record列出的runtime变更仅`configs/ppo_semantic_v3/stage_task_spec.yaml`和`src/wlr50_clean/ppo/semantic_supervisor.py`，增加既有workspace的predecessor前准备信用；无新增reward权重、nominal/物理/history阈值变更。
- 来源checkpoint46464 SHA=`fc1fd6710965d627aff6e6ecc91e02c5ed9c144f7637cba9a50cfa079cf5513d`。保存所有actor参数（包括已学习state-dependent std）、critic及相同324观测/12raw动作/identity normalizer；old_rollout_buffer_inherited=false、physical_state_inherited=false，恢复已核实RNG后采新MDP rollout。Adam moments按显式MDP边界清空、initial LR3e−5；这不是悄悄重置网络或继续旧buffer。
- source/origin与已花预算保持：原v3 origin10112，48512阶段账full_episode14848（本run前12800+2048）、phase_suffix23552、smoke0。改变reward语义并没有重发预算。
- checkpoint48512 SHA256=`285dd33f0a72c3144cc193257bde0830a32bc185fd88958b6efe3fb73985d8db`，本次PowerShell实际文件hash与pointer/sidecar匹配；sidecar `save_load_round_trip=true`。sidecar SHA由pointer记录为`ccf92afc69e2188fd0490381a87eefb4abdfc92496a876d51b9c01928e5bccf0`。
- 仅纳入global46465–48512的16个完整128决策rollout；update329–344全部actor_parameters_changed=true、finite_nonzero_gradient_observed=true、各20个optimizer steps，global+128及相邻actor hash链无不一致。首update的actor_before=`571f19b7ccc64593456eb4a90cd9163c38b273d2e65305b2b87f27b6c41df205`，与source46464 actor相同；边界actor=`0eb13040406392f3b4ec2cbcf5648b3f4bd04fce102f425b44ae53f20b8d701a`。
- 16次完成update的实际LR均1e−5，为原adaptive更新后的记录，与initial3e−5分开。最后update344：gradient norm1.011574814～1.414213339，KL0.0225843022、clip fraction0.284375、entropy−5.56472468、value loss0.00872399622、surrogate loss−0.0349934289。这里记录真实更新，不拿测试通过代替optimizer evidence。

## 全P01信用、native和raw分布绑定

两次实际episode均从P01开始：第一条global46465、第二条global47615都对应decision_count1、episode tick8 /0.066666667s。2048条逐行decision_count、physics_tick=8×decision_count、120Hz时钟及global连续性均匹配；无short interval，无time_outs。此边界**2048 PPO决策=16384实际物理ticks**，没有teacher物理tick或teacher policy credit混入。

确切依据是实际调用参数from_phase=P01、checkpoint `sampling=P01_full_task_only_initial_version`、`curriculum_epoch.prefix_request=null`、每次从tick8计入第一条PPO行为、以及本run没有`prefix_evidence.jsonl`。自然P01分支没有suffix专用`prefix_teacher_data_in_ppo_storage/curriculum_start`字段；不把“字段缺失”伪当作读取到false，而是以上独立执行/时钟证据共同确认不运行teacher。

2048条policy audit全native verified=16384、actual target effect=16384、own-phase request effect=16368。16tick归属排除不计作当期独立request effect；无未验证native或非零root pose/root velocity/force/gravity写入。接口有效不等于任务可完成。

全部行包含12维finite mean、12维finite且正std。以各自raw/mean/std独立重算12维Gaussian old log-prob，最大误差**1.3946199999e−6**；没有以投影后动作代替RSL原始sample。全通道std范围0.106615014～0.195385709，RR hip/knee分别0.142136618～0.159585029 /0.144520417～0.186022967，RL hip/knee分别0.107574299～0.158453181 /0.135271832～0.178963035。这些范围混合状态和在线权重，不能单独归因mean/σ导致物理进展或失败。本报告仅重算JSONL，不声称重新反序列化tensor rollout。

## 两次自然P06通过，与首回合真实未完成链

| 回合 | 已优化global范围 | 决策/ticks | 边界结果 |
|---|---|---|---|
| ep0完整 | 46465–47614 | 1150 /9200 | tick9200 /76.666667s，P09 INCOMPLETE_CONTROLLER_BLOCKED |
| ep1尾 | 47615–48512 | 898 /7184 | tick7184 /59.866667s，P09未终结，reason=null |
| 合计 | 46465–48512 | 2048 /16384 | 1个完整失败+1已优化物理尾段，success0 |

**P06真实准备完成不是标签跳转：**

| 回合 | P06策略样本 | P06→P07 tick/time/global | RR/RL front（m） | 当时后腿history |
|---|---:|---|---|---|
| ep0 | 367 | 5568 /46.4s /47160 | −0.193440225 /−0.219808009 | RR/RL Q/C/P均false |
| ep1 | 314 | 5168 /43.066667s /48260 | −0.192305204 /−0.214472559 | RR/RL Q/C/P均false |

两条真实transition均`completion_values.rear_approach=1`、reason=`current physical goal set satisfied`、terminal=false；两轮都由本策略自P01得到FR/FL放置并进入准备，没有teacher免费前驱。进入P07不结束episode，不借阶段label发终止奖。两次通过也不是新增信用的配对因果验证：当前为seed1001随机训练且期间持续更新，不同于之前checkpoint的seed2001固定mean评估。

前1024条新增reward依赖的独立重建见同目录`preparation_credit_live_reward.md`：P06中旧orderedΦ之外仅新增`.02125*workspace_RL`，保存新Φ重建误差≤1.21e−16，既有`5*(.995*Φ_after−Φ_before)`误差0。该报告还记录同旧Φ下RL接近/退回的真实差值。此处引用其固定47488边界证据，不将它误扩为本2048行全部进行了同项重建；本次全2048核验的是信用、phase/native/分布及更新链。

### ep0：RR合格后落地撤销，未越沿/放置

- 实际FR Q44/C1423/P1447，FL Q1541/C2557/P2630；都发生在当前策略从P01的行为段。完成前腿放置是历史事件，随后是否承载必须另读当前contact。
- RR first initial tick4969在P06准备中，之后5654/5777/5881等初始离地尝试；hard Q6080 /50.666667s，随后**6342 /52.85s GROUND-before-cross资格撤销**。未再得到hard Q，整回合无RR cross/placed。RL有多次initial，但无hard Q/C/P；重复initial不等于已合格。
- 原终局为P09 INCOMPLETE_CONTROLLER_BLOCKED，PPO return−57.67625997305633；首未完成仍是RR在保有合格抬升后实际越沿/放置，不是P06准备条件尚未满足。
- 终止tick9200时RR AIR但front−101.723131mm/clear−46.540851mm/load0、active=false；AIR不代表 above-top或保有资格。RL GROUND，front−254.323598mm/clear−50.610157mm/load0.508639。FL当前TOP但load仅0.015981（front+258.481387mm/clear+.328747mm），不能据TOP标签声称承担主要支撑。
- 无RR/RL placed、无后续P10/P12/P13样本，task/full-task success=false；保留原INCOMPLETE，不另造wheel-only等分类。

### ep1已优化尾：到P09，但截至48512尚未qualified RR

第二轮从真实新P01 reset重新建立FR Q50/C1459/P1467、FL Q1563/C2591/P2653；不继承第一轮RR history。P06实际314决策通过后，已有250条P09决策，但截止tick7184仍无RR/RL hard Q/C/P。

边界RR GROUND，front−54.792192mm/clear−48.577550mm/load0.040983；RL GROUND，front−237.808045mm/clear−51.279703mm/load0.472715；FL AIR/load0（front+261.966828mm/clear+12.800341mm）。FR/FL历史placed并不表示当前FL还在TOP。PPO return−10.90511100956875，terminal=false、reason=null；既不能预写第二轮失败，也不能因它已进P09称成功。

## 截至48512准确phase ledger

| source phase | ep0完整 | ep1已优化尾 | 总已优化样本 |
|---|---:|---:|---:|
| P01 | 1 | 1 | 2 |
| P02 | 176 | 179 | 355 |
| P03 | 4 | 4 | 8 |
| P04 | 1 | 1 | 2 |
| P05 | 147 | 147 | 294 |
| P06 | 367 | 314 | 681 |
| P07 | 3 | 1 | 4 |
| P08 | 1 | 1 | 2 |
| P09 | 450 | 250 | 700 |
| P10–P13 | 0 | 0 | 0 |
| 合计 | 1150 | 898 | 2048 |

本块确实补入自然P01前驱准备的策略信用，也仍缺RR通过后的实际样本；不能根据两次P06通过推断全任务稳定性。余下2048请求与最终50560未纳入本有界账。报告完成后停写，等待根代理给出实际终局边界。

所有操作限PowerShell读取与此独占报告写入；没有Python/Isaac并行、没有生产/master修改或提交。

## 最终追加：50560 /360 /7200，全P01实际4096决策完成

本节来自已关闭的本run最终`run_manifest.json`、checkpoint50560 sidecar、32条optimizer更新及4096条policy audit。根代理确认session68920 exit0；SUCCEEDED仅指训练执行成功。旧48512原文的“半块/第二回合尾”保留当时事实，下述补齐其后真实发生的数据。

### MDP迁移链、源checkpoint保护与实际优化

- source46464→target50560；requested/planned/actual均4096，32次完整128决策update、640 optimizer steps。`unconsumed_requested_policy_decisions=0`、`rounding_overrun=0`；wall **1553.4874382000417s（约25.8915分钟）**。
- checkpoint50560 SHA=`6ccf5b26784e4db7056b1ba6521d0ac34c3c49cf915ff618d4410c9f14d7fd99`，sidecar SHA=`69bb996011ce3f12730bcfc0511887174631e669dd5600366ec1e5bf74f88702`。根代理已实际核验两者；本报告读取sidecar确认累计50560/360/7200及roundtrip=true，不重复hash最终文件。
- 源46464 immutable文件本次另用PowerShell实际hash，仍为`fc1fd6710965d627aff6e6ecc91e02c5ed9c144f7637cba9a50cfa079cf5513d`，与迁移前记录一致。原checkpoint未被覆盖成新MDP版本。此次只有明确版本的workspace准备potential依赖变更；旧观测/动作/网络参数继续使用，不将旧rollout或物理环境状态带入新MDP。
- 最终policy仍`heteroscedastic_log_v1`；原v3 origin10112保留，full_episode16896（本run前12800+4096）、phase_suffix23552、smoke0。不是重新初始化actor，也不是重发预算。stage/full P01 metadata、`physical_env_state_saved=false`均与实际执行路径一致。
- 32次真实update329–360全部actor改变、finite_nonzero_gradient_observed=true、每次20 optimizer steps；global+128及相邻actor hash链连续。首actor-before仍是源46464的`571f19b7ccc64593456eb4a90cd9163c38b273d2e65305b2b87f27b6c41df205`，最终actor=`2f2bfb60ae85605afdbf50d919d6b6b16b49f33400bf9dc70d404abb13d9c207`。
- 每次完成update实际LR均1e−5，区别于new-MDP Adam初始化3e−5。末update360：gradient norm1.002003804～1.151601316、KL0.0173583050、clip fraction0.2640625、entropy−5.35651011、value loss0.00295676495、surrogate loss−0.0409618979。

### 无teacher的全部物理样本与RSL分布证据

4096行policy global46465–50560连续，真实物理ticks=**32768**；core同为4096决策/32768ticks/4个episode，没有额外teacher/reset-roll-in决策。四次首信用global46465/47615/48713/49842，均P01、decision1、tick8；逐行decision_count、tick=8×decision_count、120Hz时间检查全一致，无time_outs、无short interval。原始suffix专用字段仍未出现在自然P01路径中，不将缺字段当作读取到false；`prefix_request=null`、P01-full sampling、无prefix文件、完整core/策略计数和从tick8起信用共同支持teacher=0。

全部32768ticks native verified且有actual target effect；own-phase request effect32736，32tick归属排除不冒充当期独立请求。全部行无未验证审计或非零四类in-episode状态写入。4096行12维mean finite、12维std finite/positive；按各自raw重新计算Gaussian old log-prob最大绝对误差**1.5347344489e−6**。无投影后动作替换raw采样的证据；本报告不额外重载tensor storage。

全部通道σ范围0.104595229～0.211361378；RR hip/knee分别0.139086053～0.159585029 /0.144520417～0.186022967，RL hip/knee分别0.107571997～0.162859634 /0.133032814～0.178963035。范围跨状态/更新，不能单独区分探索噪声与mean偏置的因果贡献。前1024的workspace公式验证仍以独立`preparation_credit_live_reward.md`为证，不误称此4096流式核验重做了其完整公式重建。

### 3个完整P09失败与第4回合719决策尾

| ep | global范围 | PPO决策/ticks | 终止/当前时间 | 实际结果 |
|---|---|---|---|---|
| 0 | 46465–47614 | 1150 /9200 | tick9200 /76.666667s | P09 INCOMPLETE_CONTROLLER_BLOCKED |
| 1 | 47615–48712 | 1098 /8784 | tick8784 /73.2s | P09 INCOMPLETE_CONTROLLER_BLOCKED |
| 2 | 48713–49841 | 1129 /9032 | tick9032 /75.266667s | P09 INCOMPLETE_CONTROLLER_BLOCKED |
| 3尾 | 49842–50560 | 719 /5752 | tick5752 /47.933333s | P09，terminal=false、reason=null |
| 合计 | 46465–50560 | 4096 /32768 | 3次完整终止3377 + 尾719 | task success0 |

全部四回合均由当前策略从P01建立前腿历史。FR/FL最终placed tick依次为ep0=1447/2630、ep1=1467/2653、ep2=1549/2726、ep3=1529/2715；不存在teacher带入或跨reset继承。

- ep0事实不变：RR初始4969，hard Q6080→GROUND撤销6342，未C/P；RL无hard Q/C/P。
- ep1后200个决策补齐旧898尾：最终1098，整回合RR无hard Q/C/P；RR终止GROUND，front−120.073106mm/clear−48.856935mm/load0.236408；RL GROUND，front−294.370469mm/clear−51.040837mm/load0.318131；FL当前TOP/load0.129893。PPO return−56.76891884721516。首未完成为RR合格抬升/越沿，不是P06。
- ep2同样RR/RL无hard Q/C/P；终止RR GROUND，front−110.808965mm/clear−49.702289mm/load0.563673，RL GROUND/front−284.430366mm/load0.021884，FL TOP/load0.414443。PPO return−57.19912248166586。不同分载不据此归为相同噪声机制，仍保留原P09 INCOMPLETE分类。
- ep3尾719已进入最后完整update，但物理回合尚未终结。RR AIR，front−184.204396mm/clear−47.519843mm/load0，RL GROUND，front−195.230689mm/clear−51.578658mm/load0.521872，FL当前AIR/load0、clear+153.000405mm。FL历史placed不能冒充当前TOP；RR AIR且低于top不是hard Q。当前RR/RL无Q/C/P，PPO return−7.692259478383993；只在P09采样18决策，不预写为第4次失败。

整块只有ep0一次RR hard Q，之后撤销；四回合全无RR/RL cross/placed，未到P10–P13。首次未完成已从先前单次确定性C46464的P06条件，变为本块三条随机训练回合的RR P09任务，但策略采样、seed和权重更新不同，不能把这项结果差异直接归因新增reward。

### 四次P06真实通过与最终phase ledger

| ep | P06样本 | P06→P07 tick/time/global | RR/RL front m |
|---|---:|---|---|
| 0 | 367 | 5568 /46.4s /47160 | −.193440225 /−.219808009 |
| 1 | 314 | 5168 /43.066667s /48260 | −.192305204 /−.214472559 |
| 2 | 336 | 5416 /45.133333s /49389 | −.217160520 /−.204043564 |
| 3 | 358 | 5584 /46.533333s /50539 | −.202892151 /−.188757804 |

四条transition均rear_approach=1、terminal=false；使用原RR/RL workspace条件，不是改变nominal或跳过准备。阶段完成不是新episode，也没有被当成全任务成功。

| source phase | ep0 | ep1 | ep2 | ep3尾 | 最终优化样本 |
|---|---:|---:|---:|---:|---:|
| P01 | 1 | 1 | 1 | 1 | 4 |
| P02 | 176 | 179 | 189 | 187 | 731 |
| P03 | 4 | 4 | 4 | 4 | 16 |
| P04 | 1 | 1 | 1 | 1 | 4 |
| P05 | 147 | 147 | 146 | 147 | 587 |
| P06 | 367 | 314 | 336 | 358 | 1375 |
| P07 | 3 | 1 | 1 | 1 | 6 |
| P08 | 1 | 1 | 1 | 2 | 5 |
| P09 | 450 | 450 | 450 | 18 | 1368 |
| P10–P13 | 0 | 0 | 0 | 0 | 0 |
| 合计 | 1150 | 1098 | 1129 | 719 | 4096 |

此训练块真实执行已完成，但不存在“全任务成功”或“稳定性胜过A”的新证据。根代理已启动独立C50560自然P01固定mean评估，本报告不预填其阶段/结果，不拿当前训练统计替代它。

最终报告至此定稿并停写。仅PowerShell只读和此报告编辑；未运行Python/Isaac、未修改生产/master、未提交。
