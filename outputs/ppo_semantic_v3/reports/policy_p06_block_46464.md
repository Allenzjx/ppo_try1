# FINAL｜State-dependent log-std P06课程：实际完成46464，训练完成不等于任务成功

最终run manifest为**SUCCEEDED（训练执行完成）**：源38272→实际已保存46464，新增**8192决策 /64次PPO更新 /1280个optimizer steps**；累计**46464 /328 /6560**。10次INCOMPLETE_CONTROLLER_BLOCKED共8032决策，另有160决策已优化但未终结的P06尾段；suffix/full task success均0，无P13样本。请求8192已耗尽、未消费0、舍入超额0。下方38784、39296、40832、45440原有界报告保留为历史附录，最终追加见文末。**后腿子事件属于教师初始化后缀，不是自然P01全任务成功，也不证明sigma改动导致进步。新checkpoint的自然P01确定性评估独立进行，本报告不预填其结果。**

## 来源与独立policy迁移：0新增训练计数，不重置预算

- Run：`runs/ppo_semantic_v3/train/20260906T0946289788164Z_g49749aa527a4_d866c10e684d4248b10521cb7d32bf09`；HEAD=`49749aa527a4a00901e434104821d7e8241ea8da`。N1、seed1001、P06 offset0、stage=phase_suffix、planned8192；显式`policy_distribution_migration=true`。
- 源为38272 /264 /5280，旧策略`gaussian_scalar_v1`→新`heteroscedastic_log_v1`。324观测/12 raw动作、原投影与物理MDP保留；该变更不是new-MDP warm start，也不复用旧rollout。
- 独立immutable initial文件：`checkpoint_initial_policy_from_000038272_sf8b6e1923328_g49749aa527a4_b6fb79ec05cd8925253063170b07aedd10afc0c8d0cc2cd1bb6e37825c365391_p54871747db0d.pt`。sidecar stage=`initial_policy_distribution_migration`、roundtrip=true、SHA=`052ca0b75082fa4c34c1112bb4a3c127c427a71bd0d735b4e572dce1efd5e251`（实际文件核验由根代理完成，此处不重复hash）。
- initial保留global38272 /264 /5280，stage账full_episode12800、phase_suffix15360、smoke0，原new-MDP origin仍10112。没有因策略头变更重新发放训练预算。初始migration evidence明确policy decisions/physical steps/PPO updates/optimizer steps added均0、rollout_step=0、fresh_Adam_no_old_moments、source RNG restored=true、源真实LR=1e−5。
- 真实当前Isaac观测上，映射前后mean最大差0、value最大差0、std最大差1.4901161194e−8、KL最大绝对值4.2632564146e−14；hidden/mean权重逐tensor保留，std-head weights初始全0，log-std bias由源12σ换算，sigma_clipped=false。映射没有采样或推进物理，不声明后续随机轨迹逐位相同。
- 注意“迁移新增物理步0”指映射/比较操作本身；为了得到当前合法P06观测，后述教师roll-in已经真实执行3584ticks，不能从映射计数0误推整个run没有prefix物理消耗。

## 教师初始化与PPO信用边界

首次prefix实际448决策/3584ticks，所有`reset_only_prefix_decision`的policy_credit=false，raw/projected residual全0，3584ticks native verified。没有将这些教师行为放入PPO storage或global计数。

- 接管发生于episode tick3584 /29.866666667s，requested=actual=P06、offset0，remaining task time170.133333333s；整个200s任务时钟包含prefix，不在接管时重新计时。
- 真实handoff证据source_control_tick3583、teacher_calls3584、target_first_observed_tick3584。command_physics_tick3763属于包含reset settle的dispatch时钟，与episode tick3584不是同一原点，不能据179tick差值宣称时钟跳变。
- 接管原始观测：RR GROUND，front−494.883957mm、clearance−50.949587mm、ground6.553654N/load0.228028；FL真实TOP，obstacle6.463661N/load0.224897。FR/FL已有Q/C/P是teacher形成的历史，RR/RL均无Q/C/P。
- 第一条PPO行为global38273、decision1，推进到tick3592 /29.933333333s；RR仍GROUND、FL已AIR/load0。接管时TOP不能冒充随后一直承载。
- 截至global38784：core包含prefix共960决策/7680ticks；其中448/3584是teacher、512/4096是当前策略。512条PPO audit均明确`prefix_teacher_data_in_ppo_storage=false`、`task_result_scope=teacher_initialized_suffix`。

## 已保存优化边界与真实native审计

| durable checkpoint | 新增决策 /PPO /optimizer | 累计PPO /optimizer | 实际LR |
|---|---|---|---:|
| initial38272 | 0 /0 /0 | 264 /5280 | 1e−5 |
| 38400 | 128 /1 /20 | 265 /5300 | 5.0625e−5 |
| **38784（本报告固定）** | **512 /4 /80** | **268 /5360** | **1e−5** |

38400 SHA=`497b27efd6ac568e0d5907ac1e8a054320e41327de03ef2ca1c09ed6199e0468`、roundtrip=true。38784 SHA=`0c7e7099ef779e7f2de5f1f0a08412b7c9808e54cdc38146193c99ef75b3a873`，本次PowerShell实际hash与sidecar/pointer匹配、roundtrip=true。中间38528/38656的优化更新行属于这4次已完成更新，不把超出38784的采集中行纳入本账。

4次真实update265–268均actor_parameters_changed=true、finite_nonzero_gradient_observed=true、各20个optimizer steps，actor hash链连续。初始actor SHA=`3ba295ffd216eb1c2e87f476559f36c8ff89e122979c607bcb41583597fba517`，38784 actor SHA=`8420a8db5b85c76f63672e7fdaea7d4cf707db98f420a0f3e78c300dc62033b8`。LR按原adaptive机制先升至5.0625e−5，之后回到1e−5；没有在报告中改写为额外调参。

512条实际策略audit推进4096ticks，verified=4096、actual native target effect=4096，无未验证记录；四类in-episode root pose/velocity/force/gravity写入全0。加教师段共7680ticks。策略/任务失败与接口失败严格分开；本边界尚无terminal。

## 已采样的12维mean/std与raw/log-prob绑定

512条已优化审计都具有12个finite mean、12个finite且正的std，以及raw action、old_log_probability/value、reward/terminal和applied/native证据。PowerShell用同条raw/mean/std按独立12维高斯重算log-prob，最大绝对差**1.2142847599e−6**（double重算对比原float计算）；没有拿projection后的动作替代PPO原始sample。

本报告不使用Python打开rollout张量文件；目录已存在各实际rollout文件，训练端完整storage绑定由当前运行代码硬校验，本报告额外验证的是可读JSONL的逐行分布/log-prob一致性，不伪称重新反序列化了tensor storage。

迁移刚完成的首批128条std逐通道完全恒定，与零std-head weights的初始化一致。完成第一次真实update后，在**同一个第二rollout**里出现以下各通道std范围；这是固定该rollout权重下不同实际观测对应的分布输出，不等于已证明有利的探索安排。

| canonical raw通道 | 首批σ（128条恒定） | 第二rollout σ范围 |
|---|---:|---|
| FL hip | 0.145220608 | 0.142416462–0.145737678 |
| FL knee | 0.158400297 | 0.158002704–0.161563993 |
| FR hip | 0.149555817 | 0.149675950–0.152592823 |
| FR knee | 0.148399696 | 0.148785308–0.151941165 |
| RL hip | 0.157569066 | 0.155591041–0.157769427 |
| RL knee | 0.159262627 | 0.155482903–0.156797543 |
| RR hip | 0.166641071 | 0.163229555–0.166638210 |
| RR knee | 0.171698615 | 0.170391247–0.174759850 |
| FL wheel | 0.156868264 | 0.156780720–0.159360215 |
| FR wheel | 0.164305925 | 0.162543789–0.163223311 |
| RL wheel | 0.150284633 | 0.147750989–0.150922865 |
| RR wheel | 0.143310592 | 0.142328173–0.143339440 |

σ属于unbounded Gaussian latent，不是物理deg或rad/s幅度；后续仍通过原tanh/projection/mapper与真实native约束。mean也随状态变化；有限范围表不代表在相同状态配对比较了旧、新策略表现。

## 后腿已优化样本与首回合局部事件

| source phase | 本边界真实优化样本 |
|---|---:|
| P06 | 256 |
| P07 | 1 |
| P08 | 1 |
| P09 | 91 |
| P10 | 1 |
| P11 | 1 |
| P12 | 161 |
| 其他 | 0 |
| 合计 | 512 |

首回合RR的以下事件均晚于teacher接管3584，属于当前PPO行为段，不是teacher RR历史：initial5719/5820/5848/5914；**hard Q6221（51.841667s，gain11.283197mm、joint motion20.553484deg、above-top+.299514mm）→cross6368→placed6369**。

但到本边界tick7680 /64.0s、P12：RR已GROUND、front−65.151062mm、clearance−49.564036mm、load0.401598；其历史Q/C/P仍true，不代表当前TOP。FL当前TOP/load0.469238；RL既非AIR、亦非GROUND/TOP，load0.129164，且无hard Q/C/P。首个尚未完成后腿环节是RL合格抬升/越沿/放置，同时RR放置保持也已失去。未终结不能填写为INCOMPLETE或成功，当前task_success=false。

仅凭这一条随机后缀、4次在线更新，不能归因“state-dependent sigma导致RR进展”；旧策略在先前训练与评估也曾出现RR子事件，入口、随机动作和权重序列不同。运行继续由根代理控制，本报告不暂停优化器、不加门槛、不预记剩余7680请求量已训练。

本报告仅PowerShell读取现有记录并写入此独占报告；无Python/Isaac并行、无生产/master修改、无提交。后续终局和最终整块结果须在新的真实完整保存边界追加。

## 有界追加至39296：首回合正式终局 + 第二回合223决策尾段

本追加严格截至global38273–39296，包含8个完整128决策rollout/update。后续采集/更新不计入本账；训练仍由根代理继续运行。38784附录中“首回合未终结”是当时事实，现由下述真实terminal补齐。

- 实际checkpoint39296 SHA256=`de00d5fe892f030f84aa1f9024f7e79355e44a785e9edfe0e6134c5a68e5b2c3`，本次PowerShell实际文件hash与pointer/sidecar一致，`save_load_round_trip=true`。
- 新增1024 /8 /160，累计39296 /272 /5440；8次更新均actor改变、有限非零梯度，global+128及相邻actor hash链无不一致。边界actor SHA=`8b993f7e8826c79f0074483d19d94c3968139a8c3ebdc554d3b972e7ce72ca11`，实际LR1e−5。
- stage账full_episode12800未变、phase_suffix16384（源15360+1024）、smoke0，origin10112未变。本run8192请求尚有7168不在本已保存账内，未预记其已完成。
- 1024条policy audit推进8192ticks，native verified/effect均8192、四类in-episode写入全0、teacher_in_PPO_storage为false。12维finite mean/positive finite std全部存在；高斯log-prob重算最大绝对差仍1.2142847599e−6。
- 两次真实教师prefix均448决策/3584ticks，合**896决策/7168ticks**；所有teacher raw/projected残差0、policy_credit=false、native verified7168。加本段策略，实际core累计**1920决策/15360ticks**。teacher和PPO分别记账，不给prefix额外学习信用。

| 回合 | 已优化global范围 | PPO决策/ticks | 教师决策/ticks | 当前结果 |
|---|---|---|---|---|
| ep0 | 38273–39073 | 801 /6408 | 448 /3584 | tick9992 /83.266666667s，P12 INCOMPLETE_CONTROLLER_BLOCKED |
| ep1尾 | 39074–39296 | 223 /1784 | 448 /3584 | tick5368 /44.733333333s，P06未终结 |
| 合计 | 38273–39296 | 1024 /8192 | 896 /7168 | 1个完整失败 +1个未终结尾段，success0 |

### ep0完整后腿事件与真实终止

RR事件仍为PPO段Q6221→C6368→P6369，全部发生在teacher接管3584之后。其放置保持很短：**首个decision-end已见RR重新GROUND的样本是tick6440 /53.666667s（global38629）**，front−51.846858mm、clearance−50.962309mm、load0.593361；FL TOP/load0.406639。这是15Hz策略末采样定位，不伪称该tick一定是120Hz精确最早落地。

RL全回合29次initial中，tick7/1723两次由teacher产生并排除PPO信用；其余27次在当前PPO段（包括P06准备tick3695及后续P12尝试），**没有任何hard Q/C/P**。重复initial不能记作27次合格抬升。P12决策末AIR样本最高净空为tick6400的−11.534123mm，front−124.765124mm；该数是15Hz最高样本，不是完整120Hz连续峰值。

ep0真实终止tick9992 /83.266666667s：

- 原结果`INCOMPLETE_CONTROLLER_BLOCKED`、phase/end_phase=P12，policy reward return−58.065344988904364；包含teacher的physical episode return−59.3241215864556另记，不能把二者混成PPO回报。
- RR当前GROUND，front−64.038422mm、clearance−49.708201mm、load0.375561；其历史Q/C/P保留不代表TOP。
- RL当前障碍pair active，非AIR、非GROUND、非TOP，front−49.884346mm、clearance−47.343382mm、load0.095140；仍未越沿，不能把有载荷误称为顶面放置。FL当前TOP/load0.441577。
- 共享physical evaluator success=false、final_region=false、final_controlled=false，其额外hard termination_reason=null；不另造wheel-only或body collision。task/full_task success=false，time_outs=false、terminal_bootstrap_allowed=false。
- 首个未完成环节为RL合格抬升/越沿/放置；同时RR先前放置已失去。这里确认后腿接续仍未完成，不是当前策略从P01全程成功。

### ep1尾段与最终本次phase ledger

ep1通过新的真实448决策teacher prefix后才开始global39074；边界223个PPO决策全在P06，physical core decision_count671=448+223，tick5368。RR/RL无hard Q/C/P；RR GROUND/front−246.549007mm/clearance−50.632617mm/load0.029590，RL GROUND/load0.559657，FL AIR/load0。termination_reason=null、task_success=false。尚未终结不能预写为失败或成功，也不能把ep0的RR历史跨reset带到ep1。

| source phase | ep0完整 | ep1尾 | 截至39296已优化 |
|---|---:|---:|---:|
| P06 | 256 | 223 | 479 |
| P07 | 1 | 0 | 1 |
| P08 | 1 | 0 | 1 |
| P09 | 91 | 0 | 91 |
| P10 | 1 | 0 | 1 |
| P11 | 1 | 0 | 1 |
| P12 | 450 | 0 | 450 |
| P01–P05/P13 | 0 | 0 | 0 |
| 合计 | 801 | 223 | 1024 |

第一条后缀轨迹有RR子目标进展、随后失去支撑并在RL阻塞，但没有同入口/同随机数/冻结旧权重的对照，不将其归因于新sigma头；旧mean/critic映射保持及σ可随状态变化的实现证据也不等于效果提升证据。本次到39296的有界报告完成后停写，等待根代理指定下一真实保存边界；不密集轮询、不运行Python/Isaac、不改生产/master、不提交。

## 有界追加至40832：2560决策，前三回合完成，第4回合仍未终结

此节仅统计global38273–40832；以实际immutable checkpoint为边界，不把更晚的采集或优化行计入。请求8192中本账实际2560，剩余5632尚未计为完成；46464仍只是计划终点。

- checkpoint40832实际SHA256=`4afab9654294f1ac83aec6079b3b472a4c617e46111b6b74b4a2af11d29e73f4`，本次PowerShell文件hash与sidecar一致，roundtrip=true。累计40832 /284 /5680；新增20个完整128决策update，各20个optimizer steps，共400。
- 全20次actor改变、有限非零梯度、global+128和actor hash链均验证无不一致。边界actor SHA=`1d6971d119e547e18c5fdc0a7da8b854a3482eb02c13606825549f91db629679`；当前LR1.5e−5，本run截至该边界LR范围[1e−5,5.0625e−5]，仍为原adaptive机制。
- policy_contract仍`heteroscedastic_log_v1`，origin10112未变；full_episode12800、phase_suffix17920（源15360+2560）、smoke0。没有将policy转换initial或teacher重新计入预算。
- **2560条policy audit /20480ticks**，全部native verified且有actual target effect，四类in-episode写入0，global及每回合物理时钟连续，3条真实terminal。无short decision interval。
- 共4次实际teacher prefix，各448决策/3584ticks，总1792决策/14336ticks；全部零raw/projected、policy_credit=false、native verified。与PPO合计core4352决策/34816ticks。第4个prefix是当前未终结回合的真实初始化，不漏记，也不计PPO信用。
- 每条policy行均有12维finite mean和positive finite std；以对应raw重算高斯old log-prob最大误差1.6214673835e−6，无缺字段、非正sigma或teacher-in-storage行。跨20次更新/各状态，σ全通道范围约[0.130568,0.188650]；RR hip/knee分别[0.151341,0.170198]/[0.161480,0.188650]，RL hip/knee分别[0.137368,0.163430]/[0.143222,0.160803]。这些跨权重/状态的范围不是同状态探索效果对照，不能单独归因任务变化。

### 完成回合与当前接触，不借用历史placed

| ep | global范围 | PPO决策/ticks | episode终止时间 | 实际结果/首个未完成环节 |
|---|---|---|---|---|
| 0 | 38273–39073 | 801 /6408 | tick9992 /83.266667s | P12 INCOMPLETE；RR曾Q/C/P但回地，RL无hard Q/C/P |
| 1 | 39074–39895 | 822 /6576 | tick10160 /84.666667s | P12 INCOMPLETE；RR曾Q/C/P，RL两次Q均在cross前GROUND撤销 |
| 2 | 39896–40599 | 704 /5632 | tick9216 /76.8s | P09 INCOMPLETE；RR曾Q但cross前GROUND撤销，无C/P |
| 3尾 | 40600–40832 | 233 /1864 | 当前tick5448 /45.4s | P06未终结；RR/RL无hard Q/C/P |
| 合计 | 38273–40832 | 2560 /20480 | 3次完成终止+1尾段 | task/full_task success均未获得 |

三次completed episode共2327决策，另233决策尾段已经进入完成的优化更新，但其物理回合尚未结束。三次正式reason均INCOMPLETE_CONTROLLER_BLOCKED；独立physical evaluator没有额外hard termination_reason，不另造wheel-only或body collision。

**ep1（第二回合）的真实RL资格不是放置成功：**

- RR initial5897/5926/6075/6161/6229；Q6297（52.475s，gain39.388087mm、joint9.657246deg、above-top+.268908mm）→C6536→P6543，均晚于teacher3584，属于当前PPO行为段。
- RL Q6729（56.075s，gain33.206411mm、joint12.322591deg、above-top+.324072mm），GROUND-before-cross撤销6799（56.658333s）；再次Q6893（57.441667s，gain17.035760mm、joint13.424747deg、above-top+.821130mm），再次GROUND撤销7225（60.208333s）。两次都没有RL cross/placed。
- 相邻decision-end样本6736/6800/6896/7232的RR均为**当前真实TOP**，load分别0.464717/0.284743/0.461219/0.348734，FL load分别0.440533/0.409693/0.538781/0.316747。因此不能把这两次RL落地简单叙述为“RR已经先失去全部顶部支撑”。
- P12最高AIR决策末样本tick6936 /57.8s：RL净空+86.800682mm、front−152.132895mm；当时RR TOP/load0.490825、FL TOP/load0.509175。确实抬高，但仍在前沿后方；这是15Hz采样峰值，不冒充120Hz连续最大值。高净空本身不是越沿或放置。
- RR后来首个已见GROUND的decision-end样本为tick7616 /63.466667s，front−53.860303mm、clearance−48.961448mm、load0.585523。终止10160时RR仍GROUND（front−52.918967mm、clearance−49.126229mm、load0.535917），RL AIR/load0（front−52.815941mm、clearance−48.798265mm），FL TOP/load0.464083。RL当前active=false，事件字典保留首次Q6729只是历史账。
- 此回合PPO return−58.54298206123593；未获得suffix成功，更不是natural P01 full成功。

**ep2（第三回合）停在RR资格撤销后：**

- RR first initial5866；Q5949（49.575s，gain15.763007mm、joint15.034838deg、above-top+.631423mm），在未cross前GROUND撤销6453（53.775s）。随后虽反复initial，未再获Q，整回合无RR C/P。
- 终止9216时RR obstacle_pair_active、非TOP/非AIR/非GROUND，front−47.261354mm、clearance−46.272515mm、load0.402584；RL GROUND/load0.295051，FL AIR/load0。障碍载荷不等于顶部放置；保留P09 INCOMPLETE原分类。
- RL没有hard Q/C/P；没有实际进入P10–P12，不能把它算成第三次独立RL阶段失败。PPO return−53.98730664164896。

截至前三个完成回合：RR曾hard Q为3回合，cross/placed为ep0与ep1共2回合，ep2资格已撤销；RL仅ep1有两次hard Q，均被撤销，全3回合无RL C/P。ep3尾段无后腿hard事件，不能将前三回合历史跨reset继承到它。

### 当前第4回合尾与准确阶段样本

边界40832时ep3已执行233个PPO决策，core681=448teacher+233PPO，tick5448/45.4s；仍P06、termination_reason=null、task_success=false。RR GROUND，front−264.860269mm/clearance−50.312548mm/load0.046734；RL GROUND/load0.529581，FL AIR/load0。不能预写后续成功、失败或P09进入。

| source phase | ep0 | ep1 | ep2 | ep3尾 | 截至40832已优化 |
|---|---:|---:|---:|---:|---:|
| P06 | 256 | 283 | 252 | 233 | 1024 |
| P07 | 1 | 1 | 1 | 0 | 3 |
| P08 | 1 | 2 | 1 | 0 | 4 |
| P09 | 91 | 84 | 450 | 0 | 625 |
| P10 | 1 | 1 | 0 | 0 | 2 |
| P11 | 1 | 1 | 0 | 0 | 2 |
| P12 | 450 | 450 | 0 | 0 | 900 |
| P01–P05/P13 | 0 | 0 | 0 | 0 | 0 |
| 合计 | 801 | 822 | 704 | 233 | 2560 |

两次P12阻塞与一次P09阻塞说明局部抬升/越沿/保持仍不稳定；新σ头输出有效且可随状态变化，只是实现和数据证据，不证明改善或退化的因果来源。此40832有界追加后停写，保留旧附录；没有运行Python/Isaac、修改生产/master或提交，未纳入未优化尾段，未预填46464。

## 有界追加至45440：7168决策，8个完整回合 + 第9回合860决策尾段

仅纳入global38273–45440对应的56个完整更新。读取期间实时文件已有更晚采集和第10个prefix，均不混入本账；最终run manifest尚未出现。本节的第9回合虽然其860条行为已经优化，物理回合仍未终结，不能写成完成或成功。

- immutable checkpoint45440实际SHA256=`eb458846fd34d4d972d97de7d3f0ae0b7c3e8866473a7ab3b3daff4b92005b5d`，本次PowerShell文件hash与sidecar一致，roundtrip=true。新增7168 /56 /1120，累计45440 /320 /6400。
- 56次更新均actor_parameters_changed=true、finite_nonzero_gradient_observed=true、optimizer_steps=20；global每次+128，actor前后hash链连续。边界actor SHA=`f30604b01031f40917e380ddb7c953e9feedc00dc54b1aff3e87461059e6a9e9`。最后LR1e−5、KL0.024790693、clip fraction0.2828125、entropy−5.9994804、value loss0.0055243681。连续分布微分熵可为负，不据此误报无效分布。
- `heteroscedastic_log_v1`、原origin10112保持；阶段累计full_episode12800、phase_suffix22528（本run前15360+7168）、smoke0。独立policy initial仍增加0训练计数，不发放新预算。
- **7168条PPO audit，57337实际physics ticks**；native verified=57337、actual native target effect=57337，own-phase request effect=57295。差42属于逐tick归属审计，不把有handoff的效果冒充当期独立请求；没有未验证行或非零in-episode四类状态写入。唯一短决策是ep3/global41342的真实terminal，实际1tick而非8，因此总tick比7168×8少7；保留terminal、不bootstrap，不伪填缺少的物理步。
- 对应前9次真实teacher prefix均448决策/3584ticks，共**4032决策/32256ticks**；9次接管均P06/tick3584，全部teacher raw/projected为0、policy_credit=false、native verified。与上述PPO合计core**11200决策/89593ticks**。第10个已实时生成的prefix不属于45440之前的信用窗口，此处排除而非否认其物理消耗。

### 12维分布、raw log-prob与实际执行

7168条已优化行的12维mean有限、12维std有限且正，无缺失或非正σ；逐条以对应raw/mean/std重算12维高斯old log-prob，最大绝对误差**1.9514380512e−6**。仍仅核对JSONL，不声称用Python重新打开tensor storage。该检查没有以tanh/projected/applied动作替代PPO raw sample。

| canonical通道 | 本窗口mean范围 | 本窗口std范围 |
|---|---|---|
| RL hip | −0.248564～−0.031590 | 0.107758～0.163430 |
| RL knee | −0.304234～0.348502 | 0.141403～0.168248 |
| RR hip | −0.346520～0.215946 | 0.139773～0.170198 |
| RR knee | −0.357001～0.228603 | 0.151674～0.188650 |
| RL wheel | −0.363854～0.217833 | 0.139634～0.173150 |
| RR wheel | −0.093258～0.390746 | 0.123191～0.155446 |

所有通道σ总范围0.107758～0.196893。范围跨不同状态与56次在线权重更新；可以证明实际分布输出被记录并用于采样、残差到达真实native目标，**不能**据此证明sigma本身改善任务，或把失败一概归于探索噪声/mean。首个固定权重rollout内的state-dependent变化证据仍见历史附录。

### 完整回合和已优化尾段

Q/C/P分别是hard qualified lift、真实越沿、真实放置历史。以下后腿事件tick均晚于对应teacher接管3584；每次reset的历史各自独立。

| ep | global范围 | PPO决策/ticks | 终止/当前时间、阶段 | RR/RL实际事件与首个未完成 |
|---|---|---|---|---|
| 0 | 38273–39073 | 801 /6408 | 83.266667s，P12 INCOMPLETE | RR Q6221/C6368/P6369；RL无Q/C/P |
| 1 | 39074–39895 | 822 /6576 | 84.666667s，P12 INCOMPLETE | RR Q6297/C6536/P6543；RL Q6729、6893分别在6799、7225落地撤销，无C/P |
| 2 | 39896–40599 | 704 /5632 | 76.8s，P09 INCOMPLETE | RR Q5949→6453撤销，无C/P |
| 3 | 40600–41342 | 743 /5937 | 79.341667s，P09 INCOMPLETE | RR Q6280→6687撤销，无C/P；末动作只推进1tick |
| 4 | 41343–42080 | 738 /5904 | 79.066667s，P09 INCOMPLETE | RR Q6330/C6441，但未placed；首未完成为RR真实放置 |
| 5 | 42081–42840 | 760 /6080 | 80.533333s，P09 INCOMPLETE | RR无hard Q/C/P；不能把反复初始离地记作合格抬升 |
| 6 | 42841–43707 | 867 /6936 | 87.666667s，P12 INCOMPLETE | RR Q6534/C6711/P6900；RL Q6953→7057撤销，Q7274→7522撤销，无C/P |
| 7 | 43708–44580 | 873 /6984 | 88.066667s，P12 INCOMPLETE | RR Q6507/C6667/P6945；RL Q7214→7284撤销，无C/P |
| 8尾 | 44581–45440 | 860 /6880 | 87.2s，P12未终结 | RR Q6964/C7154/P7289；RL Q7470→7498撤销，无C/P |
| 合计 | 38273–45440 | 7168 /57337 | 8个完整INCOMPLETE + 1未终结尾 | 无RL C/P、无P13样本、无suffix/full success |

ep2–5四次P09失败的具体缺口并不完全一样：ep4已cross但没placed，而ep5未qualified。因此不能把相同终止阶段归纳成同一物理失败机制。ep6/7重新到P12说明仍有不同探索结果，不是连续单调改善或退化。

### 历史placed与当前真实支撑分开

| 边界 | RR当前状态（front/clearance，mm）及load | RL当前状态及load | FL当前状态/load |
|---|---|---|---|
| ep6终止tick10520 | GROUND，−70.627/−49.267，0.207390 | GROUND，front−54.798/clear−50.115，0.453700 | TOP /0.151436 |
| ep7终止tick10568 | GROUND，−86.255/−48.920，0.176820 | GROUND，front−65.998/clear−50.228，0.439268 | AIR /0 |
| ep8尾tick10464 | GROUND，−123.275/−49.435，0.094771 | GROUND，front−63.940/clear−51.031，0.486669 | AIR /0 |

这三行RR虽然history Q/C/P均true，当前都不是TOP；RL曾qualified但已在cross前撤销，也没有placed。当前P12的首个未完成任务仍是RL合格空中越沿与放置，且RR放置后保持也已失去。终止分类保留原INCOMPLETE，不自行改成wheel-only。接管/后续阶段曾TOP不能代替这些具体时点的接触证据。

### 截至45440精确phase ledger

| source phase | 已优化样本 |
|---|---:|
| P01–P05 | 0 |
| P06 | 2709 |
| P07 | 9 |
| P08 | 11 |
| P09 | 2235 |
| P10 | 5 |
| P11 | 5 |
| P12 | 2194 |
| P13 | 0 |
| 合计 | 7168 |

完整ep0–7共6308样本；尾ep8的860样本分布P06=375、P07=1、P08=1、P09=87、P10=1、P11=1、P12=394。课程P06准备持续采样，但P07/P08较短；P12样本来自5个实际回合（其中1尾），没有任何P13优化样本。不把后缀结果称作自然P01完整成功，也不以达到success才允许后续训练。

本次只用PowerShell流式只读及修改此独占报告；无Python/Isaac并行、无生产/master编辑、无提交。请求余下1024和最终46464尚未纳入，等待真实终局后另行追加。

## 最终追加：46464 /328 /6560，8192实际优化决策全部完成

本节以已关闭的run文件、最终`run_manifest.json`与immutable checkpoint46464 sidecar为准；与此前45440快照的差值是随后实际完成的1024决策/8次更新/160个optimizer steps，不覆盖当时尚未完成的历史事实。

- 原请求/最终actual均8192，64个完整128决策rollout均已更新；`rounding_overrun=0`、`unconsumed_requested_policy_decisions=0`。run lifecycle=SUCCEEDED，根代理已确认session51079 exit0；这不是机器人任务成功。
- 最终checkpoint SHA256=`fc1fd6710965d627aff6e6ecc91e02c5ed9c144f7637cba9a50cfa079cf5513d`；sidecar SHA256=`10b34e9c076a83c22f4af4871c6d837795cc12c59200db21d2df2bc0d5d19a40`。根代理已实际核验两文件hash，本报告读取sidecar确认global46464 /PPO328 /optimizer6560及`save_load_round_trip=true`，不重复大文件hash。
- source38272 /264 /5280与origin10112保留；最终阶段账full_episode12800、phase_suffix23552（原15360+8192）、smoke0。独立policy分布迁移的initial新增计数仍为0，teacher未进入预算。
- wall time **4776.261459799949s（约79.6044分钟）**。telemetry另记teacher roll-in1807.6189648001455s、reset38.28408839995973s，均为本run开销分项，不能再加到wall总量冒充额外总时间。
- 64个真实更新均actor改变、有限非零梯度、每次20个optimizer steps；global+128与相邻actor hash链全部连续。actor首hash=`3ba295ffd216eb1c2e87f476559f36c8ff89e122979c607bcb41583597fba517`，最终=`571f19b7ccc64593456eb4a90cd9163c38b273d2e65305b2b87f27b6c41df205`。gradient norm总范围1.000203966～1.414213605，更新KL范围0.012429387～0.028379551。
- 最后update328：LR1e−5、KL0.0206072454、clip fraction0.2453125、entropy−5.81217878、value loss0.00108260048、surrogate loss−0.0392653219。整个run LR范围1e−5～5.0625e−5，为既有adaptive规则的实际记录，无额外调参。

### 完整信用、native及分布审计

一次PowerShell流式核验覆盖global38273–46464全部8192行，顺序连续；与最终telemetry阶段计数及物理tick一致：

| 部分 | 决策 | 实际physics ticks | 学习信用 |
|---|---:|---:|---|
| PPO当前策略 | 8192 | 65529 | 全部属于64次已完成更新 |
| reset-only teacher，11个prefix | 4928 | 39424 | 0；不放入PPO storage/global |
| 真实core合计 | 13120 | 104953 | 不把core总决策当PPO决策 |

全部11次prefix accepted=true、miss=null，各448决策/3584ticks；不存在失败prefix fallback。11次teacher raw/projected全0、policy_credit=false、native verified39424。初始接管与末次P06尾段的teacher均实际消耗时间，不因其无学习信用而漏记。

8192条PPO行全部12维finite mean/positive finite std，高斯old log-prob重算最大绝对差仍**1.9514380512e−6**；没有缺分布字段、非正σ、teacher-in-storage或非零四类in-episode状态写入。actual native verified/effect均**65529ticks**；own-phase request effect65484，45ticks的归属排除不冒充当期独立请求效果。唯一short decision仍是ep3/global41342的1tick真实terminal，故65529=8192×8−7；没有填补或遗漏终止动作。

最终整个run跨状态和更新的σ范围0.106848493～0.196892634。RL hip/knee范围分别0.106848493～0.163430080、0.139736027～0.168248266；RR hip/knee分别0.135443792～0.170197591、0.145371988～0.188650176。raw分布与实际native作用均有证据，任务因果改善仍没有同状态/同权重配对证明。上述JSON检查不等价于另行重载rollout tensor；保存与roundtrip证据来自真实训练结果，不来自测试模拟。

### 最终11回合分账（10次终止 + 1个已优化未终结尾段）

| ep | global范围 | PPO决策/ticks | 最终时钟 | 实际终止/预算末阶段 |
|---|---|---|---|---|
| 0 | 38273–39073 | 801 /6408 | tick9992，83.266667s | P12 INCOMPLETE |
| 1 | 39074–39895 | 822 /6576 | tick10160，84.666667s | P12 INCOMPLETE |
| 2 | 39896–40599 | 704 /5632 | tick9216，76.8s | P09 INCOMPLETE |
| 3 | 40600–41342 | 743 /5937 | tick9521，79.341667s | P09 INCOMPLETE |
| 4 | 41343–42080 | 738 /5904 | tick9488，79.066667s | P09 INCOMPLETE |
| 5 | 42081–42840 | 760 /6080 | tick9664，80.533333s | P09 INCOMPLETE |
| 6 | 42841–43707 | 867 /6936 | tick10520，87.666667s | P12 INCOMPLETE |
| 7 | 43708–44580 | 873 /6984 | tick10568，88.066667s | P12 INCOMPLETE |
| 8 | 44581–45496 | 916 /7328 | tick10912，90.933333s | P12 INCOMPLETE |
| 9 | 45497–46304 | 808 /6464 | tick10048，83.733333s | P09 INCOMPLETE |
| 10尾 | 46305–46464 | 160 /1280 | tick4864，40.533333s | P06；terminal=false、reason=null |
| 合计 | 38273–46464 | 8192 /65529 | 10回合8032 + 尾160 | 5个P12失败、5个P09失败，success0 |

每个时间都包含该回合真实teacher的29.866667s，没有在接管时重启200s任务时钟。最终尾160样本是合法的已优化rollout部分，不能将“训练预算结束”改写为物理失败；其物理环境状态不作为checkpoint连续仿真状态保存。

**45440之后的具体物理补齐：**

- ep8原860尾继续56决策，到global45496正式P12 INCOMPLETE。RR Q6964/C7154/P7289；RL Q7470在7498落地撤销，直到终止无C/P。终止时RR已GROUND，front−111.723509mm/clear−49.389071mm/load0.148079；RL GROUND，front−61.978694mm/clear−51.074995mm/load0.465596；FL AIR/load0。PPO return−59.87098612292425。不是RR仍在顶部支持下完成后腿任务。
- ep9通过第10个真实prefix后开始。RR Q6805，在6876落地撤销，无cross/placed；RL也无hard Q/C/P。终止tick10048时RR AIR但clear−49.078210mm/front−55.249092mm/load0，RL GROUND/load0.494993、front−218.945768mm，FL AIR/load0；保留P09 INCOMPLETE，PPO return−55.39885509739361。AIR标签不代表 above-top合格，也不代表仍保有历史资格。
- ep10第11次teacher接管后只执行160个P06决策，core608=448teacher+160PPO，tick4864/40.533333s。RR GROUND，front−320.791301mm/clear−50.991519mm/load0.054713；RL GROUND，front−357.871812mm/clear−50.367141mm/load0.506157；FL AIR/load0。RR/RL均无Q/C/P、没有终止，当前PPO return−2.427562487941101。最终保存并不把ep9历史带入此回合。

**整块后腿事实：**10个完整回合中RR曾hard Q的有9个（除ep5），曾cross的6个（ep0/1/4/6/7/8），曾placed的5个（ep0/1/6/7/8）。这5个placed回合的终止时RR全部GROUND，而非当前TOP。RL在ep1/6/7/8共获得6次hard Q，全部在cross前GROUND撤销；整个8192信用段无RL cross/placed，无P13访问，teacher形成的FR/FL历史不计作PPO新增后腿事件。因此阶段进展不等于支撑保持，也不能凭样本先后声称新分布造成单调改善。

### 最终phase ledger与first unfinished

| source phase | 最终已优化样本 |
|---|---:|
| P01–P05 | 0 |
| P06 | 3225 |
| P07 | 10 |
| P08 | 12 |
| P09 | 2685 |
| P10 | 5 |
| P11 | 5 |
| P12 | 2250 |
| P13 | 0 |
| 合计 | 8192 |

P12的2250样本来自5个完整回合各450；P09五次完整失败中ep4首未完成为RR放置，ep2/3/9为RR资格落地撤销后未越沿，ep5为未hard-qualified。五次P12失败均未完成RL空中越沿/放置，且终止时RR放置保持也已经丢失。末尾P06准备尚未结束，不能再归到第六次RR或RL终止。

训练执行已完成，suffix success0、fresh-P01 current-policy success0；后者也不能作为此P06课程已经实际尝试完整P01的分母，本块P01–P05策略信用为0。根代理另行进行cp46464自然P01确定性评估，本报告不替它预判结果，不把训练成功或历史子事件写成稳定性胜过A。

最终报告到此定稿并停止写入；保留全部历史有界附录。只执行PowerShell只读检查和此报告修改，无Python/Isaac并行、无生产/master修改、无提交。
