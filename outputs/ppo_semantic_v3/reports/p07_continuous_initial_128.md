# RUNNING：P07 / offset0，同MDP普通resume首128已保存窗口

Run：`runs/ppo_semantic_v3/train/20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5`。

HEAD `c34262abffc16847ff32d15ecbf790dd60803e0a`，v3/N1/seed1001，P07 offset0，phase_suffix。本报告只到真实首保存 **73216 / 537 PPO updates / 10740 optimizer steps**，即globals **73089–73216**、新增128/1/20；计划4096没有作为完成计数。仅PowerShell只读清单与JSONL，不加载tensor、不运行Python/Isaac、不更改生产或重复checkpoint大文件哈希。

## 1. 普通resume，不是再次new-MDP初始化

启动参数明确指向源`checkpoint_step_000073088.pt`，`new_mdp_warm_start=false`、`resume_migration=null`、`policy_distribution_migration=false`。没有本run的`new_mdp_warm_start.json`。第73216与源73088的完整runtime contract、policy contract及runner config JSON完全相等；本次只是固定课程P06/240→P07/0的run边界选择。

| 源73088实际sidecar字段 | 值 |
|---|---|
| 累计global / PPO / optimizer | 73088 / 536 / 10720 |
| actor SHA（含learned state-dependent std分支） | `5031de525ff8d43dfcc9ea315a5efc4aba8999e51607644255aa94e765c64589` |
| critic SHA | `4310da2896ea00dca18004321de54f19018f0f9a8bb8dd2f09839e038d84af69` |
| normalizer SHA | `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` |
| Adam SHA / 已记录LR | `cbfe0ed12021b6947cb4515f9384e744cb9b219cb645cf7e8cdaf6c8fe6dd566` / 1e−5 |
| stage full / suffix / smoke | 29184 / 33792 / 0 |

首update的`actor_parameter_sha256_before`与上述源actor完整相等。正常loader路径要求actual actor/critic/optimizer/normalizer的加载后摘要与源sidecar一致，恢复源`training_rng_state`，并拒绝非空storage或旧transition；它不清空Adam。运行已成功进入真实update，因此没有绕过这些加载检查。此项依据生产加载路径及运行证据，**不是本报告另外打开tensor或读取加载瞬间RNG/Adam内存的独立测量**。首保存normalizer仍相同，policy保持324/12的heteroscedastic版本。

历史new-MDP provenance即使由源继续保存，也不表示本run重置过Adam；以实际参数和普通加载分支区分。没有继承旧rollout或物理环境，真实P01教师reset在本run重新执行，随后空official storage只收C动作。

## 2. 首个直接P07接管：READY仍P07，RR真实GROUND

第一段前缀accepted=true、miss=null、无fallback；744 teacher decisions / 5952 physics ticks，所有raw为0，native审计和no-state-write证据通过，教师不获PPO credit。

- target首次观察5952，offset0，handoff与policy_credit_start均t5952 / 49.6s；剩余任务150.4s。
- `mode=teacher_initialized_suffix`、requested/actual均P07、`requested_phase_still_active_at_credit=true`。
- last executed source_control_tick5951、command/ACK tick6131；controller bias全12维0，没有额外bias takeover消耗P07。
- 接管使用实际live observation、刚执行nominal/mapper历史与已有supervisor；未应用历史姿态snapshot或重置任务时钟。

| 首次实际READY状态 | RR | RL |
|---|---:|---:|
| AIR / GROUND | false / true | false / true |
| front距离mm | −205.028680 | −219.366219 |
| clearance mm | −50.919979 | −50.144145 |
| load fraction | 0.247819573 | 0.303504492 |
| 硬Q / C / P | false / false / false | false / false / false |

RR当前initial_clearance也false，无障碍pair或TOP。FR教师Q/C/P为71/1665/1695，FL为2461/3115/3583；这些前腿历史全部在reset-only前缀形成，不记作C新增事件。此直接证据补足先前`p07_pre_lift_course_evidence.md`中“尚无P07专用start”的当时限制，但不改写旧报告的历史证据范围。

## 3. 首128实际样本：两个准备入口，不跳过P07/P08

第一个策略回合在78个decision后终止，自动reset重新执行第二次真实前缀，之后收50个decision，合成同一个完整128 rollout。两次前缀均744/5952、accepted、无fallback、P07/READY t5952且bias0；第二次所核RR接触/几何及QCP状态与上表一致。**总1488 teacher decisions / 11904 ticks不入PPO/global/stage ledger。**

| 首rollout部分 | globals | 策略decisions | P07 / P08 / P09 | credited ticks | 结果 |
|---|---|---:|---|---:|---|
| episode0 | 73089–73166 | 78 | 1 / 1 / 76 | 622 | P09 BODY_COLLISION |
| episode1尾 | 73167–73216 | 50 | 1 / 1 / 48 | 400 | P09非终止 |
| 合计 | 73089–73216 | 128 | **2 / 2 / 124** | **1022** | 1真实终止+1未终结尾 |

两次P07→P08均在t5960，P08→P09均在t5968；这是实际连续策略采样，不用老师后续阶段代替C。P07/P08确有样本，但每次仅1个decision，不能夸为大量准备训练或全P01学习。

episode0正式completed记录：78 PPO decisions，绝对任务t6574 / 54.7833333333s，末decision仅6ticks，因此128×8−1022=2的差额恰由这个真实BODY_COLLISION短步产生。physical evaluator valid=true，reason为central body/obstacle collision；time_outs=false、terminal_bootstrap_allowed=false，合法失败留在本次PPO更新，不筛掉。

## 4. 新RR事件只来自C信用段，尚未越沿/放置

episode0的RR initial事件t6037、6233均晚于接管5952；硬Q t6374，记录上向excursion36.955312mm、当时净空+0.186214mm。RR未cross/placed。终止t6574时RR AIR、front−32.144036mm、clearance+1.844810mm、load0；这不是RR放置成功，失败由身体碰撞终止。

第二回合截止t6352 / 52.9333333333s仍P09非终止；RR AIR、front−47.034648mm、clearance−30.129074mm、load0，尚无硬Q/C/P。该尾段状态不是预测其下一步失败或成功。两个回合的FR/FL placed是教师历史，不能和这些新增RR动作事件混记。

## 5. 首update、原生审计与immutable保存

128行raw/old mean/old std均12维、finite、std正，实际std范围0.086482078–0.252441317；按同次Gaussian raw/mean/std重算old LP最大绝对误差1.1929321e−6。这里保留raw policy likelihood，不用投影后动作替代PPO action。

- 1022/1022物理ticks verified，actual native effect1022、own-phase effect1018；128行末tickactual mapping/dispatch/setter/same-tick counterfactual均通过，完整tick审计无缺失。
- 四类回合内root pose、root velocity、force/impulse、gravity写入均0；全部128行`prefix_teacher_data_in_ppo_storage=false`。
- update537真实20个optimizer steps，actor `5031de52…` → `3998fe702c5b3cd29a8d35826b71c028a17f294a87f923f4c9e97f7822439709`，changed=true、finite_nonzero_gradient=true。
- gradient norm范围1.414213140–1.414213642，KL均值0.034125834、clip fraction0.375、entropy−4.444464421、value loss102.814252853、surrogate loss−0.038194843。update结束LR1e−5；未逐minibatch记录LR，不断言全程恒定。

已读取`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000073216_manifest.json`：73216/537/10740，`save_load_round_trip=true`，记录checkpoint SHA `0d1b8d398d2415fd54566cc5abe4dffeb7a92955798dadf79b9ce296a2c32e91`，actor与update after相同。stage full29184 / suffix33920 / smoke0，恰只增加128 suffix decisions。

`implemented_reset_sampling`与`curriculum_epoch`均为`natural_P01_A_teacher_prefix_then_semantic_suffix_N1.v2:P07:offset_0`，且声明课程只在完整rollout/update边界之间改变。没有新initial checkpoint增加计数，没有将教师1488决策或仍在采集的后续行合入本窗口。

结论仅为：真实P07 pre-lift入口、准备阶段C信用与普通resume已走通，首含失败rollout已进行真实优化并保存；不是后缀成功、自然P01成功或计划4096完成。首128固定核验完成，停止扩展本报告。
