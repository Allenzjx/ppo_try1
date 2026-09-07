# 当前后腿链：已完成回合的只读诊断

2026-09-06。运行中的版本为 `49749aa527a4a00901e434104821d7e8241ea8da`；本次 policy-only 迁移没有修改 `64abc5357d00763419fe51c8126db8454fcf7697` 的物理、任务、reward、nominal 或 observation 配置。仅 PowerShell 读取及本报告写入；未运行 Python/Isaac、未改生产或参数。当前8192决策块继续固定运行。

## 样本边界和结论

当前 run：`runs/ppo_semantic_v3/train/20260906T0946289788164Z_g49749aa527a4_d866c10e684d4248b10521cb7d32bf09`。依据 `completed_episodes.jsonl`、`residual_and_projection_audit.jsonl`、`optimizer_updates.jsonl`，只统计 ep0/ep1 的 **1623个完整已优化决策**：global38273–39895。已保存的 update277/global39936 确认覆盖这些样本，记录本次20 optimizer steps、actor变化和有限非零梯度；该边界另外包含的41个 ep2 决策以及之后所有数据均不纳入本报告。

- ep0 的首个未完成物理子任务是 **RL 合格抬升**；ep1 已两次合格抬升，但首个未完成子任务是 **在保持有效净空/过程的同时前进越沿**。两者都无 RL crossing/placed，不能把 initial 或历史直方图当完成。
- RR 的 placed 是真实发生过的历史事件，但之后确实退到前沿后方/回地。ep1 的 RL 最后一次资格撤销早于 RR 首次采样负 front，因此 **RR 退回不能单独解释该回合最初的 RL 失败**；后来它仍可能妨碍恢复。
- 当前 observation 已区分 RR 当前位置/载荷与历史 placed；all-placed 前的 task potential 则缺 RR 当前区域的直接保持/恢复项，同时仍有通过 RL unload/support 的间接反馈。这个局部信号缺口不是已证实失败主因。
- 实际控制请求进入了 native targets；数值审计未发现 reward/Phi/action smoothness 或保存 Gaussian log-prob 的反号、重复计入、倍率接线错误。有限样本既不能证明权重最优，也不能把失败归因于“只需更多训练”。

## 1. 当前 P06 两回合：历史与当前状态分开

两次 reset 均记录真实 A teacher prefix448决策/3584 ticks，teacher credit=false；P06 自身 credit 从29.866667s开始。下表 RR 的 Q/C/P 均发生在 credit 开始以后，并非 teacher 继承。Q=qualified，C=cross，P=placed。

| 项目 | ep0 | ep1 |
|---|---:|---:|
| credited global / 数量 | 38273–39073 / 801 | 39074–39895 / 822 |
| phase决策数 | P06:256,P07:1,P08:1,P09:91,P10:1,P11:1,P12:450 | P06:283,P07:1,P08:2,P09:84,P10:1,P11:1,P12:450 |
| 终止 | 83.266667s, tick9992, P12 `INCOMPLETE_CONTROLLER_BLOCKED` | 84.666667s, tick10160, 同原因 |
| episode return | −58.065345 | −58.542982 |
| RR Q/C/P tick | 6221 / 6368 / 6369 | 6297 / 6536 / 6543 |
| RL Q；GROUND撤销 | 无 | 6729→6799；6893→7225 |
| RL C/P | 均无 | 均无 |
| P12决策末 AIR / qualified 数量 | 97 / 0（共450） | 232 / 50（共450） |

`history.event_ticks.active_lift.RL` 保留首次6729，并不表示末尾仍有资格；第二次6893及撤销用 `lift_attempt_events` 核对。ep0/ep1 自身 credit 内分别有27/25次 initial 事件，均不能当作 Q/C/P。

以下几何单位为 mm，clearance 相对障碍顶面，front 相对前沿。Q/C/P/撤销事件 tick 来自120Hz内部事件；表中“首次”位置/接触是 **每8tick决策末首次采样**，不是声称精确接触发生时刻。

| 回合 / tick | RL 当前几何和过程 | RR 当前状态 |
|---|---|---|
| ep0 / 6384，首负front样本 | front−92.707，clear−11.565，AIR | front−5.001，非TOP/非GROUND，load .4544 |
| ep0 / 6400，P12入口及该阶段AIR最高 | front−124.765，clear−11.534，无Q | front−19.813，非TOP/非GROUND，load .4869 |
| ep0 / 6440，RR首GROUND样本 | front−136.310，clear−47.396，AIR | front−51.847，GROUND，load .5934 |
| ep0 / 9104，P12最近front | front−48.613，clear−38.834，AIR，无Q | front−60.564，GROUND |
| ep1 / 6736，首Q后样本 | front−67.308，clear+11.102 | front+46.198，TOP |
| ep1 / 6936，P12最高AIR | front−152.133，clear+86.801，Q有效 | front+29.783，TOP，load .4908；FL亦TOP |
| ep1 / 7008，Q期间最近front | front−48.962，clear−32.642，Q尚有效，但已非AIR | front+37.417，TOP；FL亦TOP |
| ep1 / 7232，最后Q撤销之后 | front−51.128，clear−50.675 | front+5.918，仍TOP |
| ep1 / 7520，RR首负front样本 | front−49.922，clear−46.185，无Q、非AIR | front−3.902，非TOP/非GROUND |
| ep1 / 7616，RR首GROUND样本 | front−54.228，clear−50.176，无Q | front−53.860，GROUND |

ep1 在 RL 最高时离前沿仍152mm；后来最接近前沿的 qualified 样本仍差49mm且已低于顶面33mm。最后GROUND撤销 tick7225 比 RR 首负front样本7520早295tick（2.458s）。因此“有高抬升”与“有向前进展”并未同时满足合法 crossing，不是 RL 从未动，也不能从 RR 后续退回倒推它是唯一先因。

终止时 ep0 RR为GROUND/front−64.038、RL front−49.884且非AIR/非GROUND/非TOP、load .0951；ep1 RR为GROUND/front−52.919、RL AIR/front−52.816/clear−48.798。**RR历史 placed=true 不等于当前TOP支撑；RL非AIR也不自动等于GROUND或TOP。**

## 2. 实际策略与目标：非被整段交接吞掉

两回合各450个P12决策均核验3600/3600 native audit ticks；own-phase-request-effect 为3599/3600，排除了阶段首个handoff tick。因此没有“P12整决策始终只用旧 residual”的证据。

下表从每个样本自己的 `old_distribution_mean_full12` / `old_distribution_std_full12` 与 raw action 计算。σ是当前 state-dependent 策略实测值；范围同时混合不同 observation 和多次在线更新，不能单独归因于状态依赖。噪声RMS用 sample−**该样本自己的**mean，不使用最新checkpoint mean。

| P12通道 | ep0 mean均值 / σ范围 / 噪声RMS | ep1 mean均值 / σ范围 / 噪声RMS |
|---|---|---|
| RL hip，index4 | −.09624 / .14937–.15663 / .15103 | −.09266 / .13737–.16026 / .14364 |
| RL knee，index5 | +.03749 / .15213–.15777 / .15983 | +.03585 / .14827–.16080 / .15687 |
| RL wheel，index10 | +.06511 / .15064–.16303 / .16357 | +.05151 / .14838–.16391 / .15980 |
| RR wheel，index11 | +.23027 / .14351–.14926 / .14477 | +.20003 / .13932–.14897 / .15110 |

确有非零均值与随机探索同时存在，但不能把均值正负直接解释为整机前/后退，也不能把跨时序动作当独立试验推整回合成功概率。

ep1 已有连续、实际生效的名义/残差目标变化。以下“drive”是经mapper、反馈和slew后 dispatch 的逻辑目标（servo deg、wheel rad/s），不是测得的关节位置或轮实际转速；native audit另核对其float32实发值。

| tick | nominal RL hip/knee | drive RL hip/knee | nominal四轮 | drive四轮 |
|---|---|---|---|---|
| 6568 | 15.4 / 22.6 | 19.588 / 21.393 | [−1.025,0,0,0] | [−1.0638,.0805,.1177,.2079] |
| 6736 | .5 / 35.3 | −.595 / 28.735 | [−.3,−.3,−.3,−.3] | [−.2541,−.2074,−.2768,−.1621] |
| 6936，AIR最高 | 30.2 / 35.3 | 30.877 / 39.495 | [.3,.3,.3,.3] | [.3273,.4090,.3242,.5127] |
| 7232 | −10.1 / −18.7 | −15.654 / −17.000 | [0,0,0,0] | [.1129,−.0880,−.0059,.1199] |
| 10160，终止 | −10.1 / −18.7 | −10.964 / −19.464 | [0,0,0,0] | [.0196,.0328,.0491,.0989] |

相应 body_forward 在6736为.30556m，6936为.27859m，7232为.23133m。它证明动作/身体/几何均在变化，不证明某个轮目标符号错或某个 advisory 单独导致退回；没有把同一物理状态下的反事实 rollout 做出来。

## 3. 当前可观测性、reward依赖与局部高度/前进取舍

[semantic_observation.py:230](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_observation.py:230) 编入每腿当前 clearance/front/load，历史位另列；当前 RR 退回对324维输入不是隐藏状态。

[semantic_supervisor.py:553](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:553) 对历史placed腿固定贡献1；all-placed前 finish=0。当前FR/FL/RR已placed、RL未placed时：

`Φ = .85 * (3 + RL_progress) / 4`。

所以 RR 当前 front/区域损失没有自身直接 potential 差分；但是 RR/FL 当前 support/load 可通过 RL unload 改变Φ，身体、接触、实际目标smoothness成本也仍有。不得由此建议把所有已placed腿AIR都扣分：FL协同抬起必须仍可合法发生；也不应删掉真实合法的历史事件。更完整依赖见同目录 `placed_history_current_region_reward_readonly.md`；尚未选定任何实现修改。

### 新补查：hard-qualified 后下降并前进的相邻样本

[semantic_supervisor.py:558](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:558) 的 carry 使用 hard_lift 与 front，不乘当前净空；lift自身则使用 `.008/(.008+max(0,−clearance))`，cross之前不会因为曾经qualified而永久为1。对当前RL，隔离的 global Φ 分量为：

`Φ_carry=.074375*clip(1+front/.25)`；`Φ_lift=.03984375*current_height_fraction`。

ep1 两段有效qualified期间，共7对相邻8tick样本符合“当前下降到顶面以下、同时front前进”。其中2对 carry增益大于lift损失，但 **7对实际总Φ均下降**。下面全部仍未cross，RR/FL两端都TOP；数值为 potential 差，不是把独立 per-frame bonus 加到reward。

| ticks | RL clear mm；front mm | ΔΦ carry | ΔΦ lift | 两项合计 | 其他ΔΦ | 实际ΔΦ / 后样本task reward |
|---|---|---:|---:|---:|---:|---:|
| 6776→6784 | −9.334→−22.045；−100.659→−94.766 | +.0017533 | −.0077800 | −.0060267 | 约0 | −.0060267 / −.0502187 |
| 6984→6992 | +10.146→−7.868；−99.549→−77.525 | +.0065521 | −.0197560 | −.0132039 | 约0 | −.0132039 / −.0864698 |
| 7000→7008 | −22.809→−32.642；−58.360→−48.962 | +.0027961 | −.0025032 | +.0002928 | −.0050404 | −.0047475 / −.0439684 |
| 7160→7168 | −39.381→−39.390；−49.517→−49.503 | +.00000416 | −.00000115 | +.00000301 | −.00042687 | −.00042387 / −.0223205 |

前两对 RL 一直AIR、load0、其他support数2、workspace饱和、initial位不变；故其他Φ变化为浮点零，可直接看出下降损失大于前进增益。7000→7008 则 RL从AIR转为接触，load0→.389755；unload由1下降到.762806，额外Φ损失恰为−.00504038。RR/FL仍TOP，RR front+36.485→+37.417mm；并不是 RR退回把总Φ拉低。7160→7168主要是RL load .370038→.386108，其他support2→3均足够，unload差恰解释其余损失。

7008实际weighted body/contact/smooth costs分别−.00382284 / −.00001517 / −.00307413；6992分别−.00095437 / 0 / −.00306312。加上折扣项与时间成本，不能把表中某个孤立两项正和称为整步获得正reward。

**确实存在一个可描述的局部取舍**：在hard资格尚未撤销、越沿未完成时，carry仍给低于顶面的前进连续增益；净空越低时 reciprocal lift 的边际损失变小。固定其他量下，有些前进/下降组合可使这两项净增。**但此处真实7对总Φ全降，没有证据称总reward正在奖励这7次下降，更没有因果证据称这就是失败主因。** 瞬时potential取舍也不等于折扣、终止正确的PBRS改变了基础任务的最优目标。

### 数值接线复核

对上述1623个真实已优化样本，以保存字段重算：

- `5*(.995*potential_after−potential_before)` 与 `potential_shaping` 最大误差0。
- 已加权五family和与stored float32 reward最大误差2.18e−7。
- smooth family与 `−.05*(actual_drive_first_difference+actual_drive_second_difference)` 最大误差1.73e−18；不是再把nominal/residual差分各罚一遍。
- 用每条自身 raw/mean/std 重算12维Gaussian old log-prob，最大误差1.62e−6。

普通phase切换不设done；终止zero-next-potential、无bootstrap，与 [semantic_reward.py:179](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:179) 一致。熵正则在PPO loss，不混进上述环境reward。以上未发现可证实的反号/缩放接线bug，不能证明奖励权重或状态分布已足够。

## 4. 同物理版本的旧 P10：另一处首个未完成任务

对照 run：`runs/ppo_semantic_v3/train/20260906T0748083849450Z_g64abc5357d00_525e380c18ae4039ad93d764e7a06f50`，实际cp28032→30080，共2048 credited决策/16updates/320steps。两完整回合各有948决策/7584tick teacher prefix，RR Q6938/C7109/P7579 均在 credit之前，不能记作该P10策略自身完成RR。RL则在credit内合法完成：ep0 Q7912/C8066/P8122；ep1 Q7960/C8064/P8132。

两回合终止均为 P13 `INCOMPLETE_CONTROLLER_BLOCKED`；final_region/support均true、controlled=false、stable0。首个剩余任务是实际物理停稳，而不是RL越沿。P13总1841决策包括两终止及第三回合41个未终止尾样本，不能把全部1841称为三个完成回合。

| 终止实测 | ep0：127.7333s，968决策 | ep1：127.8s，969决策 | 既有硬标准 |
|---|---:|---:|---:|
| body linear m/s | .059984 | .118722 | ≤.05 |
| body angular rad/s | .231826 | .349317 | ≤.30 |
| actual wheel max rad/s | .159298 | .147229 | ≤.25 |
| actual command max rad/s | .112195 | .113466 | ≤.02 |

两终止nominal四轮均0，实发目标仍非零；FR/RL当时TOP、FL/RR AIR。这既不表示全部历史placed仍在当前支撑，也不能把全placed腿AIR自动定义失败。既有 `p10_block_30080.md` 中 ep1 的628个全12维nominal零样本：command不达标627、body speed372、angular32、actualwheelrate142；不是只有残留nominal rolling，也不是只缺命令停稳。

该旧P10 JSON没有逐样本mean/std；本次未解包 `.pt`，不能从这些JSON独立分离其策略偏置与探索噪声，也不借用其他rollout的cp28032 actor-only结果冒充本块测量。旧P10与当前P06虽然物理MDP相同，但checkpoint、策略分布、真实prefix位置/历史不同，不能当配对消融或据此判断新分布更好/更差。

## 后续决策边界（不改当前块）

本报告支持保留三个分开的待判断问题：RL同时抬升和前进的控制学习；已placed腿当前可恢复区域的直接task信号缺口；allplaced后的实际停稳/探索。当前数据没有把它们归约为一个已证实主因，也没有证明新增架构/阈值必要。

先完成正在进行的固定8192块，使用原定保存checkpoint后的真实P01确定性mean评估，报告真实Q/C/P和当前接触而不是只看历史直方图。若之后需要专门归因探索，才考虑经授权、同一个checkpoint/同seed与同真实prefix分布、只改变 sampled/mean执行的比较；这里不声称现有CLI已经提供P06 suffix-eval入口，也不把该比较设为继续optimizer的门禁。若讨论reward单因子实验，应与policy/entropy/nominal/cap变化分开，保留hard任务和合法全身协同；本报告没有选择或实施该修改。
