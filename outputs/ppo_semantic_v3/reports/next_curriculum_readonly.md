# 下一块课程建议：补P10连续后腿覆盖，保留P01/P06

2026-09-06。只读PowerShell/报告工作；未运行Python/Isaac、未改生产或参数。当前9d70aae的P01/4096块继续固定运行。本报告的**完成训练计数截止已最终保存的46464**；不把当前块的计划50560或尚未最终化部分并入完成账，也不修改C46464报告。

## 建议结论

**有足够的实际覆盖依据，把下一块安排为当前新reward、heteroscedastic策略的 P10 offset0 → 后续连续阶段训练。建议先2048个credited decisions，N1、seed1001，普通exact-resume。** 不需等P01完整成功才补后段，也不需新架构/新课程工程。

其价值是让当前网络直接采到第二后腿准备、RL越沿/放置以及有机会进入P13停稳的数据，**不是“P10能修好RR”或“新分布尚未训练RL”**。P10并不能替代前驱由C自己控制的P01/P06；也没有证据保证这个2048块必到P13或成功。

后续最小轮换建议（均为未来分配，不记作已训练）：

| 顺序 | 起点 / stage | 建议credited预算 | 目的 |
|---|---|---:|---|
| 下一块 | P10 offset0 / phase_suffix | **2048** | 补RL连续后段、尽可能取得当前分布的P13直接样本 |
| 随后 | P01 / full_episode | **2048** | 保留前腿与自生成后腿入口，检查/训练后缀知识能否用于自然链 |
| 再随后 | P06 offset0 / phase_suffix | **4096** | 强化RR前驱与两后腿连续链，减少只会教师提供P10入口的偏置 |

这个8192决策小周期为P10/P01/P06=25%/25%/50%，是根据当前后段缺口的工程起点，不是最佳比例或必须整周期执行的硬规则。当前正在跑的P01块在它实际结束后另行记账，不为凑比例修改它。先做2048 P10而非立刻再做8192固定P10，能够以较小块获得当前策略的真实后段信息，同时限制单一teacher入口占用训练预算；后续按已经完成的phase计数和物理结果再分配。

## 1. 截至46464：缺的是哪些数据，不是哪些通道

已通过文件搜索核实，用户提到的旧P01报告实际名为 `p01_block_38272.md`，没有将不存在的 `policy_p01_block_38272.md` 当证据。以下均使用各报告最终追加账，不使用其中历史尚在运行的附录数值。

| 已完成块 | 分布 / 实际新增决策 | P10 / P11 / P12 / P13实际决策 | 后腿与结果 |
|---|---|---|---|
| 28032→30080，P10 | 旧scalar Gaussian / 2048 | 3 / 3 / 201 / **1841** | 3次teacher后新增RL Q/C/P；2个P13正式未完成，另111个决策未终结尾；suffix成功0 |
| 30080→38272，P01 | 旧scalar Gaussian / 8192 | 1 / 1 / 450 / **0** | 7个正式未完成、1尾；仅首回合到P12，RL Q后撤销，无RL C/P |
| 38272→46464，P06 | **heteroscedastic log-std / 8192** | **5 / 5 / 2250 / 0** | 10个正式未完成（5 P09、5 P12）、1个160决策P06尾；RL共6次Q均在cross前GROUND撤销，RL C/P0 |

最新完成的hetero块还实际采到P06=3225、P07=10、P08=12、P09=2685。P12的2250样本来自5个完整回合各450；因此RL/P12已有相当直接失败信用，不能称“RL完全没训练”。RL hip/knee在其他阶段也参与全身控制，不能因P13缺样本说这几个通道没更新。

真正清楚的缺口是：**当前state-dependent分布没有任何P13直接on-policy样本，P10/P11直接样本各只有5；已有P12样本没有RL crossing/placed后继续停稳的轨迹。** 旧scalar策略的P13经历通过保留权重继续存在，但其后的P01/hetero更新会改变共享特征和动作输出，不能等同于当前分布已在P13接受直接训练。state-dependent std只是可表达“运动与收尾采用不同探索”的能力；没有P13数据，不代表这种能力已学好。

旧P10路径提供了必要但有限的可行性证据：三次真实roll-in后，PPO自己的RL Q/C/P分别完成于7912/8066/8122、7960/8064/8132、7789/8075/8139。并非全程成功；其中两个终局仍有实际body/command停稳条件未满足，且nominal已经归零。不能只把问题归因于sigma，也不修改stop标准。

## 2. 为什么从P10，而不是只从P12/P13或继续无限P06

主规范第9–10节明确要求P06/P07→后段和P10/P11→P13两类真实课程，不能只练第一后腿；第4/13节要求前驱动作和后续回报连续；第20节仍要求最终保存checkpoint从自然P01评估。已完整读取主规范与补充，两者的建议采样比例不是硬验收条件。

P10 offset0是现成的较早第二后腿起点：保留P10→P11→P12→P13的动态连续性，而非在RL已经AIR或placed之后才交给策略。即使P10/P11在某次真实轨迹中各只有一个decision，仍如实记录其余physics ticks的自身请求与后续动作，不人为延长phase或恢复历史姿态。

P06继续很重要，但它已经暴露一个实际筛选瓶颈：最新hetero块10个完整回合有5个没通过P09，另5个到P12也未RL cross；最终保存46464的自然P01 mean评估甚至在P06因RL workspace差16.834mm未完成。后者已促成当前新的preparation信用，但不保证以后每次都到RL。只继续P01/P06可能把下一段预算再消耗于第一后腿之前，P13仍拿不到当前分布的反馈。

P10可暂时绕开“必须先由C完成前三腿才能采到后段”的数据筛选；这是课程用途，不是绕过最终任务。它**不能证明C已学会RR准备、放置保持或创造P10入口**。`current_rear_chain_diagnosis.md` 中ep1的RL资格撤销早于RR退回，说明RL自己的净空/前进协调也值得独立学习，但不能证明稳定的teacher RR入口一定解决它。

## 3. 固定prefix风险与最小限制

既有P10三次真实前缀各948教师决策/7584ticks，63.2s接管，200s任务时钟包含前缀。接管RR已有教师Q/C/P6938/7109/7579、当前TOP/front约+92mm；而当前P06策略自然RR放置后的入口和支撑通常更脆弱。**这些不是同一reset分布**。

- 教师的FR/FL/RR完成事件、前缀动作和时间单独记录，不入PPO storage/global。只把接管后的RL与停稳动作计作PPO信用；最终即使成功也标 `SUFFIX_SUCCESS`，不能写自然P01成功。
- 这个固定teacher前缀没有reset状态随机化；相同/不同seed均不自动提供物理参数或入口鲁棒性证据。策略之后的随机动作产生变化，不等于已覆盖多样C入口。
- 首块保持P10 offset0；不同时引入offset随机化、改nominal、改entropy或改std。P10可能很短，随意增加teacher offset会直接错过目标phase/准备过程，反而削弱课程目的。
- 用已有P01/P06轮换补自生成入口，而不是立即新增C-prefix库、混合采样器或高层网络。只需沿用已有prefix证据记录实际handoff phase、miss/fallback和教师排除信用。
- 若实际发生fallback，按真实P01采样记账，不假称这批获得P10覆盖；它仍是合法训练数据，不因任务失败自动禁止optimizer。预算结束的有效尾样本可以已优化，但不能假称完整成功/失败回合。

2048是16个128决策完整更新的兼容预算。历史P10的两个完整后缀分别约968/969 credited决策，因而这个量级曾覆盖两次完整停稳期限及一个尾段；**这只是量级依据，不承诺新策略同样经历两回合或到达P13**。历史此块训练函数wall约24.18分钟、外层约30.81分钟，roll-in约17.40分钟；这些旧实测显示初始化开销不能忽略，不当作下一块耗时保证，也不能计入PPO决策预算。

## 4. 现成exact-resume路径，不再次清模型

下一块来源应是**当前9d70aae运行在完整update边界实际保存的最新checkpoint**，而不是预写计划终点，也不回退使用旧reward的46464。9d70、reward配置、策略分布和执行拓扑保持不变，仅在新run/课程epoch切换固定起点。

现有接口已支持：`scripts/run_semantic_ppo.ps1` 的 `-Command train -SemanticVersion v3 -NumEnvs 1 -FromPhase P10 -TeacherOffsetDecisions 0 -Stage phase_suffix -Decisions 2048 -Seed 1001`，附实际checkpoint和固定HEAD。这里只列参数语义，**未执行命令**。

不传 `-NewMdpWarmStart`，不再传 `-PolicyDistributionMigration`。`semantic_cli.py`从checkpoint metadata自动解析hetero版本；`semantic_training.py:350–419`普通load验证同runtime/seed/拓扑，实际加载actor/critic/Adam/normalizer、恢复训练RNG，要求fresh storage。`train_semantic:665–695`允许新run的固定采样epoch并拒绝run内变动，继承lifetime与stage_spent账。P01→P10改变reset采样分布，不声称物理状态逐位续接，也不伪称另一个policy architecture或reward修改。

现有exact-resume绑定seed，因此先继续1001；不能为表面多样性随意换seed而绕过现有源checkpoint校验。将来若另行需要分支种子实验应显式记录，不是本建议的必要前置工程。

截至46464真实预算账为full_episode12800、phase_suffix23552、origin10112。当前正在进行的P01预算必须在其实际完成后增加；后续P10/P06消费phase_suffix、P01消费full_episode，均沿用现有余额检查，不重置origin或已消费预算。

## 证据文件与停止点

- 主规范：[continuous_transition_revision.md](C:/Users/kskzz/Downloads/codex_residual_ppo_continuous_transition_revision.md)，特别第4、9、10、13、20节；补充：[rear_leg_continuation.md](C:/Users/kskzz/Downloads/codex_residual_ppo_rear_leg_continuation.md)。
- 最终hetero课程：[policy_p06_block_46464.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/policy_p06_block_46464.md)。
- 旧自然链：[p01_block_38272.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p01_block_38272.md)；旧P10：[p10_block_30080.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_block_30080.md)。
- 入口与因果边界：[current_rear_chain_diagnosis.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/current_rear_chain_diagnosis.md)；独立mean评估：[eval_46464_diagnosis.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_46464_diagnosis.md)。

有界建议到此结束。不暂停当前训练、不新增成功探针或启动门槛，不把后缀/教师成功当最终交付；下一步只需要在现有固定版本与保存边界上分配真实采样。
