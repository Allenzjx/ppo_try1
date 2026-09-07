# 学习信号与时间信用：PBRS / critic / 128-step GAE 只读核验

2026-09-06；当前训练保持不变。只用本地源码、已完成 episode 的 JSONL 和 PowerShell/double 数学计算；没有 Python、Isaac、tensor load、生产修改或新门禁。主规范要求的**同一 global potential、真正终止吸收态**保留，不建议添加 FL/AIR/阶段/入口奖金，也不建议错误保留 terminal phi。

## 结论

1. 当前 PBRS/终止/GAE 接线合法，没有发现阶段切换丢 bootstrap、误用 timeout 或单步进度反号的实现错误。真实局部物理进展确实改变 reward；不能把单步总 reward 为负称为 bug。
2. 但 PBRS 不会在“最终仍失败”的完整折扣轨迹上额外留下阶段成就回报。相同起点、终止 phi=0 时，其累计恒为 `−5 phi_start`。它提供局部 TD 学习提示、改变价值表示，不等价于增加新的任务目标。
3. `gamma=.995` 的回报跨度与 `gamma*lambda=.94525` 的直接 TD trace 衰减必须分开。128 decisions=8.533 s，每块独立计算 GAE；几十秒后的后腿/停稳结果主要依赖 critic 跨 rollout 的 Bellman 学习，而不是直接回填早期 actor 样本。
4. 已完成首回合存在具体、可复核的近 deadline critic 误差：终止前已记录 value −3.450804，实际 terminal reward −42.274586，TD residual −38.823782。这证明该样本尚未预测好失败结果，不证明算法接线错误、所有 critic 状态都不准或需要立即调参。

## 1. 实际生产链

- `semantic_training.py:34,101–114`：rollout128、gamma .995、lambda .95、5 epochs×4 minibatches；`rl_library_wrapper.py:649` 设置整块 advantage normalization，而非每 minibatch。
- `semantic_supervisor.py:647–663,681–682`：普通物理阶段完成只前进 stage、更新其计时；v3 始终覆盖为同一个 `physical_potential`。阶段 deadline 或全任务200s为真实 `INCOMPLETE_CONTROLLER_BLOCKED`，不是“可继续同任务”的外部截断。
- `semantic_reward.py:179–190`：`F=5*(.995*phi_after−phi_before)`；真正 terminal 的 phi_after=0；success +40，其它真实 terminal −40；时间/质量成本按实际物理 dt 积分。没有阶段奖金。
- `SemanticRslAdapter.step`（`semantic_training.py:167–205`）：done来自core.terminated，time_outs全部false，真实final observation先保存再reset；相同 rollout 可以包含下一回合，但done负责隔离。
- `train_semantic`（692–734）调用官方 `act → env.step → process_env_step`，收满128后 `compute_returns`，保存真实raw/actions/logprob/value/return/advantage，再更新。
- 本地 RSL-RL5.0.1 `algorithms/ppo.py:187–209`：最后一槽使用 `critic(next_obs)`；`delta=r+(1-done)*gamma*Vnext−V`，`A=delta+(1-done)*gamma*lambda*A_next`，从 `A_next=0` 逆序开始。`ppo.py:403` 更新后clear storage。普通rollout边界并不done，但只经value bootstrap衔接，不保留跨批次GAE trace。

终止槽即使 next_obs 已是新reset状态，其 `(1-done)=0` 同时切断 Vnext 与后续GAE，因此不会把新回合价值借给旧任务。官方 timeout补偿分支也因time_outs=false不生效。上述行为符合有限任务终止语义，不应为拉长信用擅自改成bootstrap。

## 2. 完整失败轨迹的 PBRS，已用真实日志验证

对 T 个已执行动作，令最后状态为吸收态 phi_T=0：

`sum[t=0..T-1] gamma^t * 5*(gamma*phi_(t+1)−phi_t) = −5*phi_0`。

只读重算两个已经结束且全部位于完成优化范围内的 episode；相邻reward phi_before/前一phi_after连续误差均为0：

| 完成数据 | 动作数 / global范围 | 起点phi | 实测折扣shaping和 | −5phi_start | 差值 |
|---|---|---:|---:|---:|---:|
| 134849 P01 首回合，P06 incomplete | 939 / 54657–55595 | .05420549241807721 | −.2710274620903856 | −.2710274620903861 | +5.0e−16 |
| 121130 P10 suffix 首回合，P13 incomplete | 970 / 50561–51530 | .68 | −3.400000000000001 | −3.4000000000000004 | −4.4e−16 |

源run分别为：

- `runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189`。
- `runs/ppo_semantic_v3/train/20260906T1211309264074Z_g9d70aae58243_42ace02dcac84aaa83ee7e1ee578f58e`。

这是reward_breakdown双精度数值的直接求和，不冒充float32 PPO storage逐位复算。P10 teacher prefix不参与求和或PPOcredit；它让suffix起始phi较高，所以−3.4不是“老师数据被扣分”，也不能与P01的−.271直接比较策略好坏。

具体反例：同一起始物理/历史状态的两条轨迹，即使一条先完成更多合法后腿进展、一条没完成，只要最终吸收态phi为0，纯PBRS的完整折扣和仍相同。若二者T、终止结果及其它成本也相同，它不会另行偏好“最终失败但阶段更多”那条轨迹。这是保持原任务目标的数学性质，不是遗漏某个事件bonus的实现错误。

它仍可以提供可学习的局部变化。既有 `p06_first_episode_134849_readonly.md` 的真实tick5768→5776，RR/RL同时前送，phi增长 .000448625460267，隔离正向 `5Δphi=+.0022431273`；实际net shaping为负来自原有discount项，而非前进被反号。近似critic/有限数据时这些局部TD目标可以帮助学习；但不能保证它在无成功样本时自动替代远期任务价值。

进一步，如果critic已精确学会变换 `V_shaped(s)=V_base(s)−5phi(s)`，则shaped TD residual与base TD residual相同，GAE也相同。故不能一面要求严格PBRS/真终止，一面期待它永久额外奖励已失败轨迹中的阶段数。

## 3. 回报discount与直接GAE trace不是同一个跨度

下表是距未来一项TD residual或reward为n个决策时的代数系数，**不是实际成功概率或总advantage**。前四行只有在相应样本位于同一128槽内时才是本次GAE直接路径。

| 滞后 | gamma^n：未来reward在回报中的系数 | (gamma lambda)^n：单个未来TD residual的直接系数 |
|---|---:|---:|
| 1s / 15dec | .927569 | .429735 |
| 4s / 60dec | .740261 | .034104 |
| 8s / 120dec | .547986 | .001163 |
| 127dec / 8.4667s | .529092 | .000784 |
| 20s / 300dec | .222292 | 4.61e−8（理论未截断trace；实际跨块无直接路径） |
| 40s / 600dec | .049414 | 2.13e−15（同上） |
| 60s / 900dec | .010984 | 9.82e−23（同上） |

当前P01首回合P06入口22.6s，到62.6s失败相隔40s/600dec；前腿FR placement约12.7s到失败约49.9s。旧完成P10首回合RL placement67.85s到P13终止127.8667s也约60s。不能把这些真实几十秒链说成由一批128槽内的−40直接教会；它们主要依赖不断更新的critic在相邻状态/批次间传播、以及更近的物理成本和进度差分。

反过来也不能说128把“任务回报”硬截成8.53s：非终止末槽有critic bootstrap，理论价值可以包含更远结果。lambda控制估计器偏差/方差与直接trace，而不是把环境目标的gamma改为.94525。当前前腿已有技能及suffix课程，也不能仅由这个衰减表推断网络无法学习后腿。

## 4. 真实近deadline值误差与最后rollout边界

当前首回合939dec结束；最后episode内片段落在run第8批：episode dec897–939，共43槽，global55553–55595。该128槽的其余85槽是下一回合，true done会隔离。

利用该43槽日志中的实际float32 old_value/reward，按官方公式用PowerShell double逆算其**归一化前**GAE；终止把后续切断，因此不用读取下一回合或猜末端bootstrap：

| global / episode决策 | 距终止dec | old_value | 单步TD residual | 归一化前GAE，double重算 |
|---|---:|---:|---:|---:|
| 55553 / 897 | 42 | −3.513250 | +.016390 | −3.595963 |
| 55565 / 909 | 30 | −3.482691 | +.037024 | −7.149493 |
| 55580 / 924 | 15 | −3.469562 | +.001137 | −16.670650 |
| 55595 / 939 | 0 | −3.450804 | −38.823782 | −38.823782 |

末槽实际reward为−42.27458572387695；不是只有原始−40事件，另有terminalphi归零和实际成本。前42槽同批可收到明显信号，且不依赖阶段label奖金；但上一批已更新/清空的dec896及更早样本不会被这次终止重新回填。其学习依赖先前保存的bootstrap估计及未来采样，而不是离线重放旧buffer。

这不是对 `.pt` 中advantages的逐位比对，更没有声称这些是PPO最终使用的归一化值。官方会把整个128槽（包含下一回合）统一归一化，所以局部负GAE也不能直接当最终标准化优势或某个动作必然被压低的证明。

可确认的具体学习问题是：这个近deadline状态的已学value对本次真实−42.27结果仍有约38.82的on-sample TD误差。不能用小的平均value_loss宣称所有deadline附近已校准，也不能单次认定网络容量或gamma有bug。已有每批保存的values/returns/advantages/observations足够供后续固定边界抽样复核；当前不加载它们、不新增门禁。

## 5. 有限失败下的真实目标取舍，不是加奖建议

PBRS以外仍有时间、body/contact/smooth及终止事件。对于“其它成本相同且最终必失败”的纯数学比较，晚收到负终止事件会因discount减轻：以完整15Hz动作、只计−.02/s时间与−40终止，T=300/20s的起点回报约−9.143759，T=600/40s约−2.239975。PBRS相同起点只加同一个常数。这反映原discount负惩罚目标对延后失败的取舍，不是某次实际policy被证明故意拖延；成功+40、实际质量成本、阶段推进后新的物理机会都会改变比较。

真实当前939dec首回合的折扣终止事件为−.363169803、时间−.264257640、PBRS−.271027462；task合−.898454906，总reward合−1.520721701。它们与未折扣报表中的−40、task−50.536等不矛盾：汇总用途不同，不能用未折扣长轨迹累计量直接代替critic的discounted目标。

## 审查边界

目前证据支持优先观察已存critic在阶段期限附近及跨rollout状态的预测/TD误差，以及同一MDP连续训练后是否改善；**不支持把合法PBRS单步负值判bug，也不支持通过新增阶段/入口/AIR奖金或保留true-terminalphi“修复”**。

若以后明确比较gamma/lambda/rollout长度，它属于独立的学习估计/目标配置实验，不能由本报告自动决定，更不要求先全任务成功才能继续optimizer。当前训练、reward、nominal、P13 control、entropy和std均不改。记录当前未见完整任务成功的事实，不把prefix或历史placed当正终止样本。

本次第一次只读求和命令因PowerShell自动变量名冲突失败；已改用任务专属变量重新运行，以上表格仅来自成功重算（exit0），不是采用失败输出。没有生产异常或文件修改由该诊断产生。仅新增本报告，写完停止。
