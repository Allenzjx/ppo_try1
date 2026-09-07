# P10 capture：前缀与首 512 已优化样本

**固定窗口 g62849–63360 / updates457–460；训练仍运行，窗口无终止。** 仅 PowerShell 只读核验，不运行 Python/Isaac，不改生产或 sigma，也不延长窗口。

运行：[20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc)，N1 / seed1001 / v3 / P10 offset0。完整请求4096不记作已完成；本报告只计实际512。

## 1. 原 checkpoint 的实际续接依据

source 为 immutable checkpoint62848，累计456 PPO updates / 9120 optimizer steps。当前 started manifest 明确 `new_mdp_warm_start=false`、`resume_migration=null`、`policy_distribution_migration=false`；已解析 policy 仍为 heteroscedastic_log_v1，当前 runtime 与 source runtime content 一致。是同一任务/奖励契约下切换已支持的固定 P10 reset curriculum，不是重置策略、复制旧物理状态或新 MDP warm-start。

- source actor hash `7d8b3ea0b862a587ed65877e84da836033c14c8275a3d46f694ebd8cb885f56b` 与首 update457 的 actor-before 精确相等。
- source optimizer-state hash `55e8d8a85e279f3b48b0e1b392f178afc0656633e1ede90614bf61becbb6e9d5`；source LR=1e−5。
- source normalizer hash `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` 与首个真实保存62976相等。
- 首保存62976 sidecar：457 updates / 9140 optimizer steps、roundtrip=true，resume ancestry指向62848/456/9120及上述source actor。
- 4次实际更新严格global每次+128、update每次+1，3个相邻actor-before/previous-after均相等；80 optimizer steps、全部finite gradient=true，4次记录LR均1e−5。

只读源码 `semantic_training.load_semantic_checkpoint` 的普通分支在开始学习前要求 fresh empty storage，实际restore后核验actor/critic/optimizer/normalizer状态hash，再恢复训练RNG；Adam重置在未被此次调用的 warm-start 分支。以上运行参数、成功进入真实update及保存的lineage构成续接依据。本报告没有额外反序列化Adam或把训练后的optimizer hash误当source原值；“exact resume”指模型/优化器/normalizer/RNG/计数续接，不指相同reset分布或物理轨迹续接。

## 2. 教师前缀与策略信用严格分开

前缀从episode tick0 / 0s开始：**948个reset-only decisions / 7584 physics ticks**，每行 `policy_credit=false`、raw PPO action全零、projected residual全零、native audit verified。实际A动作本身当然可非零。

最后前缀决策在t7584 / 63.2s由P09到P10；accepted=true、无miss/fallback。请求P10，teacher offset0，最大前缀1800/最大takeover30均只是既有预算上限，没有补造历史入口。

接管收据：requested/actual phase均P10、`requested_phase_still_active_at_credit=true`、`from_P01_current_policy=false`，剩余任务时间136.8s。source-control tick7583、command-physics tick7763、handoff tick7584属于收据明确区分的时钟；不把含reset底层偏移的command tick作为episode elapsed tick。首个真正PPO决策是 **g62849 / t7592 / decision1 / P10**。

接管时：

| 腿 | 教师建立的硬事件 | 当前物理状态 |
|---|---|---|
| RR | Q6938 / C7109 / P7579 | TOP，front+91.978375mm，clear−1.638344mm，load .484429155 |
| FL | Q2461 / C3115 / P3583 | TOP，front+529.826585mm，clear+2.120805mm，load .425281204 |
| RL | 无Q/C/P | GROUND，front−87.233629mm，clear−47.864353mm，load .090289642 |

后续RL **Q7959 / C8070 / P8113** 都发生在t7584接管之后，属于当前策略信用段的真实事件；教师RR/前腿事件不能算为当前PPO学得的全P01通过。

首512策略决策共4096 physics ticks，512行native verified和no-state-writes verified均true，无terminal。实际总episode elapsed ticks到11680=`7584教师+4096策略`，全200s任务时钟没有重置；PPO global只增加512，不增加948前缀决策。

| 信用阶段 | P10 | P11 | P12 | P13 | 合计 |
|---|---:|---:|---:|---:|---:|
| 实际已优化样本 | 1 | 1 | 65 | 445 | 512 |

## 3. P13且四轮nominal严格0的实际子集

共有 **173** 条：**g63188–63360 / t10304–11680 / 85.866667–97.333333s**。选择条件是每行起始phase_id=P13且nominal四轮逐值等于0；不把先前有限非零nominal混入统计。

| 实际量 | 最小 | 最大 | 均值 |
|---|---:|---:|---:|
| 四轮actual command最大绝对值（rad/s） | .214346345 | .524636556 | .369606284 |
| 四轮实测speed最大绝对值（rad/s） | .196875036 | .597563386 | .382628401 |
| body linear speed（m/s） | .006485348 | .249566509 | .064051862 |
| body angular speed（rad/s） | .033880783 | .648582781 | .192359295 |

- actual maxcommand≤.02：**0/173**。
- 实测maxwheel-speed≤.25：**6/173**。
- final region=true：**124/173**；current final support=true：**173/173**。
- final controlled=true：**0/173**。完整物理受控收尾未成立；不能只看历史四腿placed。

末行g63360：actual wheels `[+.231671448, −.267636210, −.420365727, −.166562086]`，实测maxspeed=.421241581，region=false/support=true/controlled=false，stable duration=0。当前FL/RR是TOP，loads .492981/.507019；FR/RL是AIR、load0。四腿历史placed保留，但不是四腿当前都承载，更不是task success。

## 4. 同实际访问状态的 raw mean/std 与建议、真实命令比较

下表只针对上述173条。raw mean/std无物理单位；其余列为rad/s。`0.6*tanh(mean)`只是该条实际采样状态上保存的行为策略均值经过幅度映射，**未执行历史滤波/速率限制**。

| 轮 | raw mean均值 [范围] | raw std均值 [范围] | 未滤波mean建议均值 | 未滤波sample建议均值 | 实际 filtered canonical 命令均值 |
|---|---|---|---:|---:|---:|
| FL | +.372537 [.293955,.433919] | .120235 [.111637,.129711] | +.213598 | +.214008 | +.214089 |
| FR | −.384037 [−.499391,−.282973] | .201437 [.192463,.208662] | −.219371 | −.218393 | −.221241 |
| RL | −.707188 [−.851130,−.525585] | .203933 [.187077,.221992] | −.364435 | −.360466 | −.365154 |
| RR | −.198339 [−.441477,−.055962] | .259240 [.233256,.287886] | −.116881 | −.108882 | −.113306 |

最后一列来自 `actual_drive_target_full12` 的 canonical 逻辑应用命令；native 下发验证在独立 audit 中完成，不把此 double 向量当成经过左轴符号映射后的 float32 native readback。

未滤波mean建议的逐通道范围：FL `[+.171463,+.245155]`、FR `[−.276983,−.165393]`、RL `[−.414996,−.289199]`、RR `[−.248921,−.033542]`。**0/173** 条静态mean建议满足四轮绝对值全≤.02，因此不能把该子集未停轮只解释为采样std造成；这也不推断成功概率或要求重置sigma。

例如g63360：

- `0.6*tanh(mean)`：`[+.208145, −.203777, −.355455, −.103193]`。
- `0.6*tanh(sampled raw)`：`[+.231671, −.267636, −.422978, +.141436]`。
- 真实命令：`[+.231671, −.267636, −.420366, −.166562]`。

RR在同一条里未滤波sample建议为正、真实命令仍负，正说明不能丢掉既有滤波/速率/前序动作历史，把静态建议当本次真实执行。raw mean和std是当前真实随机访问状态、各自采样时行为策略的记录，窗口期间发生真实PPO更新；不是把source62848冻结后从P01执行的确定性mean轨迹，也不是对成功率的估计。

结论限定：已确认同契约模型/Adam/normalizer计数续接、教师信用排除、RL在PPO段完成子事件，以及零nominal P13样本中实际命令和静态均值建议都明显非零。当前只是P10初始化suffix，窗口没有受控成功；后续结果不预填。报告到63360停止。
