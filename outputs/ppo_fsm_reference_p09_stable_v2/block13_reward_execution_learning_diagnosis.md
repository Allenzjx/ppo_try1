# Block13：奖励尺度与实际残差消费者核查

固定来源 `train/20260910T1451368221058Z_g7db0d17f398d_19bb6607a9384f08b00d341ba21060b7`：2048条审计，global149889–151936，16次PPO更新/320 optimizer steps。旧7db/v1 nominal语义，不代表当前v2分布。无活动数据、physical大流、rollout tensor或模型字节读取，无参数/生产修改、无CSV。

**结论：未证明训练消费者错误或奖励符号缺陷；存在可量化的局部进展—平稳性权衡。** P06/P09的PBRS本身平均为负，不能只归咎稳定性惩罚。观察到的高限速比例符合已有执行合同，不等于残差通道关闭或PPO用错动作。

## 1. 日常尺度：排除真正终态后，每条15Hz决策的加权奖励

| 源phase | 样本：全部/非终态 | PBRS均值 | task_progress均值 | body成本均值 | contact成本均值 | applied smooth成本均值 | 总reward均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P05 | 746/744 | +.0040283 | +.0026950 | −.0001658 | −.00000355 | −.0018472 | +.0006785 |
| P06 | 266/265 | −.0025110 | −.0038444 | −.0003922 | −.00000975 | −.0024294 | −.0066757 |
| P09 | 116/113 | −.0079064 | −.0092398 | −.0011564 | −.00001723 | −.0022742 | −.0126876 |

PBRS是 `5×(.9985×Φ_next−Φ_previous)`，非终态task_progress还扣除`.02×实际秒数`（完整8ticks为−.00133333）。P05/P06/P09非终态PBRS中位数−.001444/−.003297/−.003583，95分位+.093554/+.063037/+.051854：正进展脉冲存在，但P06/P09正PBRS总和2.57034/.95073小于负PBRS绝对总和3.23577/1.84416。

终态必须另看：全块7个真实独立安全中止=6 FALL+1 HARD，事件成本均−40、next Φ=0。P05两个终态reward−41.71455/−41.46443；P06一个−42.24540；P09三个−42.25620/−42.65154/−42.21132。191条末回合采集尾不terminal，不伪造第8次失败。

局部正PBRS却负总reward：P05 **64/343**、P06 **14/80**、P09 **5/25**。例如P09 global150321：PBRS+.0020041，time−.0013333、body−.0053459、smooth−.0020260、contact−.0003971，最终−.0070982。这证明成本可以抵消某次小进展，**不证明该姿态/加速度必不可少，也不证明稳定性符号错误**。body成本绝对总和占正PBRS总和仅0.94%/4.04%/13.74%；不要把净进展的正负抵消误读成body成本在每条动作上压倒任务目标。

## 2. 计算与跨阶段信用没有发现断裂

- 2048/2048：PBRS公式、family权重、实际tick时长及总和一致；加权求和最大浮点误差7.11e−15，collector reward与breakdown转float32 **完全相同**。
- body分项等于−.4×(attitude+Euler rates+angular acceleration积分)/3；smooth只等于−.1×(actual一阶+二阶积分)/2。nominal/residual变化仅诊断，没有重复相加扣分；residual magnitude regularization全2048条为0。
- 接触chatter积分P05/P06/P09为2.6875/1.58333/.725，但在本版本只是diagnostic，不能直接当罚分；confirmed passive rebound积分全为0。未发现“主动离地统一扣分”的数据证据。
- **42次普通phase交接全部done=false、bootstrap=true、terminal_event=0**；包括4次P05→06、3条P06→07→08→09链。非terminal相邻Φ精确连续，未在phase边界清零。各episode折扣PBRS望远镜恒等式最大误差1.69e−15。
- 7个真实终态全部bootstrap=false、next Φ=0；全2048条time_outs=false，短终态按实际tick成本、仍每policy decision一次gamma。正ΔΦ但PBRS≤0的105/80/43条是折扣项的数学结果，不是符号反转。
- 审计raw与applied-audit raw全相同，mean/std/reward/logprob全finite、std全正；独立float64 Gaussian联合logprob与记录float32最大差3.07e−6。更新消费者源码在存储和minibatch处核对未变换raw，16个成功更新日志均actor hash变化且梯度finite/nonzero。

限制：本轮未加载rollout tensor，所以不冒称再次逐元素重算了storage/GAE；上述是审计数值、边界标记、已成功执行的消费者guard及更新记录核验。reward/profile/消费者源码均与该run绑定的7db Git字节hash吻合。

## 3. Raw→filtered request→headroom→final drive：不要混称“饱和”

下表的“限速一致未达目标”是**至少一个通道**在决策末端仍未达`tanh(raw)×cap`；全部这些差异分量都吻合60deg/s、1.8rad/s²及记录handoff保留tick的端点限速公式。阈值`.95/.99`仅统计描述，不是新门禁。

| phase | 限速一致未达目标：决策数 | tanh幅值≥.95：通道×决策 | ≥.99：通道×决策 | headroom额外裁剪：决策数 | candidate→final不同：决策数 |
|---|---:|---:|---:|---:|---:|
| P05 | 616/746 =82.57% | 853/8952 =9.53% | 374/8952 =4.18% | 16/746 =2.14% | 9/746 =1.21% |
| P06 | 264/266 =99.25% | 399/3192 =12.50% | 173/3192 =5.42% | 1/266 =.38% | 0/266 |
| P09 | 110/116 =94.83% | 135/1392 =9.70% | 57/1392 =4.09% | 4/116 =3.45% | 15/116 =12.93% |

headroom裁剪具体是P05的FL knee16条、P06的FR knee1条、P09的RR knee4条。最终candidate差异可能包含最后clamp/slew，不是“实测关节已追上目标”；本轮不作逐物理tick完整projector重放。

一个重要尺度例子：P06 FR knee残差cap=112deg，235/266条末端仍在接近请求目标，虽然raw tanh≥.95有47条，filtered residual≥.95 cap仅4条；P09相同通道106/116条未达请求目标，filtered近cap为0。大raw探索请求与实际已施加幅度不是同一个量，不能只依据raw方差说关节实际扫过全部范围，也不能因此认定likelihood消费者有错。

全2048条12通道mask均开；16365 native ticks全部verified/actual effect，own-phase16323，42个交接tick另计；无in-episode状态写入。限速保留历史与“清零动作”不同。

## 4. 对训练分配的证据边界

P09仅116条、来自3个回合；P06来自4个回合，不能将这份分布推成全阶段稳定性结论。既有物理诊断已说明前驱P06可建立RR Q并连续进入P09，但RR C/P尚无；本轮不重做那些物理因果分析。样本深度与前驱连续性可以支持后续课程关注，**没有据此修改算法、超参、reward或成功门槛的证据**。

16次更新的clip fraction均值.24131、KL均值.020836（分别是PPO分布比值统计，不是执行器裁剪率）。有效LR为adaptive：14次更新末1e−5，update1147为5.0625e−5、1149为2.25e−5，末1152回到1e−5；不得写“整个block固定1e−5”。仅描述真实日志，不建议改调度。

短证据：`block13_reward_execution_learning_summary.json`；完整分位数/逐通道统计：`block13_reward_execution_learning_diagnosis.json`。物理过程复用 `natural_p01_block13_episode0_rl_knee_limit.md`、`natural_p01_block13_episode1_p09_fall.md`、`natural_p01_block13_episodes2_to7.md`。未读取当前活动block15。
