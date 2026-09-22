# Block14：FL 捕获 → P06 失载的真实信用（只读）

结论：**当前支撑损失没有漏罚；但本次有限 rollout/critic/标准化形成的学习信用，并不逐拍等于即时奖励符号。** 两段证据不足以证明整个目标冲突，也不能把它概括为“正确动作只差多训练”。

范围：仅封存 rollout1508/1509，各128样本；真实 optimizer 的20 minibatch × 5次/样本已有记录。没有新 forward、优化、物理或生产修改。方法仍为 PPO + LIMITED AUX；此 P05/P06 窗口 receiving ×3 均未激活。明细与四个封存文件 SHA：[JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/block14_capture_hold_credit_readonly.json)。

| 末拍 tick / learner | 状态 | 即时 reward | raw GAE | 实际标准化 A |
|---|---|---:|---:|---:|
| 2960 / 370 | FL placed2957，P05→P06 | +0.283563 | −0.314463 | +0.786560 |
| 2968 / 371 | 首 P06，仍 TOP | −0.001656 | +0.233656 | +2.037519 |
| 3104 / 388 | 首已记录 AIR，gap +0.397 mm | −0.123335 | +0.890005 | +1.004012 |
| 3496 / 437 | AIR，gap 29.864 mm | −0.031711 | +0.574926 | −0.246074 |
| 4096 / 512 | AIR，gap 40.084 mm | −0.005248 | −0.038845 | −2.681232 |

支撑损失确实可见：tick3096→3104，全局 Phi 0.478650→0.454932；FL retention 1→0.441532，soft window 权重仍1，FL 已放置分量 PBRS = −0.120090，全局 PBRS = −0.122001，再扣时间0.001333。tick3496 retention 降至0.045643。用生产 pure potential 重算256个实际快照，与日志误差为0。捕获收益来自势函数，不是额外捕获 bonus（terminal event=0）。在这两段中，geometry 最小分离71.537 mm >20 mm margin，实际 geometry/body 代价全部0；contact/smoothness 有诊断但权重0，未与支撑惩罚争抢。

信用边界：capture370距首次失载388仅18决策，但1508在384结束；捕获之后该 rollout 仅有14个实际后续奖励，然后用 critic 尾值−19.669006 bootstrap。1509尾值−21.090839。两段均无 done，普通阶段交接未断回报；gamma=.9985、lambda=.99。1508 的 GAE 不直接含1509真实失载奖励，不能说“没有 bootstrap”。官方实际使用**整 rollout 标准化**（按保存 returns−V、sample std 重算误差0，实际5次 minibatch A 均逐位匹配）。首次失载 TD delta 已为−0.178558，但后续 GAE令 raw A 为正；1509的116个AIR样本，标准化A为正43、负73（均值−0.078110），不是全在奖励悬空。27个历史已捕获且当前TOP样本实际A全正，均值1.202093。

HISTORY/梯度：P05→P06 的FL hip/knee cap 18/24→32/36°，H按上拍已过滤物理请求换算为−0.065802/−0.162487，并未清零；rho=.9。首失载拍 hip 的(base,H,mu,sample)=(−.254012,−.197370,−.203034,−.156633)，knee=(−.343597,−.319724,−.322111,−.326693)，投影残差−4.972/−11.360°（不是关节实测角）。实际网络均值头 loss 导数五次均值分别−0.028137/+0.002757，但该样本4/5次已走PPO clipped分支；仅局部未clip方向偏向增加hip均值、减小knee均值，**不等于物理改善方向**。捕获拍hip sample几乎等于mu，实际均值导数仅+0.000074，而knee为−0.022082。

116个AIR样本的真实均值头导数有明显抵消：FL hip/knee signed mean +0.000949/+0.000460，mean absolute .026286/.020583；FR/RR wheel mean absolute .064775/.061220，RR knee .128581。只能说明这些输出敏感度较大，不能据此认定旁路造成全局clip或推断Adam最终均值变化。旧logp与rollout记录严格一致；没有额外采样/forward。支持损失已进目标，现有证据更具体地指出**边界 bootstrap、critic 相对基线、整段标准化和 clipping 下的信用分配**，并未证明一个可直接修复的奖励符号/概率实现错误。此报告不建议改 reward/rho/LR，也不阻塞继续训练。

tick4120属decision515/update1510，超出选定两段，不补造信用。

