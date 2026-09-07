# Entropy、stop shaping 与 state-dependent std：只读结论

2026-09-06，源码 HEAD `64abc5357d00763419fe51c8126db8454fcf7697`。只用 PowerShell 读取本地代码、已完成 run 和已有报告；没有 Python、tensor load、Isaac、生产修改或参数决定。C38272 的确定性 P01→P09 wheel-only 失败是根代理已报告事实，不是本报告新运行的评估。B 正在原版本执行。

## 结论

官方 state-dependent log-std **提供分开表达不同状态探索幅度的能力，不自动解决探索与停稳目标的权衡**。迁移时保留当前 mean 和 sigma，初始策略分布并未变得更容易停止；此后正熵正则仍在所有采样状态鼓励更大的熵。现有证据不足以认定熵是 P09 或 P13 失败主因，也不足以支持和架构迁移同轮修改 entropy。C38272 eval 使用固定 mean，因此该次失败不能直接归咎于评估时 Gaussian 抽样；训练探索可能影响已学 mean，但没有配对证据可分离其因果份额。

## 实际 entropy 调度与 PPO loss

`semantic_training.py:29/597–599`：`c = .005 - .004*min(lifetime_global/210000,1)`，每次完整 rollout 后、官方 update 前赋值；不是每回合重启，也不是按本次8192或 v3 已花预算重新归零。

已完成 P01 run `20260906T0819495765677Z_g64abc5357d00_b22cdf1222cd4e759de72734f0d5fa7a`，64行 optimizer_updates 的只读计算：

| 完整更新边界 | c | 已记录12维 raw Gaussian entropy H | 由记录计算 −cH |
|---|---:|---:|---:|
| global30208 | .004424610 | −5.492495 | +.024302146 |
| global38272 | .004271010 | −5.289020 | +.022589453 |

64次 `−cH` 平均 .023598090。负数 differential entropy 合法；正 `−cH` 不是符号错误，最小化它仍鼓励 H 增大。官方 `distribution.py:206–208` 已按12维求和，`ppo.py:313` 再按 minibatch 求均值，不见额外重复乘12或误乘物理 dt。

官方 loss 是 `surrogate + value_coef*value_loss - c*entropy.mean()`；entropy **不在** env reward、returns 或五个 reward families 内。官方 `PPO.compute_returns:205–209` 当前按整个 rollout 标准化 advantage，且 clipped surrogate、value error 与 entropy 梯度的来源不同。因此 `.0236` 的 loss 数字不能直接与 `.025` 的环境 reward 相减，也不能据此说 entropy 占主导。没有保存的分项梯度或配对运行，无法量化其控制贡献。

## 现有环境奖励与停止信号

v3 配置固定 `gamma=.995`、potential weight=5、成功/失败事件 ±40、时间成本 .02/s；body/contact/smooth 权重 .4/.2/.1，residual-magnitude regularization=0。`semantic_reward.py:112–175` 将质量成本按实际120Hz dt 积分；完整8tick decision 时间成本 .001333333，不见重复积分/Hz缩放错误。

当前 stop command 项确实存在且只进入一次 global potential（`semantic_supervisor.py:480–500/546–566`）：

- 仅所有 leg 已有 placed 后，finish 才进入 global phi；因此它不会在当前 C38272 尚未完成 RR 的 P09提前给予停止奖励。
- command progress `q = 1`（max actual wheel command ≤.02），否则 `.02/command`。v3 stop 平均4项，finish 中 stop 权重 .2，global finish 权重 .15；故 command 对 `phi` 的系数为 `.0075*q`，再统一进入 `5*(gamma*phi_next-phi_before)`，不是新增第二个 family。
- 仅 command 从 .16→.08→.04→.02、其余条件相同且非 terminal 时，隔离出的 command shaping 分别是 +.004640625、+.009281250、+.018562500。不是每帧发放该进度值；持续保持相同 q 时该子项是小的负折扣项。
- 实际 body linear/angular/wheel speed 的三个 stop 子项仍为 clipped `1-speed/tolerance`：速度高于对应阈值时该子项为0，减速但尚未进入阈值内可能没有这一子项的 dense 改善。这是现有信号局限，不是已证明的缩放 bug 或失败主因；command 连续项和其他奖励仍可能变化。
- hard success 仍要求实际速率、command、区域、当前支撑、安全与持续 .5s 全部满足。不能把 q 增大或 suffix placed 当成成功。

同 HEAD、已完成 P10 run `20260906T0748083849450Z_g64abc5357d00_525e380c18ae4039ad93d764e7a06f50` 的 **1839个 P13非终止已优化 decision**，本次 PS 重新汇总：

| 已加权 family | 平均/decision | 合计 |
|---|---:|---:|
| task_progress | −.025386327 | −46.685456 |
| body_stability | −.000503871 | −.926619 |
| contact_motion_quality | −.000005276 | −.009703 |
| control_smoothness | −.003048414 | −5.606034 |
| control_regularization | 0 | 0 |

这里特意排除2个终止事件，不能当全回合总回报。已有 `p10_block_30080.md` 的第二回合零 nominal 段628样本中 command 超标627、body linear超标372、angular超标32、actual wheel超标142；不是只有命令抽样噪声。已有 `lift_credit_live_reward.md` 则验证上升 lift 子项 +.003921498 时总 task 仍为−.009431366，是折扣/时间及其他phi项的结果，不是高度信号反号。

## Potential、discount、terminal：未发现本次可证实接线错误

`semantic_reward.py:179–190` 是 `F=5*(.995*phi_next-phi_before)`，真终止将下一 phi 置0；事件另加±40。普通 phase 不 terminal、不重置 global phi/history。官方 `ppo.py:194–204` 用相同 .995 构造 TD/GAE，并以 done 切断 bootstrap；N1 task timeout/失败/成功均 time_outs=false。非终止 rollout 边界正常使用最终状态 value，与任务终止不同。

在这套“每次 issued action 一次 gamma”的决策时间约定下，shaping 内的 gamma 和回报外的 gamma 并非重复错误：逐轨迹 `sum(gamma^t*F_t) = -5*phi_start + 5*gamma^T*phi_terminal`，terminal phi=0。负的累计 shaping 或进度上升时负的单步总 reward 不足以证明 reward 冲突；理想价值表示下该项是 potential-based shaping。实际近似 critic、稀少成功和有限训练仍影响学得多快，不能由这个恒等式保证训练效果。

gamma=.995 在15Hz下的有效折扣跨度约200 decisions/13.33s，是既有建模选择；本次没有证据把它判为实现 bug。short terminal interval 的成本只积分实际已执行 tick，仍按一个 issued decision 终止、下一phi/value为0，源码已显式记录这一约定。

已有真实分解还验证 applied-only smoothness，没有独立重复惩罚 nominal 和 residual；25个 soft-earned P09 AIR 样本 confirmed rebound=0。没有发现“主动 AIR 自动罚 rebound”、entropy 混入环境 reward、错误终止 bootstrap 或重复缩放的具体证据。

## 是否同轮改 entropy

不建议基于目前证据把 entropy 改动绑到候选分布迁移同轮。保留调度便可单独考察新增状态相关表达能力；如果另行修改 entropy，应诚实标为优化目标/训练超参数变化（不是物理 MDP reward 变化），不要凭停不住就固定 P13 std=0。此建议仅维护结果可解释性，不新增实验门禁，不要求等待全任务成功才能继续优化。当前固定版本继续，是否实施仍由根代理在本块及 fresh evaluation 后决定。
