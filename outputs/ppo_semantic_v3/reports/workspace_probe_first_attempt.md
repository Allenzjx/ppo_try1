# 首次真实 workspace probe：接口拒绝，非完整控制权验证

源 run：`runs/ppo_semantic_v3/interface_checks/20260906T0233067086105Z_g2248cd1_workspace`。源 HEAD `2248cd1940ad7a37d7a283f277732a366519db8d`，seed1001，P06 teacher-initialized suffix，offset0，请求每段64 decisions，raw magnitude .5。本文只读 PowerShell/.NET 解析实际文件，未启动 Python/Isaac，未修改任何源 run。

## 结果分类

总生命周期 `FAILED`，确切错误：`drive feedback bias exceeds the bounded mapper envelope`。调用栈终止于 `RobotAdapter.apply_full12 → _full12_drive_feedback_bias`，不是环境返回的碰撞、纯轮爬升或其他 physical task failure。新范围段因此 **incomplete/interface failure**；不可声称完整控制权、完整新范围或成功通过已经验证。探针不是训练，optimizer updates/steps/credit均为0。

| 段 | 已完成决策 | 实际120Hz响应ticks | 阶段 | native verified | 真实native effect ticks | 结束解释 |
|---|---:|---:|---|---:|---:|---|
| nominal_zero | 64 | 512 | 全P06 | 512/512 | 0 | 有界窗口结束，无task termination |
| old_range | 64 | 512 | 全P06 | 512/512 | 512 | 有界窗口结束，无task termination |
| new_range | 2完整，第三决策仅4ticks | 20 | 全P06 | 20/20 | 20 | 下一次dispatch被feedback接口拒绝；无segment_manifest |

前两段逻辑physics tick3585–4096（29.875–34.133333s）；新段仅3585–3604（到30.033333s）。所有记录的1044响应ticks均finite、body collision=false。此事实仅覆盖已执行样本，不能外推未执行的新范围或后续任务。

## 配对初始条件

三段均独立物理reset后，用同一冻结teacher到达P06，开始响应的逻辑tick3584/time29.8666667s。逐字段、逐数组数值核对三段 **完整已记录响应初始 physical observation 相同**（不只是同seed），包括base、joints、wheel geometry、contacts/force、bodies、CoM、commanded targets。nominal_zero与old_range的segment_manifest初始观测一致；new_range的prefix最后`credit_start_observation`经不依赖JSON属性排序的递归比较，与之0项差异。

- 共同source nominal：`[22.8,-13.4,0,45.9,6.9,0,0,0,.3,.3,.3,.3]`，controller bias全0。
- 共同上次ACK drive：`[21.55,-12.15,-.7996284,45.3470521,5.65,.6421600,-1.3499030,.9543499,.3,.3,.3,.3]`。
- 共同base位置 `[.330085814,-.004596896,.091114365]m`，body speed .0425169m/s；四轮均实际接触，FL/FR在台面，RL/RR在地面。
- old/new整份451行prefix字节相同，SHA256 `83af2467213a6c6dd3f218c033549ad55f86b72185d7a696ef46344ae4256926`。
- nominal_zero prefix SHA不同（`d93aa6cf3f004d8564830bfdbaaf51bfabae65b17146c145a7b4fe8cb944c282`）。差异只在第1行reset原始观测的3个数值：FL lower-link vx/vy各约5.82e−11m/s，CoM z约1.64e−9m；余450行字节相同。**不能声称每个reset原始bit完全一致，但在实际响应credit起点，记录中的物理状态已完全一致。** 未记录的内部状态不能由此额外证明。

## nominal_zero 与 old_range 的真实响应

两段相同4.266667s响应窗口、相同P06 nominal，全程controller drive bias=0。

old_range稳态projected residual为八舵机交替 `+1.848468629/-1.848468629°`、四轮 `+.05545405887rad/s`，与`.5 raw → tanh(.5)`及旧4°/.12 scale一致；不是策略优化学出的动作。每个tick实际float32 target有变化。单tick相同pre-state zero-current-residual反事实差的peak：各servo约.02789855rad（1.5985°）、各wheel .05545405rad/s。这个反事实只去掉当前residual，保留t−1 target状态；不能误当整段zero rollout与old rollout的因果差值。

末帧实测关节（FLH,FLK,FRH,FRK,RLH,RLK,RRH,RRK，单位°）：

| 段 | 实测8关节 |
|---|---|
| zero | `[21.9134,-12.1593,-.3817,45.5538,6.2852,.4838,-.8896,.7687]` |
| old | `[23.6892,-13.9386,1.5901,43.7338,8.2881,-1.3961,.7998,-1.0719]` |

old−zero末帧各关节差约+1.776/−1.779/+1.972/−1.820/+2.003/−1.880/+1.689/−1.841°，体现真实机械响应而非仅命令计数。

| 指标（相同512tick终点） | zero | old |
|---|---:|---:|
| body x相对共同初态位移 | +63.110mm | +74.676mm |
| body z相对共同初态位移 | +2.935mm | −1.586mm |
| body speed | .017885m/s | .021140m/s |
| FL实际载荷占比 | 21.26% | 19.38% |
| FR实际载荷占比 | 24.27% | 25.86% |
| RL实际载荷占比 | 28.31% | 29.90% |
| RR实际载荷占比 | 26.16% | 24.86% |

这支持“旧范围能实际改变关节、推进与载荷”，不支持“旧范围已能完成后腿跨越”。四轮末端仍接触，RR距前沿zero −431.913mm、old −420.035mm，均远未越沿。FL是真实接触载荷，不是以CoM位置替代。

## new_range：最后安全样本与拒绝点

20tick/0.166667s时，八servo projected/combined bias交替±10°（最大10，controller bias仍全0）；四轮bias+.2772702944rad/s。实际ACK wheel target+.5772702944rad/s，真实buffer按轮符号为±.5772702694rad/s。旧冻结feedback接口仅允许servo bias≤`SERVO_TRACKING_COMPENSATION_MAX_DEG=10°`，把扩大后的PPO residual与小型feedback补偿共用此入口；下一tick按60°/s slew将请求±10.5°，超出接口包络，在生成成功ACK/物理推进前拒绝。这里不是absolute hip/knee limit或轮hard limit触发。

新请求最终tanh尺度应为hips约±11.0908126°、knees约±16.6362189°、wheels+.2772702944rad/s，但servo只执行到±10°就停止，不能声称这些完整请求均已送达。20个有效tick的单ticksame-state native反事实差peak为servo约.03054330rad（1.75°），wheel .2772702575rad/s；同样不等于整段机械姿态变化。

在相同20tick时刻进行三段对照：

| 指标 | zero | old | new部分 |
|---|---:|---:|---:|
| body x位移 | +3.414mm | +4.045mm | +5.077mm |
| body z位移 | +1.872mm | −1.919mm | −11.779mm |
| body speed | .021185m/s | .020645m/s | .110129m/s |
| RR实测hip/knee | −.9974/.9838° | .4504/−.4384° | 4.4241/−4.2977° |
| FL实际载荷占比 | 17.77% | 16.34% | 26.16% |
| RR实际载荷占比 | 21.25% | 20.08% | 26.86% |

new末帧CoM vertical velocity −.058389m/s，base z=.079335m，四个真实支撑仍在，finite=true、body collision=false。较大的下沉/载荷变化是**已发生的物理响应**，不是物理失败标签，也不是稳定性通过证明。下一步需修复明确的residual/feedback接口契约并重跑完整有界响应；无需新历史入口门，也不把本次接口失败当训练任务失败。
