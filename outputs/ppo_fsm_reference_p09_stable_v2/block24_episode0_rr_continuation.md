# Block24 episode0：初始净空未建立 RR Q，P09 机身碰撞

范围：`train/20260911T0005451949265Z_gd4e46006b382_08d59fe9ad444f60b1098f90dbe1fca0`。2026-09-11 00:22:46.047–00:22:46.123 UTC 单次解析 audit 前180行、逻辑字节 `[0,14128580)`，另读 completed 首行；未解析后续活动样本，未读大 physical 流或 checkpoint。

## 信用与真实终态

教师从 P01 运行至 P06/t3584（29.866667s）；PPO 信用为 global161665–161844，首端点3592。`prefix_teacher_data_in_ppo_storage=false`，不是“没有教师”。180条来源阶段：P06=147、P07=1、P08=1、P09=31；179×8+末3=1435新物理ticks，全部 native verified/actual effect，own-phase effect=1432；180条均无 episode 内状态写入。

唯一终态为 t5019/41.825s、P09 `BODY_COLLISION`，物理评价 `TASK_FAILURE_BODY_COLLISION / BODY_CONTACT`、`VALID / VERIFIED`，reason 为 central body/obstacle collision；没有完整成功，terminal bootstrap=false。可用证据确认机身碰障，不提供具体 collider/pair、接触力或 signed BODY distance，不能由姿态补造。主控先前“128已更新、52待满rollout”只是当时保存快照；本报告不把后52认作当前仍未优化，也不把回合结束写成训练退出。

## RR 事件：I≠Q，P09 入口≠已经抬起

- 首3592虽 AIR，I/Q/current/C/P 均false，地面相对净空仅0.796632mm。
- 新 I 仅一次：**4379，P06**；事件记录 upward excursion=3.217089mm、全身关节响应12.638789°。端点4384仍 AIR/initial=true，recent clearance gain=3.564626mm、当前地面相对净空2.179124mm、front=−376.182672mm、台面净空=−47.917969mm。此时 RR 自身近期关节响应8.839026°、全身23.305047°；这是实测全身运动下的小净空，不是已受控越障。
- 4392已 GROUND/initial=false。全180条历史与事件列表中 **RR Q=0、资格撤销=0、C=0、P=0，current有效端点=0**。没有Q可撤销；不能把这次 I 后回地计为Q撤销。末5019 AIR/0.929624mm也未形成新I/Q。
- 接触端点为 AIR=9、GROUND=171，无 RR 边缘/TOP 事件。最靠前端点4784仍GROUND：front=−215.133574mm、台面净空=−49.999242mm；其后退至末端−222.062346mm。最大端点地面相对净空/台面净空出现在4384；这些是决策端点极值，不冒充逐物理tick极值。

因此首个尚未达成的后腿子任务是 **RR 功能性抬升资格**，之后越沿和受控放置均未完成。RL 的新Q4342及4374回地撤销是另一条腿，不能误计为RR Q。

## 连续交接与真实承载

| 实际边界tick | 转移 | 当时 RR / FL | 后续交接审计 |
|---|---|---|---|
|4760|P06→P07|RR GROUND/3.729638N；FL AIR/0N|4768记录，Full12 residual原值继承|
|4768|P07→P08|RR GROUND/2.422692N；FL AIR/0N|4776记录，Full12 residual原值继承|
|4776|P08→P09|RR GROUND/2.558237N；FL AIR/0N|4784记录，Full12 residual原值继承|

三次普通切换均 done=false、entry_valid=true；禁止通道和phase-scale裁剪列表为空，残差数值误差至多3.6e−15°。P06 completion 为 rear proximity/role prepared，P07 为 transfer ready；P08→P09只进入待完成的 `placed_RR=0`，没有虚构Q。边界前4752至首P09端点4784 nominal四轮始终+0.3rad/s；真实最终wheel仍非零，如4760 `[−.764672,−.086133,−.208676,+.848475]`，4784 `[−.703113,−.215747,+.086898,+.851602]`。前腿/后腿 nominal 随新source owner产生新请求，不等于残差或mapper重置。

180条Full12 mask均开放，native映射与setter dispatch校验全部通过。RR nominal hip/knee从P06 `[0,0]` 到4784 `[1.6,0]`，末5019 `[−6.9,−37.8]`；最终target分别为4384 `[15.230850,−10.773727]`、4784 `[−3.356939,−22.101283]`、5019 `[7.329958,−56.411042]`°。4952存在一次RR knee headroom裁剪（索引7），不是通道关闭。末实测joint-range margin反算RR `[9.098433,−55.475714]`°，与target不同；命令已发送不等于轮端取得净空，角度方向也不足以单独证明轮端下降原因。上述动作曲线限记录端点，不杜撰未保存的逐tick nominal曲线。

FL Q2417/C3115/P3583均来自教师历史。策略期间FL 104端点AIR、76端点TOP且verified-bearing；首3592AIR，首TOP3608，最后TOP4736。从三次交接直到末端FL都AIR/0N，不能把历史P当当前承载；当时独立实测支撑为FR TOP与RL GROUND（4776分别13.430514N、17.074566N）。终态FR也AIR/0N，仅RL地面11.590720N；这是瞬时接触证据，不是稳定性或单因证明。

结论：未发现新的可确认 reset、mask、owner丢请求或dispatch缺陷。已发送连续全身动作仍未建立RR Q，随后真实机身碰撞；保留为本次策略任务失败，不建立新的optimizer门槛或修改安全规则。
