# 四腿连续转移 v1：生产替换与证据边界

本轮从实际最新123136 decisions / 927 PPO updates / 18540 optimizer steps恢复；源checkpoint没有完整P01成功证明。原A、历史数据、robot USD、质量、摩擦、障碍和执行器物理参数不改。B/C共享本轮新nominal和评价规则，调度修复不是PPO学习收益。

## 确认和替换

旧workspace只检查前缘x区间/横向ROI，旧support只数另外两接触，旧load_ready仅低载；它们不能证明对角转移。生产阶段完成列表已移除这三类旧准备门，前缘量明确命名edge_proximity，独立于receiver workspace代理。

新四腿映射为FR→RL、FL→RR、RR→FL、RL→FR。preferred bridge是建议，当前support取真实GROUND或TOP接触及原力噪声阈值，不用历史placed创造载荷，不把墙接触当顶面承载。FR新支撑参与FL准备；RR新支撑参与RL准备；接收侧AIR与支撑退化分别记录，不要求提前接触。

新增TransferRoleTracker使用原全机器人质量加权CoM、速度、实测轮端/机身几何、关节硬限余量及独立0.5s身体/指令/载荷窗口。方向以每个短窗口起点的世界坐标定义；CoM位移和receiver轮端位移分开。workspace明确是关节/几何代理，不是精确FK可行域或Newton–Euler安全证明；没有完整接触wrench/惯量，不虚构水平冲量或角动量。

准备允许接收空间变化、向接收方向身体运动、实测载荷再分配或全身初始离地等不同路线。转移不能仅靠瞬时低载：另需主动真实响应及连续短证据。支撑连续性只检查配置0.0667s后缀，旧0.5s中的掉点不强制额外等待；不要求静止或指定两腿组合。两点支撑既不被三腿门禁止，也不伪造静态稳定结论。合格下游运动已经发生时，准备阶段整体连续接管，原几何数值如实保留，不伪报edge满足。

原全身lift/初始AIR→合格Q→越沿C→真实两采样TOP放置P没有改低标准。RR自己的关节不必先大动；GROUND发生在越沿前会撤销当次资格。机身碰障碍和无主动抬升的wheel-only越沿仍失败；跌倒/硬限/NaN/爆炸独立中止。

## P05/P06时序缺口

已录制合同P05末FL AIR/0N，P06连续轮行后TOP/5.510N。旧监督器先要求placed_FL才开放P06建议，确有动作依赖缺口，但这不是所有历史P05失败的唯一因果证明。

P05内新增pending_capture：已有合格FL抬升、未placed、非GROUND、合法当前近沿几何及真实其它支撑时，提供与P06同类+.3rad/s轮行建议，原连杆层继续；第一个TOP样本不撤回建议，第二个真实TOP才可placed。不跳P06、不伪造AIR接触、不重置wheel/residual/mapper/history/GAE。

## A此次未通过（保留原结果，不重跑）

既有A运行20260905T0902190128552Z_g00b94fbb276a_e0ddaa746bc84bddb8e4f80506e09fac：65.3667s / tick7844 / P10 WAIT_ENTRY，任务未完成、传感/执行有效、无任务硬失败。RR knee −45.8583与冻结入口−50.3976差4.5393°，速度2.6107与23.5853差约−20.9746°/s，不满足旧入口；FR/FL/RR已有placement，RL未完成。任务未完成已由传感和控制事件独立确认，不是仅录像验收失败；989帧15fps H264已有完整解码证据，但容器问题另列。历史Trial043不能把这次结果改成功；180 settle加64额外tick不能无证据宣称唯一原因。本轮不重跑A/FSM/Recording。

## 网络与迁移

补充录像文件边界：旧A的完整解码/连续PTS虽通过，MP4容器duration字段为无效大值，并未通过本轮严格发布验证；这与P10任务未完成分别记录在video_manifest.json。本轮不为训练启动去重跑或修复这段旧录像。

324维/12动作、共享256×256 actor/critic、RSL、120/15Hz、history rho=.9、状态相关std、gamma=.9985/lambda=.99均保留。只调整两列FR knee残差历史尺度4→6，首层输入列210/222×1.5补偿，保持旧未clip域相同物理输入下网络函数（浮点容差），不宣称新动作域/任务标量/轨迹等价。其它参数/identity normalizer/RNG及终生计数保留；新MDP重建Adam lr3e−5并丢弃旧未完成rollout。

角色由阶段/历史可辨识，当前准备判定通过既有phase_progress、全局Phi表达；详细窗口锚点和workspace分解没有逐项追加，不能声称单帧324完全重建内部短时状态或完整Markov。

只扩有物理依据的请求范围：早期轮±.6可抵消并反向+.3建议；后腿起FR knee±112在mapper/controller各+10的保守基线65.9下可到−40，仍有原硬限/2°reserve/60°s residual及最终slew。−40不是目标角门或奖励。

## 固定版本训练检查块

先1024自然P01，再1024当前checkpoint策略roll-in到P04的FL准备，另1024 P06前驱和1024 P10对应RL前驱（必要最短真实FSM训练roll-in单独记账，失败回自然P01）；总4096新采样后重新加载自然P01确定性评估。所有正式失败样本保留，prefix不计PPO量或完整成功。实际运行/恢复命令和偏差写training_manifest，不用建议数代替实际量。

尚未成功不生成success/improved名称的视频或checkpoint。旧A若条件不一致只作非配对参考，任务完成与稳定性分开报告。
