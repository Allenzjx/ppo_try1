# Block10：RR 放置后保持丢失的已封闭窗口

对象：`20260921T1043344171428Z_g97ecd305afb5_0168bb1943e74402accb92ea31b10de3` 的首回合，随机训练中且包含更新，不是固定 checkpoint 评估。只解析残差日志到 tick 6600，主体为 5200–6600 的 176 个完整 8-tick 端点；前缀摘要而非活动文件全量 hash 见同名 JSON。另只读 `completed_episodes.jsonl` 第一行补充真实终态，未扫描第二回合。

## 第一处丢失与实际运动

RR 曾真实 qualified=4759、cross=5322、placed=5542。但放置后的首个已存端点 5544 已是 AIR/0 N；因此首次失去当前 TOP 保持发生在 **(5542,5544]**，不能把 15 Hz 端点当作精确 120 Hz 失接触时刻。RR 中心仍短暂前送到 5560（前沿距离 +49.313 mm），之后回撤；5624 尚在 top XY，5632 首次已存为外侧（−11.347 mm）。之后出现间歇 TOP 与 AIR，不能把一帧 AIR 就判失败，也不能把边缘 TOP 接触与中心 within_top_xy 混为一谈。

6212 有直接 `current_lift_revoked_ground` 事件；6216 首存 GROUND，当前资格 false，但已完成 cross/placed 的历史仍保留。6600 是 GROUND，前沿距离 −51.530 mm、相对障碍顶 gap −51.664 mm。历史 placed 不是当前顶部承载。

| 已存窗口 | 真实 body 前向位移 | RR 中心位移 | RR 当前 TOP 承载端点 | FL 当前 TOP 承载端点 |
|---|---:|---:|---:|---:|
| 5200→5544 | +46.380 mm | +79.749 mm | 0/44* | 30/44 |
| 5544→6600 | −77.898 mm | −98.402 mm | 39/133 | 24/133 |

*真实 capture=5542 落在采样端点之间，0/44 不否认该物理事件。占比是端点数量，不是完整接触时长。后窗 FL AIR 为 109/133；不能虚构 FL 持续承载，更不能用 FL 轮端移动替代机身位移。

## 四轮指令与阶段交接

四轮顺序 FL/FR/RL/RR，单位 canonical rad/s。下表全部来自同一次真实派发；N=mapped wheel baseline。完整 raw policy、REQUEST、effective、final、实测及当前接触已保留在 JSON。

| tick | N / mapped 四轮 | effective residual 四轮 | final 四轮 | 实测 canonical 四轮 |
|---|---|---|---|---|
| 5200 | .300,.300,.300,.300 | −.090,−.036,.024,−.143 | .210,.264,.324,.157 | .197,.312,.397,.157 |
| 5384 | 0,0,0,0 | −.246,.020,.150,−.147 | 同 residual | .061,−.082,.316,−.145 |
| 5544 | −1.070,0,0,0 | .540,−.204,−.125,−.185 | −.530,−.204,−.125,−.185 | −.530,−.217,−.061,−.224 |
| 5608 | 0,0,0,0 | .670,−.163,.017,−.220 | 同 residual | .670,−.297,.098,−.221 |
| 6216 | 0,0,0,0 | −.670,−.070,.008,−.096 | 同 residual | −.670,−.232,.141,+.137 |

N 四轮 .3 在 5384 变零、5392 变为 FL −1.07/其余 0；此前的同窗源诊断将 5384 绑定到 P09 late-group 开始、`finite_ordered_source_wheel_owner`。这些变化早于 P09→P10=5560 和 P10→P11=5600；跨这两个阶段端点 N 保持 FL −1.07/其余 0，5608 才观察到 N 全零。源码 `semantic_supervisor.py:2247–2271` 明确保留有限源 wheel owner 和 authored stops，不以额外 rolling suggestion 覆盖。**这里存在源时序的零/反向指令，并非阶段切换把四轮无条件清零；完整原子 stop event 的逐通道最后 owner ledger 未在本提取保存，故为 null，不能补造。**

这段 12 维残差许可全 1，所有 endpoint 与 1400 个区间物理步派发验证通过；有 2 次 handoff hold，不是把 residual 归零；独立 final-stop owner 0 次取得。wheel final − 同拍 mapped − effective 最大误差为 0。原始 native 实测 qd 未单列，保留 null；上表是已记录的 canonical 实测，不拿 native target 冒充读回。

后窗 RR final 132/133 端点为负，mean −.1120 rad/s；actual canonical 102/133 为负，mean −.0911 rad/s。N 为 0 时这是可见的策略负向修正，而不是“把仍为正的 N 抵消至零”或 mask 丢失。6216 的负 target 对应正 actual，说明接触负载/耦合下跟踪并非逐帧相等。轮速反向与退回同时可见，但本观察没有牵引力/反事实实验，**不能把全部机身后退唯一归因 RR 电机**。

## 已结束回合的终态附录

独立读取已完成首回合记录：1039 decisions，8306 tick / 69.216667 s，P11 `INCOMPLETE_CONTROLLER_BLOCKED`；physical evaluator valid=true、termination=null、success=false，不是 BODY_COLLISION，也不是预算尾。time_outs=false，terminal bootstrap=false，terminal event −40，末次总 reward −43.157002。

RL 无 qualified/cross/placed 事件；因此第一未完成的顺序任务是 RL 接续抬升/越沿，而已有 RR 放置也没有持续保持到终点。末 RR GROUND、current_lift_valid=false、前沿距离 −52.996 mm、gap −49.997 mm；RL GROUND、前沿距离 −134.776 mm；FL AIR、gap +85.591 mm/0 N。这个 first-unmet 是对真实事件表的推导，不是日志中不存在的独立字段。

结论仅用于后续课程判断：已有真实 RR 抬升、越沿与短暂放置样本；需要关注放置后到 RL 接续的当前接触与机身进展，而不能以历史事件当持续承载。本次未改 reward、控制、mask 或训练门槛，未启动额外物理仿真。
