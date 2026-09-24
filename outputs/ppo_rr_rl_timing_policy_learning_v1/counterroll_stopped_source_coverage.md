# 停止源指令后的反向轮速：只读反事实覆盖率

范围：仅已封存 `20260924T0130272021180Z_g49eb23163a6e_5e6b94b24cce4a2fb9edc117565be855`，384 个已优化决策、3072 个实际奖励物理拍；不读取当前 CP225280 录像，不修改奖励/模型/控制。此前零信用前缀中的 RR 放置不归为 learner 成功。此块是预算结束的非终态 partial，不是完整越障。

## 实际缺口与计数

冻结 `fl_counterroll_sample` 必须 `source_FL > 0`，且进展分母是 `source_FL × wheel_radius`。因此仅把判断改成 `>=0` **不够**：N=0 时分母退化到 `1e-12`，任意极小正位移都会清掉惩罚。现有 3072 拍 audit 的 eligible、positive cost、总 cost **全部为 0**；这是真实旧奖励，不是重算结果。

|已封存端点筛选（15 Hz；不是 120 Hz 时长）|数量|
|---|---:|
|全部端点；N_FL 为负 / 为零 / 为正|384；42 / 342 / 0|
|N=0 且原 policy request、最终下发、实测 FL 都反向|285|
|保留其它现有物理门，仅允许 N≥0 后符合候选条件|62，全部 N=0|
|上述 62 中前一端点仍是合法 source reverse pulse，保守排除跨脉冲区间|1|
|余下同为非负 source 的相邻 8-tick 区间|61|
|前送或合法下降 ≥ 既有 `0.005 m/s`，应完全豁免|37|
|有较小合法下降，应部分减罚；保守版本可完全豁免|1|
|无净前送、无合法净下降，严格 nonprogress 候选|23|

候选仍要求：后腿准备相关；RR 当前合格 AIR、当前未承载；FL 非 AIR、有真实 ground/TOP 支撑、bearing verified、实测力≥原 `0.2 N`；RL 尚无当前合格持续摆腿；传感有效。独立、可重叠的失败门：FL 实际支撑不足245、RR非当前合格AIR288、RL有效摆腿9、准备不相关233。因此不能把 285 个反向端点全部称为应罚动作。61 个区间中60个两端都在合法 TOP XY；另外一个只能用真实前送证据，不能给合法下降信用。

## 最小候选含义（未实施、未选定新奖励版本）

保留上述门，显式 source 负向脉冲始终豁免；source=0 不改变 nominal stop，也不生成轮速目标。仅当 **最终命令与实测 FL 都反向** 时，定义

`reverse = min(max(−final_FL,0), max(−actual_FL,0))`。

反向量沿用实际 residual cap `1.2000000477 rad/s` 归一化；进展可用已配置的 transfer-role `velocity_scale_m_s=0.005` 归一化，**不依赖 source 速度**：

`raw_cost = clip(reverse/cap,0,1) × [1 − clip(max(forward,legal_descent)/0.005,0,1)]`。

这里只是借用既有量纲尺度，不声称该尺度已验证适合作为新 counterroll 超参数。`forward` 用 RR wheel center x 的实测差分；日志的 `front_distance=center_x−fixed_front` 与其差分等价。`legal_descent` 用两端都合法 TOP XY/lateral、非 ground 时 `max(gap,0)` 的下降，不能用 body z、轮心 z、向墙下降或更深负 gap 冒充。若严格保护**任何**实测正进展，则只保留表中23个零净进展候选，不将那1个小下降算负例；噪声/逐拍有效性仍需真实相邻测量验证。

保持现有系数仅作量级比较：单拍代价有界于 `.01×dt`，没有“停止就加分”；原 `.02/s` 时间成本、任务潜势、终止判据不撤销，也不延时或重复触顶给奖励。这避免直接奖励 loiter，**但不能保证新最优策略不会停滞**。本方案不是全阶段轮速符号正则、不是强制前送，也不解决旧 nominal 已下发关节目标继续追赶的问题。

## 实际正反例（原始 canonical rad/s、tick）

|tick|事实|候选解释|
|---:|---|---|
|6144|N=−1.07，final=−.950，actual=−.973|保护 authored reverse，不能罚成“错误反转”|
|6192|RL 当前有效摆腿=true|保护正在进行的 RL swing，不重做准备|
|6480→6488|N均0；final=−.96481、actual=−.93074；前送 `.07359 m/s`、合法下降 `.12936 m/s`|即使反转，真实进展足够，cost=0|
|6496→6504|N=0，final=−1.01983、actual=−.98370；FL实载1.2488N；RR gap44.218mm；两项净进展均0|严格 nonprogress 候选，dimensionless proxy=.81975|
|6632|FL AIR、force=0，final=−.87495|无地面牵引证据，豁免|
|6888→6896|合法下降 `.0007868 m/s`，gap72.215mm|不能叫“完全无进展”；连续方案部分减罚，保守方案全豁免|

前段 P01–P06 没有本块真实样本：保护来自明确 phase 排除，不能冒称已在本块物理验证。RR 后来落地/失去当前资格后也不属于本候选；hist placed 不能替代 current AIR 或真实承载。

## 覆盖限制与结论

旧逐拍 audit 提前返回时未存前后 RR 几何/FL 接触中间值，只有 eligible=false、cost=0；因此上面的61区间是**端点净进展代理**，不是61个完整物理区间持续符合门，更不能乘8冒充488个新奖励样本。未重算实际 PPO reward/GAE、未运行优化、未声称 reverse 是失载唯一原因。证据支持“source=0 的局部奖励盲区值得小范围检查”，同时证明不少反转伴随真实合法进展，不能全罚。

复现：Base Python 执行同目录 `counterroll_stopped_source_readonly.py`（stdlib；只读同一封存 journal/config，输出紧凑 JSON）。本次执行已退出；没有 Torch/PXR/Isaac 或生产写入。
