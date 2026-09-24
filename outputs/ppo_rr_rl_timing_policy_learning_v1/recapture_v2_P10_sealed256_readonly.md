# recapture v2：封存 P10 256 决策审计

范围仅 `20260923T2112002124392Z_g44219b4fdc4d_719169cdf58947d3bd595c9e8efd9674` 的 **221057–221312**。HEAD `44219b4fdc4d36d33be489b833c03b897766045b`。SUCCEEDED 仅表示训练预算封存，不是越障成功。

- 已完成 **256 decisions / 2 PPO updates（1693–1694）/ 40 Adam steps**；P10=1、P11=1、P12=254，学习率 1e-5，两次均非零梯度且 actor 改变。CP221312 SHA（父代理封存记录）`d3760840db9b4e4291055e76822285c8125f45c739a72e34ce1317cc3b4cd6df`。
- N 实际前缀到 P10：51.133333 s/t6136，0 policy credit。RR 已放置来自前缀，不能计本块网络首次捕获。256 末态均 nonterminal，RL 未越沿/placed；不是自然 P01 确定性成功。
- 256/256 rear assists OFF、RR14零、全12 mask开放、派发核验通过；未运行模型、Torch、Isaac 或 FFmpeg。

## 首次掉载：v2 recapture / sigma / retention 已接通

| decision / s | RR / force N / gap mm | RR retention | 输入任务 / σ(FRk,FLk,RRh) | P12 输入 main/RL ticks |
|---|---|---:|---|---|
|221070 / 52.0667|TOP / 5.383 / 0.0066|0.999869|prep / 2,2,1|89 / 88|
|221071 / 52.1333|AIR / 0 / −0.0047|0.499907|prep / 2,2,1|97 / 96|
|221072 / 52.2000|AIR / 0 / 2.1924|0.459687|recapture / 1,1,4|105 / 100|
|221073 / 52.2667|AIR / 0 / 7.3589|0.386292|recapture / 1,1,4|113 / 100|
|221123 / 55.6000|TOP / 4.898 / −0.5704|0.988846|prep / 2,2,1|513 / 148|
|221312 / 68.2000|AIR / 0 / 49.3012|0.168234|recapture / 1,1,4|2025 / 249|

输入任务是该决策开始前状态，contact/gap 是末态，故首掉载后下一决策切换正确；255 对 next-input 均匹配当前 RR bearing。RR TOP/当前 bearing 34，AIR 222，历史 placed 256；输入 recapture 221、prep 35。potential 逐行复算最大误差 1.11e−16；不是只改了标签。

P12 RL lane 等待末态 225（除 RR 失载还包含 bridge 不许可），217 个决策的 RL cursor 完全未推进。混合接触的一个 8-tick 决策中 cursor 可推进 1–7 tick，再在掉载后暂停；不能用末态倒推整拍许可。末段 RL cursor 保持249，主 wheel/source clock 继续到2033。没有 catch-up 或全身 freeze 的证据。

## 确认的 RL 当前资格接口缺口（需下一冻结版本修）

RL 实际5个 AIR末态；其中221062–64、221066共4个有本次 active_attempt 与 qualified lift。t6183/51.525 获本次 qualified_measured_upward_lift；221065 是 OBSTACLE_AMBIGUOUS，非 AIR，不能算 swing；t6217/51.8083 返回ground撤销 active_lift/attempt。旧 first-event tick6183保留，之后不能据此恢复资格。短AIR未到顶部、未越沿，也不是可达性证明。

生产 `semantic_supervisor.py:875–895` 只给 RR 合入 current_lift_valid / motion_continuation_allowed，RL 256 行都无此字段；`semantic_rear_policy_timing.py:40–42` 却要求这两个 RL 字段，因此 rl_current_swing 和输入 swing 位全256为false。早期真实RL AIR时 RR仍承载，故**不能归因为本次末端停滞的唯一原因**。需 evaluator 本次RL资格与ground无条件撤销（包括crossed历史保留），再复用现有RL观测位；不能只补历史fallback。

## FR / FL 实际动作和 CoM

FR knee nominal始终+31.1°，final范围−39.874…+29.221°、actual−37.260…+29.577°；最小负硬限余量22.740°，不是“仍固定高直”的事实。
FL nominal−31.4°，final−58…−39.15°，209/256精确裁到−58；actual最低−58.590°、负硬限余量最低1.410°。**未看到有效正向行程恢复**，这是后续实质阻碍候选，不能把phase推进称为支撑架构完成。
末端 FR final/actual=−19.717/−19.057°；FL=−58/−58.448°；RR再次悬空49.30 mm，RL仍ground。

角色接收方向256/256为FR。日志 mass-weighted CoM 的0.5s短窗方向每窗更新，不能当整次转移固定参考。固定首learner端51.2s日志d=(0.842499,−0.538698,0)，COM=(0.719667,−0.129259,0.164700)，重算末端投影 **+75.109 mm**（全过程0…95.573 mm）；这是有向位移，不等于FR已承载、稳定或RL完成。该参考不是实际转移最初时刻，前缀不补计。

下一修复必须保留新学 checkpoint；本审计未修改生产或读取下一活动P01。

