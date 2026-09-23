# v6 contact-onset：有界独立离线核验

0 新物理步、0 PPO 更新、0 actor forward。只新增 outputs 报告/回放脚本，未改生产或测试。被测 assist SHA256 `3075ba48e611d08a9c8429d96bb3ca5c2c8a3eae9703d4f0def95ccfd15bda14`，revision=`progress_reserve_contact_onset_incremental_v5`（整合版本简称 v6）。完整输入与结果见 `v6_contact_onset_actual_replay.json`。

输入严格来自已封存 c53119a v5：末两拍 native audit、终态真实传感器和前述封存分析。旧 v5 snapshot 被新 validator 明确拒绝；离线升级只替换三项 revision/semantics metadata，14 个数值字段均不改。previous FINAL8+4 使用本拍独立 pre-dispatch 审计值，并与前一拍实际输出逐值核对；新一步 dispatch tick 邻接14774→14775，传感器来自真实14595末态，未制造接触。

| 路径 | 实际输入/明确假设 | 离线结果 |
|---|---|---|
| 已封存 v5 终态直接换 metadata | 原 mode3/BLOCKED，travel≈52，peak0.138727mm、gap0.026958mm | 仍 BLOCKED/fresh_near_top_progress_required，knee变化0。只有0.111769mm新下降，不能凭临近或metadata换版复活信用 |
| v6 回放原最后一拍的真实 prestate/context | 原 state_before 是 mode6，travel51.991667，有公开新鲜信用 | 到52°仍为 DESCEND_PROGRESS；目标/行程/暴露与实际旧末拍相同，只有预算状态解释不同 |
| 接上真实终态传感器的邻接下一步 | 保留上行 v6 mode6，实际Q/cross/XY、三个支撑和AIR无力 | knee +0.008333333°，travel52.008333°，contact_seen仍0；没有把27µm当接触 |
| 独立反例：physical invalid / support<2 / XY丢失 / 跟踪失效 | 只改对应 context 的纯反例，不称物理样本 | 均 BLOCKED、额外行程0 |
| 独立反例：gap反弹到1.1mm | 同一已有mode6状态，超近面包络 | BLOCKED/finite_search_travel_or_margin，额外行程0 |

另以重复固定末态传感器的纯控制器边界循环（不是未来物理预测）验证额外行程封顶 0.9999999999999432°、总53°后 BLOCKED，未产生任何接触或重新充值。实际新回合从P01运行与“从旧已BLOCKED终态续接”是两条不同路径，不得混称。

CPU1 定向测试：`python -m pytest tests/unit/test_semantic_rr_contact_onset.py tests/unit/test_semantic_rr_capture_progress_incremental.py -q`，**39项通过**。现有显式合成fixture验证真实TOP输入立即停止下降、GROUND/壁面不冒充TOP、接触丢失/窗口reset不充值、远处52°预算保持、近面无下降不能赚取信用及先前captured增量接续。实际sealed回放从头到尾保留AIR/0力；未向实际记录插入人工TOP。

未发现本次近面预算接线阻断。重要边界：这是一次最多1°/1s的额外动作许可，不是触地保证；旧终态仍拒绝属于正确的信用保留规则，不应手工把mode3改回6。源码的peak进展设计允许1mm以内短时反弹消耗既有最多2s局部信用，并非每个上升tick都立刻停止；超过1mm时本次额外额度不允许。实际接触与placed标准未变。所有分析/测试Python已退出。
