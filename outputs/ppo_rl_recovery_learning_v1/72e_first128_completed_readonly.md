# 72e：首个完整 update 的真实 RL 窗口

已完成 update **1732**：新增 **128** 个有效 PPO 决策（226049–226176）、20 Adam steps，学习率 1e−5；actor 参数哈希从 `435d95f5…` 变为 `f2e9b1f5…`。这是 optimizer 已确认的统计，只读首128行，不读取未完成 rollout 认领学习。运行仍继续，本报告不是终态评估或视频。

来源：`train/20260924T0731401970292Z_g72e63592bdf4_8abca665b0f64439bdd233a9d189e513`。完整边界、哈希、更新记录和关键端点见同名 JSON。

## 前缀与 learner 边界

prefix 为真实连续 `successful_nominal`，767 决策、6136 ticks / 51.1333 s，全程 **0 PPO credit**，无 teleport。P10 learner 从 tick6136 接管，首个有效决策终点6144。RR placed6133 属于前缀，**不是本轮 learner 首次捕获**。

RL qualified lift 事件发生于 **6198 / 51.6500 s**，比 learner 接管晚62 ticks；在有效决策226056的6200端点仍保持资格。因此这不是教师前缀里已有的 RL qualification。它同时也不是此次 PPO update 的收益证明：首 rollout 用的是尚未执行1732优化的 CP226048 权重，新的 reward-only 修订不改变该冻结 actor 的输入／动作计算。

| learner 实测端点 | RR | RL | 解释 |
| --- | --- | --- | --- |
| 6200 / 51.6667 s | TOP，12.69 N，gap −0.879 mm；另有 FL 支撑 | qualified AIR，0 N，gap −41.14 mm，距前缘 −53.00 mm | learner 接管后短暂形成真实有效抬升，未越沿 |
| 6208 / 51.7333 s | TOP | 仍 qualified，但 obstacle ambiguous，距前缘 −50.60 mm | 唯一 qualified edge-recovery 决策窗口，不能说已越过 |
| 6216 / 51.8000 s | AIR，0 N，gap −0.483 mm | GROUND，15.14 N，资格撤销 | 既有历史 lift 不能替代当前 qualification；RR 掉载与 RL 落地同期出现，未证明谁是唯一因果 |
| 7160 / 59.6667 s，完整 update 尾 | qualified AIR，gap 34.34 mm，距前缘 +77.17 mm，0 N | GROUND_AND_OBSTACLE，资格 false，距前缘 −50.62 mm，bearing 未验证 | 回合未终止；RL 仍未越沿／落脚，RR 需保持或恢复可用接触 |

6200 时 FR 方向短窗保存的固定世界轴为 `(0.8364,−0.5481,0)`，0.5 s CoM xyz 位移 `(72.86,−28.15,8.72)` mm、body `(93.40,−24.57,28.36)` mm；此时 RL 实测0 N且 qualified。它比“仅正投影／载荷比例下降”更明确，但仍是短窗口，**不是 RL 任务成功或后续支撑持续稳定的证明**。

## 已优化样本的非互斥物理覆盖

| 窗口 | 决策端点数 |
| --- | ---: |
| RR 真实 TOP bearing＋前腿准备 | 31 |
| RR 合格 AIR 可落脚准备 | 84 |
| CoM/body 沿固定 FR 轴正投影＋RL 比例下降（代理指标） | 11 |
| RL 当前 qualified | 2 |
| 其中 qualified edge recovery | 1 |
| RL 合格 AIR 捕获区／TOP bearing／placed | 0 / 0 / 0 |
| P13／完整任务完成 | 0 / 0 |

RR 存在多段真实重新 TOP 接触；但本128端点内尚无 RR ground。新 retention 修订在8个当前 AIR、XY区外端点产生非零局部 `delta_global_potential`，范围0.003670–0.004895，仍在既有份额内。其余合法 AIR／TOP 原值保持；6198资格附近 delta=0。这里是**实际奖励覆盖**，不是“新奖励已教会 RL”的因果证明。

原始动作12通道许可保持全1，后腿任务补全器关闭。后续真实难点是：保持 RR/FL 可用支撑同时保住 RL 的短暂卸载，继续越沿与放置。照既定课程继续采样即可；不因本窗口增加新门禁，也不把此成功前缀初始化后缀当成自然 P01 全程能力。
