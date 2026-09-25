# CP227840 确定性正式评估：前驱保持，RR 捕获未完成

来源：`runs/ppo_rr_capture_first_cp225280_v1/eval_CP227840_DET_tracking_1e10d39/source`，源码 `1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81`。自然 P01、保存后严格重载的确定性 composite，FL assist ON、后腿 task assist/owner projection/诊断覆盖 OFF。

91.166667 s / 10,940 物理 tick / 1,368 决策后正常封存。结果为 **INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED**；首个未完成条件是 **RR 合法顶部实触并形成可用的 0.5 s 承载保持**，不是 RL 阶段。local/full success 均 false。

## 前驱保护与计数

998 条未激活请求的 local mean/applied delta 严格为零、local forward 为零、HISTORY 仅一次，实际 issued 等于 frozen prior conditional mean。与已接受 CP225280 源逐条同 tick 对齐，raw / FINAL 最大绝对差均 **0**，无缺行。FR placed 2761（23.008333 s）、FL placed 5165（43.041667 s）、RR qualified 6995（58.291667 s）、RR cross/gate 7979（66.491667 s），均与该参考相同。

此评估新增 PPO **0**。被评估 checkpoint 为 `checkpoint_CP227840_local002560_lineage448_v2_g1e10d39c9a80.pt`，累计 local 2,560 决策 / 5 PPO / 100 Adam；其中 task-v2 512 / 1 PPO，AUX 0。不得将这些累计数全部标为 tracking 修复后新训练量。

## RR 实际响应

表内顺序均为 RR [hip, knee]，raw 无量纲、目标/actual 为度、gap 为 mm。gate 在请求中途 7979 激活，因此第一行仍是 prior-only 请求；第一个 active 请求结束于 7992。

| 决策端 tick / s | prior raw mean | local raw mean delta | 实际 issued raw | FINAL ° | actual ° | gap mm |
|---|---|---|---|---|---|---|
| 7984 / 66.533333（入口端点） | [0.77738, −0.60753] | [0, 0] | [0.77784, −0.60870] | [7.48514, −58] | [7.22345, −57.76043] | 58.98812 |
| 8224 / 68.533333（最小采样 gap） | [0.78572, −0.62467] | [−0.05353, −0.12904] | [0.73139, −0.74495] | [6.82393, −58] | [6.56410, −57.76375] | 54.74732 |
| 10940 / 91.166667（终端） | [0.79585, −0.61826] | [−0.04306, −0.12494] | [0.75154, −0.74014] | [7.11553, −58] | [6.84695, −57.76624] | 58.10951 |

三行均为合法顶部 XY、AIR、0 N、hold 0。最小值是 **15 Hz 决策端点最小**，不冒充 120 Hz 几何最小。另已读取全部 10,940 行原始 capture stream（episode tick 1–10940）：RR `top_surface_contact=True` 为 **0 行**，不是只凭稀疏端点宣布没有接触。

370 条 active 请求全部 selected raw == issued raw，12 维 mask 均为 1、全 12 policy 通道未被执行器层改写，无诊断覆盖。RR knee REQUEST 范围 **[−22.90985, −19.86555]°**，370/370 条 FINAL knee = **−58°**。最小 gap 行的 REQUEST [14.97393, −22.75659]° 经 headroom 后为 [14.97393, −18.95]°。实际 hip 只比入口端点减少 0.65935°，knee 约不变；不能称为有效持续下降，也不能把裁剪说成 mask 丢失。

## Source tracking 修复的实测

元数据明确 `source_tracking_owner_revision=pending_source_tracking_inheritance_v1`。370 条 active dispatch 均 `tracking_servo_names=[]`，P09 carrier 继承 `previous_sample_before_unconsumed_late` 的空 tracking，而非从未消费 late full12 偷启 RR 跟踪。P09 late pending 始终未消费，policy 无额外许可限制、无 HISTORY reset/额外写入。

RR source 恒为 [−6.9, −37.8]°；mapped N 恒为 [−8.15, −39.05]°，RR controller bias 恒为 0。已有 mapper 补偿 [−1.25, −1.25]°仍保留，故不能称“整个控制补偿为零”。source tick 864 stop 首次记录于决策端点 8200，tracking 仍空；其他 source layer 的轮速贡献仍可能存在，8200/8224 composite nominal 四轮仍各 +0.3 rad/s，不把 stop receipt 误解成当拍四轮皆停。

FL wheel 在最小 gap 行的 FINAL / measured canonical 为 −0.69608 / −0.80628 rad/s，终端为 −0.99795 / −1.07401 rad/s。此处是实际持续反向，非仅 native 符号；本报告不将其单独判为 RR 悬空的已证因果。

结论：**前驱保护和 tracking 继承修复成立；RR 确定性捕获能力仍未取得。** 该控制修复不等于网络已学会下降。本报告只读 sealed JSON，不运行模型、修改控制或自行导出视频。完整数值见同名 JSON；较完整选点与前驱对齐见 `CP227840_DET_tracking_formal_actual_metrics.json`。
