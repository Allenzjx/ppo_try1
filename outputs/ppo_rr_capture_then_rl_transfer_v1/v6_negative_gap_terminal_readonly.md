# v6 首个负净空拍被局部 deadline 截断：独立末窗核验

源 `20260923T0829436381354Z_g97c4367ee293_5795b2a4e4384ec6be3589fa43477a5c/source` 已封存，HEAD `97c4367ee293cb4dd191944ef663bbda63a5b235`，实际CP SHA `0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01`。只读三条流最后241拍（14360–14600，2s）及末2个decision，约24MB尾部读取，没有全历史重扫。0新physics/actor/PPO，未编辑生产。证据及各尾部SHA见同名JSON。

## 第一处实际失败，不是额度耗尽

| post-physics tick | RR gap µm | mode | 累计行程 ° | recovery / local_warning |
|---|---:|---|---:|---|
| 14595 | +26.957549 | DESCEND_PROGRESS | 52.000000 | true / true |
| 14596 | +20.992126 | DESCEND_PROGRESS | 52.008333 | true / true |
| 14597 | +14.843011 | DESCEND_PROGRESS | 52.016667 | true / true |
| 14598 | +8.689687 | DESCEND_PROGRESS | 52.025000 | true / true |
| 14599 | +2.610613 | DESCEND_PROGRESS | 52.033333 | true / true |
| **14600** | **−3.537004** | **DESCEND_PROGRESS** | **52.041667** | **false / false** |

最后一拍 pre-dispatch context 来自14599，native dispatch tick14779；它仍Q/cross/XY/physical-valid且三个其他支撑。派发成功后14600真实sensor净空变负，fresh已提交state仍mode6，但当前`rr_top_reachable=false`；正是 context 的 AIR `gap>=0` 条件失去。此时stage age52.533333s已超过原30+进展延长7.225=37.225s，supervisor同拍设置 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。本拍没有执行下一次assist，不能报告“assist已经因负gap BLOCKED”。它下一拍潜在的signed-gap信用冲突仍须与context一起修正。

排除同期预算/跟踪/支撑原因：travel52.041667<53；exposure42.041667<45s；public peak=+0.138727mm，local window=.208333<2s，hold=0，retired=0、contact_seen=0。RR actual hip/knee [-2.122588,-13.254651]°，FINAL [-1.785913,-13.319510]°，target−actual [+0.336675,−0.064859]°，均远小于原3°tracking门。current-Q、lift-established、within_top_xy/lateral均true；front20.146016mm，ground-relative lift仍有效。

当前其他 verified support：FL TOP3.178248N、FR TOP13.586666N、RL GROUND11.973246N，全部support=true/air=false。RR仍非承载。

## 原始接触证据：没有TOP，不放宽验收

241末拍均RR AIR、两pair active=false；所有原始 force vectors 均[0,0,0]，最大norm0N。终态ground mean point=null；obstacle mean point为[.540282786,−.316741526,.051250271]m，但pair_verified=true/active=false、当前力和三拍力历史全零，不能因该坐标存在就声称触地。历史RRcross=true、placed=false，当前TOP=false/bearing0N。负3.537µm只是collider几何与顶面的有符号差，不是支撑或传感器故障证据。

## 最小有限 formation 语义及定向反例建议（尚未实施）

应统一 context恢复许可、assist进展信用与search-limit窗口峰值对**有符号顶面附近状态**的解释；只删`air_candidate gap>=0`不足以避免下一拍撤销mode6。保持已有公开信用、余量、2s进展窗及53°/45s绝对上限，不充值、不添加固定1s等待。配置顶面带[-15,+25]mm是必要几何边界，不是允许无条件往下穿15mm；额外contact-onset仍保留已有近面1mm上界及真实下降/追踪约束。

| 用例 | 应有行为 |
|---|---|
| 实际14599 +2.611µm、14600 −3.537µm，Q/cross/XY/3支持、mode6新鲜且有剩余额度 | 只允许有限形成接触/继续既有有界动作；仍AIR、support=false、TOP=false、placed=false |
| 同态零净空且力0 | 不授予接触/承载/放置，不刷新预算或窗口 |
| 带外gap、GROUND、壁面、XY/lateral丢失、invalid/stale sensor、其他支持不足、tracking失效、额度耗尽 | 撤销形成许可；硬安全和global200优先 |
| 已BLOCKED无新鲜下降信用，只因接近或重置窗口变为近面 | 不重新赚取mode6/行程 |
| signed peak也变负但仍在带内且真实进展有效 | 不能因标量符号切换暗中从53降为52或冒充新额度；需显式统一版本语义 |
| 明确合成小力：norm .24N，首次pair未过.25N | 原classifier保持inactive/AIR，有限formation不等于TOP；不可降低classifier门槛 |
| 明确合成raw active但upward<.2N或力向/点仍不满足TOP | 停止继续下探；绝不冒充TOP/bearing。若允许顶面相容的有限确认，只能用既有累计HOLD/确认边界，排除壁面/无效point，不能无限等待 |
| 真TOP/向上bearing满足原规则 | 当拍物理事件照旧；下一助控停止下降，沿原minimum_top_samples=2及正常15Hz交接确认，不先改history |

当前微力/AMBIGUOUS分支并未在这个尾窗实际发生（全部原始力0），只列为新修订应覆盖的合成正反例。真正致命分歧已限定为14600的恢复许可丢失；不是actor更新、source stop、native派发失败，也不是放置标准应降低的证据。全部CPU helper已退出。
