# P12+8 第二回合：有局部前送/卸载，仍未越沿

限定来源：同一72e run `20260924T0835078527430Z_g72e63592bdf4_eb09a77b75b0420997a80291feb869d8` 的第二条 sealed episode，以及 global **227013–227465** 的453条 learner记录。只跳过前452行而不重新解码它们；未读取第三段前缀。

**结果仍为 INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED**，tick9835 /81.9583s。P12 elapsed30.6917s，额度30.0+0.68422s；物理安全终止为空，不是硬限。读取时更新1742/global227456已完成，本回合 **444已优化、9尾段未优化**。

## 与第一回合的区别

| 实际指标 | 第一回合 | 第二回合 |
| --- | ---: | ---: |
| learner采样数 | 452 | 453 |
| learner新RL资格次数（排除prefix6212） | 2 | 3 |
| 新资格qualified端点 | 23 | 20 |
| RL越沿 / TOP放置 | 0 / 0 | 0 / 0 |
| 首个RR sensor TOP丢失端点 | 6448 /53.733s | 6248 /52.067s |
| 首个RR GROUND端点 | 7432 /61.933s | 7064 /58.867s |
| RR-ground采样端点 | 295 | 346 |
| 全段body forward变化 | −139.35mm | −151.63mm |
| 固定首端点FR轴CoM净投影 | −105.02mm | −107.14mm |

这不是同权重同随机序列配对，不能将差异当改进/退化的因果证明；两个回合中权重均在正常update边界更新。

## RL：新增资格存在，但未形成越沿

两回合handoff都在6216/51.8s，初次RL资格6212和RR placed6133都由真实nominal前缀形成，零credit。

第二回合：继承6212资格6端点后6272见GROUND；learner新资格 **6323（3端点）、6747（10）、6880（7）**，分别6352、6832、6936首次见失资格且GROUND。最高gap分别−35.74、+51.80、−38.80mm；各段最近前缘距离仍−71.96、−49.39、−50.70mm。RL cross/placed、合法TOP、P13均为0。

**有一个真实局部进展**：6688→6752（完全在learner，0.5333s），CoM世界Δxyz=(+28.91,+8.08,+1.50)mm，固定FR轴投影+19.92mm；body forward+35.54mm，RL绝对bearing从10.59N降到0，current qualified false→true，RR/FL合法TOP分别8/9、9/9观测端点。不是仅载荷占比下降。但CoM的+y与初始FR方向的−y相反，正投影主要是向前贡献，**不是已证明向FR侧完成转移**，之后仍落地。

另外两段新增资格的匹配0.533s窗，CoM固定FR投影分别−16.90、−29.41mm，body forward−16.11、−35.11mm；两端RL力都是0，不能凭资格事件宣称这两个整窗有净卸载载荷变化。

## RR/FL支撑与恢复

第二回合RR首次AIR在6248时仍withinXY，front+52.40mm，不同于第一回合先退离XY但仍TOP承载。此时FL仅0.41N、FR TOP13.60N；6272 FL AIR、RL GROUND14.21N，RR仍TOP但仅0.43N。有后续短暂恢复，不把第一次loss当永久掉载。

最后合法TOP端点为RR **6976**、FL **7000**。RR **7064**落地后，346个ground端点（截至1742其中337已优化、9尾段）中没有新的TOP接触或current qualification。终端RR/RL在ground、FR在TOP、FL AIR gap88.10mm，历史placed_RR不等于当前可用支撑。

## 最早可证实的source与policy/历史贡献

四轮顺序FL/FR/RL/RR，rad/s。以下来自同期source N/FINAL/实测，未独立重算nominal相减归因。

- **6248首次RR掉TOP时**：source N全为−0.3；FINAL=[−.601,−.780,−.154,−.164]，actual=[−.599,−.764,−.140,−.164]。源有限反向段仍存在，前轮learned/HISTORY流又改变了最终请求；不能把早期掉支撑全部归因于stop后残差，也不能只归咎nominal而排除关节/载荷耦合。
- **6480首次采样到source全零**（不是精确atomic stop-event证明）：FL FINAL−1.147、actual−1.154；当时FL无TOP承载，不能算地面牵引。
- **6584最早观察到stop后FL带载反向**：N=0；FL raw=-1.132530、conditional mean=-1.134898、safe projected residual=-0.974260；FINAL−.974、actual−.878；FL/RR真实TOP力3.52/3.34N。其余FINAL=[FR+.494,RL+.095,RR−.044]。这是真正的stop后策略/历史反向执行，不是mask或画面误判；raw不等同最终当拍作用，未归因成唯一原因。
- N=0后共有417个FL负FINAL端点，其中44有FL合法TOP承载、36同时RR/FL合法TOP承载。其余不能当作地面反向推进。

从首learner端点6224到9835，CoM净Δxyz=(−145.25,−27.63,−24.68)mm；RR前缘+45.94→−55.36mm。最初0.5s正投影窗含nominal前缀，不算网络学会。

只读分析；未改阈值/配置/生产代码，未加载Torch/PXR/模型、未运行或停止仿真。当前1536课程保持不变。

