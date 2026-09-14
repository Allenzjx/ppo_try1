# Block11 episode1–2：真实FL放置与RR连续抬升，原失败保留

固定读取同run `train/20260910T1326149079057Z_g7db0d17f398d_ff94228e94e648f5b9fe4b59228b03f1` 的前三条completed及截至第三terminal的556条audit，只解析尚未报告的245–556行，不重复解析ep0 native。分析原始bytes18,219,679–41,222,320（含端点，23,002,642 bytes），SHA256 `37110da0df6f3117b9f6099eee9be1934b798984dd5252db952149d68b37e9cc`。两回合均在真实教师前缀P04 tick1696/14.133333 s后开启信用；教师数据不进入PPO，FR Q23/C1665/P1695不计新策略成绩。

| 回合 | policy范围/样本 | 实际发出动作相位 | 物理/native | 原终态 |
|---|---|---|---|---|
| ep1 | global148597–148719；123 | P04=1/P05=72/P06=50 | 983 ticks全部verified；own981 | P06 FALL，2679/22.325 s |
| ep2 | global148720–148908；189 | P04=1/P05=146/P06=24/P07=1/P08=1/P09=16 | 1505 ticks全部verified；own1500 | P09 BODY_COLLISION，3201/26.675 s |

**ep1简述。** 新FL I1755/Q1759/C1889/P2277，2280进P06；RR无I/Q/C/P。终态372维policy/critic相同、finite_fallback=false，固定schema及locked底面回解base_z=13.987787 mm<15 mm、gravity_z=-0.992386，线/角速度模长0.171534/0.315193，支持真实低高度FALL而非翻转。FL当前TOP、bearing14.546523 N验证有效；但FR bearing不可验证，使所有load_fraction_valid=false，**不能把FL数值0.53113写作已验证载荷比例**。载荷缺失不取消独立高度安全证据。当前P06 rear_approach未完成，保留失败。

## ep2 RR事件发生在P06，不是P09才抬起

- FL在P05产生I1766/Q1772，C2680/P2870；2872进入P06。此入口FL真实TOP，bearing4.286169 N、有效load0.161046，不只是历史placed。
- RR **I3051/25.425 s/global148889**：excursion3.540499 mm，全身实测关节运动34.987045°；**Q3057/25.475 s/global148890**：excursion8.334816 mm，RR自身实测关节运动8.785076°。两事件都发生于发出P06动作期间。
- P06→P07在3064/25.533333 s，P07→P08在3072/25.6 s，P08→P09在3080/25.666667 s，均由当前functional lift连续接管。rear_approach仅0.146687、后续RR edge_proximity约0.32，不能写作后腿已靠近到最终目标。
- 同一次Q跨普通阶段保持，**没有GROUND撤销或再次I/Q**；18个决策末态current_lift_valid=true。RR final[hip,knee]在三个边界为[-11.704,-19.966]→[-8.505,-16.466]→[-12.005,-19.615]°，residual未归零、wheel持续非零，普通handoff hold延续执行。

RR抬升交接的3064/3072/3080以及末态，FL均为 **AIR、bearing0、有效load0、support=false**，不能用P2870声称FL此时承载。末态实际支撑是FR TOP（11.729867 N，load0.563850）与RL GROUND（9.073318 N，load0.436150）；不据此猜CoM方向或建立固定支撑组合要求。

末态RR AIR、ground-relative lift207.012 mm、台面净空156.844 mm，但仍在前缘 **-310.411 mm**，**C/P均false**。history/lift_established保留true，current_lift_valid/continuation=false、reason=`physical_safety_abort`；大离地和曾经成立不能替代当前受控，更不是越沿放置。首个未完成后腿任务为继续受控前送、越沿及放置。

## BODY依据、执行与信用边界

ep2 audit/completed一致为BODY_COLLISION；evaluator valid=true、evidence VERIFIED、source=`BODY_CONTACT`、reason=`central body/obstacle collision`。既有生产仅消费authoritative exact base_link/Obstacle detector结果，不把腿/wheel接触等同BODY失败。但本训练audit未保存原始BODY pair持续次数、penetration和检测器细分reason，**不能独立指定这次是持续接触还是穿透分支**。已存body AABB min z=0.049989955 m不是signed penetration读数，不能据此补造穿透量或判检测错误。

终态372维policy/critic相同、finite_fallback=false；float32回解base_z18.135391 mm、gravity_z=-0.932240，线/角速度模长0.047210/0.429331，8个实测关节在各自硬限内；未显示低高度/翻转/速度爆炸或硬限条件。保留BODY失败，不降阈值。上述高度来自locked底面0减实测relative plane的固定schema回解，不冒称独立原始双精度物理流复核。

两回合共2488/2488 native-effect verified、7个普通交接hold，禁止的episode内state write=0；未发现新增下发/普通切换reset缺陷。按13:47:24.2032615Z主控快照，1128/global148864已优化保存本块512条；固定第三terminal时另44条已采样、尚未进入该checkpoint，**包含RR I/Q这组新事件**。不预支后续更新，不把跨更新随机后缀训练称为固定评估改善、完整后缀成功或自然P01全程成功。

仅新增本报告；未读第四前缀、physical大流或checkpoint，未改生产、四主报告或CSV。
