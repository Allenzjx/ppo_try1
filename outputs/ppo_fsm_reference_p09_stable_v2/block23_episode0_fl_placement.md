# Block23 ep0：信用段内 FL 新放置成立，P06 低机身高度跌倒

固定读取 run `train/20260910T2345331327840Z_gd4e46006b382_b30b9a723a7c414bbbb503960c5886e4` 的audit前100行、bytes [0,7314726)，global161153–161252（2026-09-10 23:52:14.084–23:52:14.125 UTC，一次解析），及completed第一行。未解析活动尾或巨型物理流，未读取checkpoint或主ledger。

真实教师前缀至P03/tick1656，首有信用决策1656→1664。100个新策略样本为 **P03=5/P04=1/P05=64/P06=30**，即前驱准备也有6个真实策略样本；800物理ticks全部verified/native effect有效，own-phase effect=797，无episode内状态写入。终态2456/20.466667s包含前缀，策略信用段物理时长为6.666667s；不是自然P01全程策略回合。

## FL 事件明确发生在 PPO 信用之后

| 新事件 | tick / 源阶段 | 与信用起点的关系 |
|---|---|---|
| 初始离地 I | 1710，P05 | >1656，首尝试未建立Q |
| 新 I → Q | 1757 → 1760，P05 | Q事件实测向上变化8.965903mm |
| 越沿 C | 1923，P05 | event_ticks首次记录，非教师历史 |
| 顶部放置 P | 2211，P05 | event_ticks首次记录，非教师历史 |

因此FL新I2/Q1/C1/P1，无Q回地撤销记录。FR Q23属于教师；FR C1664/P1693则发生在本次P03信用段。不能把整条前缀或教师Q加进新增策略训练量。

FL当前接触与P历史分开：2208仍AIR、bearing=0；P2211后，2216的FL实际TOP/support=true、bearing_verified=true、bearing=4.882817N、front+66.640857mm、台面净空约+0.001135mm；2224又AIR/bearing=0。最终2456重新TOP、verified bearing=7.723222N，front+74.245754mm、净空−0.710199mm。放置事件真实成立，但不能据历史P声称全程持续承载。

## 普通阶段交接连续，后腿任务仍未完成

实际切换P03→04=1696、P04→05=1704、P05→06=2216，均done=false且入口valid。下一决策的三个jump审计均显示全12通道previous/carried residual相等，forbidden/phase-scale-clipped为空。

2208/2216/2224 nominal四轮均+0.3；2216 residual=[−0.433313,−0.249729,−0.010531,+0.539047]，final=[−0.133313,+0.050271,+0.289469,+0.839047]，随后继续变化，没有P05→P06清零wheel或residual证据。

**RR整个信用段无新I/Q/C/P，current Q端点计数为0**，并非仅看末态false。终态RRGROUND、verified bearing=9.344727N、front−513.240673mm、台面净空−49.968558mm。终止前一拍2448及终态的 `rear_approach=0`；首次未完成任务是P06的后腿接近/入口准备，而非已经抬起后未放置。

## 跌倒证据与回解范围

completed首行确认ep0/seed1001/100decisions/FALL/full_task_success=false。物理评价为 `SAFETY_ABORT` / `PHYSICAL_SAFETY` / `fall or physics explosion`，run VALID、physical evidence VERIFIED；本报告不把它改成成功。

终态actor观测372维全部finite，`terminal_observation_finite_fallback=false`。按现行schema及固定chassis/body identity回解：gravity=[0.240953,−0.007565,−0.970507]，线速度模0.199027m/s，角速度模0.333262rad/s。obstacle-relative planes的bottom项（index97）为−0.0145770088m；结合 `configs/environment_lock.json` 中固定障碍物bottom_z=0，推得**base_z≈14.577009mm，低于现有15mm跌倒阈值**。gravity_z未越过−0.30，速度也未达到5m/s或20rad/s爆炸阈值。

这是由持久化float32观测和锁定几何回解的高度，不是重新读取的原始双精度base坐标；没有读取完整物理流复核。已足以支持低高度FALL分支，不需要单因归责某个关节或放宽阈值。首100样本仅按已观察信用统计，当前优化/保存进度由主控另行核验；无生产、官方checkpoint或四主报告修改。
