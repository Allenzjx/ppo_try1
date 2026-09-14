# Block24 episode3：两次 RR Q、两次回地撤销，末尾为非终止采集尾

范围：`train/20260911T0005451949265Z_gd4e46006b382_08d59fe9ad444f60b1098f90dbe1fca0`，仅解析audit **630–1024**（global162294–162688），逻辑字节 `[48923592,82770214)`；2026-09-11 00:57:27.050–00:57:27.347 UTC 单次扫描，前629只定位。未读其他回合、checkpoint、大物理流或活动block25。

教师P01→P06信用起点3584，prefix attempt=3，教师不进PPO。395样本来源为P06=220、P07=1、P08=1、P09=173；3160物理ticks全部native verified/actual effect，own-phase=3157，395条无episode内状态写入。主控已确认全部优化保存；这里不另读CP或推算其他回合事件。

## 实际抬升/重试

| 本回合尝试 | 新事件与回地 | 连续有效决策端点（非逐tick完整状态） |
|---|---|---|
|1|I5552 → **Q5612** → 回地撤销5662|5616–5656，每8tick一条，共6条；FRONT_WALL/OBSTACLE_AMBIGUOUS，非AIR悬停。|
|2|I5834 → **Q5853** → 回地撤销5956|5856–5952，每8tick一条，共13条；边缘/不明障碍接触与AIR之间变化，没有因轻触边缘清空同次资格。|
|3|I6616，无新Q；6648端点已回地|current有效0；6632最高端点地面相对净空7.271415mm，尚未形成8mm资格。|

本回合直接计数 **I=3、Q=2、真实回地撤销=2、current有效端点=19、C/P=0**。每次回地同时有`qualification_revoked_ground_before_cross`和`current_lift_revoked_ground`两个标签，仅计一个物理事件。初始I后的第三次回地不是Q撤销。所有新事件均在P09；末RR Q/current=false没有抹掉前两次真实Q。精确Q到回地事件时点可读，但审计没有全部中间tick的current快照，不能将端点连续性冒充每tick可控保证。

第一Q后5616地面相对净空9.511897mm、front−46.009716mm；5632增至11.075591mm，5656降至4.527025mm，5664已GROUND。Q5612事件的`upward_excursion_m=5.784131mm`是通用短窗gain，不是functional RR判据所用的地面相对gain（`semantic_supervisor.py:814–818`）；不能拿它替换8mm条件或反推放宽阈值。

第二Q后5856净空9.040616mm、front−44.800219mm。**5920**是本范围最靠前且地面相对净空最高的记录端点：AIR/current=true，front **−34.274843mm**、台面净空 **−17.679342mm**、地面相对净空 **32.673119mm**。仍未到前缘/台面，不是C。至5952仍current=true但已退至−57.440546mm、净空降至8.121251mm；5956回地撤销。表现为实际抬升/接近后失去净空、退回，并未完成越沿或放置。

19个current有效端点均body_control_evidence=true，独立FR TOP与RL GROUND支撑均verified。FL并非固定承载：第一次窗口全部AIR/0N；第二次5856短暂TOP/.507575N，另5880/5888/5896有TOP承载，其余多为AIR。5920最佳端点FL AIR/0N，FR/RL分别12.899176N/14.044702N。故不能以“CoM向FL侧”或FL历史P代替当前FL受力。本回合FL总199个AIR、196个TOP有效承载端点；这仍不是稳定性/单因证明。

## 连续动作及源后段时序

真实边界5344 P06→P07、5352→P08、5360→P09均非done、RR仍GROUND无I/Q。交接记录5352/5360/5368的Full12 residual完全继承，无forbidden或phase-scale丢弃；全部mask开放、setter/native mapping通过。nominal四轮按P06实时几何退役从+.3经+.235487、+.001509至0，而最终wheel一直保留非零residual（5368约`[−.580805,−.339258,+.641741,+.502155]`）。并非阶段切换清零动作。

两次Q的P09阶段年龄为**2.10s、4.108333s**，都早于源5.4s后段原子请求。端点6008仍旧FL/RL nominal，6016才已见新全身请求`FL=[−18.5,−31.4], RL=[15.4,19.4], FLwheel=−1.07`；第三次I6616为10.466667s，发生在该后段之后但没有Q。不能把前两次Q或回地归因于尚未下发的5.4s段。

第一次Q至第二次撤销期间RR nominal保持`[−6.9,−37.8]`°，residual与final持续变化：5616 final `[−11.282889,−36.659877]`，5656 `[−21.384776,−52.819100]`；5920 `[−31.978519,−46.893363]`，5952 `[−12.804940,−54.724449]`。实际净空/前缘曲线才是运动证据，不能把负膝角或某个请求变号直接命名为物理下降原因。

## 封存末态

6744/56.2s仍P09，stage age=11.533333s，**395条全非终止**；reason/source=null，time_outs=false、terminal_bootstrap_allowed=true。它是正常外部采集尾，不是第四次终态/新失败，更非成功。RR GROUND/front−120.673862mm/台面净空−50.026382mm，I/Q/current=false；motion_continuation_allowed=true，允许下一次实测重试。FL AIR/0N，FR TOP11.731868N verified；RL虽记录ground/support/9.352833N但bearing_verified=false，不能宣称其已验证承载，整体CONTACT_BEARING_UNVERIFIED。

首未完成任务仍为RR重新取得可用抬升并连续越沿、放置。两次Q成立、随后正确回地撤销、仍可继续探查均有直接证据；未发现新的明确dispatch/交接缺陷，不增门禁或调整生产规则。
