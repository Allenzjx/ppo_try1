# 0.25后腿512块：首33拍有效FALL，不据此判定接管bug

仅只读 `20260916T0628168580885Z_gfc14a68c037a_fe364dae15434293be866a2fe834d015` 已完整写出的首33条审计；global177281–177313，P06/P07/P08/P09按实际请求阶段样本数11/1/1/20。未读活内存或未保存rollout，未运行actor，未重算GAE。原日志确有当拍old distribution mean/std，可直接引用，不用新网络伪代。

终止tick2944 / 24.533333s（N-prefix后2.2秒），reason=FALL，source=PHYSICAL_SAFETY，无任务成功。终态保存的float32观测按固定schema反归一化：roll−19.987729°、pitch2.572985°，projected gravity z=−.938818；以同拍确切CoM世界z减去保存的CoM-relative-base z，得到base-origin z约 **.014920801m**。现有 `_fall_and_explosion` 判据为 `base_z<.015 OR gravity_z>−.30`，本次满足前者，不满足后者；线速度 .07513m/s、角速度 .23821rad/s及高度均未达explosion阈值。raw guard_values未单独持久化，数值来自明确标注的有限精度终态观测解码；不是新仿真。机身包围盒底z .055236m是另一个几何量，不能替代base-origin安全阈值或把既有FALL改名成功。

## 为什么早进P09

- tick2764：RR真实初始上行3.7768mm。
- tick2768：RR qualified上行8.4956mm，current_lift_valid=true；FR/RL有当前承载。P06→P07。
- tick2776：P07→P08；tick2784：P08→P09。三次均为记录在案的 `qualified downstream motion already active; continuous takeover`。
- 当时rear_approach/edge_proximity=0，RR距前缘约−.526m，并非已越沿或完成后腿；末拍仍距−.493m。

源码RR特例使用 **current_lift_valid** 而非单一历史active_lift：包括建立/持续净空、至少2个其他支撑的短连续性、当前非ground、现行body safety及适用的接触response条件。它不是未来静态稳定保证，也没有要求先回地再抬。日志未出现该判据与同期支持/净空的矛盾。远离前缘可能是合法carry入口，不能只因时序提前或随后失败就加回固定距离门。

## 首个偏离和下沉的先后

| tick | 请求阶段 | FR knee final / actual°（actual前1tick） | FL hip final° | 机身bbox底world-z m | 当前支撑 |
|---|---|---|---:|---:|---|
| 2688 | P06 | 41.959 /45.557 | 18.400 | .127422 | 全4 |
| 2752 | P06 | 10.739 /17.568 | 13.179 | .133931 | 全4 |
| 2760 | P06 | 14.739 /15.664 | 17.179 | .132438 | FR/RL |
| 2784 | P08→P09 | 11.739 /15.686 | 41.179 | .129034 | FR/RL |
| 2880 | P09 | 42.630 /35.616 | 66.605 | .077194 | FR/RL |
| 2944 | P09 FALL | 65.975 /65.214 | 60.833 | .055236 | FL/FR/RL |

P06就已有实际残差执行偏离：FR knee持续降低，tick2752四轮target `[.780591,.079475,−.097463,.201165]` rad/s，实际 `[.782320,−.248814,.080417,.165770]`；target与actual不能混为一谈。tick2760 FL/RR卸载发生在接管前，但机身底z至2752仍上升。显著持续下沉在进入P09之后。跨阶段FL hip final从21.179→31.179→41.179→51.179°，其N及残差都保留在JSON；未发现非法后写覆盖、mask泄漏或重置历史证据。此时序不能独立证明由N造成跌倒。

首33拍 `max(abs(raw))=1.069314`、`abs(raw)>3`为0拍，有效sigma峰.466954：不是上一温度高raw饱和现象的简单重演，较小raw仍能通过现有幅度/时序影响载荷。终态任务reward −42.600857（含−40 safety terminal，跨阶段数据未切断为done），原始失败应保留参与真实更新。

结论分类：**当前资格合法、后续物理结果不利；未确认实现错误；单通道/nominal净因果仍未知。** 继续现有采样/完整update有依据，不因一次失败设新optimizer门禁，不改N、距离门或安全guard。详细33拍实值与终态解码见 `quarter512_first33_FALL_diagnosis.json`；提取脚本 `analyze_quarter512_first33.py` 为output-only。
