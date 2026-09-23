# RR → RL 角色、reward 与固定方向边界（只读）

2026-09-22；检查 HEAD `a54678ceef5868e419e55840cea34899854bc726`。仅源码/当前配置核对，无 Python、测试、仿真、轨迹重放或训练；没有修改运行中的生产文件。本报告不声称已量测本次 RR/RL 的实际 CoM 转移方向。

## 1. 切换不是“全局只保留一个方向”

| 范围 | 当前明确语义 |
|---|---|
| P09 | `active_leg=RR`，接收 FL、优选桥接 FR+RL；目标仍为真实 `placed_RR`。 |
| P09 → P10 | 在合法 entry、`placed_RR` 谓词满足、无终止且 `tick % 8 == 0` 时切换。`active_leg` 与摘要 `transfer_role_context` 在该拍切为 RL。 |
| P10 / P11 / P12 | 一直是 RL → 接收 FR、优选桥接 FL+RR；分别为接收侧准备、RL 转移、RL 捕获。P10/P11 可连续接管已真实合格的 RL AIR 运动，不要求重落地再起摆。 |
| 每拍全部角色 | `TransferRoleTracker.observe()` 始终同时计算 FL/FR/RL/RR 四行；48 维角色观测也全部保留，不因阶段切换清空 RR 行。桥接列表是偏好，非强制接触模板。 |

证据：`stage_task_spec.yaml:27–40,240–274`；`semantic_supervisor.py:1497–1566,1665–1672`；`semantic_transfer_roles.py:149–268`。

**重要限制：** 当前 `placed_RR` 的 `_current_rr_placement_usable()`（supervisor:186）允许历史放置后、当前合法且 current-Q 有效的 AIR。因此“已切换 P10”不等于“RR 正在承载”。新的 `rr_current_bearing` 严格区分此事；`rl_transfer_ready` 要求当前 RR verified TOP bearing + RL `preparation_ready`，但目前该位只进入 backend info、actor 的 7 bool 追加观测和日志，**没有成为 P10/P11/P12 的 source/transition 卸载许可门**。这与首版保持 source 不变的范围一致，不能报告成承载门已经部署。`rr_capture_recovery_allowed` 单独用于 P09 有效 DESCEND 的局部 timeout，wheel guidance 当前 off。

## 2. 旧 RR post-cross 信用没有因切阶段失效

当前 spec 仍是 `established_RR_over_top_receiver_retirement_v2`。`_current_rr_receiver_preparation_retired()` 不读阶段标签；它要求已有真实 RR Q+cross、当前 lift_established / motion_continuation_allowed、初始抬升增益、无 GROUND、合法 TOP XY/lateral 及 AIR/TOP 条件。瞬时 `body_control_evidence/currentQ=false` 单独不再撤销这份已建立信用。

`physical_potential()` 的处理顺序很关键：

- RR **未放置**：满足上述退休条件时，其 workspace 的 FL receiver 项固定为 1，不再要求继续 FL 收拢/增加其关节余量。GROUND、越界或失效确实可撤销这份准备信用，但那不是 P10/P11 标签造成的。
- RR **已放置**：首先进入 `.8 + .2 * current_retention` 分支，根本不再调用 RR workspace/receiver 准备项；即使后来 RR 失去当前 TOP，也不会重新以 RR preparation 去奖励折 FL。此时惩罚取决于当前区域 retention；合法 AIR 本身不是被禁止的姿态。
- 全局 Φ 会提前给 RL workspace 准备信用（原有 `.1` 份额），所以 **RR 尚未退休时，FL 与 FR 两个 receiver 的 workspace 项可能同时存在**。这是关节余量/相对机身径向收缩 proxy，不是两个“CoM 必须同时朝 FL/FR”的显式方向 reward，也不是当前已证明的矛盾。
- RR 合法 post-cross 后，RR unload 份额亦固定为 1；RR placed 后只余 retention，RL 才取得完整 unload/lift/carry/capture 的前驱信用。RL 前驱判断用历史 RR placed，不额外要求新的 current-bearing bool。因此若 RR 已失载，不能靠该 bool 的存在推断 RL reward/卸载已被阻止。

证据：supervisor:89–98,275–314,1301–1350,1392–1417,1450–1476。receiver 退休最多固定原 Φ 中 `.85/4 * .1 * .5 = .010625` 的份额，未增加新 bonus。reward 仍是 `5 * (.9985 * Φ_next - Φ_prev)` 加原时间/终态项；普通阶段切换不重置 Φ。当前 P09 只有无方向性的 collider separation quality，P10–P12 不在这项质量窗口；接触/平滑/regularization family 权重均为 0。不要把 `physical_transfer_fraction=max(eligible motion_fraction)` 描述成新增方向奖励。

## 3. CoM 数据真实可用，但现有方向仍是滑动窗口

tracker 使用 `center_of_mass.position_w_m/velocity_w_m_s`（必须 valid）、当前 wheel center 和 body pose。质量中心来自所有锁定刚体的 `body_com_pos_w` / `default_mass` 加权，无 base-only 回退（`sensing/com_diagnostics.py:18–60`）。RR 对应接收点是 FL wheel center；RL 对应 FR wheel center。

`window_s=.5`，每拍移除旧样本；`fixed_direction_world` 只固定于**当前滑动窗口最早一拍**：`normalize((receiver_first - massCOM_first).xy)`。`com_toward_receiver_m` 投影的是 CoM 自身位移，接收脚位移另列，故单纯移脚不会直接伪造这一投影；但窗口推进会重新锚定方向。接收脚相对机身收缩另进入 workspace proxy，也不能被称为 CoM 位移。actor 看到各角色方向 xy、进展/成熟度及当前 CoM 相对 base/速度，未保存“本次转移开始 anchor + 全部窗口历史”，不能声称已经具备固定转移起点或完全 Markov。

## 4. 单次固定 d 的最小只读诊断方案（尚未执行）

在本次封存后，对同一 run 预先声明两个参考起点：RR 准备的 P07 首入口、RL 接续的 P10 首入口；它们是**调度入口参考**，不冒充力学卸载的精确开始。如果有效转移早于入口，明确未覆盖前序，不事后移动起点挑更好的结果。

每段从同拍真实 raw 记录固定 `t0, COM0, receiver0, d0=normalize((receiver0-COM0).xy)`；之后只计算 `dot(COM(t)-COM0,d0)`、`dot(vCOM(t),d0)`，另列 receiver 位移、base 位移、RR/FL/FR/RL 当前接触/verified bearing、history 与 phase/source 事件。跨普通阶段、接触闪断不重锚；真实 GROUND attempt 失效结束这段诊断，新尝试另起独立编号。若 raw receiver0/CoM 无效、缺测或平面向量退化，则结果 null，不用当前脚位或滑动方向补造。此方案只读、不给 reward/GAE/optimizer 信用，也不证明正投影必然产生正确载荷转移。

结论：角色映射无左右/前后颠倒；RR v2 退休仍生效。当前最需如实披露的是“新 current-bearing/RL-ready 已可观测但未硬接卸载调度”，以及“固定转移起点方向尚未实现/实测”；没有证据支持在活动运行中修改 reward 或 source。
