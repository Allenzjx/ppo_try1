# 局部 world-down nominal 修正：最小集成设计（未实施）

2026-09-06；基于当前 `9d70` 代码的只读设计。未运行 Python/Isaac，未修改生产文件、参数或当前 P10 训练。主规范和补充规范沿用此前完整阅读；真实轨迹依据见 `nominal_preplace_timing_readonly.md`。本方案不是新的训练成功门禁，不要求 B/C 先完成任务。

## 1. 选择唯一真实 mapper 之后的 seam

不在 `NominalMotionProvider` 前复制 mapper、对 shadow mapper 调用 `advance()`，也不再运行一次控制器来取候选。当前 mapper 有反馈采样、跟踪补偿、前一 native/final 状态；把 logical 角度差直接当下一机械关节增量同样不成立。

建议复用 `semantic_residual_adapter.apply_semantic_residual()` 的唯一真实 `mapper.advance()`：先拿到本 tick 的原生 nominal，再计算 B/C 独立 nominal 几何修正 `g`，最后沿原接口一次合成 controller bias 与 PPO residual、一次 hard/slew、一次写入。它是 **post-mapper nominal advisory correction**，不是修改冻结 mapper，也不是把大 residual 再塞回有 ±10° 验证的 teacher/controller bias。

最小数据流：

`真实 q / world J / 当前几何 → 固定本 tick context`

`source logical nominal n → 唯一 mapper.advance → native m → nominal-only g → 原 final bound(m + g, controller c + PPO r, 上一实际 final h) → 原 Full12 write → 现有 native audit → 唯一 physics step`

`g` 的函数输入不能包含本 tick raw action、projected residual、actor mean/std 或 PPO value。输入可以包含真实当前 q/J/geometry、当前阶段与上一实际 final；后者自然包含过去真实动作历史，不能另建“零 residual 轨迹历史”。用同一 context 重建零当前 residual 分支时必须复用同一个 `g`。

## 2. 最少文件与注入点

| 文件 / seam | 最小职责 |
|---|---|
| 新 `ppo/semantic_nominal_geometry.py` | 一个小的只读 context 采集器和一个纯 2-DOF 约束投影函数；不拥有 simulator/controller/optimizer，不写姿态或推进 mapper。可只依赖现有 torch 与标量代数，无新求解器框架。 |
| `semantic_backend.py:187–215` | v3 execution profile 显式 opt-in；在本 tick dispatch 前，从已更新的当前 articulation 和 authoritative source frame 采集 context，传入现有 `SemanticActuationDispatch`。用传给 `_atomic_apply` 的真实 physical tick 绑定记录，另记 source controller tick；不能把含 settle/prefix 偏移的物理 tick 当控制 tick。reset 清理 context。 |
| `semantic_residual_adapter.py:24–118` | opt-in 路径在一次真实 native 产生后求 `g`；无修正时逐值沿原路径。开启修正时，zero residual 也必须经过同一几何处理，不能被当前 `if not any(residual): adapter.apply_full12(...)` 快路绕过。依旧只更新同一 `_final_drive_servo_deg` 一次。 |
| `actuator_target_effect.py:49` | 可选新 composition 的重建；同时保留 raw native 与 corrected nominal，actual 和 zero-current-PPO 分支使用同一 corrected nominal。旧字段/旧模式不能静默换含义，v2 分支不变。 |
| v3 `execution_profile.yaml` | 一个版本化 nominal execution opt-in，不改 action caps、reward、TaskEvaluator 几何阈值或 task 顺序。未设置的 v2 保持原行为。 |
| 新 focused tests | 纯数学、真实 mapper/adapter CPU seam、prefix/视频角色接线；不建立额外证书/多层门禁框架。 |

Controller 和视频通常**不需要新增执行路径**：

- `SemanticControllerAdapter.step()` 已经按当前 observation 更新共享 TaskEvaluator，再产生 source nominal；其 frame / authoritative `semantic_task` 已含 `current_legs.within_top_xy`、AIR/GROUND、净空与阶段。`NominalMotionProvider` 的各 layer 时钟持续，不暂停/重置，不把修正值写回录制 layer 或篡改其 endpoint。source logical nominal 仍诚实记录为修正之前的建议，新的执行修正另外记录。
- 首版只处理实际当前 P09 的 RR hip/knee、P12 的 RL hip/knee，不顺便建立其它腿/全身求解器。四轮、其它腿不参与投影；不能用“历史 placed=true”替代当前 place XY 或当前支撑。
- `PrefixSemanticIsaacBackend` 已继承同一个 B/C backend。teacher 的 `TEACHER`、bias 尚未退尽的 `TAKEOVER`、以及精确 receipt 接管首 tick 都 bypass。可在 prefix 的现有 frame info 中仅补一个 `nominal_geometry_allowed` / handoff tick 字段，或用现有 mode 与 receipt tick 作同等显式判断；不改 teacher 命令，也不修改 `from_live_prefix()` 的原始接管 ACK。接管后的 READY tick 才允许处理；fallback P01 使用普通 B/C 路径。teacher 仍 reset-only、零 PPO credit。
- `semantic_video_cli.build_video_core()` 的 B/C 已构造带 v3 profile 的 `SemanticIsaacBackend`；普通 train/eval CLI 同样如此，故共享修正。A 视频/`legacy_fsm_eval` 仍构造原 `IsaacFSMBackend`，绝不注入本 helper。无需为视频再写一份投影，也不增加转换开关。
- 29 个冻结 A/FSM/mapper/physics 文件均不编辑；只导入其既有转换和 scalar bound 函数。若直接读取其对象状态，也不能写入或 monkeypatch。

## 3. 真实增量、单位与测量点必须区分

定义当前 measured physical joint positions 为 `q`（rad），原始 mapper 输出 `m` 为 canonical drive degrees，controller bias 为 `c`，上一真实 final 为 `h`。先用原 `bounded_drive_feedback_step` 与 `build_physical_batch` 计算 **本 tick 零当前 residual、未作几何修正的 bounded nominal physical target** `t0`，然后 `d0 = t0 - q`。

`d0` 是从真实当前关节位置指向 bounded nominal target 的一阶运动学候选；不是 `n_t - n_(t-1)`，也不是实际会在 1/120 秒执行完的关节位移，不能当速度或精确下一 tick 响应。不得把含当前 residual 的 actual target 拿来求 `g`，否则投影会直接改写 PPO 通道含义。

原转换在 `infrastructure/command_batch.py:199–216`：`physical_rad = radians(standing_deg + SERVO_COMMAND_SIGN * canonical_deg)`。四个 rear servo 的 sign 均为 −1，front 为 +1；standing offset 不可遗漏。实际 servo IDs 来自 `adapter.joint_map`，轮体名用既有 `WHEEL_JOINT_TO_BODY`，不能猜 tensor 下标。`g` 回写到 canonical/native 时使用此逆关系；最终仍由原函数作 hard/slew 与 float32 target cast。

本机 IsaacLab `task_space_actions.py:74–82,143–154` 已有 world Jacobian：floating base 的 joint 列为 DOF index +6，body 行不减 1；fixed base 则 body 行减 1。必须读取实际 `is_fixed_base`、body names、joint names 与 tensor shape 验证，不能照抄固定机械臂索引。Jacobian 来源与 q、source observation 必须属于同一已经 readback 的物理状态；不能用上一轮缓存或为“刷新”额外 `sim.step()`。

参考点现已由 primary source 查明：[PhysX 5.6.1 Articulations — Jacobian](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#jacobian)（Jacobian 段，网页文本行 791）说明 dense Jacobian 对应各 link **COM 处的 world-frame 线/角速度**。本机 `C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/isaacsim/extscache/omni.physics.tensors-107.3.26+107.3.3.wx64.r.cp311.u353/omni/physics/tensors/impl/api.py:2093–2120` 的 `get_jacobians()` 说明 global link velocity，并给出实际 tensor 构造；结合前述本机 `C:/robotics_sim/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py` 接线，不能省略 COM→sensor link origin 转换。

项目 `sensing/sensor_reader.py:266–276,396–401` 用 **body_link_pos_w** 送入轮几何，full-body CoM 使用另一组 **body_com_pos_w**。故取同一真实 tick 的 `offset_world = body_link_pos_w - body_com_pos_w`，并用 `J_link_linear = J_com_linear - skew(offset_world) * J_angular`；其速度等式是 `v_link = v_com + omega × offset_world`。不能把 COM Jacobian 直接当轮底 Jacobian。

`sensing/geometry.py:120–177` 的 `WheelGeometry.center_w_m` 就是传入的 body link position，**不是 AABB bounds 的几何中心**；现 bottom 为 `(center.x, center.y, cached_bounds_min.z)`。保持这个既有几何定义，其局部平移导数使用上述 link origin Jacobian，不另加随姿态转动的“轮底点”而暗改 TaskEvaluator。primary 文档仍不能代替本机器人 live tensor identity/速度核验：实际 probe 需核 body/DOF 索引、COM/link offset、world 线角速度关系、standing/rad 与左右符号，浮动基座的速度身份验证还须包含其 6 列；局部关节候选才固定 base 取两列。当前仅补来源和数学，尚未进行该 live 验证。

## 4. 可测试的局部约束，不做完整 IK

对当前 active rear leg 只取 hip/knee 两列，`Jx,Jz` 为已验证测量点的 world x/z 行。若当前 `within_top_xy=true`，直接 `g=0` 放行 nominal 下降；不额外要求已 cross/placed/TOP、固定关节姿态或前驱静止，成功仍由原共享 evaluator 判断。当前 GROUND 时也不以旧 lift 历史锁住它，保持原 nominal 和 residual 的重试自由。

在有效当前几何、未进入 place XY 且处于需要处理的 airborne / 非 GROUND 前沿接近过程中，使用原 15 mm clearance 参数，不新增 margin：

- `allowance = max(0, current_clearance - existing_0.015_m)`；
- 候选 target 位于由 `h`、原 1.25°/tick slew、原 canonical hard limits 变换得到的 physical target box；
- 只在 `Jz*d0 < -allowance` 时需要修正，目标满足 `Jz*d >= -allowance`；这是目标方向约束，不是实际下一 tick 净空保证；
- 若在该 box 内能保持 `Jx*d = Jx*d0`，取离 `d0` 最近的解。若不可兼得，优先保留原有前送而不是反向/放大：在原前送的可行区间内取最接近原 `Jx*d0` 的值，再最小化关节增量变化。不能为了凑约束增加腿以外的自由度、四轮动作或主动完整抬腿轨迹。

只有两个变量，可按有限约束边界/投影求最近点，无需 full-body optimizer。不得把“rank 小”直接当除零；near-singular 时不用未经界定的伪逆放大。若当前 box、向下约束和保持前送约束无共同可行解，必须明确报告，不能宣称已同时保证。纯函数返回 `unchanged / projected / infeasible / unavailable` 等有限状态及原因，不发任务成功、失败或 reset 指令。

只有实际发生投影时才反解回填：求出 desired corrected **zero-PPO final nominal** canonical target `y`（已在同一可行 box），令 `corrected_native = y - c`，即 `g = y - (m + c)`，只作用于受影响两个 servo；实际写入仍一次执行原 `bound(h, native=m+g, bias=c+r)`。其零当前 residual 分支必须经原 bound 重建为 `y`。**不能**用 `g = y - old_bounded_nominal` 再加到 raw `m`，因为超出 slew 的原始距离尚未消除。用于求候选/审计的 bound 是纯标量计算，不产生第二次实际写入。

`unchanged` / `degraded_bypass`（包括投影没有改变 nominal target）必须直接保留 `corrected_native=m, g=0`，而不是把原 bounded target 反解成新 native。否则即使零 residual 结果相同，非零 residual 下的饱和映射也会改变。投影 active 的 action 映射变化则是明确的新 MDP 内容；`g` 始终不含当前 `r`。三分支审计使用同一个 corrected native，并由真实 frozen scalar bound 重建，不能假设 bound 与加 residual 可交换。

数值正反例（单 servo，canonical 度，hard limit 不触发，`h=0, m=20, c=0`，slew 为 ±1.25°）：

| 分支 | native / g | `r=0` 的最终 target | 说明 |
|---|---|---:|---|
| 原路径 | `m=20, g=0` | 1.25 | `old_bounded_nominal=1.25` |
| 错误 delta 回填，desired=0 | `g=0−1.25=−1.25`，corrected native=18.75 | 1.25 | 修正被同一侧 slew 再次吞掉，未得到 desired |
| 正确 active 反解，desired=0 | `g=0−(20+0)=−20`，corrected native=0 | 0 | `r=+0.5` 时最终为 +0.5；`r=+2` 时仍被原 slew 限为 +1.25 |

反向检查 identity：若无需投影、desired 仍为原 bounded 的 1.25，保留 `m=20` 时 `r=−0.5` 仍输出 +1.25；若错误地反解成 native=1.25，则同一 `r=−0.5` 输出 +0.75，已经改变旧路径。故 identity/degraded 不反解。以上仅证明 target 算术与审计接线，不证明关节实际移动量或连续物理净空。

该设计只拒绝/衰减当前建议的局部预下降分量，不设置历史目标高度或自动重试轨迹。源时钟不暂停，整个阶段不停等；跨阶段继承和其它 owner 沿旧逻辑继续。它也不会自动修复 current wheel prior 在低净空退出的问题，本轮不能顺便调四轮行为。

## 5. 缺证据和审计

缺失/非有限 Jacobian、测量点或时钟不匹配时，helper 不得猜符号、补零 Jacobian、宣称成功投影或继续使用过期缓存。ABI/索引配置错误属于明确接口错误；当前物理 invalid/terminal 继续由已有接口处理，不发明新的 task failure。有限但奇异/约束不可行时可退回原 bounded nominal 并记 `degraded_bypass`，residual 仍自由；这叫“不引入未经验证的新指令”，**不是保证不下降的物理 fail-safe**。具体 fallback 需在实施版本中明确，不能把未处理 tick 计作有效修正。

ACK 与 native audit 保留至少三组相同 pre-tick 状态下的目标：

1. `raw_nominal_zero_policy = bound(h, m, c)`；
2. `geometry_nominal_zero_policy = bound(h, m+g, c)`；
3. `actual = bound(h, m+g, c+r)`。

PPO effect 只能是 3−2；nominal geometry effect 单独记 2−1。两者不是独立物理轨迹。`native_drive_target_full12` 保留真实 mapper 原始 m，不伪装为已调整 native；新增 `nominal_geometry_adjustment_full12` 与 `geometry_adjusted_native_full12`，不改变已有 controller bias ±10° 或 independent residual 字段含义。读取四个 float32 setter/dispatch buffers 的审计仍在原 write 后、physics step 前；重建必须逐值匹配，zero raw/PPO 可有 geometry effect，但 PPO same-tick effect 必须为零。

每 tick 紧凑记录两个关节的 q/t0/修正 target、Jx/Jz、clearance/within_top_xy、状态、g、两个 tick 标识即可；body/DOF/offset 解析的完整映射每 reset 记一次，不把整套矩阵和初始状态在每 decision 重复。可在 CPU audit 中用同一纯函数重算 g，不能通过再次 advance 或写 simulator 来证明它。原 raw action/log-prob/value/reward/done 与实际 residual 审计保留。

## 6. 最小测试与版本边界

必要 focused tests：

- 已知 2×2 J：原前送保持、超额向下被抑制、未越界 target 逐值不变、place XY 与 GROUND 放行；正负 rear sign、standing offset 与 rad 变换往返。
- 退化 J、不可行 box、缺失/过期测量、非有限值均有诚实状态，不静默声称修正；改变当前 raw/residual 而固定 context 时 g 完全相同。
- 真实 frozen mapper + PPO adapter seam：mapper 1 次、write 1 次、final history 更新 1 次；zero/nonzero 分支同时覆盖；原 controller bias cap、hard/slew 不变；几何三分支的 float32 重建和 effect 分账一致。
- Prefix teacher/TAKEOVER/首接管 ACK 逐值不变，READY 后启用、reset 清空；PPO credit 不包含 teacher。A 与旧 v2 路径逐值不变；B/C train/eval/video 读取同一 v3 execution option，不能视频与训练各写一种控制。

这会改变 actuator target 的 nominal 中心以及 final hard/slew 下 residual 的有效作用，即使 caps 和 actor 维数不变，也应作为 **新 MDP / execution revision**，不是 instrumentation-only exact resume。使用已有显式新 MDP continuation 保存旧 checkpoint/计数与来源、清空旧 rollout、按既有流程从合法 reset 重采；不把当前 P10 在旧执行语义下的 samples 混入新 rollout。无需再增一套迁移证书或完整 A/B 成功发布门禁。

本方案未实现、未测试、未预测成功率；真实 base/contact 动力学不在该一阶模型中。最小实际响应检查可以验证方向与接口，但不能把静态 Jacobian target preview 称为物理因果证明。当前 P10 块按原计划继续，待完整边界后由 root 决定是否实施。
