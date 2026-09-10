# 共同物理验收：只读问题与最小修复边界

审查基线 HEAD `285307603f43bfc2846dada5f3ede73b0dc5f536`。已完整读取本轮893行正式规范。范围是共同 evaluator、接触/几何传感语义、四腿主动越沿及 P13 结果分层；仅源码、既有 B1 和 P13 小报告。没有重读 raw、加载 tensor、测试、仿真、修改生产或历史判定。本报告是静态审查，不是新版物理验证。

## 确证问题

### 1. 缺旧 Q 不等于已证明纯轮违规

`semantic_supervisor.py:462–506` 对四腿使用相同旧资格：连续 AIR≥2 physics ticks、轮底≥台面、窗口上升≥8mm、主动作用证据及前缘区间。首次 `distance>=0` 时，`not active_lift or not crossing_geometry` 直接报 `TASK_FAILURE_WHEEL_ONLY_CLIMB`；只有已 Q 的轻微负净空 AIR 有 pending 例外。

这确实违反新规范的证据分层。已有全身运动判断并非“只看目标腿自身关节”，不能把这部分合理实现推翻。现有测试 `test_active_clearance_then_early_top_landing_can_cross_after_multiple_contact_ticks` 也已允许**先 Q，再提前接触，再轮心越线**；真正缺口是合法主动过程必须先符合旧 AIR-above-top 模板，且其缺失被当成纯轮的肯定证据。

最小修复：同一评价器、每腿本次尝试保存主动作用/真实上升/接触面序列，保留旧 Q 为一种充分证据；另容纳可靠的主动抬升→提前顶部接触→轮心越线。证据不足单列 unverified/pending，不能自动成功，也不能确定纯轮。纯轮结论须有当前过程的正证据，不由缺标签代替。真实前缘前重新落地仍撤销本次尝试；RR_FIRST、其他腿已发生历史保留。

**B1 已有证据的重新解释边界：**5993 initial（全身19.499°、own1.683°已被认可）；最高 AIR 6016 仍 gap −.886mm；6057 首越线，obstacle pair 13.758N，旧 Q=false。因此确实命中旧规则，但按新规范不足以确定唯一 wheel 驱动或合法主动跨越。旧结果不得改写成成功；缺失后续轨迹不能补造。

### 2. 整障碍反作用被当成 TOP/承载，未独立核接触面

`semantic_supervisor.py:402–406,478–479,512–537` 的 `top_active` 实际是整个 obstacle exact pair；`loaded/top_contact` 只另加轮底薄层、XY、轮心已越前缘。`ContactClassifier` 正确只报告 GROUND/OBSTACLE/AIR，并不声称面分类；其 `normal_force_n` 是 `force_w_n` 范数。evaluator 却把地面与障碍范数相加，作为 support/load_fraction；水平 wall 反作用可被算成承载。

现有 `contact_point_w_m` 是可选的 filter mean，`force_w_n` 是聚合向量，**没有单独逐接触法向字段**。不能将 force 向量或混合 mean 直接冒充唯一接触面的法向/点。`com_diagnostics.compute_support_diagnostics` 同样把任意有效 obstacle pair 的点纳入 XY hull，故该 hull 不能无条件称为承载稳定性证明。

最小修复：保留原始 pair/force，新增有效性明确的 surface/contact-reaction/bearing 三种量；可靠点与几何、受力方向相符时识别 TOP/WALL，混合/缺失证据标 ambiguous。当前 TOP 累计应与轮心越线历史分开，允许真实提前顶部接触。反例：同轮心/轮底，分别给顶部向上反作用、立面水平反作用、混合 mean/缺点，不能得到同样的已验证 TOP 承载。接触反作用不是禁止项，不能把合法腿/轮顶障改为 BODY。

### 3. 旋转后的 collider 世界尺寸没有更新（优先底层交叉核验）

`sensing/geometry.py:127–159` 首次取得 world AABB 后缓存相对原点的 world min/max offsets；之后仅 `center+offset`，不再使用当前 orientation。底层 `UsdCollisionBoundsProvider.collision_bounds` 本来已有 body-local points + 当前四元数变换（约244–259），但外层缓存让后续调用不到这里。轮底和 base penetration 都继承该问题。

明确反例：半长 .20/.02 的非球形 collider 绕 Y 转90°且中心不变，真实 z 半径应由 .02 变 .20，缓存结果仍 .02。它影响具有倾斜变化的几何，不能预断 B1 亚毫米差值就是该原因。

最小修复是缓存**body-local**不可变形状，每 tick 以真实 pose 变换，或复用现有 provider 的轻量变换路径；不改资产/质量/摩擦，也不每 tick 重扫 USD。新 P13 body 已通过判断应在这项修正后使用实测 collider 范围，不继续使用原点替身。

### 4. 缺机身接触证据可仍显示 evaluator valid=true

`BodyCollisionDetector.evaluate` 在 base observation/pair 不可用时返回 detected=false；`TaskEvaluator.observe` 只要求这个布尔，逐腿验 wheel pairs，但没有独立验 base exact pair。SensorReader 的 data_quality 会记录单体 sensor 缺失；`guard_state._all_finite` 仅查数值，不把该缺失变为非有限。

明确反例：四轮 exact pairs 正常、base obstacle pair unverified、所有运动量有限，可得到 detected=false 且 evaluator valid=true。这应是 BODY evidence unverified，而不是已证明无碰撞，更不是自动 BODY 碰撞。真实 base pair 2 tick 持续或实时 penetration 的原有安全分支应保留；只对缺失增加有效性，不拉长去抖或隐藏碰撞。

### 5. P13 基本越障、当前受控、严格恢复仍混为单一 success

`semantic_supervisor.py:545–557`：四轮都须在旧 top-gap [−15,+25]mm 薄层、front≥20mm，base-link 原点 front≥150mm 且仍在平台 XY；再要求 body linear≤.05m/s、angular≤.30rad/s、实测 wheel≤.25rad/s、command≤.02rad/s、至少2 TOP支持，持续.5s才 success。v3 `physical_stable_pose` **已解除硬 home**，无需重复“去 home”修复。平台目标也没有要求驶下后缘，这一点保留。

确证缺口：没有独立 traversal event/current task complete/recovery quality；base 原点不是机身后缘，四轮薄 z 层和很小 command 是严格恢复要求而非完整物理通过的唯一证明。历史 P13 报告 `p10_145301_ep0_ep1_p13_terminal_readonly.md` 的 ep1 四轮 XY 均在平台，FR AIR +28.314mm 单独使 region=false；但实测 wheel .421rad/s、command .461rad/s 等也未满足旧 stop。**不能把该旧回合直接重新宣称基本受控成功。**另 `p13_headroom_first_episode_readonly.md` 的 RL Q/C/P 是真实 PPO suffix 进展，FR/FL/RR 来自教师，仍非自然 P01 成功。

最小修复：分别输出 `traversal_event_observed`（可锁存）、`traversal_task_complete`（当前区域/物理受控/有效证据/无安全失败）、`recovery_quality`（实际速度、command、姿态、冲击、jerk等）。基准区域使用同一平台和真实 body 范围；基本受控阈值须预先定义，不能为旧单帧拟合。固定有限收尾观察后才结束；事件之后退回/跌倒/碰撞照常影响最终结果，不能 milestone terminal 后继续追加同 rollout。同步 `whole_task_success`/Phi、terminal reward、env/backend/recorder，避免基本完成但质量未达仍统一吃 −40 的“完全失败”结果。

## 应保留与共同接线

- 四腿主循环统一，RL 的 RR_FIRST 特殊顺序是本轮明确要求。120Hz 连续 tick 验证、同 tick 不重复计数、当前支撑与历史 placed 分开、重新落地撤销尝试、真实 BODY/实测硬限/FALL/非有限独立安全中止均有保留依据。
- A/B/C 可复用 `TaskEvaluator` 与 `PhysicalEvaluationRecorder(task_spec_path=...)`，必须显式传同一新版配置；默认仍是 v2，不能靠相同类名宣称同评价。
- 旧 `trial_analyzer.py:250` 甚至将缺 crossing 与缺 lift 都归 wheel_only；`physical_success.py` 还沿用旧 guard/manifest 证据。这些旧产物保留，不作为新共同物理判定输入。R/A 原始控制路径不改；新增离线共同评估结果另存旧/新分歧，不从旧 phase 或 success 导出新成功。

## 最小定向测试位置（只建议，未执行）

1. `tests/unit/test_semantic_supervisor.py`：四腿参数化旧AIR充分证据、主动提前TOP而未旧Q、纯轮立面正证、缺点/缺运动未验证、落地后旧证据不能复用、真实重试、RR_FIRST。
2. `tests/unit/test_sensing_stack.py`：沿现有 active pair/BODY 测试补缺 base pair 有效性；同一 cache 连续两姿态测试（现有 local-point 单次变换测试不覆盖外层缓存）；TOP/WALL/edge-mixed 及 bearing/reaction 分离，真实2tick BODY不能被减弱。
3. `test_semantic_continuous_stop_progress.py` + `test_semantic_capture_retention.py`：四腿历史不足、body 原点通过但后缘未过、当前完成但严格质量不佳、区域外退回、完成事件后跌倒、有限收尾且无二次 terminal/重复奖金。
4. `test_semantic_legacy_evaluation.py`（若现有文件名不同沿 recorder 测试入口）：相同原始片段/同spec 的 R/A/B/C 评价一致；未访问质量保持 unavailable，不用教师 suffix 替代自然 P01。

上述确证是代码语义/数据有效性问题；除已引用历史局部事实外，新面分类、body through 与分层完成均 **PHYSICALLY_UNVERIFIED**。没有为训练增加 A重跑、人工探针成功或13阶段全物理通过门槛。
