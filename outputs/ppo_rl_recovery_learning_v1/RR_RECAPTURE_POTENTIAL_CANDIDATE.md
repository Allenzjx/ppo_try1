# RR 重捕获 potential 隔离候选（未应用）

结论：可以在**现有 RR retention 的 0.2 份额内**，恢复地面／越出台面区域的连续几何准备信号，而不开放依赖 RR 承重的卸载，不增加事件奖金。它只解决一处稀疏区；不能据此声称策略或捕获能力已改善。

只使用已完成的 225281–225664（updates1726–1728、384 个 PPO 样本）。生产源码、reward/config、运行 rollout、模型与安全规则均未改；无 Torch/Isaac 调用、无辅助拟合。先交付当前实际模型视频的优先级不变。

## 最小设计

替换的数学子份额来自 `semantic_supervisor._current_capture_retention()` 中已 placed 的 RR、且 RL 尚未形成真实连续 swing／placement 的现有分支；**不在该 supervisor 方法或共享 task snapshot 上落地修改**。确切实施候选是在 `SemanticRewardCalculator.evaluate()` 的 reward-local Φ 中以 `new − old` 替换同一份额，保持 actor/critic 共用观测中的旧 `task_progress_potential` 原值和含义。其余腿、未首次放置 RR、合法 RL 摆腿后续、历史 0.8 份额、全局系数全部不动。

设：

```text
g = clip(1 − current_top_xy_outside_distance / 0.25, 0, 1)
    × 0.025 / (0.025 + abs(current_RR_gap))

K = 原有 legal AIR/TOP 几何条件
    或 当前有效、未回地面的本次 RR 抬升资格

C = 原有当前 TOP、实测 bearing 成立时的连续 TOP 样本比例；否则 0

R_candidate = 0.5 × g × (0.25 + 0.75 × K) + 0.5 × C
ΔΦ = (0.85 / 4) × 0.2 × (R_candidate − R_current)
```

0.25 m、0.025 m、TOP 样本数和承载条件均来自当前实现。**几何半份额中的 1/4 是待验证的软分配假设，不是接触门槛、真实可达性证明或已经学到的参数。**没有提高 retention、potential_weight 或 smoothness 权重。

| 当前物理状态 | 几何准备／接近度上限 | 接触半份额 | 含义 |
| --- | ---: | ---: | --- |
| GROUND／GROUND+OBSTACLE | 0.125 | 0 | 仅准备，不是抬升、顶部支撑或可卸载许可 |
| 非法区域内未重新取得资格的 AIR/EDGE | 0.125 | 0 | 小范围恢复几何，不沿用历史 placed 作为当前资格 |
| 当前有效 AIR/EDGE；或原有合法 AIR 区域 | 0.5 | 0 | 接近度，不是承载；边缘接触不自动失败 |
| 实际 TOP，但载荷未验证 | 0.5 | 0 | 接触候选，不冒充 bearing |
| 当前实际 TOP 且 bearing 验证 | 0.5 | 0.5 | 最高 retention，随真实连续 TOP 形成 |

保留原有合法 AIR 的数值，即使当前 qualification 因其他支撑证据短暂不足，也不新增一次降分。这个几何分数本来就不是新主动抬升资格。初稿曾降低这类旧合法 AIR 的值，现已由 `offline_v2` 修正；初稿 JSON 保留，不能作为采用版本。

同一物理资格下，跨过旧 XY／负 gap 截断处不再突然清零。接触／资格是真实离散证据，**不声称整个函数在这些事件上处处连续**。AIR 与 bearing、ground 与新有效抬升仍严格分开。给定传感器字段不相容时，保守 TOP 谓词优先拒绝接触份额。

## 确切 reward-only 接入点：不能改观测数值

当前已核对的调用链（`f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b`）：

- `src/wlr50_clean/ppo/semantic_supervisor.py:1895` 把旧 `physical_potential(evaluation)` 写入共享 `_snapshot["task_progress_potential"]`。
- `src/wlr50_clean/ppo/semantic_observation.py:408` 将它放入 `groups["task_progress"]`；这是实际 actor/critic 输入，不仅是日志。
- `src/wlr50_clean/ppo/semantic_env.py:193` 把本决策的真实起止 `SemanticObservationFrame` 与 120 Hz samples 交给 `SemanticRewardCalculator.evaluate()`，随后才 encode observation。N1 使用这一路。`semantic_vector_env.py:113` 也调用同一个 calculator。
- 唯一计划调整位置是 `src/wlr50_clean/ppo/semantic_reward.py:458–460`：当前这里读取 shared Φ 两端，并只做一次 `potential_weight × (γ × Φ_after − Φ_before)`。

候选伪代码（**未写入生产**）：

```python
shared_before = finite(previous.task["task_progress_potential"], "shared Phi before")
delta_before = rr_retention_new_minus_old(previous.task, frozen_reward_binding)
phi_before = shared_before + delta_before

if termination_reason:
    phi_after = 0.0  # 包括无效终止观测；不先计算它的 geometry delta
else:
    shared_after = finite(current.task["task_progress_potential"], "shared Phi after")
    delta_after = rr_retention_new_minus_old(current.task, frozen_reward_binding)
    phi_after = shared_after + delta_after

potential = potential_weight * (gamma * phi_after - phi_before)
```

`rr_retention_new_minus_old` 是纯计算，不写 `task`、`groups`、history、supervisor、nominal、owner 或 observation；仅返回原预算内 `(0.85/4)×0.2×(R_new−R_old)` 和独立 audit。值域/资格/配置检查使用与冻结任务 spec 相同的尺度、TOP 计数、force floor 和 rear 模式，启动时精确绑定，不能运行中读一个可变全局配置，也不能复制一组日后悄悄漂移的默认值。可复用纯函数计算现有 retention，但其旧路径数值和 supervisor 调用结果必须完全保持。

这不是另加一个 reward family，也不能在上面替换后再把 `Δr` 额外加入 total，或对每个 120 Hz tick 重复累积同一次 15 Hz potential 差。保留旧 event/time/counterroll 和所有其它 family；普通 phase、rollout 尾与 terminal/bootstrap 规则不动。

审计需明确区分 `shared_observation_potential_before/after`、`reward_potential_before/after`、`RR_old_retention`、`RR_new_retention` 和 `RR_retention_delta`，含同拍 evaluator tick。旧 `potential_before/after` 若继续表示实际 reward Φ，必须声明新 semantics；下游工具不能继续假设它必等于共享观测 Φ。本文表格／JSON 的 `candidate_global_phi` **仅指拟采用的 reward-local Φ，不是修改后的观测值**。

因此真实采用时可称为“reward-only 连续性修订”，但不得改 supervisor 的共用 scalar 后仍声称输入完全相同。当前 RR 几何、接触、recapture、owner 等原始观测已可见，不新增隐藏控制状态，不必为纯 reward 参数增加维度。冻结模型、同一物理状态/历史/RNG 下，改 reward 配置不会自行改变 mean、sigma、raw、log-prob 或 final action；只有之后真正 PPO 更新才能产生学习变化。

## 同轨迹离线数值

| 窗口／时刻 | 当前 RR retention | 候选 RR retention | 当前全局 Φ | 候选全局 Φ |
| --- | ---: | ---: | ---: | ---: |
| 第三回合 75 个 RR-ground 样本范围 | 0 | 0.031907–0.034322 | 0.610938–0.616250 | 0.612324–0.617709 |
| tick6000：实际 TOP/bearing | 0.982825 | 0.982825 | 0.670857 | 0.670857 |
| tick6056：late 开始、仍实际 TOP | 0.979255 | 0.979255 | 0.669847 | 0.669847 |
| tick6064：失去 TOP，合法 AIR | 0.478958 | 0.478958 | 0.627302 | 0.627302 |
| tick6128：仍有资格、越出台面 XY | 0 | 0.157860 | 0.610938 | 0.617647 |
| tick6328：已回地面、无资格 | 0 | 0.031934 | 0.616250 | 0.617607 |
| tick7000：仍地面／障碍接触 | 0 | 0.034319 | 0.616250 | 0.617709 |

75 个 ground 样本的 `ΔΦ` 仅 0.001356–0.001459。按真实连续前后端点、原 γ=0.9985 与 potential_weight=5 离线替换，单步 shaping 差 −0.000154～+0.000131，合计 **−0.000574**；没有把这段失败尾部整体改成获奖轨迹。它是微弱的方向分辨信号，不是增大奖励或成功证据。

掉载损失保留：tick6056→6064 的 retention 0.979255→0.478958 完全不变，原总奖励 −0.218760 也不变。同几何下 TOP 1→AIR 0.5；回地面上限仅 0.125。停在地面不动时，γΦ−Φ 为负；回到同一完整状态的折扣闭环也是非正，不增加历史 placed 或重复接触的独立奖金。

一个必须保留的反例：tick6120→6128，RR 候选 retention 仍下降 0.197063→0.157860，但其他原有项变化可使**总候选奖励**由原 −0.027078 变成 +0.006417。因此不能以“新总 reward 更高”宣称这一步是有效 RR 恢复；必须单列 RR 几何、接触及其他腿 potential。该候选并未修复 cooperative relevance 退出后的其他 proxy 切换，也不恢复缺失的 FL 行程／RL 避障准备项。

## 定向正反例与应用边界

- 固定 gap，ground 落点向合法 XY 靠近：低准备值增加；bearing、lift、placed 和 source 许可不变。
- ground/障碍接触即使反力很大，也不获得 TOP/contact 半份额；历史 placed 不解锁当前资格。
- 同一当前有效 AIR 在 XY 边界和旧负 gap 边界两侧，几何分数连续；不能用这个分数证明摆腿成功。
- 原合法 AIR/TOP 数值保持，包含 TOP 真承载但因其他支撑变化暂时 current_lift_valid=False 的情况。
- TOP 失载、回地面或 unverified force：保留实际损失，AIR 从不计入实测支撑。
- 同位置的 unqualified AIR 不新增主动抬升证据；纯轮爬升／body collision／硬安全保持原终止规则。
- 静止、来回刷同一状态、重复 placed 历史：不得获得新事件奖金；不更新任何历史事件。
- RL 已进入真实合法 swing 或已 placed：走原有 continuation，不能以 RR 重捕获分支拦截。
- 终止 potential 仍为 0；普通 phase 切换、GAE/bootstrap、log-prob 和所有动作通路不变。
- 输入不变正例：对同一完整 frame 分别启用/关闭 reward 修订，`task`、`groups`、编码后的全部 439 输入数值逐项相等（同时覆盖两个 `task_progress` 数值）；只允许 reward/audit 变化。保存/重载冻结 actor/critic 与相同 RNG 的动作证据也须相等。这是后续落地测试要求，本次没有加载模型执行。
- 输入污染反例：若实现写回 `task["task_progress_potential"]`、重建 `groups["task_progress"]`、在 reward 内修改接触/history，测试必须拒绝，不能靠“还是 439 维”通过。
- 双计账反例：检查总 task family 恰为一次 `w[γ(Φ_old_after+Δ_after)−(Φ_old_before+Δ_before)]` 加原 event/time/counterroll；不得再加 `w(γΔ_after−Δ_before)`，不得乘物理 tick 数。
- 终止反例：即使 current geometry 不完整或含 terminal fallback，`reward Φ_after=0`，而 previous 端仍采用同版本 delta；不能先对坏终止几何计算 delta 再掩盖错误，不能把普通 P09→P10 当终止。
- 模式/范围反例：未 placed RR、真实 RL 连续 swing/placed、旧模式关闭时 delta 必须为 0；旧合法 AIR/TOP 的 delta 为 0，既有观测 scalar 对所有模式保持原值。

隔离 scalar 检查已通过（代码内正反例）；并非真实物理验证。上述 reward-only 接入／439 输入不变／模型同输出测试是**待实施验收设计**，本次仅核对调用点并补充文档，未运行新物理或加载模型。若下一合法边界采用，需要独立版本、严格声明 reward 语义变更、保留权重/Adam/LR/normalizer、丢弃旧未完成 rollout、重新采集；保持共享观测的旧 scalar，不以维度不变代替数值语义一致性。仍须先审查新 reward 统计和真实动作，不能先提高 sigma/权重补偿。

复现文件：`rr_recapture_potential_candidate.py`（独立 stdlib 脚本，未被生产 import）；权威离线结果：`rr_recapture_potential_candidate_offline_v2.json`。旧 v1 JSON 只作被修正草稿保留。
