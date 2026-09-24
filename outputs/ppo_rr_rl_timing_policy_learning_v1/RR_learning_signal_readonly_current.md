# RR 学习信号：有界只读核查

核查生产 HEAD：`d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd`。新 P02 422 观察版本保留后腿 `rr_live_swing_evidence_v3`；本核查不修改模型、配置、生产代码或训练数据，没有运行 Python/Torch/Isaac。当前正在运行的 DET 未作为已封存结果使用。

## 结论

**尚无证据证明当前 reward 在普遍阻止 RR 下降或真实承载。更直接的限制是：有效接触前的联合动作样本不足，已有负向 hip 探索与很深的 knee、全身运动混在一起；一次更低的 RR 净空又伴随真实机身碰撞。不能靠继续增大已有的 RR hip 4 倍 sigma 解决这一诊断。**

存在需留意的信用分配竞争：RR 当前抬升资格失去时，其势能份额可下降 `0.03984375`（5 倍 shaping 约 `-0.19922`），远大于单拍几毫米下降的几何增益。但资格包含实际支撑/运动条件，不是 hip 角度门槛；尚未证明这些下降在当前版本是物理有效且遭到错误撤销，故不建议据此立即减弱安全资格或改 reward。

## 已证事实

| 证据 | 实际结果 | 能说明／不能说明 |
|---|---|---|
| v2 P07 封存 512 decisions／4 PPO | P07=1、P08=1、P09=510；RR TOP、承载、placed 全为 0；结束是非终态采样边界 | 确有后腿学习，但没有实际捕获正例；不是任务已失败或成功 |
| 同块晚 P09 窗口 | actual hip ≤ −20° 共 11 行，RR gap 19.549–49.403 mm；knee FINAL 全窗口 −58…−51.979°，104/385 行 ≤ −57° | 负向 hip 已真实到达，不是通道被 mask；并未隔离“绝对 hip −20°附近、近保持当时 FINAL knee”的联合候选，不能证明角度保证接触或保证无效 |
| 同块相邻实际 hip 下降 >0.5° 的 127 对 | gap 降 70 对、升 57 对 | 全身耦合显著，不可把 hip 单关节当确定因果 |
| v2 P10 封存 256 decisions | 34 个真实 RR TOP/承载端点、222 AIR；历史 placed 全 256；下一输入有 221 recapture、35 prep | 新 recapture 状态实际重开，4 倍 hip sigma 恢复；不是只依历史 placed 永久切掉 RR 任务 |
| 同块 retention | TOP 5.383 N、gap 0.0066 mm：0.999869；AIR 0 N、gap 2.1924 mm：0.459687；AIR gap 49.3012 mm：0.168234 | 已保留历史 0.8 信用，同时现有 0.2 项确实反映当前接触/空间接近；旧 v1 的 AIR 满分缺口已修，不应重复报告为当前问题 |
| v3 第一集 | 45 learner decisions，45.958333 s，真实 `BODY_COLLISION`；RR AIR、gap 7.680 mm、0 N、未 TOP | 曾比上述窗口更接近顶面，但同时违反机身碰撞任务条件；不能算有效接触正例，也不能假定是下降本身的错误惩罚 |

v3 第一集原 journal 的 reward 合计：task `-42.60558588`，body `-0.01241333`，contact/smoothness/regularization **均 0**。终拍 reward `-42.94697122`，其中真实碰撞 event `-40`、终态 Phi 清零 `-2.94572230`、时间 `-0.0005`、几何成本 `-0.00074892`；禁止 bootstrap。保存的终拍 raw GAE `-28.89900208`、标准化 advantage `-1.91820610`。这不是“sigma 抖动成本压过任务进度”的证据。第一集失败样本实际保留并进入 PPO；随后 83 行为另一真实前驱后的未终止片段。

## 当前源码含义

- `semantic_supervisor.py::_current_rr_receiver_preparation_retired` 保留已赚取的越沿后 receiver 准备份额；依据真实持续抬升、几何、非落地状态，不能被解释为要求继续回到旧入口姿态。
- `physical_potential` / `_current_capture_progress` 在合法 XY 下，用 `0.5 * 0.025/(0.025+abs(gap)) + 0.5 * current_TOP_fraction` 给 RR 接近及真实接触信用。正 gap 降低会增加几何势能，无 hip/knee 参考角奖励。历史 placed 后的当前 retention 已有上述实测响应。
- `_current_lift_credit` 当前资格丢失仍会撤销当前 lift 份额。这是可能掩盖小幅下降信号的竞争项，但既有历史样本的撤销伴随支撑/短窗口证据不足，不能直接定性判定器错误。
- `semantic_rear_policy_timing.py::public_timing` 在 RL 尚未真实有效 AIR 时重新标记 RR recapture；`rear_dependency` 要当前 RR 承载和前腿桥接，v3 已补齐 RL 当前资格生产者。v3 本次早碰撞块没有 P12/RL 学习样本，不能据它评价 RL 修复成败。
- `semantic_rr_capture_context.py::rr_capture_transfer_context` 的 reachable 是合法 XY、当前 AIR/真实承载、至少两条其他支撑腿、关节余量等代理条件，不是 IK 证明，更不证明 −20° hip 可安全落脚。
- `semantic_reward.py` 使用 `5*(gamma*Phi_after-Phi_before)`，gamma 每个实际 policy decision 一次；真正终态清零 Phi、无 bootstrap，普通阶段切换不 done。单拍负 shaping 不等于下降使折扣总目标变坏；不能忽略后继状态与 GAE 就宣称目标冲突。

## 最多两个下一步学习调整建议（未实施）

1. **使一小部分 RR capture 探索覆盖“hip 向绝对 −20°附近连续移动、knee 近保持该真实入口的 FINAL”的联合动作族，而非再增大独立 hip 噪声。** 它只是待检验候选，不是固定 nominal、奖励角度、成功门槛或必经方法；其他支撑腿和轮仍需协同。现有 knee 深弯/限幅样本不能替代这项联合覆盖。若通过分布修改实施，须版本化、重新采集，raw sample 与真实条件分布 logp 一致；若先用独立控制干预验证，明确诊断身份，不把它当 PPO 样本或纯策略成功。
2. **优先给真实 P07 前驱后的连续接近／接触／保持窗口足够的 on-policy 采样预算，并单列首次接触和丢载 recapture 覆盖，而不是先重调 reward。** 已知当前相关块只有 4 次无接触更新加 1 次含早碰撞更新；P10 块的真实接触保持证据不能伪称从 P01 或未捕获入口学成。保留所有失败、安全终止及跨阶段回报，不从已抬 RR 快照绕过前驱；固定范围看真实接触率、当前支撑、knee 跟踪和碰撞，再决定是否需要窄 reward 修正。没有现有证据支持把机身碰撞惩罚降低。

## 有限证据范围

- `outputs/ppo_rr_rl_timing_policy_learning_v1/recapture_v2_P07_sealed512_readonly.md`
- `outputs/ppo_rr_rl_timing_policy_learning_v1/recapture_v2_P10_sealed256_readonly.md`
- `outputs/ppo_rr_rl_timing_policy_learning_v1/RR_capture_reward_review.md`（仅用于区分已修复的历史 v1 问题）
- `runs/ppo_rr_rl_timing_policy_learning_v1/train/20260923T2202400655573Z_g45862675a18a_4cb1b080d26342afb27a51eb821bbe47/` 中封存的 `completed_episodes.jsonl`、`residual_and_projection_audit.jsonl` 前 45 行及 `advantage_audit.jsonl` 的已存终态统计。未重扫物理逐 tick 历史，未读取活动 DET stream。
