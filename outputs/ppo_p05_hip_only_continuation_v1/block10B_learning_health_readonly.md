# Block10B 普通 PPO 学习健康：有限只读检查

范围：CP216960 → CP218496，已封存的 12 次更新（+1536 decisions / +240 Adam steps）；524 个 P09 已执行动作分布，以及 rollout 1666 的 128 个固定真实输入（217601–217728）。本检查不训练、不改模型/生产/配置、不创建 checkpoint；CPU helper 已退出。

## 结论

现有证据不支持“optimizer 没工作”“RR 被全部 mask”或“总体探索关闭”。更符合**原已存在明显非零残差偏置，而这 12 次 PPO 对确定性 RR 均值只作了很小调整**；不能据此把偏置方向认定为悬停的物理原因。RR σ 没有整体变小，P09 也没有普遍 tanh 饱和。不能从裁剪后的梯度范数判断原始 actor 梯度很弱。

## 实际更新与探索

- 所有更新 LR = 1e-5，actor hash 连续且每次实变，记录的梯度全部有限非零。更新 KL 平均 0.01987（0.01416–0.03114），clip fraction 平均 24.23%（19.53–33.44%），surrogate loss 平均 −0.01741。这些是更新诊断，不是成功指标。
- 记录梯度为 actor/critic 分别裁剪后的合范数：约 1.0–1.4142，不是裁剪前 actor-only 范数。连续分布 entropy 为 −26.03 至 −17.38；负值本身不是异常。
- Value loss 在更新 1669 达到 193.3645；其他 11 次均不超过 0.854。这是需保留的 critic/return 尺度异常峰，本检查没有进一步证明其原因或 reward 实现错误。
- 524 个 P09 的 RR raw 条件均值（hip/knee/wheel）平均为 +1.03467/−0.51689/−0.18134，σ 为 0.22558/0.02606/0.04042。各通道 `(raw−μ)/σ` 的实测标准差 0.947–1.031，采样确实发生。
- 样本 `|tanh(raw)| ≥ .95`：RR hip 6.87%，RR knee/wheel 0%；FL wheel 6.30%。`.99` 阈值 RR 三通道均 0%。这仅是描述性统计，非新 guard。

## 固定同输入的均值变化

以下为 `cap * tanh(conditional raw μ)` 的平均变化，是 REQUEST，不是最终下发或实际关节运动。保持相同 389 输入及其 HISTORY，因此隔离了两端参数变化，但不代表闭环轨迹。

| 腿 | Δhip（°） | Δknee（°） | Δwheel（rad/s） |
|---|---:|---:|---:|
| FL | −0.07993 | −0.05989 | −0.000674 |
| FR | −0.02905 | −0.37908 | +0.004048 |
| RL | −0.14761 | +0.03978 | +0.009590 |
| RR | +0.17049 | −0.01667 | −0.000520 |

RR hip REQUEST 由 +15.2895° 到 +15.4600°，knee 由 −17.9574° 到 −17.9741°。RR σ 的平均逐行后/前比为 hip 1.2032、knee 1.1226、wheel 0.9433；因此不能把本块归类为 RR 探索整体收缩。固定输入的完整 Gaussian KL（前→后）平均 0.2582，含其他通道与 σ 变化，不应冒称全部来自 RR 均值。

## 执行约束与统计修正

所有 P09 endpoint 的 residual mask 均为全 1，实际 target-effect 审计通过；RR hip 对同拍 nominal 反事实 target 有效影响比例 99.62%，其余 11 通道均 100%。这证明指令作用而非实际物理跟踪。9/524 个 endpoint 有 FL hip/knee 的 capture-assist owner，未称全部纯 policy。

最后 native tick 的显式 servo headroom clipping：FL knee 380/524 = 72.52%，RR knee 20/524 = 3.82%，其余 0%。RR hip REQUEST 与末 tick filter 输出不同为 15.08%，不能将 filter/HISTORY/rate-limit 差异一律称为硬限。以上都不是全部 8 个 native tick 的占比。

初版 JSON 错把 `native_drive_target_full12`（mapped nominal 输入）当最终 target，**初版的 `final_clamp_slew_or_cast_difference` 与 `final_hard_bound_near` 均无效，不得引用**。原文件保留以明确更正过程。权威结果为 `block10B_learning_health_readonly_corrected.json`：使用 `actual_native_targets` 的物理 rad/rad/s，经同拍 standing pose、servo/wheel 符号转回 canonical deg/rad/s，并先隔离 assist transform。逆映射与既有 final-servo receipt 的误差 <1e-4°。修正无需新的模型 forward。

修正后 post-assist candidate 与最终 dispatch 的差异比例（包含末端 clamp/slew/cast）：RR hip 3/524 = 0.57%，RR knee 2/524 = 0.38%，FL hip 2/524，FR knee/RL knee 各 1/524，其余 0%。真实 canonical 硬限边界只有 FL wheel 9/524 = 1.72%，其余 0；轮硬限按生产值 ±2.094395102 rad/s，不用近似 ±2.1。

524 个 P09 endpoint 的 RR placed 均为 false。本检查不推断“改变某一关节方向即可落脚”，也不据此改安全、噪声或 reward。下一真实 PPO 块/重载视频的结果仍是任务能力证据。

权威 JSON SHA256：`40d50318f0cd078138430540580db533cc63e7bea17b965a151de8fd086febbb`。
