# Block09 rollout 1648：RR v2 的实际消费

结论：**新 v2 已用于该块真实奖励，但该块没有遇到 v2 新增退休条件与 v1 分歧的状态。** 不能把“v2 配置下采了 128 步”说成“128 步都得到新增奖励”。

范围仅封存 `rollout_001648.pt` 的 128 步（global **215297–215424**），加同回合前一端点 215296；输入 tick 5408–6424，下一状态 5416–6432。全为 P09、无 done。这是第二回合 P04 checkpoint-policy-prefix 课程的真实 learner 数据，不是新重放／新 credit。

## 已核对的实际接线

- rollout runtime 的 task-spec 哈希与当前 v2 文件逐字节一致。
- 128 条 stored action/reward/old-logp/done 与同期 audit 对齐；policy request currentQ 与前一实际端点一致。
- 128 条 `potential_before/after` 和 PBRS 均与当前生产 `physical_potential` 相符：`5 × (0.9985 Phi_next − Phi_input)`。只离线比较显式 v1 mode；未改变 evaluator/history 对象、buffer、GAE 或优化器。

| 状态集合 | 数量 | currentQ true / false | v1 退休 | v2 退休 | 非零 Phi 差 |
|---|---:|---:|---:|---:|---:|
| 全部实际输入 | 128 | 128 / 0 | 55 | 55 | 0 |
| 全部下一状态 | 128 | 128 / 0 | 56 | 56 | 0 |
| 已 cross 的实际输入 | 55 | 55 / 0 | 55 | 55 | 0 |

v1→v2 的单步 shaping 差也全部严格为零。这里没有“currentQ 因 body-control 短暂 false 而 receiver 信用重新启用”的状态；因此不能从本块证明 v2 修复改变了动作或解决 hover。

## RR 触碰、越沿与落脚

RR 历史 lift=5359，cross=**5988**，没有 placed。唯一保存的 RR obstacle-contact 端点是 **5976 / global 215367**：TOP surface、bearing **9.0859 N**，但 front_distance=**−6.5742 mm**、within_top_xy=false、qualified top_contact=false、top_count=0；这是实际短暂顶面接触，不是合格捕获。

cross 后的 56 个保存端点均无 TOP 接触／placed，gap 范围 **8.6769–81.3169 mm**，末值 **65.6619 mm**。未保存每个物理 tick 的完整 RR contact 状态，因此不声称不存在端点之间的瞬时接触；实际 placed 历史始终 false。

## 下降与上升：局部 capture 信号存在，但不等于总回报

只取已经 cross、相邻两端点都是 AIR 的实际转移；下降／上升按测量 gap 变化分类，不由膝角推断。局部 capture 项直接使用生产函数，Phi 系数为 `.85/4 × .2`；当前 RR 几何半份随净空减小而增大，真实 contact 半份未取得。

| 转移 | 步数 | 平均 gap 变化 (mm) | RR capture PBRS 均值 | 全局 PBRS 均值 | 总 reward 均值 |
|---|---:|---:|---:|---:|---:|
| post-cross 下降 | 23 | −2.3798 | +0.000942 | −0.001382 | −0.002715 |
| post-cross 上升 | 32 | +3.4913 | −0.002308 | −0.009267 | −0.010600 |
| 首次 cross 所在步 | 1 | +8.6136 | +0.078756 | +0.079995 | +0.078662 |

下降局部 capture 累计 +0.021664，上升累计 −0.073849；下降也有极小一步为 −0.000049（折扣存在）。全局 Phi 还包含其他腿和 RR 其他子目标，所以不能要求下降每步总 reward 必须为正，亦不能把分组相关性当作某关节的因果梯度。

该块全部实际加权 quality family（body/contact/smoothness/regularization）为 **0**，几何成本也是 0；每步时间成本 −0.00133333，无 terminal event。因此这 128 步的下降负总回报**不是稳定性质量惩罚压过下降奖励**，而是全局进展合成与时间／折扣共同作用。未做 reward 调参或新设计建议。

[结构化证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_p05_hip_only_continuation_v1/block09_rollout8_RR_v2_readonly.json)；rollout SHA256 `82fecce6ec643f1f490d6786c6e88d429243e7b5f6308edd29c46c348045dc72`。全部比较仅使用该封存块与一个前端点，不涉及其他回合或新物理。

