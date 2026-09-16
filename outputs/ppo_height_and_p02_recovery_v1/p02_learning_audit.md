# P02 学习路径只读审计

范围：当前生产代码及已完成 P01 512 决策块（167425–167936）；不读活动 zero、不加载 checkpoint/rollout tensor、不重哈希、不改生产。数据细项见 [p02_training_reward_audit.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p02_training_reward_audit.json)。

## 372 实际可见性

固定 schema 先除工程 scale、再 clip ±20；RSL actor/critic normalizer 均为 Identity，没有评估数据拟合。两者接收相同 372 数值。网络是 MLP，不是 RNN；HISTORY 只是观测 [195,207) 的上一实际 raw，在网络输入和固定均值核中都可见：mean = 0.1×base_mean + 0.9×previous_raw，sigma 不被该核缩放。普通切阶段不重置这一历史。

- 直接可见：阶段/进度/时间；FR 净空、前距、载荷代理；8 关节实测位置/速度、轮速、姿态/速度/CoM；ground/obstacle 两对法向力与 active；Q/C/P 四腿位 [146,158)；当前 nominal [171,183)、上一 dispatch/mapper 状态；4 腿各 12 个 role 摘要 [324,372)，含 motion_fraction、preparation/transfer_ready、方向、支撑连续比例和窗口成熟度。
- 可由现有字段组合得到：FR 的 Q 已获得但 P 未完成、当下 AIR（两对 active 均 false）、净空是否 ≥15 mm、纵向接近前缘、载荷代理变化趋势。HISTORY/相邻 applied 是有限动作信息，不是完整物理时间窗。
- **未直接编码**：active_attempt；TOP/FRONT_WALL 等 surface 分类与逐腿 bearing_verified/load_fraction_valid；精确 placement event tick/捕获后年龄；consecutive_top_samples；轮中心横向位置/within_lateral_span（至多由关节/机身和学习到的运动学间接推断）。role.valid 不是 normalized_load_valid，wheel_load_fraction 还是原始接触法向力比，不能把其非零当作已验证承载。
- pending_capture 本身不在 48 role 字段内；Q、P、ground 已可见，但横向条件未直接编码。新 body carry allowance 的 active_attempt/真实支持分类与捕获后 0.5 s 退出年龄也不是单个 372 的完整显式函数。历史 Q/P 位不能替代当前有效性或事件年龄。
- nominal layer 起点/挂起时钟、捕获 owner retirement、P09 height recovery offset 有内部状态；当前输出 nominal 可见，不等于所有将来调度状态可反推。P07–P09 的新状态不能倒推为旧 P02 首次失稳原因。

证据：[semantic_observation.py:123](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_observation.py:123)、[:250](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_observation.py:250)、[role 字段:23](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_transfer_roles.py:23)、[pending:254](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_transfer_roles.py:254)、[carry:38](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:38)、[HISTORY:61](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_history_actor.py:61)。

## 实际 512 决策数据

P02 219 个决策（100 正、119 负），总奖 +0.141173；progress +0.620078、body −0.062252、smoothness −0.415182。Q+AIR+横向有效+纵向区域内+净空 ≥15 mm 的 190 端点合计 −0.411088，其中 progress +0.007013、body −0.057258、smoothness −0.359460。这是保持/前送中的成本与进度组合，不证明网络偏好降低身体或失去净空。

以相邻 P02 **决策端点**比较，净空下降 97 点合计 −0.499757，未下降 120 点 +0.620387（两个首点不分组）。两次自然 P01 回合的 FR 峰值为 130.435/141.100 mm；P02→P03 末端仍为 96.018/54.288 mm。唯一向下穿 15 mm 的端点 448→456 tick，净空 20.256→9.787 mm，回报 −0.010985→−0.002959；后点全身 potential 仍略正，不是 FR 单项净空奖励。端点统计不能描述区间内每个 120 Hz tick，也不是同状态反事实或因果分解。

该块真实终态为决策 167829 / P06 / BODY_COLLISION：reward −42.247150，终态 event −40，potential −2.241000，保留真正 terminal observation、bootstrap=false。其后自然重置继续采集；块末 P03 未终止不能称为完成越障。

## 信用与优化一致性

raw Gaussian sample、old mean/std/log-prob/value 均在物理步之前固定；store 和 update 都断言使用同一个 raw，未把 projected/最终目标冒充 likelihood action。phase 改变不产生 done；8 个真实 tick 积分为一决策奖励，短安全终态照实记录。task terminal 将下一 potential 置零，time_outs=false，GAE 用 done 阻断跨 episode；普通 rollout 尾用 critic bootstrap，128 长度并不把阶段切断。

P01 块实际 gamma=.9985、lambda=.99、rollout128×1、5 epochs×4 minibatches；4 次更新实际 LR 均 1e−5，KL 0.013499–0.018621，clip fraction 0.1625–0.259375，actor 均改变且梯度有限。末次 value loss 21.819953 较前三次升高；不能单凭它指认终态 credit 错误。已完成块的普通阶段切换记录均 done=false。

512 行 audit 只存 old_value 与 signed reward，**没有 PPO returns/advantages**。因此本次实际 advantage 分布/阶段符号归因为 unknown；episode_return 不是 advantage。代码按整个 rollout 标准化 advantages（不是每 minibatch），之后用真实旧 raw likelihood ratio 做 PPO clipping；不能用后续已更新 critic 的 old_value 伪造原 rollout 尾 bootstrap。

证据：[semantic_env.py:148](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_env.py:148)、[:196](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_env.py:196)、[terminal:234](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:234)、[采集/断言:1189](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:1189)、[实际 RSL GAE:187](C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl/algorithms/ppo.py:187)、[已完成配置/更新](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_timing_task_priority_v1/training_P01_actual.json)。

## 可选后续优先级（本次均未实施）

1. 先完成已授权 same-reset wheel4-off、必要时 FL-knee-off 诊断并保留实际 raw HISTORY；它们直接分离已观测到的早期全身控制影响。没有证据支持先改 rho/sigma 或再做一次 carry reward 折扣。
2. 下次现有训练块仅补只读 rollout 级 signed advantage/return/阶段聚合，区分 raw GAE 与标准化 advantage，连同当前 FR 净空、Q/C/P、旧值和末尾 bootstrap 来源；保留 natural P01 与后续 FR capture 数据。当前块 P01 仅4、P03仅19样本，不能把“512 P01课程”当成512个早期阶段样本。
3. 若后续匹配观察实证显示不同有效接触/捕获年龄需要不同动作，再做显式版本化的最小观察状态补充；不能悄悄改写现有372位语义或宣称已严格 Markov。该风险目前不是已证明的 RR 失稳原因，也不是 optimizer 启动新门禁。

未发现现行 raw-likelihood、普通阶段 done 或 terminal bootstrap 的明确实现损坏；没有因此修改生产、物理参数或验收规则。

统计限定：上述 190 点是写明条件的净空分组，不是逐 tick carry helper 的实际 allowance 重算；未将缺失支持/attempt 条件擅自补成有效。

后续 mask 根因核查只使用该次 getter 与同一 tick 的请求→映射→最终 target、actual q/qd、接触承载/运动证据：检查重复应用、单位/符号异常；若 implicit PD estimate 持续达到 max-force，只能称“估算驱动需求/裁剪到限”，不能称独立测得的 PhysX 关节扭矩饱和。每 8 tick 的 effort 记录不能证明中间全时段饱和；需区分估计公式的 pre/post-step 时钟，不补算未知量，不宣称完整载荷或力矩因果分解。
