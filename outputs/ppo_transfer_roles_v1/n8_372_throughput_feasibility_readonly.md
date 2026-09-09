# N8 / 372 同布局续训：只读有界结论

结论：**不值得当前切换；继续 N1 无需等待本项。** 现成 N8 能复用八行场景、独立逻辑历史、单次批量写入/physics step 和 peer-bootstrap，但它是旧 v2/P01 路径，不是当前已训练 HISTORY372 控制合同的开关。只删 N1 限制或改 observation dimension 会漏掉实际动作/任务差异。未运行 Python、测试、Torch、Isaac 或基准；未扫描 Run9/任何原始轨迹，未修改生产。

## 已存在与实际缺口

| 范围 | 源码事实与最小兼容工作 |
|---|---|
| 配置/策略入口 | [semantic_cli.py:148](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_cli.py:148) 显式拒绝 v3 N≠1；`dispatch_vector` 不传当前 execution/task/observation/reward 路径。`SemanticVectorRslEnv.__init__` 无参加载 v2 schema/reward，backend 的 controller 也使用 v2 task spec。必须完整传入当前四份配置、v3/experiment 元数据，而非只接372。 |
| HISTORY372 网络 | [semantic_vector_training.py:8](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_vector_training.py:8) factory 没传 policy_version/layout，env.cfg 无 semantic_version，实际默认旧 Gaussian/v2。批量 encoder 使用 `self.schema.dimension`，本身不硬编码324；当前 HISTORY actor/372 encoder 无需另造网络，但需显式选型、沿用相同 role schema/每行 evaluator。 |
| 当前真实控制 | [semantic_vector_backend.py:89](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_vector_backend.py:89) 拒绝当前 headroom 模式。继承的批量 apply 仅作 nominal mapper→combined bias→硬限/slew，不执行当前 nominal geometry、上一 ACK REQUEST tracking reference、same-tick headroom及其证据。须在 semantic-only 批量适配层逐行接入相同纯 helper、独立前 ACK/反馈时钟，再统一一次物理写入；不能关闭这些功能冒充同 MDP。 |
| 行视图/原生审计 | 旧行 facade 已切分目标四缓冲区并支持真实 mapper/audit，不能说原生验证完全没有。新增 geometry 所需 `root_physx_view` Jacobian/fixed-base 等接口当前 facade 不提供；需正确行切片和 COM/link 世界/局部原点一致。tracking capture 要单行 q、连续前 ACK 和 reset bootstrap，当前 `_RowMapperAdapter` 只有 mapper/final-drive等基本字段。新增三反事实/headroom/reference 的独立重建必须保持，不能让每行 wrapper 各自 physics/write。 |
| 同布局迁移 | [semantic_migration.py:770](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_migration.py:770) 的372 topology只允许N1，`source_num_envs` 对v3也固定N1；`build_execution_factor`/smoke 路径仍绑定旧324/v2。 [semantic_training.py:457](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:457) exact-migration storage检查为324，runner比较未带 layout。须显式扩展同372的1→8 topology/runtime迁移，不能重做324→372 append或泛放宽源检查。 |

因此最少涉及 `semantic_cli.py`、`semantic_vector_backend.py`、`semantic_vector_env.py`、`semantic_vector_training.py`、`semantic_migration.py`、`semantic_training.py`，另需审计/行 facade 接口适配与回归；冻结 mapper、reset 原语及现有 N1 分支应保持。只做合法 P01 N8 不必重写整个 reset；**现有后缀 reset/prefix 课程并未在 N8 实现**，若要求直接搬当前 P06/P10 后缀训练，已超出这个小兼容范围。

## reset 和长后腿片段

[semantic_vector_env.py:260](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_vector_env.py:260)：任一行真实终止，八行同时 done；未终止 peers 的 reward 先加 `gamma * V(真实 reset 前 final observation)`，其 PBRS final potential不清零，随后整体 reset，GAE不会越界接入新episode。真实失败/成功行不享受该 peer bootstrap，`time_outs=False`；普通 phase handoff 不触发 barrier。已有 `test_semantic_vector_draft.py` 明确测试 final-obs 而非 reset-obs、失败行与7个 peers分账及官方RSL更新，**这些旧测试不是当前372物理等价性证明**。

[vectorized_isaac_backend.py:1096](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/vectorized_isaac_backend.py:1096)：后续 reset 是整场景 hard reset/STOP-PLAY及view重新初始化，再重建各行 mapper/controller/contact历史并 settle；明确 `partial_subset_reset_supported=False`。独立控制器/reader/bridge/role历史不等于独立物理 reset。peer-bootstrap只补价值目标，**不能保留被切断的后腿接触/transfer窗口，也不能产生没有实际采到的后腿样本**。

小摘要交叉证据：已完成372自然P01 block7（130304→131840，1536决策）四个完整回合分别370/358/313/370决策，均P06终止（2 collision、2 fall），另125决策P02非terminal尾。仅说明当前策略存在较早终止；不能把这四次N1顺序样本当N8同时运行、估算实际peer-reset频率或速度。结构上若一行较早失败，其余已走到更晚阶段的行必被截断，当前后腿重点因此有明确采样风险。

## 迁移/吞吐判断

若以后只做同物理MDP、同372/schema、同 HISTORY核/权重/normalizer/return-profile/actions 的N8，合理边界是**显式 topology migration，保留已验证 Adam/RNG/计数并新建空 rollout、合法全P01 reset**；不是无声明 ordinary resume，也不是改 reward 的 NewMdpWarmStart。每次 update 从128总样本变1024（128×8），梯度/minibatch规模及采样分布不同，不能宣称训练轨迹等价。现有全局reset成本仍在，物理step虽然可批量化，逐行Python控制/geometry/native审计仍需执行；无计时数据不能承诺8倍或任何净提速。

当前没有低风险、已接通的372 N8收益可直接领取。保留这个可复用骨架作后续独立工程项即可，**不新增A/B任务成功门、不把N8或其接口检查作为继续N1训练的条件**。
