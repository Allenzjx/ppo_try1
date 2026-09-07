# v3 return-horizon 修订：只读最小改动面

状态：**选项评估，未实施、未选择参数、未验证训练改善**。仅阅读当前源码与本机安装的 RSL 源码，未读取 active #32、tensor、checkpoint 或哈希，未运行 Python/PT/GPU/Isaac/测试。本文只新增到 outputs，不改变正在运行的训练。

## 1. 结论与两个具体约束

保持 rollout=128、v2、A、actor 分布/动作/nominal、normalizer、324 维观测与全部任务硬判定不变，在技术上可以独立修订 v3 的回报折扣与 GAE 配置；但不能只改两个常量。

第一，当前 reward loader 有实际失败惩罚覆盖检查。保持现有五族权重、time cost=.02、potential weight=5、failure cost=40 时：

`failure_avoidance_bound = (.4+.2+.1+.0+.02) / 15 / (1−gamma) + 5`

| 候选 | GAE lambda | 当前检查的 bound | `40 > bound` |
|---|---:|---:|---|
| 当前 gamma=.995 | .95 | 14.6 | 通过 |
| 较小候选 gamma=.9985 | .99 | 37 | 通过，保留原 ±40 与检查即可 |
| 原考虑 gamma=.9995 | .99 | 101 | 不通过，当前 loader 将拒绝 |

即使将 `.9995` 的无限时域上界换成 3000 决策有限和，上界仍约 79.5875，大于 40；不能用“任务只有 200 s”绕过。按现有公式，gamma 必须严格小于约 `.9986285714`。因此 **`.9985/.99` 是能保留现有失败惩罚与 guard 的更小候选**；`.9995` 不能在“只改 gamma/lambda”的约束下直接使用。这里不建议删除、放宽或悄悄改写 guard，也不从最坏成本上界推断实际策略行为或成功排序。

第二，`policy_version_from_metadata()` 目前把源 checkpoint 的完整 `runner_config` 与**当前** `semantic_runner_config()` 比较。直接把 v3 factory 改成新 gamma/lambda，旧 heteroscedastic checkpoint 将在 `_preflight_checkpoint()` 调用此验证时被拒绝，发生在 NewMdp 分支之前。因此必须显式识别历史 return-profile，不能把旧参数伪装成当前参数，也不能删掉完整 runner 校验。

## 2. 当前真实接线

| 文件/函数 | 当前作用与需要注意的边界 |
|---|---|
| [semantic_training.py:101](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:101) `semantic_runner_config` | gamma=.995、lam=.95 独立硬编码；v3 仅 LR 不同。`construct_semantic_runner` 调 factory 并缓存 `_semantic_runner_config`。这是 PPO 侧的第二份 gamma 来源。 |
| [semantic_reward.py:25](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:25) `SemanticRewardConfig.gamma/failure_avoidance_bound`、`load_semantic_reward_config` | gamma 来自所选 reward YAML；检查 `(0,1)`、120/15 Hz、200 s、terminal Phi=0、40 大于上述 bound。 |
| [semantic_reward.py:178](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:178) `SemanticRewardCalculator.evaluate` | PBRS 为 `5*(gamma*phi_after−phi_before)`；真实 terminal 强制 phi_after=0；成本仍按真实物理 dt 积分。lambda 不参与 reward。 |
| [semantic_env.py:80](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_env.py:80) `SemanticEpisodeEnv.__init__` | 持有实际 `reward_calculator.config`。CLI v3 显式传 `configs/ppo_semantic_v3/reward_config.yaml`；v2 默认仍指向 v2。当前 N1 没有显式 PPO gamma 与该实际 calculator gamma 的相等断言。 |
| [semantic_policy_distribution.py:80](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_policy_distribution.py:80) `policy_version_from_metadata` | actor/contract 与完整 runner config 严格验证，目前无法区分历史 return-profile 与当前目标 profile。actor policy version 本身不应因 gamma/lambda 改名。 |
| [semantic_migration.py:61](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_migration.py:61) `build_v3_warm_start_record` | 已绑定源/目标 runtime、六份配置及改动文件，保留 frozen A、物理频率、200 s、no-timeout-bootstrap、runtime 版本、324 preprocessing；适合已有 NewMdp 权重兼容路径，不适合 exact-MDP resume。 |
| [semantic_training.py:511](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:511) `_load_v3_warm_start` | 目标 factory 检查、新空 storage；严格加载/证明源 actor、critic、Adam、normalizer，然后明确丢弃旧 Adam，以 3e−5 新建；恢复源 RNG，保留 lifetime/budget/origin。 |
| [semantic_cli.py:80](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_cli.py:80) `runtime_contract`、`_preflight_checkpoint`、`dispatch_live` | 已记录完整已提交源码/配置清单与 content SHA；v3 selected_configuration 绑定 reward YAML。NewMdp 发布独立 immutable initial checkpoint，写 `new_mdp_warm_start.json`，不覆盖旧 checkpoint。 |

本机 [RSL PPO.compute_returns](C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl/algorithms/ppo.py:187) 直接使用 `delta=r+not_done*gamma*V_next−V`、`A=delta+not_done*gamma*lam*A_next`；rollout 末非 terminal 用真实末 observation 的 critic，真实 terminal 切断递推。其 `PPO.load()` 加载 actor/critic/optimizer 等 state，不把 source gamma/lam 覆盖回目标 algorithm。无需复制/修改官方 GAE 或 frozen `rl_library_wrapper.py`。

## 3. 最小实现设计（仅候选）

### gamma 单一来源，lambda 明确是估计器配置

1. **目标 v3 的唯一运行 gamma 数值**仍放在版本化 `configs/ppo_semantic_v3/reward_config.yaml`，例如待批准的 `.9985`；新增明确 return-profile/revision 标识，将该标识严格关联到 `.9985/.99`。v2 文件与缺新模式的历史 v3 profile 仍为 `.995/.95`。不提供任意 CLI gamma/lambda 调参绕过契约。
2. factory 通过同一版本选择与 reward config loader 解析目标 gamma，不再给 v3 PPO 写另一份独立数值。lambda 属于 factory 的已识别 estimator profile，不进入 `SemanticRewardCalculator.evaluate`，也不能用来改 PBRS 折扣。
3. 构造 runner 后及开始训练/加载完成时，检查实际 `runner.alg.gamma == env.core.reward_calculator.config.gamma == cached_runner_config['algorithm']['gamma']`，并验证实际 lambda 与已识别 profile 一致；配置不匹配应在采样/optimizer 前拒绝。`PrefixCreditCore.__getattr__` 已透传真实 core，因此同一检查可覆盖现有 FSM/checkpoint prefix，不必加新历史或改变 reset。
4. 保持一次 policy decision 折扣约定：8 ticks 普通动作只用一次 gamma；短 terminal interval 的真实 dt 仅用于成本积分，next Phi=0/no-bootstrap 不变；不能把新 gamma 额外取 1/8 次幂，也不能按 rollout 重置 potential。
5. N8 现有 `bind_final_value_function(..., gamma=runner.alg.gamma)` 已拒绝与 env gamma 不同；其当前默认 reward loader 为 v2。此任务不顺带开放/改写 v3 N8，保留原拒绝与 v2 行为即可。

这次会改变 PBRS 数值和折扣回报目标，lambda 又改变 GAE 估计器的 bias/variance 与传播权重；它不是纯仪表修复。128 rollout 未变，跨更远未来仍依赖末值估计，不可声称仅把 lambda 调高就已直接获得全部后续真实奖励。

### 源历史配置与目标配置不能混用

- 将完整 runner 校验中的 return-profile 选择显式区分 **历史源**和**当前目标**。历史识别只接受已知完整 `.995/.95` profile（旧 v3 无新字段）及已版本化的新 profile；其余 actor、critic、normalization、optimizer factory 设置继续完整逐字段校验。不要接受任意 metadata 的 gamma/lam，也不要把 heteroscedastic 缺 contract 当旧兼容。
- 对迁移源，利用现有 source runtime 的 reward-config 路径/SHA 与 `_version_bytes` 恢复的历史字节，验证源 runner gamma 与源 reward gamma 一致；lambda 必须属于该历史已知 estimator profile。目标则绑定当前实际 config。不能读当前 YAML 去解释旧 checkpoint 的 gamma。
- `_preflight_checkpoint()` 识别源 actor 类型后，只允许已有 `--new-mdp-warm-start` 进入此奖励版本边界；无 flag 的普通 resume、旧 `--resume-migration` exact-MDP 路径继续拒绝 runtime/return-profile 改变。无需另造 policy-distribution migration，actor policy contract 不变。
- `build_v3_warm_start_record` 最小增加可读的 source/target gamma、lambda、profile 与 reward config 绑定摘要（例如 `return_horizon_transition`），保留现有 exact file-diff/full contract/frozen A 校验，不做全局 hash relaxation。迁移目标若仍是 `.9995`，必须让现有 reward guard 正常拒绝。
- 新目标加载仍走 `_load_v3_warm_start`：actor 含 hetero std、critic、normalizer、RNG、计数和 stage/origin 保留；Adam moments 清空、初始 LR=3e−5；128×1×12/324 storage step=0、无 pending transition；legal reset，不继承物理状态/旧 rollout。旧 critic 权重只是兼容 warm start，不能声称其旧值函数已经校准到新折扣。
- `save_semantic_checkpoint` 已以 runner 缓存的真实 config 覆盖 infos 中旧 config；CLI initial 与后续 train/save 也已有目标 runtime/runner 配置写入。仍应回归确认 initial、updates/result、checkpoint roundtrip 和下一次普通 resume 都识别新 profile。same-state action comparison 若通过，只证明动作接线不变，不能称新旧 reward/return/trajectory 等价。

## 4. 有限改动文件与测试

预计最小生产面为：

- `configs/ppo_semantic_v3/reward_config.yaml`：gamma 与明确 profile/revision；保留 ±40、五族权重和现有 guard。
- `semantic_reward.py`：新 profile 严格验证/共享读取，PBRS 公式与 failure bound 不变。
- `semantic_training.py`：目标 gamma 单源、已知历史 factory profile 支持、实际 runner/calculator 一致断言。
- `semantic_policy_distribution.py`：识别历史完整 return-profile，不放宽 actor 或其它 runner 字段。
- `semantic_migration.py`：源 reward 字节与 source runner 参数的显式绑定，以及新旧 return-profile 的可读迁移记录。

CLI/PS 已有 NewMdp flag，env 已能访问实际 reward config；原则上不需改它们的行为。若选择新增顶层 metadata 标签，再做最少透传，不引入第二套算法配置。无需改 supervisor、execution profile、mapper、hard task evaluator、normalizer、obs/action schema、官方 RSL 或冻结 A 文件。

足够的有限 CPU 回归（此处未运行）：

1. 扩展 `test_semantic_observation_reward_env.py`：v2 `.995` 旧结果保持；新 v3 `.9985` 的 bound=37<40；`.9995` 被拒绝；NaN/bool/非法 profile、源目标 gamma 混用拒绝。真实 terminal/短 tick Phi=0；静止/闭环的折扣 PBRS 望远镜等式按新 gamma 成立；五族、±40 和 hard task outcome 不变。
2. 新增一个小 return-profile 测试或扩展 `test_semantic_policy_distribution.py`：旧完整 metadata 仍被准确识别，新 metadata 被准确识别，未知 gamma/lambda/混合 profile/其余 runner 字段篡改仍拒绝。保留原 `changed_gamma` 负例，不改成忽略 gamma。
3. 扩展 `test_semantic_training.py`：实际 N1 calculator/PPO/cache gamma 一致；故意不匹配在 `alg.act/process_env_step/optimizer` 前停止；官方 128-step GAE 对新 gamma/lambda 的手算短例、真实 terminal 与非terminal末值 bootstrap 分别核验。lambda 改变不直接改变同状态 PBRS。
4. 扩展 `test_semantic_hetero_mdp_continuation.py` 与 `test_semantic_v3_continuation.py`：真实 CPU hetero 旧 profile→显式 NewMdp 新 profile→immutable initial→exact reload→一次 fresh update；证明全网络/normalizer/RNG/计数预算保留，fresh Adam/storage，实际 algorithm gamma/lam 与 initial/save metadata 正确。无 migration、重复 initial、源 reward SHA/参数不一致继续拒绝。

未新增实机成功门、baseline 重跑门或新的训练审计框架。是否选择 `.9985/.99` 应在 #32 与随后固定 P01 评估结束后决定；本报告不预测它能改善哪一阶段。
