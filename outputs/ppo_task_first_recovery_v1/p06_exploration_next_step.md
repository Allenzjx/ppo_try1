# 后腿课程：下一步单因素建议

结论：已有足够依据，在当前完整评估结束后的独立、显式版本迁移中，试验 **innovation 温度0.5→0.25，单个128-decision P06块**。不建议仅为等到后腿而先把相同0.5设置扩大为长块。此建议不是已证实的唯一碰撞根因，也不保证低温能通过 P06。当前生产、配置和正在运行的训练均未修改；未建立新分支。

证据范围仅为已封存 CP176896 的128个真实样本：成功 N 从 P01 roll-in，P06首段80 decisions在3317 tick机身碰撞，重置后第二段48 decisions未终止；没有 P07–P13样本。数据来自该次 update1347 的原 rollout 与同步审计，不是重算 critic、GAE 或回报。

## Mean、创新和 HISTORY 是三件事

实际采样为 `raw = 0.1 × base_mean + 0.9 × previous_raw + innovation`。saved raw与日志逐项一致，stored HISTORY与前一raw逐项一致（误差均0）；两次 N-prefix 后的12维 HISTORY均为0，没有旧episode串入的证据。下表完全由已保存的 conditional mean/sigma、raw与观测分解；没有新网络forward。

| 失败段关键拍 | base mean贡献 | HISTORY贡献 | innovation | raw |
|---|---:|---:|---:|---:|
| FL wheel，tick3192 | −.00268 | 1.45594 | 3.16945 | 4.62272 |
| FL wheel，tick3200 | −.00286 | 4.16044 | 1.41138 | 5.56896 |
| FR knee，tick3200 | .01135 | 2.36792 | .85379 | 3.23306 |

FL wheel在3192的有效sigma为1.76073，创新约1.80σ，不是需要异常随机数才出现的事件。其后大conditional mean主要是上次随机动作被rho=.9延续，不能称为base mean已经学出巨大同向输出。降低温度只缩小新的随机创新，不删除 HISTORY，也不会立即抹去已经形成的偏置。

失败80 decisions中，9个decision存在至少一个 `|raw|>3`（对应 `|tanh(raw)|>.995`）：FL wheel9次、FR knee2次、FR hip1次，存在重叠。20个decision有servo `|tanh(raw)|>.95`，13个有wheel接近该范围。状态相关有效sigma峰值：FL wheel2.27984、FR hip1.54098、FL hip1.50195、FR knee .98058。第二段48 decisions另有1次FL hip `|raw|>3`，该段不是成功episode。

## 最早的支撑变化先于饱和

P06第2个decision结束2696，支撑由4腿变为FL/RR，FR/RL暂无承载；FR knee residual已由+4°升至+8°，当拍raw仅.45716。到2720四腿又有承载。两腿支撑本身不是任务失败，也不能把这一短暂卸载全归于轮差速或视为新硬门槛。

后段有实际servo大幅偏离，而不只是wheel视觉现象：

| tick | FR knee target ° | FR knee actual °（前1物理tick） | FL wheel最终target rad/s | 机身底部world z m |
|---|---:|---:|---:|---:|
| 3160 | 114.4145 | 108.0558 | −.4585 | .14348 |
| 3192，首个raw饱和 | 130.4145 | 124.0032 | −.2185 | .13323 |
| 3208 | 138.4145 | 131.9997 | .0215 | .12208 |
| 3280 | 153.2342 | 154.4506 | 1.1015 | .09946 |
| 3317，碰撞 | 134.7342 | 140.6895 | 1.3095 | .03922 |

FL wheel raw从负值骤转大正值时，残差slew仍在保留先前负修正，所以3192/3200的最终target仍负或近零，不能把 raw峰值当作当拍已执行速度。失败段wheel raw差速分量RMS .747，大于共同分量RMS .439；末拍四轮最终target为 `[1.3095, .9280, −.2881, .2780]`，同步实测 `[1.2817, .9433, −.2895, .3003]`。这是真实混合方向/幅度，不是仅RR轮有指令，但不能由raw分量推断净地面牵引。

因此证据支持“宽创新经HISTORY持续、servo与wheel执行历史共同作用”，不支持“只改wheel mask”“只降FR knee权重”或“饱和是所有支撑变化的第一原因”。P06还未进入P09 geometry，本段不是已修P09几何重复叠加的物理验证。

## 单因素试验边界

- 从届时最新兼容checkpoint保留所有权重、sigma行、Adam、固定normalizer；仅版本化temperature 0.25，sampling/logprob/entropy/KL一致使用该值；rho=.9、均值核、动作scale、限速、N、reward、采样课程与学习率不变。
- 当前actor构造器明确只接受0.5；不能偷偷传0.25或只改采样器，必须单独支持并记录新policy contract/迁移，清空未完成旧rollout，再合法reset重新从成功N P01-prefix进入P06。
- 先128 decisions，不设全程成功门禁。记录实际P06→P07/P08/P09样本量、raw/tanh饱和频度、有效sigma、前腿支撑与机身净空、终止原因；若仍早期失败，再看第一差异，不叠加reward/HISTORY/scale变更。
- 本次证据针对当时CP176768采样策略；最新网络已更新，低温收益需要新短块验证。部分episode/后缀通过不等于从P01完整成功。

详细逐通道分解与关键样本：`p06_176896_exploration_decomposition_with_actual.json`。原始较早分解 `p06_176896_exploration_decomposition.json` 保留未覆盖。脚本：`check_p06_exploration_176896.py`。

## 只读核对：0.25候选的最小实施范围

先等待当前正式评估；若 M3 达成，先保留 best，再决定是否做后腿低温课程。以下只是待授权实现范围，没有改动生产。

现有 `exploration_temperature_factor` 不能直接用于0.5→0.25：它在 `semantic_migration.py::_build_exploration_temperature_plan`（约1774行）只接受原 HISTORY1.0→TEMPERED0.5，且只绑定 `fsm_reference_p09_stable_v2`；当前工程为 `task_first_recovery_v1`。loader约487行也固定该版本对。直接修改温度数字将被拒绝，这是正常保护。

最小建议为4份生产文件，复用当前迁移/CLI/同一namespace，不新增训练框架：

1. `semantic_history_actor.py`：保留旧0.5 actor，增加固定0.25的窄子类或等效显式版本。共享原conditional mean和完整Gaussian sampling/logprob路径，保持state_dict布局；不得改sigma权重或rho。
2. `semantic_policy_distribution.py`：注册唯一的新0.25 policy version/class/contract，明确 `effective_log_std=learned_log_std+log(0.25)`；旧0.5 contract仍原样可加载，metadata解析不能混淆二者。
3. `semantic_migration.py`：扩展现有温度因子仅认可精确的 `.5→.25 + task_first_recovery_v1` 版本对，旧1→.5迁移仍通过；6份config bytes完全相同，reviewed code范围仅上述actor/contract及必要loader。因子继续保存全部权重/Adam/RNG/normalizer/counters，丢弃未完成rollout；不能与reward、执行修复或新mean-head reset混在一次迁移。
4. `semantic_training.py`：`semantic_runner_config`认可新版本；温度因子validator核验这一个新版本对并实例化目标。现有 `semantic_cli.py::_preflight_checkpoint` 已从验证后的target contract读取policy version，预计不需改CLI/PS参数。

定向验证复用现有 tempered actor、temperature migration、loader/CLI 测试：旧版本兼容；同权重同观测deterministic mean完全相同；新有效sigma恰为旧的一半；sampling/old logprob/entropy/KL及update采用同一有效sigma；checkpoint加载后所有可学习状态与Adam/RNG精确保留；其他文件/config/因子混改拒绝。随后只运行先述短128-decision真实块，不能把单步固定观测分布比较称作物理收益。
