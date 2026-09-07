# 官方 state-dependent log-std：有界集成设计（未实施）

审查日期：2026-09-06。只读源码基准：`64abc5357d00763419fe51c8126db8454fcf7697`。当前 P01 训练保持原配置；本文不是迁移决定、训练门禁或已通过测试的声明。仅 PowerShell 读取本地源码，没有 Python、checkpoint tensor load、Isaac 或生产修改。候选源权重应在本块结束及 fresh P01 evaluation 后，再绑定当时真实完整 checkpoint；不预先锁定旧的28032。

## 1. 版本身份：策略分布改变，物理任务不变

建议独立 `policy_contract`，不把 `semantic_version=v3` 改成新的物理版本：

- `gaussian_scalar_v1`：当前官方 GaussianDistribution，12个 state-independent scalar sigma。
- `heteroscedastic_log_v1`：官方 HeteroscedasticGaussianDistribution，log-std 由观测决定；324 observation / 12 raw action / 原 hidden dimensions、activation、normalization 不变。
- 将 policy contract 与 `runner_config` 同时绑定在新 checkpoint embedded infos、sidecar 和 run manifest；互相不一致必须拒绝。只支持内部枚举到两种官方类的固定映射，不直接执行元数据中的任意类名。
- `runtime_contract` 继续代表整个已提交代码/config inventory，而不是“物理 MDP ID”。策略实现提交会使其 hash 改变，但迁移记录单列 `physical_mdp_changed=false`、`reward_changed=false`、source/target policy contract。全部现用 v3 config、supervisor、reward、env、backend、projection、prior 和冻结 A 的 hash 保持相同；不同时改变课程、动作范围、奖励或终止。
- 新记录可称 `semantic_policy_distribution_migration.v1`，不是 `new_mdp_warm_start`。保留此前真正的 `new_mdp_origin_global_policy_decisions` 及其历史记录，不重新解释或覆盖它。

## 2. 最小实际触点与选型顺序

| 当前入口 | 有限改动责任 |
|---|---|
| `semantic_training.py:97 semantic_runner_config`、`:209 construct_semantic_runner` | 加独立 policy version 输入；缺省保持旧 Gaussian。迁移/reload 不执行 zero-mean initializer。无需修改共享旧 PPO 配置或 env。 |
| `semantic_migration.py:49 checkpoint_metadata` 附近 | 新的 typed plan 构建/校验、source/target policy binding、精确文件 delta、不可变 initial-publication 名称。可放现有模块，不必新增运行时模块。 |
| `semantic_training.py:320 load_semantic_checkpoint` | 分开“旧 Gaussian exact load”“一次性显式分布转换”“新 hetero exact load”；后两者不能误走旧 `_load_v3_warm_start`。 |
| `semantic_cli.py:211 _preflight_checkpoint`、`:614 dispatch_live`、`:254 _evaluation_body` | preflight 从已核验 checkpoint/plan 解析架构；构建 runner 前即确定，不以运行的 semantic_version 猜分布。train 与 fresh eval 都传同一 resolved policy contract。 |
| `semantic_training.py:289 save_semantic_checkpoint`、`:626 train_semantic` | 复用真实 target save/load round trip；保存真实所选 runner config，透传分布迁移 ancestry。不能继续无条件生成旧 Gaussian config。 |
| `semantic_video_cli.py:16 checkpoint_loader` | 从验证后的 checkpoint 自动选型，C deterministic inference 返回 mean；A/B 无 actor，不受策略架构影响。视频不执行转换。 |
| `scripts/run_semantic_ppo.ps1:13/58/68` | 如采用独立 `-PolicyDistributionMigration <plan>`，明确绑定到唯一 CLI 参数，和 NewMdpWarmStart/其他迁移互斥。不加入任意 CliArgs 通道。 |
| `scripts/run_semantic_video.ps1:45` | 发布后的新 checkpoint 自动选型，不需要分布转换 flag；现有 Checkpoint/ResumeMigration 参数可继续承担只读 exact load/明确代码版本兼容用途。 |

具体顺序：校验 source sidecar、checkpoint SHA、seed 和 source runtime → 解析 source policy → 有 typed plan 才选择 target hetero → 构建目标 → 验证源实际 tensor/state → 转换及发布 initial checkpoint → 后续 train/eval/video 从该新 checkpoint 自动 exact resume，不重复执行转换。

历史 checkpoint 没有 policy_contract，兼容仅限其完整 runner_config 明确为旧 Gaussian/scalar/324/12/256×256/ELU/identity；真实 strict source load 再确认结构与 sigma。禁止“缺字段一律当新结构”。新 checkpoint 缺失或自相矛盾的分布身份不得靠 strict=False 猜测。

当前 exact resume 校验不能直接放宽：`semantic_cli.py:228–234` 要求 runtime/config 相符，`semantic_training.py:355–363` 验证后 strict load。当前 `NewMdpWarmStart` 在 `semantic_migration.py:111–130` 明写 new-MDP/新奖励重校准，在 `_load_v3_warm_start:403` 仍按同结构加载；标签和张量加载都不适合本任务。新 typed branch 应保留旧分支原语义。

## 3. Inventory 与新增文件

`semantic_cli.runtime_contract:80–105` 对 `src/wlr50_clean`、`scripts`、`configs`、冻结起点与 pyproject 执行 git clean 检查及 `git ls-files` hash：新增运行时 helper 未跟踪时先拒绝；提交后会自动进入新 inventory。不应把新 helper 放到 outputs/docs 来逃避该绑定。

迁移只接受经过枚举的实际 training/migration/CLI/video/script delta；不是扩大现有 INSTRUMENTATION_FILES 后宣称所有变化只是 instrumentation。若新增 helper，记录 `{before: absent, after: sha256}`；delta 按旧/新 inventory key 并集计算。`semantic_migration._version_bytes:183–185` 直接索引旧 files，因此不得对旧版本不存在的新文件调用它，更不能捏造旧 hash；既有源文件仍须按源 commit 和 hash 查验。生产文件删除或未声明文件变化拒绝。

本报告位于 outputs，现有运行时 inventory 不包含它；不改变本次 live 的 pinned bytes。测试文件也不在该 inventory 选择路径中，未来测试证据应独立记录，不据此声称已运行。

## 4. 权重、Adam、RNG、预算的诚实处理

官方本地 `distribution.py:263–297` 与 `modules/mlp.py:62–70`：新输出层为24行，reshape 为 `[N,2,12]`；mean 前12行。复制全部 hidden 与原 mean `mlp.4.weight/bias` 到前12行；后12行 weight=0、bias=`log(source_sigma)`。原 `distribution.std_param` 唯一有意删除。所有 sigma 必须有限正值；不 clamp、不恢复到 .15。覆盖官方 `log(init_std+1e-7)` 初始化，保留实际 learned sigma。critic 与 identity normalizer 完整复制，固定324预处理亦相同。

`rl_library_wrapper.initialize_zero_mean_actor:680–683` 会清整个最后 Linear，使新 log-std=0、sigma=1；迁移/恢复路径必须跳过它。真正 fresh hetero 初始化才只清 mean 半头。绝不能先转换又清 mean。旧 helper 不必改，可在 semantic 构造入口控制调用。

Adam 不能按位置原样 load：官方 PPO 以 actor.parameters 后接 critic.parameters 建 optimizer（`ppo.py:115–118`），新 actor 删除 std Parameter、扩展 mean Parameter，参数 ID/shape 已改变。也不能宣称旧 sigma 的 Adam moments 原样等于 log-sigma moments。

最小低复杂度选择是明确的 **architecture/optimizer boundary**：网络全部保留，fresh Adam，保留源实际 optimizer learning rate 和其余明确 hyperparameters；不因构造默认值重置到3e-5。这不是完整 Adam 继承，更不是新 MDP。是否接受该优化器 reset 必须写入迁移方案，不能静默执行。

若明确要求保留已有优化器状态，可按经验证的参数名称映射保留 hidden/critic 完整 Adam 状态；扩大 head 的 exp_avg/exp_avg_sq 前12行复制原 mean，后12行置0，并保留原 mean 参数 step；原独立 sigma moments 舍弃。**限制：官方合并24行 head 每个 weight/bias Parameter 仅一个 Adam step（本地 torch/optim/adam.py:166–185），因此新 std 行共享旧 mean 的 step，不能声称 std 半头具有独立从0开始的 Adam。** 这是额外的明确 state transformation，测试和记录成本高于 fresh Adam；不作为已选方案。不为此拆改官方 head 或 PPO。

无论选择哪一明确 Adam 策略：在所有源/目标构造、验证及无采样前向比较结束后恢复源 training RNG，避免初始化消费 RNG；target exact resume 后正常恢复自己保存的 Adam/RNG。保留 global_policy_decisions、ppo_updates、optimizer_steps lifetime、全部 stage_requested_decisions、原 v3 origin、课程 epoch 与来源 hash。迁移本身新增0个 PPO decision/update/step；rollout 与 transition fresh/empty，不继承旧 buffer 或 teacher prefix credit。

initial publication 名称绑定 source global+checkpoint SHA+target runtime hash+target policy-contract hash，记录转换前后 component hashes。整 actor hash 必然因结构变化而不同，不声称 actor hash 相同；已保留的 slices 与 hidden/critic hashes 单独验证。不要覆盖 checkpoint_initial_v3_* 或重复转换已发布目标。

## 5. ABI、最少验证与限制

官方 `PPO.act:144–147` 仍返回12维 raw、保存 old log-prob/value 与 `(mean,std)`；`ppo.py:254–271` 的 ratio/KL 接口不变，storage 懒初始化保存两组 `[T,N,12]` 参数。原 env/projector/GAE/terminal 不改；不能把投影动作拿去算 Gaussian likelihood。`MLPModel.forward:103–107` 的 deterministic 分支已切出 mean，视频不应手工错误取24维输出。

未来最小 CPU 验证：真实官方源 strict load；完整权重映射；真实已存324观测上 mean/value 等价、sigma 与原12值相符、初始 KL 近0（浮点容差而非承诺所有设备逐位相同）；非法 sigma/身份/delta/部分 rollout 负例；target 保存重载与 train/eval/video 自动选型；一轮官方 PPO raw/storage/log-prob 不变且 std 半头能取得有限梯度；预算/RNG和所选 Adam 语义精确核验。没有“先全任务成功才能优化”的新增门槛。

这只使 sigma 有能力依赖观测，初始时仍与源策略一样是常数。它不等于手工按 phase 清零 std，不保证运动探索/停止冲突自行消失，也不把离线分布等价当新物理评估。是否实施仍等待当前固定块和 fresh P01 evaluation。
