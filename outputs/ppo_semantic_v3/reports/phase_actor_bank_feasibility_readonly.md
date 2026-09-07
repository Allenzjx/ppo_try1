# Phase-routed actor bank：只读可行性审查

范围：当前68631e9工程与本机 official RSL-RL 5.0.1 的源代码审查；没有运行 Python/Isaac/测试，没有改生产、环境或正在执行的69590训练。这不是实施、选型或新优化器门禁。现有不同课程之后的行为差异**没有构成受控遗忘证明**，本报告不把当前失败归因于共享网络。

## 结论

**同一工程、同一个官方 PPO、324观察/12动作、共享critic及跨阶段GAE均可保留。** Actor可通过官方自定义模型工厂改为13个完整已学MLP的确定性phase路由，不需要13套PPO或环境。初始复制包含全部隐藏层、均值头和已学state-dependent log-std头；不是从零训练，也不是只复制均值。

但它不是一行配置变更。当前严格checkpoint契约、迁移器、初始化、批量路由及导出必须新增明确架构版本并验证；特别要防止“未选分支梯度为零，但Adam旧动量仍改变该分支”。从范围判断，这是**中等规模、可局部实现的策略架构修订**，不是训练工程重建。相对先核实/处理已有具体P13进度瓶颈，它明显更宽：bank不能直接提供停轮/支撑学习信号，不能修复物理可达性或奖励问题，也无法证明遗忘存在。当前证据不足以将其作为优先动作或要求暂停真实训练。

## 1. 已有精确接入点

以下路径均相对项目根；官方库根为 `C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl`。

| 位置 | 现有行为与可复用接口 |
|---|---|
| `src/wlr50_clean/ppo/semantic_training.py:101–122` | actor/critic均256×256 ELU，固定schema缩放、RSL identity normalizer；配置接入现有hetero分布 |
| `src/wlr50_clean/ppo/rl_library_wrapper.py:604–646` | actor class目前仍为官方`MLPModel`，并非已有phase-bank类；actor分布通过`configure_policy_distribution`选择官方hetero log |
| 官方 `algorithms/ppo.py:473–505` | `resolve_callable(cfg['actor']['class_name'])`后先构造actor，再建storage和同一个PPO/Adam |
| 官方 `utils/utils.py:97–140` | 支持`module.path:ClassName`，无需改site-packages或全局monkey-patch注册表 |
| 官方 `models/mlp_model.py:76–105,147–180` | 一次MLP输出后统一distribution update/sample；mean/std/entropy/log_prob/KL接口可全部保留 |
| 官方 `modules/distribution.py:228–297` | hetero没有另一个共享可训练sigma参数；MLP完整输出形状为`[...,2,12]`，第一片mean、第二片log_std；不是直接给分布一个未reshape的24向量 |

一个较窄的候选实现缝是自定义`MLPModel`兼容actor，保留`get_latent`、identity normalizer和一个官方hetero分布，将`.mlp`换成返回同样`[...,2,12]`的phase路由MLP容器。每个分支拥有独立完整MLP参数。所有分支必须在PPO构造Adam之前注册；不能在构造完成后仅替换actor而让optimizer继续引用旧参数。此处只是接口描述，不提供未经测试的可执行替换代码。

## 2. 324观察中的确定性批量路由

`configs/ppo_semantic_v3/observation_schema.json:13`将`stage_one_hot`作为首组13维、scale1；`semantic_observation.py:234`按当前task stage产生P01–P13 one-hot。因此可从送给actor的**同一份当前策略观察前13维**路由，不需要增加观察、controller隐状态或改变环境。

- 输入仍为TensorDict policy `[B,324]`；分组取各phase行，送入对应branch，再按原行顺序拼回`[B,2,12]`，最后仅进行一次官方distribution update/sample。不能每个branch各采样后再挑选，否则RNG消耗、旧logprob绑定会改变。
- 训练minibatch可混合多个phase，不能假设整个batch同phase，也不能以batch第一行选择branch；N1采样也必须走同一个逐行逻辑。
- 检查真实one-hot的有限、唯一有效位，不能把无效全零/多热输入静默argmax成P01。该检查是模型输入正确性，不是额外物理成功条件。
- 必须使用采样时的观察phase，不用`env.step`之后的`end_phase_id`或事后更新的全局phase。现有incoming handoff/实际投影照旧；不可为了bank把交接变成done、重置、截断或动作结束门。
- 固定观察schema和identity normalizer保持不变。复制网络仍接收完整324维（含phase位），不删除phase输入，不重新拟合/重置每phase归一化。

官方`PPO.act`在139–150保存当前观察、raw action、logprob、mean/std；`compute_returns`在187–209只用真实dones构造GAE。保留这些函数、共享critic和环境terminal语义即可保留跨phaseGAE。Phase变化不应清storage、切return或建立独立优势归一化。每个phase actor仍会收到跨后续phase回报产生的优势，这不是13个独立任务训练。

## 3. 最重要的隔离陷阱：零梯度不等于不更新

官方PPO在362–376执行`optimizer.zero_grad()`、backward、actor整体gradient clipping及一次Adam step。本机Torch默认`zero_grad(set_to_none=True)`（`optim/optimizer.py:932`），Adam仅处理`p.grad is not None`参数（`optim/adam.py:149`）。

若先计算13个branch再one-hot相乘/gather，未选branch可能得到**零tensor梯度**，不是None；已有非零Adam动量仍可推动这些参数。把13组权重堆为一个大Parameter后索引，也不能仅靠未选切片的零梯度保证隔离。FreshAdam只会暂时隐藏这个问题，已训练后切课程仍可能出现。

较窄做法是13个独立Module/Parameter集合，训练forward只调用本minibatch实际出现的branch；缺席分支不进入计算图，zero_grad后的grad保持None，从而跳过其Adam参数/step更新。需专门用**预置非零动量**的反例验证，而不是只测首次更新。仍有共享critic、全batch优势归一化、actor整体grad clipping和全batchKL自适应LR，故只能宣称直接actor参数路径隔离，不能宣称各phase优化完全互不影响。

路由后的未来分支函数会分化，stage交接可能出现新的raw mean/std不连续；现有projection/slew没有改变，不等于实际交接质量自动维持。共享critic也仍可能改变优势估计。Bank保护的是同输入下未更新actor分支函数，不是整个未来物理轨迹的稳定性保证。

## 4. 参数迁移与checkpoint边界

1. 在真实完整update/immutable checkpoint边界，先按旧配置严格加载并验证来源actor/critic/normalizer/RNG/计数。将**每个完整已学actor MLP**逐键复制到13个独立分支，无共享storage别名；包括已训练的非零log-std权重/偏置。官方hetero分布本身无sigma Parameter，保留其log解释。不能在复制后再次调用std/zero-mean初始化。
2. 共享critic权重逐字节保留，identity normalizer及固定schema不变。给每个目标branch保存与源MLP相等的digest，whole-bank actor digest必然改变，不能要求与单网络相同。
3. 最简单、归因最清楚的优化器迁移是同一个新Adam，明确重置actor和共享critic的moments，用来源checkpoint**实际**LR和已有Adam选项；不是擅自回到默认3e−5。也可研究按名字复制critic或branch moments，但需要另行明确语义：13份共享历史moments并非13份独立phase历史，step计数如何保留也改变后续行为。任何方案都不能称“原优化器完全相同”或直接load旧参数ID/参数组到扩容模型。
4. 保留lifetime decisions/updates/optimizer-steps、spent full/suffix预算、origin和source lineage。架构复制增加0训练步；空rollout/pending transition，合法物理reset，不继承旧物理快照。构造/验证结束后恢复源RNG，避免13次构造消耗污染下一采样；分布等价不代表跨不同GEMM分组的逐bit浮点结果无须测试。
5. 初始函数等价应在同一真实观察下、全13路由分别证明mean、log_std/std、logprob、entropy、KL及共享value一致；mixed-batch重排也要对齐。相同新模型配置的以后exact resume才恢复其完整Adam、RNG和全部bank权重，不再次克隆源网络。

当前代码会正确拒绝直接塞入bank：`semantic_policy_distribution.py:41–53,80–116`只允许两个已知policy版本并重算完整runner config；旧migration在168–174只接受Gaussian→hetero；`semantic_training.py:350–367`校验policy版本，512–547的new-MDP warm start还校验官方模型class并严格加载原state。不能复用旧migration标记绕过这些检查，需要**独立明确的架构迁移契约**，而不是把环境不变的bank迁移伪称reward/MDP变更。

`construct_semantic_runner:227–240`的当前初始化分支只找最后Linear并处理均值头，不能不审查地沿用于含13个MLP的容器。新工厂/metadata选择须贯穿train、eval和`semantic_video_cli.py:30–31`；现有CLI已从verified metadata解析policy版本，可沿用该模式，但不得靠调用者手选错误架构。`save_semantic_checkpoint:313–347`和官方PPO save/load已保存完整actor state，可复用roundtrip机制；新架构/路由顺序/schema绑定必须进入manifest。JIT/ONNX不能默认现有单MLP exporter已支持动态分组；至少需要覆盖不同phase和混合batch的导出验证，不能把trace时P01路径固化成所有phase。

## 5. 规模与成本

按当前324→256→256→24 actor和324→256→256→1 critic静态计算，不是运行性能实测：

| 项目 | 单共享actor | 13完整actor bank +共享critic |
|---|---:|---:|
| Actor参数 |155,160 |2,017,080 |
| 共享critic参数 |149,249 |149,249 |
| 合计参数 |304,409 |2,166,329 |
| FP32权重 |约1.16 MiB |约8.26 MiB |
| 权重+梯度+Adam两moments（全部已分配） |约4.64 MiB |约33.06 MiB |

不含activation、rollout、CUDA allocator、metadata和优化器step张量；实际checkpoint/state审计I/O会变大。按phase分组时每样本只过一个branch，理想总MLP算量不必13倍，但小batch划分、GPU kernel launch和scatter可能变慢；全13支都算则接近13倍actor forward，而且有上述inactive-Adam问题。物理运行占比和实际吞吐尚未测试，不能承诺提速。Rare phase的小样本更难泛化，复制已有网络可避免冷启动，但移除跨phase参数共享也会失去有益迁移。

## 6. 如果以后选择实施，最小证据范围

仅列正确性证据，不添加必须先A/B成功的优化器门：真实来源全13branch复制/无alias、mixed phase逐行等价、日志分布和raw action绑定、非选branch在**有旧动量**时权重与optimizer state不变、跨phase非terminalGAE仍连续、单次官方PPO更新/64位计数不重置、bank checkpoint完整save/load及eval/video自动选型。另保留现有物理失败、教师排除、冻结A和最终成功判据，不能以bank架构替代这些语义。

现阶段仅确认**技术可行且范围可控，但比针对已测P13进度问题的局部处理更宽，尚无遗忘因果证据支撑优先采用**。这份审查不选择架构，不修改当前训练计划，不要求额外成功门；到此停止。
