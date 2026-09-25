# 有界方案：同一策略连续收集 512 步；暂不实施

基于当前源码 `65a9255be6d9fd4e19590a48a3a650ae606b04f7` 静态核对。仅设计，不加载 Torch/PXR/Isaac、不读当前 rollout、不改生产。既有 [72e credit 证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_rl_recovery_learning_v1/72e_P12_first_episode_credit.md) 不重复审计。

## 1. 选择与边界

| 每环境收集步数 | 15 Hz物理时间 | 已知P12第452步终止能否落在首块内 | 1536决策的真实PPO/Adam增量 |
| --- | ---: | --- | ---: |
| 128（当前） | 8.533 s | 否 | 12 / 240 |
| 256 | 17.067 s | 否；可合并较近卸载/掉载 | 6 / 120 |
| **512（建议）** | **34.133 s** | **是，若从该P12真实前驱交接后开始收集** | **3 / 60** |

优先512，256仅作为资源/方差需要时的备选；不必先跑完256才能尝试512。它不保证从P07或自然P01起的任何30秒事件都恰落在一块中。gamma=0.9985、lambda=0.99、实际LR=1e−5、网络、控制、reward全部保持。GAE半衰期仍约4秒；较长块减少中途critic bootstrap依赖，**不等于终止惩罚充分传回22秒前，也不证明已纠正critic**。

保留5 epochs、4 minibatches，因此仍20 Adam/update；每minibatch由32→128样本，每样本仍五轮。固定总决策时更新频率下降是这项实验的实际代价，不能把512伪记成四个PPO更新。首个新512块保存/重载并检查terminal、raw GAE与物理覆盖；同一块不可中途更新policy。真实episode终止照常done并断开其GAE，后续合法prefix仍0credit；不跨终止硬拼连续回报，不把前缀塞满512。

## 2. 最小修改面（5个现有runtime文件；不新建庞大迁移工具）

路径均在 `C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/`：

1. **semantic_return_profile.py:15–69**：增加独立、显式collection profile（128历史默认；本候选仅新增512/N1/439，不预先开放256），允许runner返回实际L。保留旧marker/参数/历史128返回完全不变；gamma/lambda身份与收集长度分开校验，不改reward YAML。
2. **semantic_training.py:129,193,340,567,896,2150,2334**：factory/constructor显式接收已验证L，默认仍128；PBRS一致性比较数学return身份/gamma/lambda，并另验实际storage与collection marker，不能全禁用检查。增加窄collection-only加载/保存receipt通路、fresh `[L,1,12]` actions/`[L,1,439]` observations、step=0/transition为空。新receipt进入现有每次checkpoint元数据继承清单。旧迁移构造器的128断言不做全局替换。
3. **semantic_front_retention439.py:305–363**：在现有lineage入口最前增加一个窄collection分支及小型publication helper；未带新receipt仍运行现有严格逻辑。先验证receipt绑定的**原始65a源checkpoint**及其全部65a/72e谱系；再单独校验新段，不将新L计数伪装成旧128计数。保留AUX账本原样，本次不扩展长rollout后的新AUX追加功能。
4. **semantic_cli.py:1004,1359**：train和普通eval由checkpoint的新已验证collection profile构造runner；不能eval默默回128。首次边界可用上述小publication helper产生新不可覆盖checkpoint，之后走原CLI，无需再扩建通用migration dispatcher。
5. **semantic_video_cli.py:179**：正式视频的同一个官方loader也读该L；这仅保证checkpoint契约一致，视频仍逐决策执行，不能借此改动作。

主collector **已有正确动态计数**：`batch=cfg[num_steps_per_env]*num_envs`（2150），真实更新日志/全局决策（2290–2325）、最终manifest（2397–2403）均用实际batch和完成update数；保留它们，不再引入128假设。advantage审计按实存T/N维度映射。

**课程调度器另有锁定点，不得遗漏：** `semantic_curriculum.py:32–83` 及 `configs/ppo_rr_rl_timing_policy_learning_v1/curriculum_plan.json` 均固定128。最小首试继续当前显式单块CLI调用，报告collection=512，**不声称执行旧128课程计划**，旧文件保持历史身份。若要通过该调度器自动分配新课程，必须另作小型v2：从已验证L分配/核对完整update，保存独立新版计划；这是条件增加的两处修改，不是本首块必需门禁。

## 3. 最新源与计数：不得改旧receipt

等待当前正常P07 512块在完整更新边界封存；重新选实际最新兼容checkpoint及sidecar哈希，**不得固定CP229120 AUX32或猜未来编号**。旧块若完成512仍是其原128语义的+4 PPO/+80 Adam，不追溯改账。

新顶层 `collection_horizon439.v1` receipt只绑定：实际源文件/哈希、源runtime与完整runner_config、源计数 `C0/U0/O0`、目标L/目标runtime、冻结的既有receipt与AUX账本摘要、同MDP/同actor-input/同分布声明、fresh rollout。旧 `front_retention439_runtime_identity`、`rr_retention_reward_migration`、祖先receipts、AUX32事件和原文件**逐值保留**，不把其target runner_config改成512。

当前65a校验器确有 `ΔC=128×ΔU` 和 `ΔO=20×ΔU`（327、350）；72e retention校验又要求runner_config等于旧receipt（173–174、214）。因此新段必须包住已验证源：先让旧逻辑验证真实历史源，再校验新段 `C=C0+L*k, U=U0+k, O=O0+20*k`，以及实际optimizer记录、同阶段预算和新runtime；不是拿新计数重走旧128公式或篡改祖先为适配当前。若以后256→512，再新增一个段，不覆盖第一段。

首次publication增量全部0。先普通128 loader验证并加载源，再构建L storage、精确保留actor/critic/buffers、完整Adam与LR、Identity normalizer及RNG；丢弃旧未完成rollout。固定输入的mean/sigma/value应不变；保存重载后才采新数据。历史三个128 rollout绝不拼成新on-policy长块。

## 4. 仅两个必要回归组与风险

- **身份/计数正反例**：真实最新源publication前后learned/optimizer/RNG哈希及固定输入核一致、fresh512 shape；旧receipts/AUX逐值相同。新段+512/+1/+20通过，伪造+128/+1/+20、改旧receipt、遗失AUX、混旧partial rollout均拒绝；旧128源仍照原规则通过。
- **collector/GAE小fixture + 一个真实完整块**：512内第452步terminal，普通phase换不done、terminal不bootstrap、非终止尾正确bootstrap、prefix0credit、20实际minibatch更新且每原样raw/logprob样本出现5次；随后官方checkpoint reload与正式eval/video loader接受同profile。测试不要求机器人先完整成功。

存储/每minibatch约当前4倍、更新间隔4倍，实际GPU和墙钟代价需在首块测量；更长块的整批advantage标准化/状态混合也会改变优化，不将收益都归因于“看到了终止”。当前P07继续，不热改、不新增发布门禁；本方案尚未实施或产生新训练收益。
