# 有限前段 rehearsal：只读可行性结论

结论：授权允许如实标注的局部真实接触／接续 AUX，但当前旧工具不能直接执行这项 389 维、阶段输入列限定的学习。可以作小范围显式适配；本次未适配、未优化、未写 checkpoint、未运行 Isaac/GPU。

## 数据确有局部成功，但不是整集成功

唯一检查的 CP205824 自然 deterministic 源：`video_eval/validation/20260922T0740416538018Z_ga802b24d78df_626f2bc2aa0e416d8c1545accf8e2585/source`。

- FR 真实 qualified lift tick23；P02→P03 tick2696；FR crossed tick2710、placed tick2721；P03→P04 tick2728。
- P05→P06 tick4496；FL crossed3857、placed4492，带明示 FL assist。此后进入 P07/P08/P09，最终 P09 未完成；不能命名为完整成功或纯 PPO 捕获。
- 这些事件支持“前段有效且可继续”的局部资格。监督窗口仍需预先声明连续范围／有限采样规则，保留其中全部失败动作、接触丢失和后续失败说明；不得删除正式 on-policy 失败或伪造 PPO 样本。
- **当前数据缺口**：`video_policy_decisions.jsonl` 保存 raw conditional mean/action、HISTORY、sigma 和 step_info，但 `semantic_video.py` 没有保存完整的 pre-action 389 actor 输入。物理日志也不是这 389 输入本身。不能直接把这些日志交给旧数据 loader。若从已封存同步记录重建，须逐项确定 encoder/mapper/阶段/assist 历史及 reset 输入，并用原 CP205824 在重建输入上的 raw mean/HISTORY/sigma 与原记录作有界一致性验证；未验证前不能称执行就绪的数据。

## 现有接口与最小适配范围

现有工具位于 `outputs/ppo_task_conditioned_hip_wheel_v1/candidate/finite_auxiliary/`：`inspect(...)`、`fit_mean_row(..., authorized=False)`、`append_ledger(...)`、`save_auxiliary_checkpoint(...)`；CLI 默认 CPU inspect，执行需 `--execute-aux --inspection-receipt --aux-checkpoint`，并绑定同一 source/data/helper/runtime 哈希。

其严格要求 N×372（1–128 行）、旧 `SemanticTaskConditionedHipWheelHistoryMLPModel`、P05 crossed/AIR/unplaced 样本；只优化最后层 `mlp.4` 的 FL mean 行（257 参数）。数据 loader 也专用于旧 FL−6°诊断。它使用旧 sigma/history 分支，只容许已验证的 quantity-only 数据版本边界。**修改形状常量或放宽类型检查不够，也不能截成 372。**

新分支必须显式要求 `p05_hip_only_capture_assist_history_v1` / `role372_p05_capture_assist_v1` / 389 / `SemanticP05CaptureHistoryMLPModel`，调用当前 `p05_capture_request_history` 和 receiving-wheel sigma kernel（rho=.9、temperature=.25、现有 state-dependent multiplier 均不改）。仅复用有限临时 SGD、拒绝越界提案、空 rollout、精确状态保留、独立账本与正式 save/reload 机制。旧工具的 FL 单通道角度阈值不能未经说明充当全12通道 trust bound。

CP205824(a802) 与 CP209920(5fd) 虽同 389 contract，index17 potential 语义经历 RR retirement 迁移，不能声称同一 MDP。所选 P01/P02 输入需要证明 RR predicate 未激活且旧／新 potential 数值相同，绑定该已存在的精确迁移记录；不能使用旧 quantity-only 豁免。

## 第一层列限制的准确保证

当前 one-hot 占输入0–12；P01/P02 是0、1，Identity normalizer 保持其精确0/1。只允许改 `actor.mlp.0.weight[:, 0:2]`（256×2），bias、其余列、后续层、critic 均不变，则对任一合法 P03–P13 **相同输入**：

`Δz = ΔW[:,0]·x[0] + ΔW[:,1]·x[1] = 0`。

故整个 raw mean/log-sigma，以及相同 HISTORY/caps/state-sigma 条件下的 Gaussian 分布均不变。这比旧 FL 输出行更新的跨阶段保证更强。实施时须用参数白名单逐项等值验证和真实 P03+ holdout 的同输入输出验证；不能靠完整 Adam + gradient mask，因为已有动量／weight decay 仍可能改其他元素。可用仅包含这512个临时 SGD leaves的既有 functional-call模式，最后只复制这些列。

限制：P01/P02 共享 trunk 的变化会同时影响 **12个 mean 和12个 log-sigma**，不能称 mean-only 或承诺其 sigma 不变；需相应 full-Gaussian KL／sigma／真实物理 REQUEST 变化边界。它仅是阶段特定 hidden bias 偏置，表达能力有限，未证明可恢复任务。P01/P02 动作改变后续真实输入和 HISTORY，故 **不保证后续自然轨迹不变**。rho=.9 的真实 .1 导数不能放大冒充原网络优化。

## 若之后批准的保留要求

从最新实际 CP209920 出发，禁止拷贝旧 CP205824 参数或回退网络。标签应为核对过的 source raw conditional action/mean，不是最终 actuator target；是 off-policy AUX，不是随机 Gaussian PPO sample。P03/P05 结果可作资格与 holdout，不因此解冻其参数列。

保留 critic、完整 PPO Adam/moments/LR、Identity、RNG、三个 origin 和已有 AUX 7 accepted/8 attempted 记录，追加独立、有准确列名和非 mean-only 类型的新事件；PPO decisions/updates/steps 加0。旧 Adam 确实不代表 AUX 更新，后续 PPO 可能抵消该方向。使用明确 AUX 新文件名并实际 reload，fresh rollout，之后仍须自然 P01 deterministic 验证；不可声称数学局部不变已证明物理成功。

本检查只读取上述工具、当前 actor/schema 和该一条已封存源的事件/字段；没有生成 AUX 数据、训练 checkpoint 或建议自动执行。

## 补充：block03 封存 on-policy 前段是更直接的数据源

经一次 CPU-only 只读检查，**数据可用，不需重建视频 observation**。源 run 为 `runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff`；仅第一 episode，精确 P01 输入2个（global203777–203778），P02 输入278个（203779–204056）。完整前段共280个状态是候选范围，不是本次选择或批准的 AUX 数据集。

| 该 run 下的封存文件 | zero-based flat index（含两端） | global decisions | 行为策略版本的 global counter |
|---|---:|---:|---:|
| `rollouts/rollout_001558.pt` | 0–127 | 203777–203904 | 203776 |
| `rollouts/rollout_001559.pt` | 0–127 | 203905–204032 | 203904 |
| `rollouts/rollout_001560.pt` | 0–23 | 204033–204056 | 204032 |

三文件 `observations.policy` 均为128×1×389；上述行的 `actions`、`distribution_params=(mean,std)`、`actions_log_prob` 与 `residual_and_projection_audit.jsonl` 相应原记录逐值精确相等。重新计算 Normal logp 最大差3.815e−6；无 terminal、12 mask 全1、前段 assist owner 0；RR current-qualified/crossed/placed 特征均0。这里是依次更新的三个真实行为策略，不是 CP205824 冻结策略，也不是新209920 on-policy 数据。

前后绑定：FR lift tick26；P02 最后决策204056结束tick2240；随后 FR crossed2252（decision204058），placed2268（decision204060终点2272，真实 TOP，assist owner=[]）。继续 FL crossed3390、placed3687，decision204237终点3688进入P06，FL assist owners=[0,1]。后续到P12以及最后未完成沿用既有封存审计，不重新扩大本次扫描。故能称“这条真实轨迹的有效前段、后续实际接触并接续”，不能称各个早期动作单独造成成功，更不能称整集或无 assist 的全程成功。

有限 AUX 可预先声明此范围内的短窗口／有限抽样，再绑定到后续 FR placed 证据；旧工具1–128行上限仍须明示处理。若只选 P02 窗口，P01列没有有效梯度；若要两列均有监督，须明确保留2个实际P01状态。不得把280行静默塞入旧上限。只把选定前段用作 AUX，不把P12失败后段、第二失败episode、第三未完成partial episode作为正例；原训练记录和全部 PPO失败样本保持不删不改。

**标签区分**：stored `actions` 是当时实际送入执行链的随机 raw 样本，符合“真实执行过的动作”证据，但含探索噪声、源状态/HISTORY依赖；单次模仿不能保证确定性重现成功。stored μ 是该状态下的行为分布均值，适合如实标注的均值蒸馏，**不是该拍实际执行动作**，不能借这次随机轨迹声称其 deterministic μ 已成功。280行没有一行 full12 raw==μ，最大分量差0.218656。两者都不得伪充最新策略 PPO 数据或用最终 actuator target替代。

当前 first-layer列方案仍会改变P01/P02的logσ；即使 loss只拟合raw或μ，也不能宣称方差不变。适配仍须使用当前389/kernel、最新209920全状态、独立 AUX账本与trust checks；这里只消除了“完整输入缺失”的数据障碍。CPU进程已exit0，无优化／checkpoint写入／runtime修改。
