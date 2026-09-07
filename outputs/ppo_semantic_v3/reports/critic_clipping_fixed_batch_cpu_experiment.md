# 真实固定批次 critic clipping CPU 机制实验

## 结论与边界

本次唯一一次、两组各 20 步的纯 CPU 影子拟合，确实在真实 terminal batch 中观察到 **局部 clipped-flat 分支及零输出梯度**。关闭 value clipping 的最终拟合误差稍低；差异不足以推出任务失败主因，也不证明真实 PPO 当时使用的全部 minibatch 已经 flat。

- 第 15 步后，终止样本的预测跨出旧 value 的 −0.2 边界；第 18 步该样本实际进入 minibatch 时，clipping-on 的该样本 value-loss 对预测值的梯度严格为 0（autograd 断言通过）。
- 末态 clipping-on 有 46/128 样本满足严格 clipped-flat 条件（35.9375%）；20 次实际 minibatch 访问中共 55/640 次（8.59375%）。这是本影子顺序的实测，不是对原 PPO 顺序的回填。
- 20 步后终止预测 on −4.131659、off −4.171731，目标 −42.271667。两组都仍显著欠拟合。
- clipping-on 的终止预测仍移动了 −0.310054，**超过 0.2**；因此不能把该损失分支误称为每轮全模型输出最多改变 0.2。其他样本共享参数梯度与保留的 Adam 动量仍可使该预测继续移动。
- 无 actor 更新、无真实训练权重/账本写入、无模型权重输出、无策略决策；本实验不是 PPO 续训，也不是物理评估或训练启动门禁。

## 真实输入及前置验证

项目根目录：`C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`。

输入均只读：

- `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000069760.pt`，实际 global 69760 / PPO update 510 / lifetime optimizer steps 10200。
- 同名 `checkpoint_step_000069760_manifest.json`。
- `runs/ppo_semantic_v3/train/20260906T1650134938238Z_g73e937039708_7681a31f01714528b75509300dfac00f/rollouts/rollout_000511.pt`。
- 同 run 的 `residual_and_projection_audit.jsonl` 中连续 global 69761–69888，共 128 行。

先验证、后拟合，全部通过：

1. checkpoint 的 source_run、global、iteration 和 sidecar 所记 runtime/actor/critic/optimizer 元数据对应；rollout 的 runtime_contract 和 policy_contract 与前置 checkpoint 完全相等。
2. 保存的 observations 是真实 `[128,1,324]`；values/returns/rewards/dones 是 `[128,1,1]`。全部参与拟合的浮点输入有限且位于 CPU。
3. 128 行真实 audit 的 old_value、float32 reward、done、raw action 均与保存的 rollout 一一精确相等。未重造模板观测，未替换 returns 或 old values。
4. 唯一 terminal 为零基 index 45，即 global 69806；P06 / `INCOMPLETE_CONTROLLER_BLOCKED`。saved return 与 saved reward 严格相等，均为 −42.27166748046875，time_outs=false、terminal_bootstrap_allowed=false。
5. 前置 critic 在同一批真实 observation 上的 CPU 前向与 saved old values 最大绝对差 **7.152557373046875e−7**，通过明确的 atol=2e−5、rtol=2e−5 检查。CPU 与源 CUDA 计算的舍入差保留披露；saved old values 没有重算覆盖。终止 old value −3.8216054439544678；CPU 初始预测 −3.8216052055358887。
6. 实际 critic 为官方 MLPModel（324→256 ELU→256 ELU→1），normalizer 为 Identity，严格加载全部 state_dict。

本批次前 46 行包括 episode 1 的终止，后 82 行已是下一回合；因此全 batch 并非只有终止状态，相邻状态的共享梯度也会影响该终止预测。

## 唯一改变的因素与实际执行

配置核对：source `use_clipped_value_loss=true`、clip_param=0.2、value_loss_coef=1、max_grad_norm=1、5 epochs × 4 minibatches、source effective Adam LR=1e−5。

使用两个仅内存中的相同 critic 副本：

- 原 Adam 按官方 actor 参数后接 critic 参数的顺序保存。经实际模型 named_parameters 核对，取 source 参数 IDs 6–11 对应 critic 的六个 weight/bias；step、exp_avg、exp_avg_sq 精确复制且逐 tensor 相等验证。
- 每个 critic 参数的原 Adam step=240，影子结束均为 260；240 是本次新 MDP 重置后的实际 Adam 状态，**不是** lifetime optimizer 总数 10200。影子新增 20 不计入任何真实计数。
- 相同源 Adam betas/eps/其余 param-group 属性；两组每一步固定 LR=1e−5，无 adaptive KL 调整。
- 私有 CPU permutation seed=20260906，128 行分成四个 32 行 minibatch，五轮复用同一 permutation。这与官方生成器的“一次 randperm、跨 epoch 复用”结构一致，但未声称还原源 CUDA 的实际 shuffle。完整排列见 JSON。
- 每个 critic 20 次且仅 20 次 optimizer.step，梯度单独 norm clip=1，与官方 critic 梯度路径一致。没有 surrogate、entropy、actor 梯度，也未改变环境 reward。
- actor 仅构造并冻结用于核对参数顺序；前后 state tensor digest 一致。未使用影子权重训练/评估，也没有保存影子权重。
- `CUDA_VISIBLE_DEVICES=""`，torch intra-op=1、interop=1；所有 load 使用 map_location=cpu，没有调用 CUDA 或 Isaac API。
- 拟合脚本唯一执行 exit 0，工具计时 5.0438925 秒；无失败后重跑、无延长拟合、无参数搜索。此前另一次只读 CPU 检查仅打印了 checkpoint/rollout 的键与形状，没有拟合。

官方损失保持原式：

```text
clipped_value = old_value + clamp(value - old_value, -0.2, +0.2)
clipping_on  = mean(max((value-return)^2, (clipped_value-return)^2))
clipping_off = mean((value-return)^2)
```

严格 flat 计数定义为 `abs(value-old_value)>0.2` 且 `clipped_error>ordinary_error`；不把相等分支或一般零梯度混入。实际 clipping-on minibatch 中该集合的 loss 对 value 导数均通过 autograd 等于零的断言。JSON 中 clipping-off 的 `flat_count/terminal_flat` **只表示若使用 clipping 会落入该分支的反事实分类**，并不表示 off 的实际梯度为零。

## 拟合量化结果

| 指标 | 两组初始 | Clipping on：20 步 | Clipping off：20 步 |
|---|---:|---:|---:|
| 全 128 行 MSE | 107.662033 | 104.778526 | 104.398087 |
| 单终止样本 MSE | 1478.407227 | 1454.660156 | 1451.605225 |
| 单终止样本预测 | −3.821605 | −4.131659 | −4.171731 |
| 单终止目标 | −42.271667 | −42.271667 | −42.271667 |
| 终止 MSE 减少 | — | 23.747070 | 26.802002 |
| 全批末态严格 flat（on 实际；off 反事实） | 0/128 | 46/128 | 46/128 |

off 相比 on，末态全 batch MSE 低 0.380440、terminal MSE 低 3.054932，terminal 预测额外向目标移动约 0.040072。这是此固定 20 步窗口内的效果，不外推更多更新后的收益。

## 每步轨迹、梯度与平区

所有步 LR 都是 **0.00001**。星号表示该 minibatch 含终止样本（2、6、10、14、18）。预测与全批 flat 为该步更新后的值；MB flat 和 pre-clip 梯度为更新前；flat 两列仅列 on 实际分支。post norm 的约 1.000002 是 float32 分组求和的微小舍入差，未改阈值。

| 影子 step | terminal V on | terminal V off | on MB flat | on 全批 flat | grad pre on | grad pre off | grad post on | grad post off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -3.820477 | -3.820477 | 0/32 | 0/128 | 133.224640 | 133.224640 | 1.000002 | 1.000002 |
| 2* | -3.822991 | -3.822991 | 0/32 | 0/128 | 108.887962 | 108.887962 | 1.000001 | 1.000001 |
| 3 | -3.828562 | -3.828562 | 0/32 | 0/128 | 147.422775 | 147.422775 | 1.000002 | 1.000002 |
| 4 | -3.836899 | -3.836899 | 0/32 | 0/128 | 138.597168 | 138.597168 | 1.000001 | 1.000001 |
| 5 | -3.847660 | -3.847660 | 0/32 | 0/128 | 133.153503 | 133.153503 | 1.000002 | 1.000002 |
| 6* | -3.860724 | -3.860724 | 0/32 | 0/128 | 108.766228 | 108.766228 | 1.000001 | 1.000001 |
| 7 | -3.875656 | -3.875656 | 0/32 | 0/128 | 147.236099 | 147.236099 | 1.000001 | 1.000001 |
| 8 | -3.892295 | -3.892295 | 0/32 | 0/128 | 138.412506 | 138.412506 | 1.000001 | 1.000001 |
| 9 | -3.910414 | -3.910414 | 0/32 | 0/128 | 132.886719 | 132.886719 | 1.000002 | 1.000002 |
| 10* | -3.929991 | -3.929991 | 0/32 | 0/128 | 108.481560 | 108.481560 | 1.000001 | 1.000001 |
| 11 | -3.950688 | -3.950688 | 0/32 | 0/128 | 146.890900 | 146.890900 | 1.000001 | 1.000001 |
| 12 | -3.972425 | -3.972425 | 0/32 | 0/128 | 138.117279 | 138.117279 | 1.000001 | 1.000001 |
| 13 | -3.995047 | -3.995047 | 0/32 | 0/128 | 132.501022 | 132.501022 | 1.000002 | 1.000002 |
| 14* | -4.018594 | -4.018594 | 0/32 | 1/128 | 108.098183 | 108.098183 | 1.000001 | 1.000001 |
| 15 | -4.042789 | -4.042789 | 0/32 | 36/128 | 146.450790 | 146.450790 | 1.000001 | 1.000001 |
| 16 | -4.067566 | -4.067605 | 9/32 | 46/128 | 47.826900 | 137.754562 | 1.000001 | 1.000002 |
| 17 | -4.088662 | -4.092930 | 11/32 | 46/128 | 0.650390 | 132.049026 | 0.650392 | 1.000002 |
| 18* | -4.105868 | -4.118843 | 9/32 | 46/128 | 1.041852 | 107.658081 | 1.000001 | 1.000002 |
| 19 | -4.120063 | -4.145109 | 14/32 | 46/128 | 0.770674 | 145.958115 | 0.770675 | 1.000002 |
| 20 | -4.131659 | -4.171731 | 12/32 | 46/128 | 0.648155 | 137.357193 | 0.648156 | 1.000001 |

前 15 步两组一致；首次在实际 minibatch 消费 clipped-flat 样本是第 16 步。第 17 步 on 原始梯度 norm 0.650390，对照 off 132.049026；第 18 步 on 1.041852，对照 off 107.658081。这个局部梯度削减是可重复检验的机制证据；梯度 norm 的跨步变化不能自动归因成物理失败。

## 不能据此声称的结论

- 不证明原线上 PPO 的精确 minibatch 顺序、混合 actor/entropy 梯度或 adaptive LR 历史与本影子完全相同。原实际 PPO 使用 GPU；本实验是 CPU critic-only、固定数据与固定 LR。
- 不证明 clipping 是当前完整任务失败、后腿失败或低吞吐的主因；也不证明仅取消 clipping 就会解决终止目标拟合。
- 不声称所有 20 次更新或所有样本都 flat：前 15 次实际 minibatch 的 flat 数为 0，终止样本五次访问中只有最后一次落在严格 flat。
- 不推导“每次最多改变 0.2”；本次 on 反而实际改变 0.310054。共享参数、其他样本及原 Adam 动量明确不能忽略。
- 不新增 reward bonus、不改变真终止吸收态 phi、不调 entropy/std/LR、未选择或实施生产修复；主 P07 实时训练独立继续。

## 可复核产物与源实现

同目录：

- `critic_clipping_fixed_batch_cpu_experiment.py`：只读输入、仅 stdout JSON 的完整脚本。
- `critic_clipping_fixed_batch_cpu_experiment.json`：此次已执行 stdout 的完整数值记录，包括 permutation、40 个步骤的逐步 loss/MSE/预测/flat/LR/梯度和工具退出码；由 apply_patch 持久化，不是重新拟合。
- 本报告：`critic_clipping_fixed_batch_cpu_experiment.md`。

实际代码依据：本机 `C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl/algorithms/ppo.py` 的 value loss（约 306 行）、actor/critic 各自 gradient clip（约 374 行）、保存 optimizer（约 432 行）；`storage/rollout_storage.py` 约 220 行的 minibatch generator；repo `src/wlr50_clean/ppo/semantic_training.py` 约 690–744 行的采样、compute_returns、更新前 rollout 保存接线。

本任务新增真实 policy decisions=0、PPO updates=0、optimizer steps=0。两组影子各 20 步仅用于这份离线机制报告。Python 已结束；报告保存后停止本任务。

