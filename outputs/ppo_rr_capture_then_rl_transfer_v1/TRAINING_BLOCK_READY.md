# RR410：首个2048决策PPO block启动核对

## 当前有效启动边界（覆盖下方历史a546准备记录）

Frozen HEAD=`e24a3c2630b0b95439a7a71fe6a8a3f610a9a385`。源为本命名空间`checkpoints/history/checkpoint_rr_capture_knee_v3_step_000220544_ge24a3c2630b0.pt`，SHA256`db98fa88003f72504eae5af3eddeaf3d8436baf0099a3beaca7693df3106fb54`。已经官方保存并独立重载，完整410权重/Adam/LR1e-5/Identity/RNG及祖先谱系相等；v3控制迁移增加0学习。两处审计路由已在26db修复，新v3保存继承已接通，204集成CPU测试通过。N1/seed1001/cuda:0/P07/checkpoint_policy实际连续前缀，2048新增有效决策、每1update保存；不是P01完整验收。

新v3保留原hip20°/12秒，随后有限knee正向1°/秒至多20°。当前仍wheel shaping OFF、all12原始Gaussian、原reward/caps/sigma/source/物理。旧rollout不继承。当前没有新增实际训练计数；等待视频导出助手退出后启动单一Isaac。下方a546命令是历史资料，不能直接运行；应使用本段新HEAD/CP。

## 历史a546准备记录

状态：**仅准备，尚未启动/新增学习；先等待当前a546自然P01评估结束。启动前还必须在合法边界补齐下面两处审计路由遗漏并严格重新绑定runtime。** 本检查只读代码、配置、封存JSON；未运行Python/测试/仿真/优化。

## 已保存源与预算

当前410：`checkpoints/history/checkpoint_rr_capture_p05_recross_v2_step_000220544_ga54678ceef58.pt`，SHA256 `df9072ce2f523b7398f80d58ed033c8ade7a4416afd691fc95c2698f835596db`。当前HEAD `a54678ceef5868e419e55840cea34899854bc726`。

实际220544 policy decisions /1688 PPO /33760 Adam，RR branch仍0/0/0；seed1001，cuda:0，已保存有效LR=1e-5，adaptive schedule（续训后可能改变，不宣称恒定），固定工程缩放+actor/critic Identity normalizer。410观测/12原始Gaussian通道，全部旧权重、Adam、RNG和AUX谱系已保存。

继承的stage累计：phase_suffix95744/100000，剩4256；full_episode114688/131072，剩16384。2048未超预算，不需要扩预算。若之后新增checkpoint，应以实际最新兼容源重算，不能覆盖它。

## 本地CLI支持的一条候选命令

选择P07、offset0的**真实自然P01 frozen-checkpoint-policy前缀**，然后学习P07及后续；不从已抬RR快照开始，不用旧FSM代走后腿。是否采用这个课程，待当前真实RR结果后由root决定。下面是当前a546绑定下语法/预算/410接口成立的命令，**审计遗漏未修前不要执行**：

```powershell
$trainArgs = @{
  Command = 'train'
  ExpectedHead = 'a54678ceef5868e419e55840cea34899854bc726'
  SemanticVersion = 'v3'
  ExperimentId = 'rr_capture_then_rl_transfer_v1'
  Stage = 'phase_suffix'
  FromPhase = 'P07'
  TeacherOffsetDecisions = 0
  PrefixSource = 'checkpoint_policy'
  Decisions = 2048
  Seed = 1001
  NumEnvs = 1
  Device = 'cuda:0'
  CheckpointIntervalUpdates = 1
  Checkpoint = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_capture_then_rl_transfer_v1\checkpoints\history\checkpoint_rr_capture_p05_recross_v2_step_000220544_ga54678ceef58.pt'
}
& 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\scripts\run_semantic_ppo.ps1' @trainArgs
```

审计修复会改变runtime hash/HEAD：正式执行前必须把上述HEAD和Checkpoint换为**合法修复后冻结HEAD及相应严格保存/重载的同状态410 checkpoint**；不能继续直接用旧a546参数、改sidecar或绕过runtime检查。不得添加NewMdpWarmStart/PolicyDistributionMigration。脚本自选唯一run目录、headless、单Isaac锁；它不会清空网络。

前缀为独立冻结源actor的确定性conditional mean，正式learner仍采真实Gaussian。每次reset前缀0 PPO/Adam信用，不计入2048，也不存入on-policy rollout；真实前缀占用同一200秒任务时钟。准确前缀决策数只能读新run的`prefix_evidence.jsonl`，不能沿用旧版693或预先填写。上限1800前缀decisions；若前缀未到目标，现实现仅一次fresh-P01 fallback，随后真实learner从P01采样，不递归重试、不伪称P07覆盖。报告必须区分requested/actual phase和fallback。

## Rollout、回报与保存

- N1×128决策/rollout；2048恰为16个完整rollout，没有取整超额。每update5 epochs×4 minibatches=20 Adam steps；计划总16 PPO/320 Adam，**尚非实际完成**。
- 15Hz决策、120Hz物理，通常每决策8ticks；终端可短于8，实际物理ticks从日志统计。phase切换不done、不清空跨阶段回报；真正任务终端/200秒超时不bootstrap，`time_outs=False`。gamma=.9985、lambda=.99，gamma每issued decision一次（含短终端）；非终态rollout尾使用官方critic bootstrap，不因128边界造终止。
- 每完整update保存、官方roundtrip后发布latest；支持已有`stop_after_update.request.json`合法边界停止。终态尾先保存，后续reset/prefix不污染已封存更新；最后非终态partial episode只代表采样边界，不算物理失败/成功。
- 若完整实现2048：首CP220672/1689/33780；最终CP222592/1704/34080，RR branch+2048/+16/+320。均是预计，不得预记。
- 预计文件：`outputs/ppo_rr_capture_then_rl_transfer_v1/checkpoints/history/checkpoint_step_000220672.pt` … `checkpoint_step_000222592.pt`，对应manifest；正常save才更新该namespace的checkpoint_last及pointer。rollouts预计001689…001704。run路径为`runs/ppo_rr_capture_then_rl_transfer_v1/train/<UTC>_g<实际HEAD12>_<uuid>/`。

## 必须补齐的两处审计遗漏（本次未修改）

`semantic_training.py` 的 `train_semantic()` 有两个experiment-id tuple漏掉 `rr_capture_then_rl_transfer_v1`：

1. 约1928行的 `audited_history_policy_request` 开关。目前新namespace直接调用 `runner.alg.act(obs)`，**不会关闭RR assist、不会mask通道、不会改变采样或把final target当raw**；410原始input仍进actor。但decision JSON缺`policy_request`中的本次head/conditional μ、HISTORY/caps、有效sigma、一次draw无额外forward证明及RR14+context7显式观测切片。终态行同样缺该receipt。
2. 约2035行的 `audited_ppo_update(...likelihood_audit_path=...)` 开关。目前仍调用`audited_ppo_update(runner)`，仍核验每minibatch likelihood输入就是stored raw、20步finite gradients/KL/clip并记录actor变化；但**不生成update_x_likelihood.json**，也不启用逐saved-row索引、每样本5次曝光、当前μ/σ和head梯度详细记录。

保留的证据：每decision仍写原raw12、旧μ/σ/logp/value/reward/done和applied_audit；每个sealed rollout仍存精确410输入、raw、分布参数、logp、GAE/returns。storage原raw及μ/σ逐步等值检查仍执行。env/backend的RR assist真实推进、6/7 ownership、同拍state_before→state_after独立重算、float32最终派发/counterfactual审计也仍执行；endpoint完整RR receipt位于native actuator audit，每物理tick摘要保留verified/phase/effect。两遗漏不是物理执行丢失，但不满足本轮明确要求的完整采样/likelihood观测链，因此不能略过。

最小修复：只在上述两个tuple加入新namespace；已有`audited_history_policy_request`和`audited_ppo_update`内部已支持RR410，勿改PPO公式、reward、sigma、caps、assist或policy权重。修复是instrumentation runtime变更，仍须冻结并严格绑定checkpoint，不在当前Isaac期间改。

有界测试入口：

- `tests/unit/test_semantic_rr_capture_policy_migration.py::test_old_kernel_gaussian_mean_sigma_raw_likelihood_preserved_for_any_new_state` 已覆盖RR410 receipt14+7与原Gaussian；它单独调用helper，**不能捕获train_semantic的namespace路由遗漏**。
- 复用`tests/unit/test_semantic_fl_capture_quality_migration.py::test_official_cpu_migration_update_save_reload_and_both_video_modes`的真实CPU一rollout训练断言，新增RR410实际namespace128行route用例：每row存在正确RR policy_request/raw/logp/14+7，文件20minibatches、每saved row5次，ordinary save/reload完整carry。
- `tests/unit/test_task_recovery_learning_signal.py::test_likelihood_instrumentation_has_bitwise_identical_updates_optimizer_rng_and_storage`是无额外forward/RNG/参数扰动对照；可复用结构给RR410分支，不拿旧372覆盖自动当新410证明。
- `tests/unit/test_semantic_checkpoint_prefix_training.py::test_real_frozen_prefix_actions_never_enter_official_storage_across_terminal_resets`覆盖前缀0credit、terminal no-bootstrap；RR410已保存prefix/profile测试结合新route用例即可，不重建大测试框架。

首个真实128保存后再审查实际samples/μσlogp、assist可见与owner、Adam+20、LR、Identity/fullRNG、全旧谱系/AUX和roundtrip；整块统计真实P01–P13输入数和RR当前TOP/承载/后续RL，不把suffix或助控贡献命名为完整PPO成功。当前评估结果及启动决定仍待root；本文件不是训练执行或新增门禁工具。
