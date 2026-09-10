# a8b1484 → 未来视频接线版本：只读迁移核对

本次仅读已有 `video_integration_review.md`、视频/迁移/加载源码与测试；未执行 Python、Torch、测试、哈希检查、Isaac 或视频操作，未读取正在增长的训练轨迹。源指固定 a8b1484 下将实际完成并保存的 HISTORY372 检查点；本报告不预填其最终编号或任务成功。

## 结论

**现有 video-only runtime migration 的权限设计可复用，无需新 MDP、Adam 重置或模型转换；但只修改三个视频接线文件还不能直接加载当前372检查点。** 旧 exact-migration 分支存在三个明确324残留，必须一并窄修，不能靠 `NewMdpWarmStart` 绕过。仓库没有独立 `semantic_video_migration.py`：实现位于 `semantic_migration.py` 的 `VIDEO_FILES`、`_video_instrumentation_factor`、`build_migration_plan/validate_migration_plan`；对应测试为 `test_semantic_video_migration.py`。

| 当前具体位置 | 可证阻断/记录错误 | 最小兼容边界 |
|---|---|---|
| `semantic_cli._preflight_checkpoint` 的 `resume_migration` 分支（约323行） | 重建 `semantic_runner_config` 漏传 `observation_layout`；源372 actor config包含明确layout，因此预检不相等。 | 从已严格验证的源元数据/目标schema取同一layout传入；禁止猜维度、追加特征或改policy contract。 |
| `semantic_training.load_semantic_checkpoint` 的普通 `migration` 分支（约457–464行） | storage宽硬写324；其runner config重建也漏layout。即使视频loader已接372仍会拒绝。 | 仅对显式、完整验证的 HISTORY372/N1 同布局视频边界允许372；仍要求新空storage/transition、12action和完全相同runner config。旧324默认及其他迁移语义保持。 |
| `semantic_migration.build_migration_plan` 返回值（约961行） | 顶层 `observation_dimension` 固定324；当前builder本身不据此拒绝372，但会写错事实。 | 视频factor内用现有严格 `policy_observation_layout_from_metadata` / `source_num_envs` 绑定真实372/layout，保持源目标同layout；旧324计划的黄金结构不变。 |

上述三个文件已经在 `INSTRUMENTATION_FILES` 中；因此若以后获准实施，最小发布范围是 **3个视频文件 + 3个现有exact-migration兼容文件及专属测试**，而非新增训练器/迁移通道。这里只说明必要范围，不是当前修改授权。

## 原三个视频接线仍需完成

1. `semantic_video_cli.py`：`experiment_id` 传到runtime/core/capture；C loader传已预检layout给真实runner，并记录完整policy/layout；A只复用现有 `SemanticSensorReader` 的measurement依赖配方，不改冻结A控制/reset。
2. `semantic_video.py`：`video_configuration`、capture manifest、`validate_video_configuration` 和独立replay重建runtime都显式传同experiment；当前 `all_stage_acceptance_v1` 必须使用其独立六配置目录，不能误用通用v3配置。沿用同共同TaskEvaluator和真实终止。
3. `scripts/run_semantic_video.ps1`：新增显式ExperimentId及独立run/log namespace，转发CLI；保留单进程锁、固定已提交HEAD、默认v2/v3兼容。

## 保留状态与许可边界

- `VIDEO_FILES` 正好包含上述3视频文件；`video_review={reason:…}` 要求精确非空人工范围说明，绑定每个源/目标文件hash。目标必须保留完整视频文件集，声明delta必须精确；冻结A、schema、action、reward、nominal、backend/physics、选定六配置、运行库/频率/预算均不得借此改变。不能混入prior/evaluator/execution factor。
- 在未来完整update边界提交纯视频修订后，复用 `build_migration_plan` 生成检查点绑定计划，入口使用现有 `--resume-migration` / `-ResumeMigration`。**不使用** `--new-mdp-warm-start`、target-policy或distribution migration；不生成另一个空Adam的initial训练检查点。
- 正常load分支先验证plan、源sidecar/embedded infos、原runner/policy合同，再用 `load_checkpoint_round_trip` 恢复实际actor（含learned std与48列）、critic、Adam与实际LR、identity normalizer，逐项hash核对并恢复源训练RNG。返回原计数、预算和origin；只是新物理P01 reset、空rollout，不继承物理状态或旧partial rollout。视频本身optimizer updates必须0。
- C调用实际HISTORY deterministic kernel，不导出替身；layout原324位置和rho0.9不改。loader结束时既有`unchanged()`再次比较actor/critic/optimizer/normalizer。该检查不是对整个视频行为等价或稳定性的证明。
- 目前视频固定seed4001，而自然完整评估常用2001；未来视频必须自己满足共同完整成功，不能用另一次评估替代。无真实完整成功前不录制、不创建success名称。
- 历史A仍是未完成/非配对参考。`publish_success_comparison` 当前同时要求A/C success且runtime/seed等相同，不能拿历史A塞进该函数；保留A如实链接不受C加载迁移阻挡。若以后明确需要非配对参考合成，必须在原视频模块内另做诚实标记的诊断发布入口，不能放宽C成功、伪造A通过或用迁移证明改善。本次不修复/重封装/重跑A。
- 现有200秒文件context预算、同episode完成后观察与额外video post-roll差别，仍按原review披露；不得用裁尾/加速消除安全失败，也不是训练前门禁。

## 未来最小测试（本次未运行）

- 扩 `tests/unit/test_semantic_video_migration.py`：旧324 golden原样；同HISTORY372/N1视频plan绑定正确维度/layout；错layout/缺完整contract、篡改源/目标、配置/物理/奖励delta、混合NewMDP拒绝。复用已有 `test_new_and_repaired_video_bind_exact_source_and_target_without_changing_topology` 等正反例。
- 新窄 `tests/unit/test_semantic_video_372_resume.py`：真实official CPU HISTORY372带已非空Adam的保存→video-only严格加载；均值/std/critic/48列、Adam/LR、normalizer、RNG、所有counter/budget相等，storage为空，0update；调用视频loader固定输入输出等于正式eval，历史slice变化有效，错宽/contract拒绝。可复用现有372 migration测试的真实runner fixture，但**测试操作必须走普通`migration`而非warm-start**，避免测错分支。
- 扩 `tests/unit/test_semantic_video_v3.py`：experiment/配置/capture/replay全链、A仅reader替换、默认旧行为、180原settle/extra0；沿用post-hold和实际结果标签测试。保留原视频/迁移回归。

未来经授权、无Isaac并发时可用：`CUDA_VISIBLE_DEVICES=''`、`PYTHONPATH=src`、CPU线程1，指定env_isaaclab的Python执行上述三个文件及 `test_semantic_video.py`；保存独立JUnit。当前**未执行**这些命令，也没有新的实际视频或成功证据。
