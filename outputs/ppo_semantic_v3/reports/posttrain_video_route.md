# 后续 P01 评估与录像接线：只读方案，未实现/未运行

这是训练之后的交付接线说明，不是优化器启动前置门。当前先修复真实 workspace 接口并继续 PPO；录像代码若随后改变，在保存 checkpoint 边界使用现有明确 video-only migration 审核，不放宽 runtime contract。

## 1. 现有 v3 P01 评估可直接使用

`scripts/run_semantic_ppo.ps1` 已支持 `-SemanticVersion v3 -FromPhase P01 -NumEnvs 1`。`semantic_cli._dispatch_live` 的 A 分支仍构造原 `IsaacFSMBackend + ResidualEpisodeEnv`，仅向 `_evaluation_legacy` 传 v3 task/quality路径；B/C构造显式v3 backend/action/reward/observation，`PhysicalEvaluationRecorder`同样使用v3 TaskEvaluator/quality。eval拒绝suffix/teacher-offset，正式C不会偷偷用A前缀。

下列为后续串行命令，**本分析未执行**。`$ExpectedHead`须为届时冻结并审查的完整40位HEAD。新 checkpoint 数字尚未知，不捏造；从届时真实指针取得绑定的immutable路径并校验hash，不用可变`checkpoint_last.pt`冒充历史绑定：

```powershell
$CheckpointPointer = Get-Content outputs/ppo_semantic_v3/checkpoints/checkpoint_last_pointer.json -Raw | ConvertFrom-Json
$NewCheckpoint = $CheckpointPointer.checkpoint
if ((Get-FileHash -LiteralPath $NewCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant() -cne $CheckpointPointer.checkpoint_sha256) { throw 'Checkpoint hash mismatch' }

& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead $ExpectedHead -SemanticVersion v3 -FromPhase P01 -NumEnvs 1 -Mode legacy_fsm_eval -Seed 2001 -MaxDecisions 3000
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead $ExpectedHead -SemanticVersion v3 -FromPhase P01 -NumEnvs 1 -Mode semantic_prior_eval -Seed 2001 -MaxDecisions 3000
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead $ExpectedHead -SemanticVersion v3 -FromPhase P01 -NumEnvs 1 -Mode semantic_residual_eval -Seed 2001 -MaxDecisions 3000 -Checkpoint $NewCheckpoint
```

同一组其余validation seeds2002–2005相同参数仅换seed。locked seeds3001–3005只在正式锁定checkpoint之后按声明使用。4001保留给录像及同seed的录像/不录像初始化核对。若checkpoint与当前runtime不同，C必须提供已经review的`-ResumeMigration <actual-plan.json>`；不要在eval使用`-NewMdpWarmStart`。

实际wrapper输出目录：

- A：`runs/ppo_semantic_v3/baseline_A/<timestamp_gHEAD_guid>/`
- B：`runs/ppo_semantic_v3/prior_B/<...>/`
- C validation：`runs/ppo_semantic_v3/validation/<...>/`；locked：`runs/ppo_semantic_v3/locked_test/<...>/`
- 每集：`evaluation_manifest.json`、`physical_observations.jsonl`、`native_tick_audit.jsonl`、`stage_transition_evidence.jsonl`、`physics_quality_metrics.csv`、`phase_metrics.csv`；launcher目录保存arguments/stdout/stderr。物理成功与质量改善分别判断，不能把run进程SUCCEEDED当task success。

## 2. 当前录像并未支持完整 v3，具体缺口

1. `scripts/run_semantic_video.ps1`尚无SemanticVersion参数，run/log硬编码`runs/ppo_semantic_v2/video_eval/...`。保留同一个跨版本串行锁，但root路径须版本选择。
2. `semantic_video_cli.main`两次`runtime_contract`仍默认v2；B/C的backend/core也默认v2配置。仅向继承parser塞`--semantic-version v3`不够，可能preflight已因contract不符拒绝。
3. `capture_semantic_video`中的`PhysicalEvaluationRecorder`以及`validate_semantic_video_source`里的独立`TaskEvaluator`仍默认v2；validator runtime重验也默认v2。必须由同一version绑定task/quality配置，从录制到离线raw replay一致，A只改评价选择，不改动作。
4. 当前64 extra pre-action steps改变自然初态；见下节的recorder-only替代。现有额外pre-roll后的reader/controller刷新不应继续保留，否则没有额外物理推进却多重建状态。
5. `publication_name_allowed`只允许旧名；需要添加用户新要求的`fsm_baseline_clean.mp4`、`ppo_success_clean.mp4`。C physical-success命名可以成立，但仍`improved_claim=false`，不能自动命名improved。
6. 当前semantic_video没有semantic A/C comparison入口。旧`ppo/video_artifacts.py::_comparison_filter/_encode_comparison`可参考15fps、末帧clone、libx264/yuv420p参数，但不能直接调用其完整publisher：它绑定旧FSM/PPO-manifest协议并硬写`PPO improved`标签。最小additive semantic组合应先验证两个各自完整单集source，再按原速side-by-side合成；短侧保持末帧，并从该侧结束PTS开始明确标注“已完成”。不得重播、拉伸或把两段顺序录像拼成一条episode。

## 3. 最窄 recorder-only：捕获既有180 settle的最后64ticks

可在`semantic_video.py`新增一个局部录像context，仅在这次`core.reset(seed)`期间观察backend已有`_atomic_apply`调用；暂存原bound method，原参数不变、每次恰好调用原方法一次，finally恢复。不要改变Frozen A文件、SceneFactory、mapper、reset/physics生命周期，也不要另外调用sim.step、adapter.apply/update、Reader.read、controller.step或restore。

因为reset现有顺序是 `atomic_apply(k) → sim.step → adapter.update_readback`，在下次`atomic_apply(k+1)`之前，上一物理步已完成且readback已更新。利用这个边界取图，不需要新增after-step引擎hook：

- 原生command tick0–115照常完成，共116物理steps。进入`physics_tick=116`原调用前开启recorder、仅render预热，然后capture录像global0（此时是原settle第116个完成状态）。
- 后续原调用之前观察上一已完成step，保存command tick116–178的zero ACK provenance，录像global1–63；只在global8的倍数render/encode。
- `core.reset`照常完成command tick179并返回后，保存最后一条已完成settle证据、capture global64。此时任务自然tick0仍是普通eval的tick0；第一episode command应是native180，**不是244**。
- 全程仍只有180 settle steps；64是从既有时间段选取的上下文，而不是又执行64次。9张pre/action-start网格图 global0,8,…64 只来自一条连续真实过程。
- 删除当前`step_video_pre_action_hold`循环及`refresh_after_video_pre_action_hold`/`refresh_semantic_core_after_preroll`调用；让正常reset创建唯一reader/controller/evaluator状态。共同TaskEvaluator和quality从自然episode tick0开始，不给settle任何任务或optimizer credit。
- 元数据应改为明确`pre_roll_source=existing_reset_settle_tail`、`additional_pre_action_physics_ticks=0`、settle ACK116–179、capture offset116/120s；不要伪造为旧`pre_action`额外推进语义。原frame ledger仍用录像globaltick，另绑定native完成step。成功后的post-hold继续复用lastACK nominal/combined bias/tracking，不改其动作映射。

这里是尚未实现的观察方案，不能预先宣称render无扰动。对应focused CPU测试检查原调用顺序/次数/参数、无额外step/apply/read/reset、异常恢复；下一次真实录像核对自然tick0实际观测、首任务native180及所选baseline。检测不一致时如实记录像接口问题，不能写回状态强行制造一致，也不作为训练前新增门禁。

## 4. 未来录像命令与交付路径

完成上述窄接线并提交后，计划沿用当前wrapper，仅新增版本选择；下面含**尚不存在的wrapper参数**，不可现在直接当已支持命令执行：

```powershell
& .\scripts\run_semantic_video.ps1 -Command eval -ExpectedHead $VideoHead -SemanticVersion v3 -Mode legacy_fsm_eval -Seed 4001 -MaxDecisions 3000
& .\scripts\run_semantic_video.ps1 -Command eval -ExpectedHead $VideoHead -SemanticVersion v3 -Mode semantic_residual_eval -Seed 4001 -MaxDecisions 3000 -Checkpoint $NewCheckpoint -ResumeMigration $ReviewedVideoOnlyMigration
```

若checkpoint就在完全相同录像runtime下产生，省略ResumeMigration；否则必须实际生成review过的video-only plan，不能给占位/旧plan。二者均P01自然reset、N1、同camera eye[1.45,−1.25,.8]、target[.45,0,.12]、1280×720、15fps，每8个120Hz物理tick取图；C用official loader和deterministic actor，并校验actor/critic/optimizer/normalizer未变化。

录像source拟继续既有kind命名：`runs/ppo_semantic_v3/video_eval/baseline_A/<run>/source/` 与 `.../validation/<run>/source/`。raw `actual_viewport_video.mp4`、source manifest、frame/decision/native/raw ledgers继续保留；失败仅diagnostic，不能换成功文件名。

用户正式输出：

- `outputs/ppo_semantic_v3/videos/fsm_baseline_clean.mp4`
- `outputs/ppo_semantic_v3/videos/ppo_success_clean.mp4`
- `outputs/ppo_semantic_v3/videos/fsm_vs_ppo_success.mp4`
- 只有配对改善证据成立，才另外输出`ppo_improved_clean.mp4`、`fsm_vs_ppo_improved.mp4`。

保留现有`validate_mp4`完整decode、H.264/yuv420p、15fps与逐帧PTS验证，native frame ledger绑定；comparison另外绑定两个source、帧数与hold-final-frame区间。当前固定64pre+184post会给接近200s的任务额外约2.07s，`frames_for_episode<=3000`因此可能拒绝技术交付；必须区分“物理完成”与“上下文总片长超标”，不可改变任务deadline或裁剪途中动作来隐藏。实际若遇此边缘情形，再按用户≤200s且短pre/post需求确定最小上下文裁取；目前不为尚不存在的长成功视频增加新训练门。
