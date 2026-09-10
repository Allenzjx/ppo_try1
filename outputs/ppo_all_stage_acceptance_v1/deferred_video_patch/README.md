# Deferred video routing / exact HISTORY372 compatibility

**未应用、未执行。** 本目录六个生产文件是 a8 源码的修改副本，不是当前运行代码。当前 Isaac 不应导入或启动此目录内容。只使用报告文件 `apply_patch` 写入；没有改实际 `src/`、`scripts/`、`configs/`、`tests/` 或 Git 提交。

## 最小应用范围

`application_inventory.json` 列出六个对应路径及**已记录 a8 manifest 的原 SHA**；没有重算任何源文件/检查点 hash，也不伪填未来 target SHA。未来应逐文件审阅镜像与实际 source 的差异，在 root 明确许可的安全边界应用这六个差异。不要复制整个目录覆盖仓库，不要将当前正在产生的 checkpoint 提前视为视频成功来源。

- 视频三文件贯穿同一显式 `experiment_id`；缺省 v2/v3 路由和旧324字段保持。新 source manifest 仅在显式 experiment 时加字段；独立 replay 同时核验 selector、runtime 与原配置字节。
- C 使用预检验证的 layout 构造真实 HISTORY actor；仍调用正式 `load_semantic_checkpoint` 和 `actor(..., stochastic_output=False)`。不导出旧 mean，不改 rho/sigma，不加网络/列/新缓存。
- A 仅在显式 all-stage 选择下复用普通 `_evaluation_legacy` 的 `SemanticSensorReader` 配方；冻结控制器、mapper、reset/settle 不改。A 校准与 C 固定 chassis 参考不同仍须披露，不称严格动态初态配对。
- exact migration 的372特例仅允许完整 HISTORY/N1、源/目标相同 layout/policy contract、相同六个配置绑定；policy/critic/action storage 形状明确核验。旧324 plan 不添加新布局字段；旧非 video372 路径不放宽。`VIDEO_FILES` / `INSTRUMENTATION_FILES` 白名单和各 factor 互斥不改。
- 仍走原普通 exact load，保留 Adam/实际LR、actor/std、critic、normalizer、训练 RNG、计数、预算和origin；空 rollout，真实 P01 reset。**不使用 NewMdpWarmStart，不产生 fresh Adam 初始迁移检查点。**

## 待 root 许可后验证（目前一项也未运行）

新草案 `tests/unit/test_semantic_video_all_stage_deferred.py` 导入的是**未来已应用的实际 package**，不是镜像。包括 experiment/replay错配、A仅reader替换、HISTORY372 plan/预检、缺contract/错误layout/混gamma/配置/factor拒绝、真实CPU128源更新后视频 exact reload、Adam/RNG/所有模型张量与实际kernel一致。真实CPU用CPU生成源；不得删掉真实CUDA源 RNG来强行跨设备通过。草案尚未经过语法编译/collection/运行，任何失败都应作为未完成兼容性问题处理，不能放宽生产保护。

未来在无 Isaac 且获授权、审阅应用六文件后，可运行以下有限集合（不是当前执行命令）：

```powershell
$env:CUDA_VISIBLE_DEVICES = ''
$env:PYTHONPATH = 'C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src;C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/tests/unit'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' -B -m pytest -q tests/unit/test_semantic_video.py tests/unit/test_semantic_video_v3.py tests/unit/test_semantic_video_migration.py outputs/ppo_all_stage_acceptance_v1/deferred_video_patch/tests/unit/test_semantic_video_all_stage_deferred.py --junitxml=outputs/ppo_all_stage_acceptance_v1/deferred_video_patch/future_cpu_receipt.xml
```

CPU草案在真实RSL用例内部限定1线程、退出恢复；不调用 `set_num_interop_threads`。旧324 golden/旧视频测试保留，不叠加旧receipt冒充本候选通过。之后如需录制，必须针对最终 source CP 与未来已提交 target runtime 用既有 `build_migration_plan(..., video_review={"reason": ...})` 生成精确绑定文件，经 `--resume-migration`/`-ResumeMigration` 加载；现在不生成目标计划或目标hash。

## 明确保留的未解决问题

本补丁不改 `PRE_TICKS=64`、`POST_TICKS=184`、`MAX_FRAMES=3000`、frame ledger、物理endpoint、录制/解码/发布判据。现公式只容纳最多23751 episode ticks（197.925s）及完整额外首尾；共同任务200s与额外录像context的冲突仍需未来单独决定，不能裁尾/加速掩盖。本版任务已有1s完成后观察，原视频还额外hold184tick；其停止轮/保持旧命令不是继续策略。最后ACK含geometry/headroom/tracking参考时的post-hold精确行为，本草案不冒称覆盖，仍保留原专属测试并建议一个后续窄反例。

无成功CP、完整视频或稳定性优越证据由此产生；也不放宽原A/C成功合成要求，不修改旧A视频容器。当前训练和首4096/eval不等待此候选通过。
