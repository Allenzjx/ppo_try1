# 待保存边界应用：再接触窗口修正草稿

状态：**仅 outputs 草稿，未部署**。当前训练块及 Isaac 运行保持不变。

补丁：`outputs/ppo_p05_hip_only_continuation_v1/pending_recontact_fix.patch`，采用 apply_patch 格式。等待当前 2048 块自然结束、确认完整 update/checkpoint 已封存且无活动 Isaac 后，才读取补丁并通过 apply_patch 应用。

补丁仅修改一个生产文件 `src/wlr50_clean/ppo/semantic_capture_assist.py`，并新增 `tests/unit/test_semantic_capture_assist_recontact.py`：

- 已启用补全器从 HOLD 真正转为既无 TOP surface、也无 obstacle pair 时，仅以当前 gap 重建局部进度窗口。
- 保留 hip_entry、累计 hip_target、knee_hold、20° 总行程、真实机械余量、所有安全/跟踪限制；不是每拍 AIR、每次 BLOCKED 或 phase 变化都重置。
- 新常量 `CAPTURE_ASSIST_FEEDBACK_REVISION = "hold_to_air_progress_window_v2"`，通过 snapshot 的非数值 `feedback_revision` 字段记录。
- 原 `CAPTURE_ASSIST_MODE`、12 个数字状态特征、389 维策略契约、reward、caps 和 sigma 均不改。执行语义仍发生变化，须按 root 准备的严格 same389 迁移绑定保留权重/优化器/RNG，并在新版本下重新采集，不能混入旧未完成 rollout。

验证：在独立短 CPU 进程中读取补丁，仅对内存中的模块应用 hunks，然后将草稿测试作为内存 pytest module 收集执行，**14 项全部通过**；包含独立 actuator audit 的完整状态重算。没有写入生产模块或 tests，也没有进行真实仿真。这不能代替应用后运行真实测试和续采。

正式应用后的最小测试：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPATH='src'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m pytest tests/unit/test_semantic_capture_assist_recontact.py tests/unit/test_semantic_capture_assist.py tests/unit/test_actuator_target_effect.py tests/unit/test_semantic_residual_adapter.py -q
```

应用前如实际生产内容已变化，先重新核对补丁上下文，不覆盖并发工作。旧 CP201728 P06 没有触发这次误报；不能用此草稿声称已经解释或解决了其后续未完成。
