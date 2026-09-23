# P05 recross v2：严格迁移实现

只修改生产 `semantic_rr_capture_migration.py` 和其定向迁移测试；root负责supervisor/task两项配置。无actor/backend/assist/codec/奖励系数更改，未启动Isaac、CUDA发布或优化。

新增第二variant精确匹配两个已批准值：`p05_preedge_same_air_recross_v2` / `current_capture_or_completed_handoff_after_original_window_v1`。首版约束保留。首版410到目标运行时的差异必须恰好为supervisor、迁移模块、当前task_spec三路径；其余五份配置逐字节相同。旧389到目标仍执行原完整代码/配置审查，不存在SHA豁免。

`build_rr_capture_migration` 新增明确参数 `initial_checkpoint` 和 `initial_plan_path`；旧API调用不变。validate从小型 `superseded_zero_update_boundary` 重新检查，不复制大AUX账本。普通checkpoint保存继续携带完整RR migration factor及全部旧分支。

`verify_zero_update_rr_boundary` 对两个实际checkpoint只读CPU加载，核验首版plan/sidecar/embedded metadata、零新增计数、旧全部origin/AUX/RNG、source-device runner_config，并将原389映射后的完整payload（除各版本infos）与首版410逐值哈希比较：actor/critic全部参数、Adam所有groups/moments/steps、iter及所有其他保存字段均在比较范围。任何非零学习或变动拒绝，不回退丢学习。

实际CPU检查：PASS、helper exit0。原389 SHA `56239937cd0aaccdc8b7ea36c6266a41b3bed9df67b00f84a06806d2a6fa09b7`；首版410 SHA `46ff3a18024ad0fa94a30d393dbfe8bd75913a91da825c63bc524fe0e5547fff`；完整映射state SHA `41d1a096d97f8af55ffa1b63584756c4d2dc95a4babe1f655840c81d105cc663`。实际计数仍220544/1688/33760，RR0/0/0，没有新学习。

发布入口（仅root在提交后执行，保留CUDA原设备可见）：

```powershell
python outputs/ppo_rr_capture_then_rl_transfer_v1/publish_recross_migration.py --publish --expected-head <实际新HEAD>
```

无 `--publish` 时仅只读校验。唯一新文件名：`checkpoint_rr_capture_p05_recross_v2_step_000220544_g<HEAD12>.pt`；独立plan/manifest/publication receipt，不覆盖旧产物、不更新latest pointer。官方source-device保存和fresh独立重载后，再对首版410比较actor/critic/Adam/Identity/RNG/runner_config完全相同；物理结果仍为NOT_YET_EVALUATED，新增PPO/Adam/AUX全部0。

定向测试包括：首版389→410旧行为、任意附加状态下13phase分布/critic连续、第二variant官方保存/独立重载/后续普通save-carry、额外代码/奖励字节/未知或缺失mode拒绝、重哈希后actor旧列/新增列/critic/Adam矩或step/iter/RNG/AUX/counter改动拒绝。仅临时synthetic checkpoint，测试carry的计数不是真实更新。首轮28/29通过，唯一fixture未显式传临时git root已修复；最终完整29项重跑全部PASS、exit0，CPU helper已退出。`git diff --check`通过。
