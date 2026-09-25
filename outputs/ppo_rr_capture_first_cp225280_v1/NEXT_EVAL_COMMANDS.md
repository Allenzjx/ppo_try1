# 下一个已保存模型的评估与视频命令（准备稿，未执行）

当前冻结 HEAD：`0d0f8948999222f9b8710b0eb913a9419537be83`。只在当前 continuous2048 **正常结束、完整 update 保存、唯一 Isaac 进程自然退出**后执行。CP230400 当前只是中间未评估 pointer；不要预填 CP229376，也不要预猜最终 CP 编号。

## 1. 从本次正常结束结果与实际 pointer 绑定模型

以下 PowerShell 在训练结束后读取 JSON，不初始化网络、不迁移、不回退：

```powershell
$taskRepo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$taskHead = '0d0f8948999222f9b8710b0eb913a9419537be83'
$taskPython = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
$taskOutput = Join-Path $taskRepo 'outputs/ppo_rr_capture_first_cp225280_v1'
$taskTrain = Join-Path $taskRepo 'runs/ppo_rr_capture_first_cp225280_v1/train_gain10_continuous2048_from_CP229376_0d0f894'
$taskCompleted = Get-Content -Raw -LiteralPath (Join-Path $taskTrain 'run_manifest.json') | ConvertFrom-Json
if ($taskCompleted.lifecycle -ne 'COMPLETE' -or $taskCompleted.mode -ne 'train') { throw 'Training block is not sealed COMPLETE' }
$taskPointer = Get-Content -Raw -LiteralPath (Join-Path $taskOutput 'checkpoints/checkpoint_last_pointer.json') | ConvertFrom-Json
if ($taskPointer.checkpoint -ne $taskCompleted.result.checkpoint) { throw 'Pointer and just-completed block differ; inspect before selecting a model' }
$taskCheckpoint = (Resolve-Path -LiteralPath $taskPointer.checkpoint).Path
$taskMeta = Get-Content -Raw -LiteralPath $taskPointer.manifest | ConvertFrom-Json
if ($taskMeta.checkpoint -ne $taskCheckpoint -or $taskMeta.runtime_contract.source_git_commit -ne $taskHead -or -not $taskMeta.save_load_round_trip -or $taskMeta.checkpoint_sha256 -ne $taskPointer.checkpoint_sha256) { throw 'Checkpoint sidecar/runtime binding mismatch' }
$taskLocal = [int64]$taskMeta.counts.local_policy_decisions
$taskCP = 225280 + $taskLocal
$taskAux = [int64]$taskMeta.counts.auxiliary_updates
$taskStamp = Get-Date -Format 'yyyyMMddTHHmmssfff'
$taskLabel = 'CP{0}_gain10_aux{1}_DET_{2}_{3}' -f $taskCP, $taskAux, $taskHead.Substring(0,7), $taskStamp
$taskEval = Join-Path $taskRepo ('runs/ppo_rr_capture_first_cp225280_v1/eval_' + $taskLabel)
$taskVideo = Join-Path $taskOutput ('video_review/' + $taskLabel)
if ((Test-Path -LiteralPath $taskEval) -or (Test-Path -LiteralPath $taskVideo)) { throw 'Use a fresh output directory; do not overwrite old runs/videos' }
```

这不是新的 release 门禁：只防止未封存、中间 pointer、旧路径与错误标签。实际 checkpoint 字节、sidecar、完整 runtime、gain、embedded infos/AUX ledger、actor/critic/optimizer/frozen prior 状态由现有严格 `load()` 再验证；不用另造模型加载器。若本块没有完整结束，保存现状并如实报告，不把中间版本冒称整块最终版本。

## 2. 一次自然 P01 确定性评估

```powershell
Set-Location -LiteralPath $taskRepo
$env:PYTHONPATH = (Join-Path $taskRepo 'src') + [IO.Path]::PathSeparator + $env:PYTHONPATH
& $taskPython -m wlr50_clean.ppo.semantic_rr_capture_local eval --expected-head $taskHead --run-dir $taskEval --checkpoint $taskCheckpoint --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; preserve failure.json/source evidence, do not label it successful' }
```

现有入口已实际支持：`eval` 必须显式 checkpoint；严格保存后重载；`core.reset(seed=4001)` 从自然 P01 开始；`request(..., stochastic=False)`；一次 HISTORY；无独立方向 override；评估 PPO/AUX 更新均 0。不传 `--decisions` 作为视频截断，也不使用 `diagnostic`、`initialize`、`migrate` 或 `rebind`。普通 eval 的 `--checkpoint-sha256/--manifest-sha256` 参数不参与 `load()` 分支（仅 rebind 使用），不能把传入这两个 flag 冒充额外校验。

保留窗口录制默认设置：eval 是 viewport 相机路径，不擅自切 headless。运行至自己的真实 terminal，不以 P09/P10 标签人为截片；此 route 的终点可能是 **RR 局部 capture+hold**，不是整段 RR/RL/P13 全任务成功。

## 3. 同一封存 source 导出，不手填旧 CP/计数/HUD

评估自然退出后：

```powershell
$taskSource = Join-Path $taskEval 'source'
$taskEvalManifest = Get-Content -Raw -LiteralPath (Join-Path $taskEval 'run_manifest.json') | ConvertFrom-Json
$taskSourceManifest = Get-Content -Raw -LiteralPath (Join-Path $taskSource 'source_manifest.json') | ConvertFrom-Json
if ($taskEvalManifest.lifecycle -ne 'COMPLETE' -or $null -ne $taskSourceManifest.error -or $taskSourceManifest.checkpoint_load_provenance.checkpoint -ne $taskCheckpoint) { throw 'Evaluation is not cleanly sealed for the selected checkpoint; retain and report its evidence' }
$taskMediaPython = 'C:\Program Files\Python313\python.exe'
$taskExporter = Join-Path $taskOutput 'export_formal_from_sealed.py'
& $taskMediaPython $taskExporter --source $taskSource --destination $taskVideo --execute
if ($LASTEXITCODE -ne 0) { throw 'Export failed; retain source and report the actual error' }
```

可先省略 `--execute` 只查看现有 wrapper 生成的命令；不必重复预检/加新测试。wrapper 已从本条 source 的真实 load provenance 动态绑定 checkpoint、配对 sidecar、SHA、runtime、local/PPO/Adam 计数，并由 `225280 + local_policy_decisions` 生成 CP 显示名。它拒绝混用旧视频 source、诊断介入、评估时更新、不同 AUX ledger、gain 或 tracking revision；destination 必须是新的隔离目录。不要复制上一条 CP229376 的子 exporter 完整命令或旧视频标题。

交付前只需核对新导出 manifest/画面：本条 checkpoint 与 source 路径正确；正常速度/连续 episode/失败尾段保留；标明 `finite RR AUX training lineage / rear helpers OFF`（原 FL assist 另标），gain10 控制版本及真实 local/full 结果。不能叫 pure PPO，也不能把 AUX 训练谱系叫实时后腿辅助；不能用上一次视频冒充最新模型。

本次只核对了现有 `semantic_rr_capture_local.py` 的 save/load/eval/CLI 和输出侧 `export_formal_from_sealed.py`。未启动评估、导出、Torch/PXR/Isaac，未修改生产或当前运行。
