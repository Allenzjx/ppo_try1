# 可选FL诊断日志候选：未运行、未采用

本候选只准备后续证据工具；是否使用仍由根任务在新checkpoint完整det评估后决定。**未启动Isaac、未执行任何物理探针，新增PPO decisions / updates / optimizer steps均0。** 没有aux更新、教师部署、reward/HISTORY/σ/ρ/mapper变更。

## 差异范围

新增 `direction_probe_preaction_candidate.py`，直接import旧 `direction_probe.py` 的 `FiniteDirection`、`trigger`、`sha/write`，原−3°FL hip、12决策quintic ramp、75决策hold上限、capture后24决策、原release和尾窗实现完全不复制、不修改。仍为entry REQUEST anchor减3°，**不是每拍在当前策略或HISTORY上递归减3°**。仅开放 `control` 和 `FL_minus3` 两个候选case，baseline固定为checkpoint deterministic policy。

新main继续复用现有backend、env、PhysicalEvaluationRecorder、HeightDiagnostics、ProbeGeometry和单Isaac资源锁；加载改为官方 `semantic_video_cli.checkpoint_loader`，它复用正式load、N1审计forward和参数/optimizer/normalizer不变检查。没有复制仿真/控制实现，没有修改公共运行代码或旧probe。

绑定为当前 `task_conditioned_hip_wheel_v1`、`task_conditioned_hip_wheel_sigma_v1`、372维实际schema。显式CP路径须在本分支history；验证hash、roundtrip、正branch训练量、六配置和完整current runtime。**不自动选择旧checkpoint、不隐式执行迁移；archive-only HEAD差异也拒绝，需在现有正式流程中另行审核处理。** 因而未来CP189xxx只要接口/完整runtime相同即可用，不把当前187904测试fixture硬编码为运行恢复点。

## 新证据

每次动作先深拷贝当下observation/history；一次原det forward后核验物理clock/history未变，随后在`core.step`之前写入并flush `record_kind=pre_action`：

- 原始encoded全372、官方loader实际float32输入全372及其小端字节hash；实际schema解析出的history索引/缩放，拒绝当前history与输入不一致。
- 全部真实HISTORY：previous raw、previous filtered REQUEST、前后applied drive及nominal；实际原network/conditional mean、HISTORY center、σ、cap-transition状态都来自这一拍原审计request。
- 干预前det raw、干预后raw、差量、12通道物理caps、单位（前8deg/后4rad/s）；`cap*tanh(raw)`仅标注为**许可/限速前候选**，不是最终REQUEST、actuator target或实测。
- 分别记录phase residual许可、runtime许可、safety许可及乘积；人工override selector另列，不能将它当作nominal mask。原source nominal及previous ACK单列。

随后`step_result`用相同decision和pre receipt hash关联step_info/native物理证据。若step异常，pre记录仍已写出，并记录`step_exception`、last_core_frame_tick和实际physical summary；不捏造未返回的terminal step_info、reward/GAE。`probe_entry.json`也保存同一个完整pre receipt。

干预之后的“原det均值”是**当前已经受干预的真实状态/HISTORY上的det输出**，不是不存在的未干预轨迹。需要比较轨迹时，应使用同CP、seed4001、同runtime、自然P01的独立control记录；不能把同一receipt的pre/post两个动作候选冒充两次物理结果。control也是独立诊断标签，不改称正式PPO成功。

## CPU校验及未执行调用模板

`test_direction_probe_preaction_candidate.py`：14 tests PASS；覆盖全372/有限数、schema历史绑定、phase及decoded REQUEST对应、det均值一致、无额外forward/draw、只改FL hip、mask职责/nominal保留、固定−3°与原hold/release、CP187904真实封存metadata接口、错误experiment/layout/branch及隐式迁移拒绝。只读取metadata，不加载网络forward、不加载optimizer、不运行物理。

CPU复查：

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs\ppo_task_conditioned_hip_wheel_v1\test_direction_probe_preaction_candidate.py -v
```

以下**仅模板，没有执行**；需根任务先决定使用、确定实际最新兼容CP、等待唯一Isaac资源空闲。尖括号必须替换，run-dir必须是全新独立目录，control与FL_minus3不得同时启动：

```text
python outputs/ppo_task_conditioned_hip_wheel_v1/direction_probe_preaction_candidate.py
  --case <control或FL_minus3>
  --checkpoint <已核验的最新同版本checkpoint绝对路径>
  --run-dir <全新独立诊断目录>
  --max-seconds 85
  --expected-head <实际完整40位HEAD>
  --enable-physical-diagnostic
```

85s只是原允许范围15–85s的上限示例，不代表决定运行时长或采用该诊断。缺少显式enable标记立即拒绝。CPU结果仅证明日志语义与有限动作复用，不证明Isaac入口、捕获、保持、视频或完整越障成功；当前实际物理诊断数据仍为0。
