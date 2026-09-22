# 有限FL诊断日志：候选审查与首次真实运行

本工具准备时没有运行Isaac。根任务完成CP189952两种模式的完整评估后，于2026-09-21 08:35 UTC首次启动`FL_minus3_CP189952_20260921_01`（40秒上限）。当前结果以该独立run的封存manifest为准，不把候选测试当作物理结论。新增PPO decisions / updates / optimizer steps始终均0，没有aux更新或教师部署。

## 复用与边界

`direction_probe_preaction_candidate.py`直接import旧`direction_probe.py`的FiniteDirection/trigger。原FL hip −3°、12决策quintic ramp、75决策hold上限、capture后24决策、原release和尾窗保持不变；不是每拍向HISTORY递归减3°。仅开放control/FL_minus3，baseline固定为checkpoint deterministic policy。

新main复用现有backend、env、PhysicalEvaluationRecorder、HeightDiagnostics、ProbeGeometry及单Isaac资源锁。用官方semantic_video_cli.checkpoint_loader加载并审计一次真实det forward，检查权重/optimizer/normalizer不变。旧probe和所有生产代码不改。

版本绑定为task_conditioned_hip_wheel_v1 / task_conditioned_hip_wheel_sigma_v1 / 实际372维schema。显式checkpoint必须属于本分支history；验证hash、roundtrip、正branch训练量、六配置和完整current runtime。拒绝隐式迁移及archive-only HEAD差异；未来兼容CP不硬编码旧决策数，187904仅为CPU metadata测试fixture。

## 新日志

每次forward前快照当前全372及真实HISTORY；forward后校验clock/history不变，在core.step前写入并flush pre_action记录。包括原encoded观测、精确float32输入及字节hash、schema history索引/缩放、原det request/network/conditional mean/HISTORY/σ、人工raw前后与物理caps/单位、phase/runtime/safety残差许可、独立override selector、source nominal、previous ACK。

cap*tanh(raw)明确是许可/限速前候选，不冒充最终REQUEST、actuator target或实测。许可mask只作用于residual，override selector不是nominal mask。

step_result或step_exception以decision和pre receipt hash关联。异常分支只调用一次physical.summary（该函数会exclusive创建CSV），把同一缓存写入exception行及最终manifest。真实observed endpoint来自PhysicalEvaluationRecorder._last_frame；last_core_frame_tick另列，明确core.frame可能滞后、环境step未返回。不捏造terminal step_info、reward或GAE。summary写入失败记录独立错误，不重试、不替换原始step异常。

干预后的“原det均值”是在已经受干预的实际状态/HISTORY上的det输出，不是未干预轨迹。若需要轨迹对照，应使用同CP/seed4001/runtime自然P01的独立control记录，不能把两个动作候选当成两次物理结果。诊断control或FL_minus3均不能改称正式PPO成功。

## 收尾核验

旧probe的torch/tensordict导入位于旧main内部，import旧模块本身不保证Windows预载顺序。新main已显式保持torch → tensordict → AppLauncher，仍仅在未来明确启用且拿到资源锁后执行。

CPU测试文件：test_direction_probe_preaction_candidate.py。原14项加异常缓存/observed endpoint、summary失败保留原异常、显式导入顺序3项，共17项。只运行纯CPU日志测试，不加载网络、optimizer或Isaac。

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs\ppo_task_conditioned_hip_wheel_v1\test_direction_probe_preaction_candidate.py -v
```

以下是调用模板，不代表两个case都已执行；必须由根任务决定使用、确定最新兼容CP并等待唯一Isaac资源空闲。两个case不得同时启动，run-dir必须全新：

```text
python outputs/ppo_task_conditioned_hip_wheel_v1/direction_probe_preaction_candidate.py
  --case <control或FL_minus3>
  --checkpoint <已核验的最新同版本checkpoint绝对路径>
  --run-dir <全新独立诊断目录>
  --max-seconds <15至85内的显式有限预算>
  --expected-head <实际完整40位HEAD>
  --enable-physical-diagnostic
```

缺少enable标记立即拒绝。CPU测试仅验证日志语义、封存字段和有限动作复用；不证明真实接触、保持、视频或完整越障成功。新增真实物理数据保存在上述独立run，不能计入PPO采样量。
