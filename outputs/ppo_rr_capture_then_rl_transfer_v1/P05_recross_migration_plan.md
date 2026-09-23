# P05 same-AIR recross：迁移方案与实施状态

更新：当前录像自然结束后，root已选择无新增backend反馈方案；迁移模块与定向测试已实施，尚未正式CUDA发布。最终配置值是 `p05_preedge_same_air_recross_v2` 与 `current_capture_or_completed_handoff_after_original_window_v1`。实际CPU只读校验已证明首版410完整训练payload等于原389的精确零追加，映射state SHA256为 `41d1a096d97f8af55ffa1b63584756c4d2dc95a4babe1f655840c81d105cc663`；220544/1688/33760和RR branch 0/0/0保持。以下原规划文字保留其提出时的范围，最终实现优先于待定措辞。

2026-09-22。仅 PowerShell 静态读代码/已封存 JSON；未运行 Python、测试、FFmpeg、Isaac、优化或 checkpoint 写入。当前录像按原控制版本自然结束；本文件不改变其结果。

## 结论

建议在既有 `semantic_rr_capture_migration` 内增加一个**精确限定的第二目标控制版本**，保留首版条件不变。由于 RR410 初始 checkpoint 没有任何新增学习，可从同一实际389源重新执行相同的21列零追加；这是一个替代的零更新初始控制版本，不是继续了410策略训练，也不是重置已学网络。不得静默复用原迁移计划或覆盖原410文件。

若在实施边界发现已有新增 PPO/AUX、任何新增列/Adam矩有变化，或源绑定不一致，这条简化路径必须拒绝；那时另做410→410完整状态保留迁移，不能回退到389丢弃学习。

## 已核实的绑定

- 原389：`outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000220544.pt`，SHA256 `56239937cd0aaccdc8b7ea36c6266a41b3bed9df67b00f84a06806d2a6fa09b7`。
- 已发布首版410：`outputs/ppo_rr_capture_then_rl_transfer_v1/checkpoints/history/checkpoint_rr_capture_step_000220544.pt`，SHA256 `46ff3a18024ad0fa94a30d393dbfe8bd75913a91da825c63bc524fe0e5547fff`。
- 首版410 HEAD：`047e8a78be6170b09d7d082fabf720a375fd16e7`；runtime hash `595ef76307696042fee22383c777b199a32a0cc830d02762a121d3b0fab890d0`。
- 两者实际计数220544 decisions /1688 PPO /33760 Adam steps；410 `rr_capture_transfer_branch_counts` 为0/0/0，迁移新增AUX为0；有效LR为1e-5，Identity，保存设备 `cuda:0`。
- 首版官方保存/独立重载回执：`CP220544_RR410_publication.json`。这些 JSON 证明已记录的零信用边界；张量逐项同一性仍应在实际发布前的有限测试/重载检查中核验，不以 JSON 自述替代。

## 精确控制版本与配置范围

与 primary agent 协调的**待 root 确认名称**（timeout名称要随最终谓词一起确定）：

```yaml
nominal:
  p05_preedge_approach_recovery: p05_preedge_same_air_recross_v2
p05_finite_recovery_timeout_semantics: current_capture_or_completed_handoff_after_original_window_v1
```

第一项沿用既有 key，显式换成新值；第二项新 key 只管P05。两项应成对启用；未知值、只启用一项均拒绝。时间/距离仍取原spec，不加新阈值。配置迁移只允许 `stage_task_spec.yaml` 的这两处解析值变化；其余5个当前RR配置应逐字节相同，包括execution revision、410 observation schema、action/reward/quality。新控制版本通过新key值、实际新HEAD和迁移factor识别，不必再改execution revision制造额外差异。

Root正在评估更窄实现：age<原window end保留机会；之后仅当前 `allow_capture_continuation` 或本阶段真实goal全部满足、等待既有handoff可继续，其他状态恢复原终止。如果有效FL DESCEND已被当前allow覆盖，**不增加FL ACK反馈或backend修改**。本迁移方案优先采用这个无额外反馈路径；不能因本方案出现“committed DESCEND”字样就强行增加实现。最终精确值应表达实际采用的谓词，placeholder不得进入生产配置/迁移计划。

从原389源构建第二variant时，精确目标等式为：原389 task完整字典 + 首版 `rr_capture_continuation_semantics` +上述两处。其余execution/observation仍必须满足原先精确389→410追加规则。不得改成“忽略nominal所有字段/忽略revision/忽略unknown key”。

## 白名单如何严格扩展

现有 `REVIEWABLE_CODE` 已含 `semantic_supervisor.py`、`semantic_backend.py`、`semantic_rr_capture_migration.py`，因此**不需要扩大允许路径集合**。需要扩展的是可识别的精确控制variant及其校验，不是放松白名单。

第二variant额外绑定首版410 checkpoint及其sidecar/原迁移plan指纹，读取该不可变runtime作为中间审查边界。原389→新目标的完整reviewed-code映射仍逐项校验；另外把首版410→新目标runtime的**实际精确差异集合**写进factor。最窄实现为：

1. `semantic_supervisor.py`：纯same-continuous-AIR recross条件和P05局部有效恢复超时；旧模式分支不变。
2. `semantic_rr_capture_migration.py`：该精确variant、来源绑定/验证及版本记录。
3. 当前RR `stage_task_spec.yaml`：上述精确两处。

只有最终谓词审查证明必须独立消费已提交FL snapshot，且root明确采用时，才把 `semantic_backend.py` 的同拍反馈接线作为一个具体第四路径列入实际差异；不得预先强制或宽泛允许它。这个可选路径也只允许提交state_after/native dispatch tick/下一episode observation tick，不能再次advance。

本方案不需要修改actor、policy distribution、observation codec、FL/RR assist推进代码、reward config、mapper、cap、sigma或训练save-carry路由。若实现确需更多生产路径，必须明确列出原因后再审，不自动把路径加进集合。tests变化不属于运行时契约，但应正常提交。

建议factor内明确记录 `target_control_revision`、`control_semantics_changed=true`、`termination_semantics_changed=true`、`physical_dynamics_changed=false`、`reward_config_changed=false`、`same_mdp_claimed=false`。终止时间变化会改变轨迹/回报，不能因为reward系数字节没变就声称MDP相同。

被取代首版边界的来源记录应采用小型对象：checkpoint/sidecar/plan path+SHA、runtime HEAD/hash、RR branch origin/counts、零学习映射验证结论。保留原首版 `rr_capture_transfer_migration` 对象或其独立保存的不可变plan完整绑定，不覆盖原文件；不要把整个旧metadata/运行时/大AUX报告复制进新factor。

## 权重、优化器、谱系与新文件

继续使用既有官方source-device publisher和严格加载，不使用 `NewMdpWarmStart`，不改CPU device metadata，不关闭实际CUDA RNG。原389全部actor/critic参数原样复制；只新增21列0；Adam旧12参数的step/groups/LR/矩原样复制，两个first-layer矩仅新增21列0；Identity、完整Python/NumPy/Torch/CUDA RNG保持。对首版410还应验证它正好是这个映射，没有任何学习后差异。

五条既有origin、全部migration与counts、4条AUX events（front96/96、RR7/8、合计103/104）及旧独立7/8账本逐字段保留。新RR origin仍是实际220544/1688/33760，计数0/0/0；不能把旧五条origin改成当前源，也不能把首版录像或新迁移算PPO。第二控制variant无需再增加一个虚构学习分支；其实际控制版本和被取代零更新边界明确写入RR迁移factor即可。

建议唯一目标名：

`checkpoints/history/checkpoint_rr_capture_p05_recross_v2_step_000220544_g<实际新HEAD前12位>.pt`

同名manifest、独立新plan/publication receipt均不得覆盖既有文件。初始发布不修改latest pointer。保存并独立fresh reload通过后，视频/训练都直接加载这个已发布410文件，不通过旧checkpoint+临时runtime宽松覆盖。旧rollout与未完成transition不继承；只从合法自然P01新采集，迁移新增PPO/Adam/AUX全部为0。

## 有界验证清单（在当前录像结束后）

- 实施范围：首版410→新HEAD生产差异精确为最终批准的3路径（如确有独立反馈需求才为4）；5个配置字节相同；task字典只有两处；首版旧mode行为与旧migration约束回归不变。
- recross正反例：同AIR streak覆盖真实cross可用；差1tick/跨ground或壁面接触再AIR/伪旧attempt/已placed/缺失与非整数时间拒绝；原first-approach、geometry/support、source owner及 `[30,40)` 有限窗口不变。
- timeout正反例：P05原窗口后无合法路径不能无限warning；当前allow或真实goal完成等待handoff按最终谓词保持接续；P06+和P09 RR原规则不变；global200与硬安全优先。如最终确需独立DESCEND条件，另测stale/BLOCKED/耗尽反馈拒绝，不能以任意mode字符串放行。
- 接线：优先复用当前已测allow/goal，无新feedback。如果最终确需FL反馈，才核验本拍验证ACK state_after、native dispatch tick与episode observation tick区分、reset清理及wrapper委托；两种方案均不得再advance助控或nominal source时钟。
- 观察语义：仍为410且数值编码规则未换；FL snapshot已在旧389，任务时间/当前几何/nominal现有编码可读。same-AIR判据消费既有evaluator历史，**不能声称全部底层历史均已显式编码或actor观测已严格Markov**；这是显式控制语义变化，而非新增隐式计时器。
- 迁移负例：非零新PPO/AUX或变过的首版410权重/矩必须拒绝简化重追加；wrong namespace/schema/source/RNG/cap/reward/额外task键都拒绝。
- CPU合成：原389→410 actor conditional μ/σ/critic在同389输入和任意附加输入下连续；Adam零追加及全部老状态保留；正式保存+独立重载+普通后续save继续携带完整谱系。
- 实际零更新发布：沿 `cuda:0` 原可见设备恢复完整RNG，验证实际源/目标映射与计数、Identity、完整Adam、旧账本和fresh reload。发布前后不得做任何优化。

以上只确认实现/保存语义，不是新增成功门禁，也不保证新轮驱建议、RR落脚或全程越障成功。新版本物理结果必须由之后真实运行单独给出；当前旧版本录像的P05未完成如实保留。
