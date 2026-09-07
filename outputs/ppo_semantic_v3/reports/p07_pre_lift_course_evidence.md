# P07 / offset0：已有前驱证据与现成路由的有界结论

仅PowerShell只读查验；没有运行Python/Isaac、修改课程或生产、发起新训练。本次不追加训练总账。

结论：**现有接口可在当前运行结束后的完整保存边界，使用同HEAD、同N1的已核验checkpoint，启动固定P07/offset0后缀续训；无需为课程选择本身新增实现或新MDP warm start。** 但现存记录没有直接P07目标的实际`policy_credit_start`，所以不能称P07接管已被物理验证，更不能保证尚未运行的接管接触状态、P07/P08样本数或任务成功。

## 1. 直接证据缺口

检索当前`runs/ppo_semantic_v3`下34个`run_manifest.started.json`，其中19个train；`arguments.from_phase=P07`为0。两个workspace probe也都为P06。已存在的后缀目标为P06、P09、P10；没有可供完整读取的P07/offset0专用start。

因此以下P07时序来自**P10教师前缀途中**，不是P07课程实际接管，也不是对当前c342版本新一次reset的保证。

## 2. 已完成P10首个A roll-in：P07/P08确在RR硬lift事件之前

证据run：`runs/ppo_semantic_v3/train/20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc`。

只读取其第一段`prefix_evidence.jsonl`，至第一个`policy_credit_start`停止。compact记录的阶段变化行：

| 语义边界 | 实际episode tick / 时间 | 该行phase → end_phase |
|---|---:|---|
| P06开始 | 3584 / 29.866667s | P05 → P06 |
| P07开始 | 5952 / 49.600000s | P06 → P07 |
| P08开始 | 5960 / 49.666667s | P07 → P08 |
| P09开始 | 6088 / 50.733333s | P08 → P09 |
| P10教师接管 | 7584 / 63.200000s | P09 → P10 |

该段教师decision标签计数为P06=296、P07=1、P08=16、P09=187。末完整handoff保留的RR事件历史为：首`whole_body_initial_clearance` t6901，硬qualified t6938，front crossed t7109，placed t7579。RR lift attempt列表没有更早的RR事件。因此在**这次实测教师轨迹**中，P07/P08都发生于RR初始lift及硬Q/C/P之前；不是从已完成RR抬升/越沿的历史入口开始。

限制：compact prefix decision没有每tick的完整RR contact/load/geometry。P10末start的RR当前TOP、front+91.978375mm、clear−1.638344mm、load0.484429155只描述t7584，不能倒推t5952的GROUND或载荷。没有硬Q也不等于一定没有瞬时AIR。该旧run与当前run不同HEAD，不做配对因果推断。

## 3. 与当前P06 offset240首回合的有限对照

当前run：`runs/ppo_semantic_v3/train/20260906T1738509257629Z_gc34262abffc1_04bbf6558046407990f85358d738cc17`。

已有专属首128报告`capture_approach_p06_offset240_initial.md`核实：真实A前缀P06首次观察t3584，offset240后t5504打开credit，688 teacher decisions排除；实际RR GROUND、front−259.737mm、clear−51.135mm、load0.202802，RR/RL硬Q/C/P均false，accepted且无fallback。

本次仅读首条completed episode：episode0为361 PPO decisions，在t8385 / 69.875s以`INCOMPLETE_CONTROLLER_BLOCKED`结束，末phase P06，completed stage仅P01–P05，未进入P07。该终局位于已完成update533/global72704之前（本回合策略范围72321–72681）。RR/RL均无新增硬Q/C/P；终点RR front−168.085810mm、RL front−227.768984mm，二者均GROUND。这里没有将后续运行计入或预测整块结束。

这只能说明当前这次P06入口之后的策略回合未到后续准备阶段，而旧A前缀曾自然到达P07；不能证明更换课程一定改善策略，亦不据此改变当前运行。

## 4. 现成P07路由、连续性与信用边界

- `scripts/run_semantic_ppo.ps1`现有`-FromPhase P07 -TeacherOffsetDecisions 0`直接转为CLI参数；要求v3 N1 train，阶段用`phase_suffix`。P07的定义是RR/RL workspace与RR support准备，P08是workspace_RR与load_ready_RR准备，P09才以placed_RR为目标。
- `semantic_prefix.PrefixRequest`支持P06–P13。`PrefixCreditCore.reset`从真实fresh P01调用未修改`SensorFsmController`，每个prefix decision仅`core.step(ZERO12)`；教师工作位于reset内，不经过PPO `alg.act`/storage。
- reset options拒绝历史snapshot；教师保留实际物理过程，`from_live_prefix`复用正在运行的supervisor、当前observation时钟与最后实际dispatch的nominal/tracking/bias。`NominalMotionProvider.from_handoff`以该实际命令初始化建议源，而非恢复固定历史entry vector。adapter/mapper不在接管处重建或二次reset，任务200s包含前缀。
- controller在真实8tick决策边界、目标阶段有效且offset满足时接管。未进入P07不会伪造P07；若目标在offset前已离开或teacher物理失败，记录miss并使用现有一次fresh-P01 fallback，不新增成功门。
- P07/P08的物理准备逻辑未被路由跳过；正常阶段转换不终止回合，PPO继续后缀，`task_result_scope=teacher_initialized_suffix`，不能标为当前策略从P01 full success。

**不可提前保证的准备样本边界：** 若真实接管时teacher bias尚未退休，reset-only takeover仍继续若干物理tick，直到当前bias与最后dispatch bias均0才READY。P07可以很短（旧教师轨迹仅1个decision），takeover期间也可自然完成阶段。代码不强制READY时仍为requested phase，而是明确记录`actual_phase`、`requested_phase_still_active_at_credit`。因此现成P07/0保留准备过程和原生连续性，但不保证P07/P08每阶段一定获得正数PPO样本；没有直接P07 start时不能预填该结果。这里是已有证据边界，不建议增加锁阶段/强制姿态/新的训练门禁。

## 5. 是否可直接用于之后的有限课程块

可以使用现有参数组合：`-Command train -SemanticVersion v3 -Stage phase_suffix -NumEnvs 1 -FromPhase P07 -TeacherOffsetDecisions 0`，连同届时主代理选择的**同HEAD、已完整保存checkpoint**、seed与有限决策预算，在当前Isaac退出后由主代理调用。此报告没有执行该调用或选定新预算。

同HEAD普通loader验证源自身topology/curriculum元数据、324/12 policy、checkpoint/embedded infos、actor/critic/Adam/normalizer与RNG，同时要求新storage为空；它不把源课程标签锁死为新run唯一reset分布。新run按实际env.cfg记录P07/0，并在一个on-policy epoch内保持课程固定。因此只改变固定课程选择时可正常checkpoint resume，保留已学习权重/Adam/计数；不要无故使用`NewMdpWarmStart`重置Adam。若届时runtime/MDP实际另有变更，仍遵守已有显式迁移契约，不能拿当前结论绕过它。

P07方案的有界价值是把采样起点移到已有真实前驱路径中、RR硬Q/C/P之前，保留P07/P08准备和RR首次lift学习机会；代价是P01–P06由teacher完成且不获PPO credit，因此不能替代自然P01全程学习。没有以teacher全程成功、唯一姿态或非零probe作为启动先决条件。

只读源码范围：`semantic_prefix.py`（PrefixRequest、真实target handoff、reset/step信用隔离）、`semantic_supervisor.py`（from_live_prefix/from_handoff）、`semantic_cli.py`与`run_semantic_ppo.ps1`（参数接线）、`semantic_training.py`与`semantic_migration.py`（同HEAD加载、新storage与每run固定课程）、当前v3 stage spec。固定证据读取完成，停止扩展。
