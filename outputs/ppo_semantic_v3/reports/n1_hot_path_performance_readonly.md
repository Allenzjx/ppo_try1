# N1 当前热路径：三个可测候选，不改变正在执行的训练

只读范围：68631e9 的 N1 semantic training → 8×physics tick → native audit 路径；没有审计/设计 N8，没有运行 Python、profiler、微基准或第二个Isaac，没有修改生产/配置/环境。主4096块继续执行。

## 已知总量与不能推出的结论

已完成run134849实际8192 decisions /65525physics ticks /3060.860191s，折合 **21.4074 physics ticks/s、2.67637 decisions/s**。这里的wall来自`semantic_training.py:680–681`附近的训练计时，包含采样、后续episode reset、PPO更新和证据/checkpoint工作，不是独立测出的`sim.step`速度。不能把46.713ms/physics tick的总平均当成某一个函数耗时。

沿此次实际调用链，没有发现每physics tick重新读取配置文件或计算文件SHA：`SemanticIsaacBackend.__init__/reset`读取配置并在reset填文件hash（`semantic_backend.py:108,181–183`）；每tick `_build_authoritative_frame:290–337`复制已有`_reset_metadata`。`servo_limits_deg`只是常量选择，不读磁盘（`infrastructure/command_batch.py:193–196`）。动作配置、观察schema和reward YAML是在core构造时加载，不是120Hz重载。Checkpoint的模型/optimizer/hash/roundtrip工作发生于load/save/update边界，不能从其他legacy reset/replay helper里出现SHA就推断当前热路径每tick在哈希文件。

`semantic_cli.py:655–656`对N1明确开启native audit、关闭`collect_trace`；训练没有eval recorder的120Hz CSV/raw-observation写入hook。`semantic_training.py:719`每**15Hz策略决策**写一条JSON，而不是每120Hz tick写一个大JSON；750–752在128-decision update边界flush流，不是每tick fsync。该完成run的audit为273041065bytes（260.39MiB，平均33330bytes/decision、按总wall平均0.0851MiB/s）。这证明存在序列化/写入工作，**不证明磁盘吞吐已成为瓶颈**。

## 候选1：合并每tick native审计的小张量GPU→CPU往返

**代码依据：** `isaac_fsm_backend.py:1962–1976`在唯一已有`write_data_to_sim`之后、`scene.sim.step`之前调用`build_actuator_target_effect_audit`。`actuator_target_effect.py:121–160`依次读取4份staged/dispatched张量；每份`isfinite(...).all().item()`，通过路径再执行4次`torch.equal`，然后changed与actual/counterfactual/delta的6条向量分别`.cpu().tolist()`。成功的普通路径至少有4次显式finite标量取回、4次返回Python bool的tensor equality、7条单独CPU转换；geometry分支208–222还追加changed和三组记录的单独转换。这里数的是代码调用，不是测出的时间或独立PCIe事务。

**可选窄改法：** 以后测量一个packed、同tick、已dispatch的native snapshot，只做一次/少数GPU→CPU复制，再在CPU上完成同float32语义的finite、staged=dispatched、expected=actual、signed-zero不计effect与完整记录构造。现有冻结`bounded_drive_feedback_step/clamped/build_physical_batch`仍复用；不能省去反事实重构或改用pre-cast ACK冒充真实setter buffer。可缓存的仅是与adapter/reset-generation绑定的canonical IDs、固定顺序/常量；不能缓存任何实际target、t−1 final drive、controller/policy bias或changed flags。

这可能减少大量极小kernel/同步启动，但也可能只是次要耗时。必须保留原读取时机和所有fail-closed检查，不能增加第二次mapper advance、write、step，也不能把整个8tick窗口降采样为末tick审计。

## 候选2：后腿geometry上下文只缓存拓扑，合并live张量验证/输出同步

**代码依据：** `semantic_nominal_geometry.capture_nominal_geometry_context:100–123`在每个eligible tick重新构造body/joint名称tuple、核对12DOF映射、查唯一body index/fixed-base offset。126–141对joint q/qd、link/com pose/velocity和完整Jacobian逐张量执行finite归约+`.item()`；浮基座还有root velocity检查。149/153/171–177/189又单独取center、q、两项速度一致性误差、Jx/Jz和offset到CPU。这些值与当前物理state有关，不能跨tick复用。

**可选窄改法：** 将已验证的名称→index、canonical DOF映射、fixed-base约定和预期shape绑定到当前articulation/reset-generation缓存；只在重建/重新绑定时刷新，同时保留廉价instance/generation一致性检测。GPU上合并所需finite归约和待返回小数组，再统一复制到CPU，保留同state COM→link Jacobian转换及速度一致性核验。不得把live q/qd、J、world pose、contact/clearance/qualification缓存为旧值，不得跳过AIR/GROUND/XY资格选择。

**影响范围限制：** 此函数先在64–96行拒绝非P09/P12、不eligible或GROUND/已在平台区域的状态；它不是所有P01–P13 tick都取Jacobian。当前P10/P13长块或P06段是否受益取决于实际eligible tick数量；不能用局部小基准外推全部3060.86s。没有理由删除实时sensor/physical state验证来换速度。

## 候选3：15Hz policy/evidence边界合并重复host转换，避免无用深复制

**代码依据：** `semantic_training.py:171`先把raw action复制到CPU交给core；180行`jsonable(dict(step.info))`递归重建整个audit（`jsonable:53–64`）。训练循环704–728对mean/std、raw/logprob/value/reward及storage equality分别触发有限性/布尔读取；713–718又逐项将raw/mean/std和多个标量取回CPU，719再`json.dumps`整条记录。`semantic_env.py:211–233`将完整end-tick native/task/reward结构与8条compact tick证据装入info，`reward`和`reward_breakdown`当前包含相同奖励结构。不是每个字段都能省掉，也不能宣称JSON本身坏了。

**可选窄改法：** 以后将同一已采样raw/mean/std/logprob/value及检查结果按时点打包传输，复用已经合法取回的raw；对已证实只含标准JSON类型的immutable decision snapshot避免先`jsonable`深复制再dump的两遍遍历，仍保留`allow_nan=False`、字段/tuple→array语义、异常报错和每条采样/存储绑定。可预建不变schema/枚举/固定序列部分，但每条task/history/reward/nominal/native和terminal信息仍必须来自当下，不可缓存整条旧info。

日志应保持完整内容、顺序、global/tick和现有update/checkpoint持久化边界；不建议在尚未测量时删除重复字段、降低tick证据频率或扩大丢失窗口。GPU旧分布finite检查必须仍在physical step之前完成；不能把安全相关检查延后到8tick之后，仅为减少同步。普通JSON `.write`已缓冲，不应先入为主新增后台写线程、无界队列或假设再buffer一次一定变快。

## 本轮训练/评估结束后才可选的安全微基准

1. 先由主线程确认无Isaac/optimizer活动，再用**独立外部试验脚本或测试fixture**测现有函数，输出新性能artifact；不修改已完成run、不以旧轨迹播放替代新物理试验。按pre-audit / dispatch / audit / sim+readback / sensor+controller / observation+reward / JSON+write / PPO+checkpoint分区计数计时，先找占比；计时本身有扰动，记录开销。这里没有执行这些步骤。
2. 候选1/2可用固定fixture的实际float32 tensor/context，分别覆盖无geometry/有geometry、正常/NaN/错误IDs/错tick/不等target等反例。基线与候选应输出相同字段值和错误结果；一旦涉及GPU，在独占窗口warm-up后使用一致的同步边界，报告wall中位数/尾延迟和调用次数，不能只用GPU event忽略CPU等候，也不在待比较版本中插入不对称同步。
3. 候选3用有限数量真实已保存JSON行/同形分布tensor测转换、encode和写外部临时新文件，分开CPU序列化与文件写入；验证parse后数据、非有限拒绝、行数/顺序和flush边界等价。再决定是否值得单因素版本化微优化。若要验证实际end-to-end，主线程后续可选一个短、同seed/checkpoint/配置的单环境串行A/B计时，保留全部逐tick证据和真实terminal，不将短窗口当任务成功。

结论：前三个候选分别是**120Hz native审计的批量快照、eligible后腿geometry的拓扑缓存/同步合并、15Hz策略日志边界的重复转换**。目前只证明代码中存在相应工作，没有测出占比或承诺加速。常量/拓扑可按生命周期缓存；实时sensor、mapper历史、target、J、contact及安全/任务判定绝不可用旧缓存代替。审查到此停止，不取代或中断正在进行的训练。
