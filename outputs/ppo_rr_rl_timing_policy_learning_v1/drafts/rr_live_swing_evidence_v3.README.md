# RL 当前摆动接线 v3 — outputs-only 草稿

**尚未应用生产；未运行测试、模型、Torch 或 Isaac。** 依据封存 P10 221057–221312；当前生产仍为44219。两个patch通过 `git apply --check`，生成内容只做AST语法解析。

- `rr_live_swing_evidence_v3.production.patch`：只修改 semantic_rear_policy_timing.py 和 semantic_supervisor.py。
- `rr_live_swing_evidence_v3.tests.patch`：新增真实 TaskEvaluator 输入链测试，不手写 RL current_lift_valid/motion_continuation_allowed 作为成功前提。
- 配置、same419迁移及版本冻结由配套 migration 草稿处理；基于新学 CP221568，不回祖先，不重置网络/optimizer。

## 实证缺口与边界

在v2，RL本次t6183/51.525测得qualified lift，221062–64与221066是实际AIR＋active_attempt。但current_legs只为RR添加current_lift_valid/motion_continuation_allowed，RL依赖读取缺键，全部256的rl_current_swing=false。
221065是OBSTACLE_AMBIGUOUS，不是AIR；t6217/51.8083真实ground撤销本次RL资格，旧first-event6183仍保存。不能用旧tick或历史placed当当前摆动。
这个早期窗口RR仍承载；接口bug不证明它是本次RR悬空或FL负端饱和的唯一原因。

## 最小设计

新mode `rr_live_swing_evidence_v3` 保留v1/v2原行为，且显式作为RECAPTURE_MODES成员继承v2 recapture与RR retention，两处不再遗漏新mode。

v3仅给RL增加 current-qualified latch，由既有measured lift谓词取得；**任意真实ground返回都会撤销**，即使crossed/placed历史仍在。同时清当前attempt、初始净空和短窗输入，防止旧attempt/旧窗口下一拍重新授予。历史active_lift/cross/placed与first event tick按原规则保留。新的current-qualified tick仅为诊断，不是调度计时器。

当前资格在非ground edge过渡保留，但依赖仍需要实际AIR才称rl_current_swing；edge/TOP不冒充AIR、不授予承载。真实新抬升可以重新取得当前资格。所有current字段来自TaskEvaluator；不改物理阈值、接触传感器、资产、控制器能力、轮速方向或已执行source时钟。

当前RL资格映射到既有active_lift_history的RL观察槽（同RR做法），同时保留history的完整事件。419维数量/位置不变，但**该槽在v3是版本化当前资格语义，不再是cross之后纯历史值**。现有9维rear timing显式显示swing/recapture/等待；P12 lane、public task、RR retention/潜能共用同一dependency。不要宣称旧410前缀语义完全不变；迁移必须记录该差异并重新采集rollout。

## 测试草稿涵盖

本次真实资格AIR／仅短AIR／edge非AIR后恢复AIR／ground前后历史与当前分离（含已crossed/placed）／ground后无新动作不能重授／新真实抬升恢复／RR失载但RL合法在途时继续P12 RL lane与RR既有retention，随后RL落ground只暂停RL lane、wheel clock继续／现有RL观察槽更新而历史保留／身体碰撞和传感器不完整拒绝／v1与v2真实evaluator原输出与历史行为不变。

建议冻结后先跑这个新文件和既有 v2 recapture/control/观察回归；当前草稿仅静态检查，不是已通过测试或真实动作改善证据。它修RL接口，**不解决**FL knee长期负端饱和、RR捕获保持不足或整段任务能力；下一实测需验证这些阻碍，不能据此直接声明dependency全解。
