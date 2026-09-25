# 可选 RR knee +25 / +30 小诊断：仅准备，未运行

状态：INDEPENDENT_DIRECTION_DIAGNOSTIC / PPO=0 / AUX=0 / NOT LEARNED SUCCESS。
不抢当前DET；按主线程决定，先完成当前DET和新source的512步真实PPO，再决定是否使用。
本文件不是训练、正式评估或自动capture assist的配置。

## 依据与可表达性

旧run：diagnostic_hipminus25_kneeplus20_ecf205e，runtime
ecf205e80094693938057589fc91f7f31082ef89，local0的冻结CP225280组合。
在入口已下发FINAL [hip +7.4851438°, knee -58°]上平滑执行相对delta[-25,+20]；
不是每拍重复累加。93.5s以INCOMPLETE_CONTROLLER_BLOCKED结束，没有TOP/hold。

旧source manifest所封存的video_policy_decisions.jsonl SHA：
1be2baecd5a92d1ba72c24c4233a668b80a7e4e727f07f10dae06534738be31b。

同一run、同一端点8224（68.5333s）的证据：

- RR FINAL [-17.514856°, -38°]，实测[-17.701442°, -37.754707°]。
- gap=4.059061mm，合法顶部XY，前缘内47.736469mm，RR力0N，AIR，非TOP。
- FR/FL/RL真实bearing约13.0988/2.5078/12.9930N；不是把AIR RR算作支撑。
- 同拍mappedN RR=[-8.15,-39.05]，controller bias=0，RR requested=effective，
  headroom无裁剪。膝有效安全区[-58,208]；该P09膝residual cap=36°。
- 该cap也由真实请求+1.05°和实际issued raw=0.02917494的tanh关系复核，
  不是从最终角度差猜测当拍policy贡献。

| 总knee delta（从入口FINAL） | 旧入口下绝对目标 | 相对同拍N的residual | 对应raw |
|---|---:|---:|---:|
| +20°（已运行） | -38° | +1.05° | +0.029175 |
| +25°（未运行） | -33° | +6.05° | +0.169665 |
| +30°（未运行） | -28° | +11.05° | +0.317169 |

以上是同拍映射可表达性，不是未来body/contact响应预测。新入口/新baseline必须重算，
不能将旧raw直接搬到新run。额外正5°/10°不接近旧状态的物理正端或residual上限。

## 最小判断

优先+25值得作为一次后置短方向诊断：旧[-25,+20]联合变化确实使gap由约59mm到4mm，
末段关节跟踪良好且仍有明确正向命令余量。先试额外+5可以检验是否还具有下降效果。

但不预言能触地：旧run是hip/knee联合变化且body也移动，不能独立确定knee的浮动基座
Jacobian符号。目标固定后gap又到7.43mm，说明姿态/载荷也影响间隙。新source、已学
local head和其他10通道响应与旧run不同。+30仅在+25的实际反馈仍支持安全下降时作为
同一诊断的备选，不是自动升级到更大角度，也不将4mm AIR作为成功示范。

## 若随后授权运行，最小独立驱动方案

1. 读取当时生产runtime与最新兼容完整checkpoint，严格使用既有contract/load、
   mapper、HISTORY、原子派发与同一传感器任务判定。自然P01前缀，不teleport。
2. 独立脚本和run目录，显式mode=INDEPENDENT_DIRECTION_DIAGNOSTIC，PPO0/AUX0。
   在run manifest记录脚本sha、参数文件sha、生产runtime、CP/sidecar SHA；
   不修改生产diagnostic_request或formal eval分支，不改checkpoint。
3. 未激活前完全使用当前保存后重载的确定性组合policy。激活时记录当前actual FINAL
   作为唯一入口锚点；两RR通道以旧候选delta[-25,+20]平滑开始。其余10通道仍来自
   当前确定性policy，不强制wheel同速，不伪称teacher输出是policy输出。
4. 在真实反馈仍允许捕获（同attempt资格、合法XY、RR仍AIR、其他必要支撑和安全有效）
   时，可从已到达的当前FINAL平滑增加knee最多5°，以约0.75s过渡作为诊断请求，
   保留生产限速。hip不再继续负移，以便隔离“多展开一点膝”的局部方向。
   不要求固定7mm净空、不强制静止；不符合当前条件则记录“不适用”，不硬推。
5. raw由上一已提交ACK的真实baseline和当前phase cap反解，仅替换6/7。明确这不是
   同拍目标保证；逐拍记录requested/effective/final/actual、baseline tick、投影与
   限速、实际contact/力/资格/XY/gap、body/CoM、其他支撑以及wheel。
   保存原policy请求与实际override请求，不对override计算/宣称on-policy logprob。
6. +25若无改善，不自动加到+30。只有实测方向、XY与其他支撑允许且仍需下降，
   才从当前FINAL再增最多5°，不重新从入口重复叠加。若首次TOP，立即停止继续增加，
   保持当时有效捕获请求并观察当前接触/承载；用生产same-attempt/hold条件判断，
   历史placed不代替当前contact。
7. 初始诊断窗口以激活后约8s为记录预算，仍遵守全局200s和既有安全条件。
   预算用尽标DIAGNOSTIC_BUDGET_END，不伪装任务成功或额外PPO数据。记录完整尾段。
   即使取得TOP/hold，也只称独立诊断成功；未经训练与保存重载正式评估，不能称
   PPO学到或把这条诊断视频用于正式C成功。

随后按父代理指令准备了独立驱动 independent_rr_knee25_diagnostic.py，尚未运行。
实际驱动固定总delta[-25,+25]、1.5s平滑，并直接复用生产evaluate直到现有任务终止/
全局200s；不实现本文件早期可选的+30升级或8s诊断预算。原定DET→新source512→再决定
是否运行诊断的资源顺序不变。没有新增物理结果或学习样本。
