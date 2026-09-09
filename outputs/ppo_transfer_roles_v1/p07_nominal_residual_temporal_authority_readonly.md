# P07 FL hip：nominal 与残差的时域权限

范围：当前源码/配置及 `run11_first_P07_collision_readonly.md` 的既成首14决策；无新raw扫描、模型运行或仿真。以下反向极限以**既有 nominal/阶段时序、零历史、非安全覆盖**为条件，不是另一个策略的实测轨迹。

## 顺序与量纲

15Hz raw在8个120Hz tick内保持：`tanh(raw)×24° → mask → 每tick相对上次request限±0.5° → cap → safety`。这里的“filter”是有状态slew，没有额外EMA。普通跨phase且任一carried residual非零才全向量保持1tick；**零历史的首P07不是必然hold**。已有时序在第9/25个policy tick进入P08/P09，最强持续反向请求各损失1次slew机会，历史不清。

nominal经独立MotionExecutor/150°/s provider slew（每tick1.25°），然后唯一mapper advance；再rear-only geometry（不改FL）、controller bias与headroom后的residual相加，最后硬限和**合成目标**1.25°/tick slew，转换standing/sign为native rad并write。READY后controller bias=0。FL hip硬限±135°、reserve±133°，本例反向请求离硬限很远；不是headroom截断造成速度差。

## 能证明的请求权限缺口

P07源码FL目标依次为37.6°@0s、46.1°@.1333s、48.2°@.2s、49.2°@.2667s；是有限动作队列，不是单步跳到49.2。以入场nominal22.8°为基准：

| 已发policy ticks | 已报告nominal | 最大反向request幅度 | nominal+request相对22.8的最小余量 |
|---:|---:|---:|---:|
|8|31.55°|4°|+4.75°|
|24|46.1°|11.5°（1次hold）|+11.8°|
|48|49.2°|23°（2次hold）|+3.4°|
|充分时间、仍49.2|49.2°|≤24°|≥+2.4°|

即使乐观忽略所有hold，24/48tick仍余+11.3°/+2.4°。任意允许raw能从首tick反向推进request，不需要先学出某个固定姿态；但**从零历史不能逐tick完整撤销这段名义增量**。大raw不能越过slew或cap。自然P06可提前积累历史/改变真实入口，不能将此结论推广成全身无动作可用。

## 最终native不能与nominal+request混同

精确最后一级是 `f_k=f_(k−1)+clip(clamp_hard(m_k+c_k+r_eff,k)−f_(k−1),±1.25°)`。最终限速限制变化，不扩大残差权限。moving nominal每次变化会清mapper compensation并走追踪目标的slew；nominal稳定后，gain8/每4tick反馈可产生±10°补偿，previous-request参考也仅在nominal未变、已到达、采样tick且tracking开启时使用。因此**48tick的nominal+request余量不证明最终native必然仍上移**：例如到49.2且r=−23时，若mapper补偿为−4.65°，合成目标可等于入场final21.55°；能否产生该补偿取决于真实q/时钟，未实测。

摘要只有初始nominal22.8/final21.55，没有独立native0/compensation，不能默认为二者相同并伪造最优native轨迹。条件算例：若native0=final0=21.55、持续斜坡中无反馈，8tick最强r=−4可得目标26.3（仍+4.75°）；这是算术示例，**不是本次counterfactual实测**。首14实测最终hip继续升高还包含后来正残差，不能全部归因nominal。

## 边界选择

若“advisory可被零历史PPO即时充分抵消”是所需权限，本例已证明当前设计不满足，足以支持**下一版本窄FL名义blend候选**的审查；尚不证明碰撞不可避免或改变后会成功。仅降低nominal slew不能解决26.4>24的终点差；应明确候选究竟改善反应时间还是还要保证幅度可抵消，不能暗中扩大noise/cap、冻结全身或添加支撑/hold成功门。

必要反例仅围绕该候选：零/非零继承历史与首tick；合法持续反向及原raw0；不同反馈相位/compensation/headroom/final slew；正常需要该FL协同动作时仍可推进；自然P06预加载和P07/P08/P09普通交接不重置。当前不选参数、不实施、不增训练门。

来源：`execution_profile.yaml:29–43`、`stage_task_spec.yaml:84–99`；`action_projection.py:516–603`、`phase_action_masks_v2.py:333–401`；`semantic_supervisor.py:1001–1126`；`semantic_residual_adapter.py:112–162`、`servo_target_mapper.py:178–255`、`semantic_tracking_reference.py:214–239`、`robot_adapter.py:709–734`。
