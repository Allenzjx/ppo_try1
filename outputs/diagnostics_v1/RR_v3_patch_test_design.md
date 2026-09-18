# RR v3 最小补丁 / 测试设计（仅准备，生产冻结）

实施文件仅限 `semantic_supervisor.py`、`semantic_backend.py`、`semantic_transfer_roles.py` 和新增定向测试。无 geometry、reward、sigma、动作cap、网络或旧配置修改。

## 显式版本

- 保留 `P09_LIFT_MODE = "functional_lift_edge_v2"` 原值及原行为。
- 新 `P09_FREE_AIR_LIFT_MODE = "functional_free_air_lift_v3"`；共同功能接线用包含二者的明确集合，不将所有模式视为新规则。
- 新 `nominal.rr_carry_source_semantics = "current_free_lift_before_pending_knee_and_roll_v1"` 只在v3、现有连续source继承和partial-order下允许。缺该marker保持旧nominal行为；不能因默认字段缺失静默启用。
- `load_task_spec`、TaskEvaluator、stage predicate/current credit/takeover/current-observation映射、Nominal late group与carry、backend geometry兼容断言、TransferRoleTracker independent-validity均显式贯通。snapshot记实际mode，不硬写v2。

## 新资格状态与真实竖向速度

`TaskEvaluator._observe_functional_rr` 为v3增加最近接触轮底基点、连续AIR段计数、前一真实物理轮底样本。任意ground/obstacle接触刷新下一AIR段的基点；未观测到合法基点时为None，不能假设初始高度。当前AIR段自由增高到既有3 mm才记initial、到8 mm且现有whole-body response/其他真实支撑成立才qualified。ground撤销attempt信用；原qualified后合规edge/TOP连续接管规则保留。

`ground_relative_lift_m`仍为原含义；新增free-air增高/基点/计数诊断。新`wheel_bottom_vz_m_s`严格来自相邻真实physics tick的`bottom_w_m[2]`差除以实际时间差；无前样本、非相邻或无法验证时为None，绝不使用轮轴qd、名义target或FK预测代替。重复tick复用snapshot，普通phase切换不重置这些历史。诊断不扩372维actor observation、不静默改旧feature含义。

## 源时序门控

`NominalMotionProvider`从contract的waypoint `changed_channels/atomic_channels/full12` 和atomic group确定待消费carry-knee事件及首次正向四轮组；不把已知六次、tick232/640写成验收常量。`_sequence_permission`在消费事件之前检查，挂起不调用`MotionExecutor.tick()`，不跳过、不追赶、不改旧入口姿态；其余source层、12维residual、mapper/HISTORY和物理继续。

- 膝部carry事件：尚未top时，没有current有效free-lift，或AIR净空低于既有15 mm且真实轮底正在下降，则暂缓该事件。仅源码事件语义识别段落，不用负膝角推断物理下降。低净空而速度证据缺失应明确诊断，不伪造0速度。
- 首次正向四轮原子组：current有效qualified AIR（允许负topgap）或合法当前TOP continuation可通过。不能套15 mm滚动门槛。纯物理TOP条件从旧helper的endpoint/late-wait owner条件分离，防止首次组永远被拒绝。
- 由contract识别的显式stop不进入上述新事件门控，不拆四轮原子性；若source已在更早事件挂起，不能声称未来source stop的wall-clock时间仍与未挂起相同。

## 定向测试

1. 新3/8 mm自由AIR正例；RR自身关节不动而全身响应有效的正例；FL AIR不记承载；未知接触不可当零负载。
2. wall/ambiguous接触增高不得initial/Q；先自由抬升、ground再接触增高不得复用旧信用；晚短AIR不能把先前contact-rise洗白；接触后真正新自由升高仍可恢复。
3. 已qual后的合法轻触/当前TOP继续，不能自动宣称wheel-only失败；Q/C/P和最终几何判据不放宽。
4. 相邻bottom差速度与原物理输入一致；重复、首帧、非相邻样本None语义；普通phase不reset。
5. 待膝事件挂起/恢复、source clock不消费/不追赶、保持当前nominal；上升或高净空通过；无新膝事件时正常carry/landing不因低gap停住。
6. 负gap qualified AIR首次wheel组通过；未知edge/无Q拒绝；合法TOP通过；stop不被吞；all12残差许可、单一物理write链不改变。
7. 新mode snapshot、backend断言和transfer validity接线；旧v2全部定向测试保持原结果；marker非法组合拒绝。
8. 用旧sealed物理窗口验证RR组件：CP4633正、4731清除、4940负；zero5434正。用原source输入检查：CP4697首次would-hold、4833轮组等候；zero六个膝事件全通过、5641负gap AIR轮组通过。此为输入回放，不是新物理成功。

优先复用 `test_p09_functional_edge_lift_v2.py`、`test_semantic_source_partial_order_v1.py`、`test_semantic_p09_late_reconfiguration_guard.py`、`test_semantic_rr_top_continuation.py` 的fixture/边界。新增v3独立文件，不重写旧测试期待值掩盖兼容回归。完成小候选后真实运行仍可能失败/等待，需要新采样训练证明，不能由这些测试宣称成功。
