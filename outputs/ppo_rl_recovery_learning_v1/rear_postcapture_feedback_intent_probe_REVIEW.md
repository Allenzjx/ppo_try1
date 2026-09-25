# P10 单通道 feedback-intent 候选（未执行）

仅新增 outputs 中的 `rear_postcapture_feedback_intent_probe.py` 与对应 stdlib unit test；未改生产、模型或正在采集的 rollout。19项标准库/AST测试通过，无Torch/PXR导入、无Isaac启动。这些测试证明接线和生命周期，不证明物理方向或成功。

## 确切语义

- 最新兼容 `cp225280_front_preserved_v1` 历史checkpoint由CLI显式指定，要求checkpoint、sidecar的精确SHA及精确HEAD。继续官方load/439/512身份验证，无隐式迁移、无优化器更新、无实时teacher部署。
- 同一真实P01仿真先经现有 `successful_nominal` 前缀到P10 offset0。seed1001；前缀零残差与学生后缀都无PPO/AUX信用。前缀失败或跳过P10时不在fallback中触发。
- 仅一个允许通道：`FR_knee`=3、`FL_knee`=1、`RR_hip`=6；方向只能−4°或＋4°。不预选永久修复方向。
- entry锚定**第一条真实P10学生接管时最新已提交FINAL**，例如现有课程6136；不是更早的native阶段边界6133。意图为该固定角度加offset，绝不逐拍累加。
- 1秒smoothstep ramp，1秒hold，1秒smoothstep释放到当前学生REQUEST。首拍保持实际已有filtered REQUEST以免人为跳变。其余11个raw严格取当拍当前学生；干预引起后续闭环状态/策略变化是实际效果，不宣称后续11通道与未干预baseline相同。
- 普通raw→tanh/cap→REQUEST filter/HISTORY→mapper→headroom/owner→最终原子派发路径不变。没有额外执行器写入、mapper预测副本或HISTORY重写。

**这不是精确FINAL干预。** 普通raw保持8个120Hz拍，mapper与owner在其中演进。驱动器用上一已提交 `baseline_native_plus_controller_full12` 计算 `desired_FINAL − baseline` 的有界REQUEST近似；末端实际目标不保证等于±4°意图。记录每拍desired、实际FINAL及误差、source/mapped N、REQUEST、投影/限速、owner证据和实际关节/接触。

**入口FINAL＋4°也不是相对动态学生只增加4°。** 固定入口意图可能同时抵消学生随后较大的负向漂移，实际两臂差值可能远超4°。因此另有真正的 `--no-override-control` 臂：同一checkpoint、seed1001、真实P10前缀、观察/记录、学生预算；始终原样派发全部12个当前学生raw，不调用REQUEST替换器、不固定任何入口目标、不重置HISTORY。它与 `--entry-final-offset-deg` 互斥，manifest明确标为 `NO_OVERRIDE_CONTROL`。**零offset入口保持不是无干预对照**，也不提供这种模式。没有同条件真实control及入口证据，不作单因素因果归因。

## 触发与退出

只在首次P10学生接管尝试一次；要求当前RR合法TOP、真实verified bearing及**现有**`rear_dependency.support_transfer_permitted`，历史placed不能放行。允许P11/P12连续接管。entry越过当前cap/保留物理范围则不触发。

120Hz观察器锁存真实支撑/许可丢失、离开任务窗口或cap变化；下一决策不再覆写（最多剩余7个物理拍）。普通物理安全中止即时保持。陈旧/不一致ACK或基线拒绝；结束后不重触发、不重置HISTORY。有限3秒结束只撤销诊断raw覆写，不撤销原策略/nominal。

默认最多120条学生决策（8秒），可显式缩小，最大900；前缀上限1800决策，原全局200秒含前缀不变。输出 `DIAGNOSTIC_*` manifest，不能称正式C、纯policy、完整P01成功或训练收益。

## 命令模板：需要主任务审阅后才补执行flag

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' outputs/ppo_rl_recovery_learning_v1/rear_postcapture_feedback_intent_probe.py `
  --checkpoint '<本次完整更新后明确选定的front-preserved历史checkpoint>' `
  --checkpoint-sha256 '<该checkpoint的64位SHA256>' `
  --checkpoint-manifest-sha256 '<该sidecar的64位SHA256>' `
  --expected-head 892385cba8a7089b52567bb7f558c018fac82a77 `
  --channel '<FR_knee或FL_knee或RR_hip，必须择一>' `
  --entry-final-offset-deg '<4或-4>' `
  --max-student-decisions 120 `
  --run-dir '<runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe下未存在子目录>'
```

不带 `--execute-reviewed-real-diagnostic` 会直接拒绝，尚未加载checkpoint或创建run。即使补flag，普通单Isaac文件锁也必须可获得；本候选不会停止现有训练。

同条件control命令保留全部相同参数（尤其checkpoint/两种SHA、HEAD、channel、seed和预算），只把 `--entry-final-offset-deg ...` 换成 `--no-override-control`，并使用另一个未存在的隔离run目录。两臂都只是DIAGNOSTIC，不是正式C或训练收益。

清理时独立尝试learned-state核验、物理记录器关闭、末态记录、app关闭、lock关闭，再写最终manifest。任何一个失败均不跳过其余资源关闭，也不替换原始运行异常。清理错误写入manifest；manifest写入失败时不覆盖部分文件，另尝试 `cleanup_errors.json`，再失败则输出stderr。没有原始异常时，在全部清理/证据尝试结束后重新抛出首个清理异常。

测试命令（已执行，仅系统Python/stdlb）：

```powershell
& 'C:\Program Files\Python313\python.exe' -B -m unittest -v test_rear_postcapture_feedback_intent_probe
```

工作目录为本文件所在目录。覆盖：三通道×两方向、固定anchor不累加、其他11 raw不变、初始连续、ramp/hold/release、短暂失载锁存、不重触发、P10→P11→P12连续、cap变化退出、过期ACK/基线、真实支撑与历史区分、有界REQUEST裁剪可见、禁止额外派发/HISTORY写入、默认禁止执行、control全部12raw原样且不调用替换器、control与offset互斥。
