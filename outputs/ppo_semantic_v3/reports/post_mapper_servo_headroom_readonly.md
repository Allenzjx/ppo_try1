# RR knee：逻辑 nominal 预限幅与实际 post-mapper 余量

2026-09-06，只读源码及固定已完成 episode3；无 Python/Isaac、生产/配置/历史修改或新训练。本报告不放开任何真实硬限，不以范围分析解释任务失败成因。

## 结论与边界

当前 SemanticEpisodeEnv 确实把 **unmapped nominal** 传入通用 projector，并据此预先夹住 `nominal + residual` 的逻辑2°安全带；之后真实执行把同一 residual 加到 **当tick mapper/geometry 后的 drive target**。两处基准不同，前置限制没有在 v3 独立 post-mapper 模式下禁用，也没有改为当tick mapped headroom。

这不是一个对最终drive硬限数学等价的冗余检查：mapper正向补偿时会额外限制可用负残差，反向补偿时又不能独自保证最终drive仍留2°。最终真实hard clamp和slew仍存在，未发现它们被绕过。是否要保留“逻辑组合也在安全带内”的额外设计条件，须作为明确执行契约选择；不能把该条件自动叫作实际post-mapper可用余量。

最初给出的 **terminal RRK −54逻辑/−44drive不是饱和证据**。新增固定452行核对后，确实找到更早的实际边界饱和记录，同时同tick drive仍有额外余量。

## 1. 固定窗口与真正的饱和证据

Run：`runs/ppo_semantic_v3/train/20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5`。

只取 `completed_episodes.jsonl` 第4条（episode_index3）及其452条策略记录。前三回合长度78+452+452=982，故本窗口为 **global74071–74522**；没有解析后续回合。该回合 P09 INCOMPLETE_CONTROLLER_BLOCKED，teacher prefix后的452决策不是完整P01成功。

以下均为RR knee/index7，度。当前cap36；真实projector residual rate .5°/120Hz tick，8tick最多4°。`raw → 36*tanh(raw)` 是表达目标；`rate-only end` 从前一决策末 residual 在相同当前请求下按8tick slew计算，不含名义安全区预夹。

| global / episode tick / command tick | raw表达残差 | 前一末residual | rate-only end | 记录最终residual | 逻辑nominal | 当tick native drive | 最终drive |
|---|---:|---:|---:|---:|---:|---:|---:|
|74090 /6112 /6291|−25.514171343|−20.2|−24.2|−20.2|−37.8|−36.55|−56.75|
|74520 /9552 /9731|−24.983683450|−20.2|−24.2|−20.2|−37.8|−27.8|−48|
|74521 /9560 /9739|−22.708053160|−20.2|−22.708053160|−20.2|−37.8|−27.8|−48|
|74522 /9568 /9747，terminal|−14.106061620|−20.2|−16.2|−16.2|−37.8|−27.8|−44|

前三个列出的边界：raw请求比−20.2更负；前一末residual也为−20.2，纯rate不能解释保持不变；当前/前一nominal相同，8tick没有handoff hold。记录 `geometry=identity_within_descent_allowance`、RRK geometry adjustment0、controller bias0，排除这个边界的几何target调整混淆。

预限幅下界恰为 `−58 − (−37.8) = −20.2`。74520/74521的同tick native+controller基准却为−27.8，按**相同2°安全带**允许到 `−58 − (−27.8) = −30.2`：静态负向余量多10°。最终previous_drive=−48、final_drive=−48，native audit验证了实际buffer映射，而非仅raw请求。

更小、可证伪的同状态一步反例：若只允许残差从−20.2再走原rate允许的−.5°，即−20.7，保持这个当tick native−27.8/controller0/identity geometry，则desired drive=−48.5。它同时满足真实膝hard下界−60、原2°安全带−58及从previous−48出发的final slew≤1.25°。当前逻辑预夹却把该步回夹到−20.2。这是基于已记录当tick值的算术反事实，不是执行过的替代物理轨迹，也不声称后续mapper/姿态不会改变。

452个末边界中263个RRK residual落在当时逻辑安全下界；加上“表达与rate-only都更负、nominal前后相同、无hold、同tick还有余量、geometry adjustment0”筛选有229个：**184个明确identity**、44个无geometry分支、1个degraded identity-like bypass。主证据只使用184个明确identity；degraded单独排除，绝不把它当作正常几何投影成功。其余真实geometry分支在整窗口有projected_exact_forward5、projected_relaxed_forward43，未用它们证明同tick未经修正的余量。

日志完整target向量按决策末保存，不能将184个边界乘8冒充已重建每tick target向量，也不定位第一次进入预夹的未记录子tick。上述恒定nominal/持续向外请求的边界及源码足以定位约束来源。

### terminal行的纠正必须保留

74522 raw=−.4139660894870758，表达−14.106061619872143；前一末residual−20.2，8tick释放4°得到−16.2，与记录完全相同。逻辑−54仍高于安全下界−58，**这一行是slew回收而不是absolute预夹**。其previous actual drive−44.5→−44，geometry0、controller0，native−27.8+residual−16.2也直接等于−44。不能反过来用其他行饱和推翻这条事实。

## 2. 精确执行链

1. `src/wlr50_clean/ppo/semantic_backend.py:280` 的 `_build_authoritative_frame` 将 `controller_frame.full12` 定为 `AuthoritativeFrame.nominal_action_full12`。该frame的 `info.mapped_nominal_full12` 来自**上次dispatch ACK**，并明确标记 `previous_dispatch_native_before_post_mapper_bias`；它不是尚未执行tick的native预测。
2. `semantic_env.py:150–155` 原样把 `source.nominal_action_full12` 传给 `PhaseTransitionBridge.project_tick`，再把projection.applied_action传到backend。bridge没有接收mapper headroom，也没有读取上述mapped info。
3. `action_projection.py:504–588` 先tanh/cap/rate，随后 `residual_intervals=(min(0,lower−nominal), max(0,upper−nominal))`。`configs/ppo_action_projection.yaml` 膝hard−60..210、margin2；真实加载安全区−58..208。`semantic_backend.build_semantic_projector` 仅替换phase caps/masks/rates等，没有替换这些绝对区间或为post-mapper跳过preclip。
4. `semantic_backend.py:200–219` 将逻辑applied与未改nominal交给 `build_residual_actuation_plan`，取回同一delta；真实mapper输入仍是nominal，**不是nominal+residual**。
5. `semantic_residual_adapter.py:68–107` 仅一次mapper.advance，得到当tick canonical native drive；再做nominal-only geometry修正；最后才加 controller+policy residual，并调用 frozen `bounded_drive_feedback_step`。本回合实际+10已在native中，controller字段0，不能重复相加。

因此不能直接把前一步frame的 `mapped_nominal_full12` 替换成当前投影基准：那会把previous-dispatch信息假装成same-tick headroom，也忽略当tick反馈/geometry更新。此次只定位问题，不实施这种替换。

## 3. hard limits、margin、standing offsets不是一层

`infrastructure/command_batch.py:193` 的 `servo_limits_deg` 给出canonical command/drive范围。`robot_adapter.py:709` 的 `bounded_drive_feedback_step` 将 `native+bias` 按真实hard范围夹紧，再对previous final drive做1.25°/tick slew，最后再次hard clamp。`build_physical_batch` 又保留真实硬限，最终按 `radians(standing + SERVO_COMMAND_SIGN*drive)` 转为物理joint target。

RR knee sign是−1，因此canonical−44对应 `standing+44°`，而不是物理joint angle−44°。日志native实际target0.771701574rad与这一sign/standing路径一致。`native_drive_target_full12` 的“native”仍是canonical drive度，不是已含standing的rad；+10的mapper变化也不是standing offset。既有USD/PhysX limits按相同standing/sign从command硬界构造，不能把nominal域、drive域和physical rad直接比较。

反向补偿的最小结构反例：nominal−37.8，native−47.8（补偿−10），residual−20.2在逻辑安全带内，但native+residual=−68。最终adapter仍夹到−60并受原slew约束，因此真实hard limit没有失效；然而仅逻辑预夹不能保证最终drive仍保持−58的2°margin。实际文件中hard limit从未被建议取消。

## 4. 是否已有设计/测试解决

通用projector有明确的逻辑安全带设计与测试：`tests/unit/test_ppo_action_projection.py:170` 分别断言rate和absolute/margin阶段，`:207` 保证nominal在reserved band外时只削向外residual、不擅自移动nominal。这些条件是真实存在的，不能把它们说成遗留Recording百分比门；Recording headroom在此仅诊断。

`tests/unit/test_semantic_residual_adapter.py` 的真实adapter测试覆盖独立残差、controller原包络、最终hard clamp/mapper slew/一次write；但其较大残差直接构造actuation plan，并不经过SemanticEpisodeEnv的逻辑预夹。所检查的nominal geometry/audit用例也未把“同tick+10 mapper偏置、逻辑−58预夹、最终仍有10°余量”作为env→projector→adapter的成对反例。

所以：已有测试分别确认两层各自行为，未发现现有专用开关或测试已让**这个preclip与post-mapper实际安全余量等价**。静态及上述真实边界显示它确是额外限制；这不能升级为“当前失败由它导致”，也不能直接选择修复或扩大cap。任何后续执行契约修订仍需保留当tick真实mapper/geometry、final hard limits、final slew与对应反例验证，不复用过期mapped info，不以任务成功作为新增optimizer门。
