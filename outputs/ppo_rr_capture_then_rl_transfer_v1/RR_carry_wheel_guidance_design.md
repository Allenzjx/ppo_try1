# RR carry/capture：有限、当前支撑选择的前送投影草案

仅设计，未改src/config/tests、未运行actor/Isaac。当前v3训练在完整update边界前保持冻结。下述是需要版本化并实测的**控制器改变**，不是已学到的PPO行为或保证捕获的方法。

## 1. 已知事实与最小目标

26db封存视频：原P09有限FL反转脉冲及其stop确实下发；stop后N四轮为0，FL最终/实测约−0.93rad/s，来自policy作用而非mask丢失。根报告当前v3首训练回合：hip−20°再knee+20°后仍AIR，gap约34.8mm，front由约70mm退至1.4mm；本设计不重新读取live数据或据此归因于单轮。

需要在**原有限动作完成以后**给有真实牵引条件的支撑轮适量正向前送，不只是把FL负值削成0。FR/RL若也承载，应共同取得小正向建议；RR仍AIR时不能把它的旋转算成推进力。达到可用深度后停止额外前送，保留RR关节下降；不以新pose、fixedCoM或新placed门槛替代物理目标。

## 2. 实际接线与观测核对

| 层 | 当前事实 / 最小变化位置 |
|---|---|
| canonical/native | `command_batch.py`：8/9/10/11=FL/FR/RL/RR wheel；native signs=−/+/−/+；wheel单位rad/s。canonical正值是定义的前转，不能单凭qd正值认定实际牵引。索引仍由live joint_names精确解析，禁止硬写native ID。 |
| 现有mode | `semantic_backend.load_execution_profile`目前强制`rr_capture_wheel_mode=off`；context里`rr_carry_forward_v1`仅能算bool，**没有派发变换**。只翻flag并不能实现控制。 |
| raw分布 | actor的12维Gaussian、HISTORY、sigma、cap、old/current logp、entropy保持原样。新投影是环境内后续确定性动作变换；RAW许可仍12路，但所选轮的最终执行在该局部状态受到明确限制，不能宣传为最终动作完全自由。 |
| 派发 | `semantic_residual_adapter.apply_semantic_residual`：唯一mapper/geometry/headroom/FL+RRassist之后，现有`final_wheels`求和与硬夹紧处调用纯helper；8个servo完全不变，然后原`build_physical_batch`与一次write。 |
| counterfactual | `actuator_target_effect.physical_targets`必须在actual及zero-current-policy两支调用同一个helper、同一前拍实测与前拍FINAL；不能把guidance增量冒充policy贡献，不能重跑mapper/推进helper状态。 |
| 真实history | **已完整核对**：`semantic_env.reset/step`从`frame.info['drive_target_full12']`每physics tick写入`previous_applied_full12`及`previous_previous_applied_full12`；builder编码完整12维。因此四轮前次FINAL已在410中，不是只有mapper servo8。无需再追加4维。 |
| flag | 复用410的`fl_wheel_guidance_active`，新version明确为“当前RR捕获窗口内支撑轮前送/退出限速包络已接通”，不再仅指`carry && FLbearing`。ACK另报是否真的发生投影、每轮owner/gain；前者不依赖尚未采出的raw动作。 |

先前仅看mapper8而认为history缺四轮FINAL的说法已撤回。当前contact类别/source内部并非各自完整编码，不能宣称410整体严格Markov；但此方案不增加未观测latch、累计预算或私有ramp记忆。

## 3. 推荐的首个单一候选：当前支撑轮的小正向下界 + 有界目标变化

建议独立mode名`rr_capture_support_forward_projection_v1`，默认off。scope先只P09 RR捕获，不连带P10/RL转移。

### Eligibility（完全取同一当前观测/源派发上下文）

- 无task/safety终止、观测有效；RR真实当前有效lift/carry及历史cross，当前AIR、无GROUND/obstacle接触、合法TOP XY/lateral、gap≥0；不从旧placed推断承载。
- FL由共用`verified_current_support`判为当前承载，另至少一条真实其他支撑；用现有force_noise_floor，不加另一套载荷硬门。RR自身不计入此AIR支持集合。
- P09 late有限组及其FL wheel脉冲已经执行完，并且**最终stop已真正issued且跨过一个one-write/one-step后续拍**。从provider的P09 source endpoint/source_ticks、last completed source sequence与当前atomic_groups形成只读证据；不能由N=0、stage_age或changed_channels为空推断源已结束。
- 仍有未完成P07/P09有序wheel owner、当前fresh wheel group/stop、P13/独立安全stop时禁止接管。历史pulse保持，包括它的合法reverse；新的source事件始终高优先级。跳过helper不是强行把合法PPO residual清零，原source+residual语义仍保留。

可复用`_rr_carry_diagnostic.ordered_source_wheel_owner`作一项证据，但它单独不足以证明“late最后stop已执行”；须显式记录final endpoint与fresh owner。这是读取已有源时钟，不新增恢复计时器。

### Geometry增益（控制尺度，不是新验收条件）

用RR front distance `x` 与clearance `g`，建议首候选复用已有尺度：`d_full=geometry.xy_measurement_tolerance_m`（5mm）、`d_taper=min(geometry.workspace_max_m, 当前平台后缘可用深度)`（通常60mm），`g_taper=geometry.airborne_clearance_above_top_m`（15mm）。

`w_x = clip((d_taper - x)/(d_taper - d_full), 0, 1)`；`w_gap = clip(g/g_taper, 0, 1)`；`w=w_x*w_gap`。

只有合法几何才取该增益；尺度分母非正/缺测fail closed。70mm深度时额外前送为0；退向边缘时增益增加，约5mm及更浅的合法位置取满；gap接近0时额外前送渐减。这里5/60/15mm是显式待验证控制参数，**不得改变cross/placed/currentQ/安全**，也不是“必须达到60mm才算成功”。前缘/横向退出、真实接触、有效carry失去时退出前送。

### 四轮选择和投影

以现有`approach_wheel_prior_rad_s=[.3,.3,.3,.3]`为上界参考，而非新固定四轮等速指令。对**当前verified bearing的FL/FR/RL**分别：

```
candidate_i = original_post_mapper_nominal_plus_effective_residual_i
desired_i = max(candidate_i, 0.3 * w)  # w>0且该轮有真实支撑
```

不承载的FR/RL保持原candidate；AIR RR的第11通道保持原candidate，不声称其旋转提供牵引。超过下界的正向policy、差速仍保留；这不是abs(轮速)，也不永久关闭wheel residual。若只有FL及一条其他腿有实测支撑，选择这两条，不能虚构固定三脚架。

对接管轮用**前一真实FINAL wheel**为起点，按显式3rad/s²初始目标变化上限（复用nominal已有量级，不宣称它原已限制此新层）向desired推进；120Hz每拍最多0.025rad/s，最后仍原硬轮速夹紧。约−.93→+.3需至少0.41s；记录这期间暂仍为负的实际target，不声称一启用即正牵引。

相比以actual qd为slew中心，前FINAL能严格约束目标自身连续性，而且已经观测；actual qd保留为真实跟踪/滑移诊断。以qd为中心只能约束target−当前速度，遇负载/扰动不能保证相邻target变化，因此不首选。

### 退出，不能新增隐藏release状态

- 几何深度足够或gap缩小时令前送目标退回原candidate；在同一个合法P09、source已完成且FL/另外支撑仍有效的包络中保留上述slew，完成**软退出**。current TOP/contact时不再正向加速，可在同包络有限速地退回原candidate以待真实capture；不以contact_seen替代当前contact。
- `fl_wheel_guidance_active`覆盖这个“当前包络接通”事实；ACK分别记录forward_gain=0、slew-only退出、各轮实际修正。没有额外计时/释放latch。
- 当前支撑/合法性失效、phase离开P09、fresh source事件或安全停止时，本层让权。**此窄草案不承诺这些硬边界的额外平滑**；不可为平滑而延迟安全或fresh stop，也不能新增未观测的“继续顶住FL”尾巴。若物理诊断显示P09→P10硬让权造成轮速跳变，再单列基于已有前FINAL的阶段handoff规则，而非在本轮悄悄扩大P10控制。
- 不把guidance本身当作局部deadline豁免或无限有效恢复；现有global200与安全保持。若窗口没有产生实际前送/下降，报告无效，不能无限续时。

## 4. 实现接点、审计与版本边界（未实施）

最小production面：新纯`semantic_rr_wheel_guidance.py`；backend构造同拍context/版本验证；adapter唯一final_wheels插入；actuator_target_effect独立重建；context统一可观测eligibility；provider若需补现有endpoint/fresh-source只读diagnostic则仅补该scope；execution_profile显式mode/尺度；严格same410 control迁移和针对性tests。现source/N文件不改、reward/acceptance/physics/actor/distribution不改。

Audit必须在写入前独立保存前ACK wheel4、actual qd、source owner/end证据，验证与已编码前FINAL同拍；不能事后读当前ACK冒充previous。ACK记录`candidate/desired/previous_FINAL/slew_output/hard_output`四轮、选择集合、geometry gain、source stop precedence、native IDs/signs和实际下一拍qd/front/gap/body/contacts。

最小独立prestate落点：`isaac_fsm_backend.step_physics`原1933附近servo8捕获处、`_atomic_apply`之前，仅新mode取`adapter.last_ack['drive_target_full12'][8:]`；验证finite12、ACK write_count/physics_tick与adapter相符，并与当前`_authoritative_frame.info['drive_target_full12'][8:]`一致。原1979附近audit调用新增optional `previous_final_drive_wheel_rad_s=None`参数；旧A默认None且执行不变，新mode缺参数拒绝。审计使用此独立参数，不用新receipt自报previous替代；env每physics tick恰由该frame drive写previous_applied，定向测试还须逐位核对encoded history。这个共享base改动只增加受mode控制的读证据，不改变A目标、mapper或写入数；需纳入runtime迁移范围，不假称文件字节未变。

同tick PPO作用 = `helper(N+current_residual, same_context, same_previous_FINAL) - helper(N+zero_current_residual, same_context, same_previous_FINAL)`。另记controller作用 = with-helper−without-helper（同state）。两者非线性不可简单合并；全1mask只表示raw许可，不证明最终policy全通道产生了作用。禁止将after−独立N重算值全归PPO。

helper必须共同用于B/C、train、det/stoch eval及actual/zero-current counterfactual；冻结A不改。TEACHER/TAKEOVER原未计学习prefix按现有声明保持，READY及checkpoint-policy prefix走统一新规则。迁移取当前训练**实际完整update封存后的最新CP**，保完整actor/critic/Adam及实际LR/Identity/RNG、全部AUX/origin/counters；新控制MDP、新rollout，不重置mean、不复用旧未完rollout、不重复AUX。

## 5. 最小证据计划，不作为学习门禁

CPU定向正反例：off逐位identity；源early reverse/freshstop获优先；未接触FL/仅历史support不得启用；只已承载轮投影；AIR RR不改；x/g taper、几何失效、同前FINAL slew有界；raw/logp/history不篡改；actual/zero counterfactual共核且唯一write；reset不带入owner；显式source/安全退出不续顶。

物理最多先一组同CP/seed新进程的controller-off与on有限P07真实前驱对照（或复用已封存同CP off run，但标非新严格重复）。on是FL+当前承载FR/RL协调前送，不再先跑只把FL截到0的试验。需真正看到：source stop之后选择轮目标/实测转速响应、RR front保持/增加、gap趋势、body/FL当前支撑、RR真实contact/placed；任何target为正但front继续后退必须标失败/未证牵引。任务后缀或控制辅助成功不冒充从P01完整PPO成功。

本设计不预证`.3*w`及3rad/s²足够或稳定，也不从一次三轮指令推出真实载荷方向；其价值是让“几何前送—实测推进—RR下降/捕获”可辨识、可审计。当前训练不热改。
