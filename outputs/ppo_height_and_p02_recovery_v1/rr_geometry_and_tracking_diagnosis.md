# RR 几何与实际跟踪：当前结论与保留证据

最新已封存训练边界为 **CP174592 /1329 PPO updates /26580 optimizer steps**，本轮真实新增6400/50/1000。生产HEAD `4a58c0190ef744a2e40c3668d9042caf85389385`、runtime `169a0415974a16a762f753fe3b3d6dcd702a783aebf184f4a9ab551a6052d0a9`未改。同版zero已8857ticks/P13完整成功；最新正式自然P01/all12 C174592在812ticks/6.766667s/P02膝硬限失败，首个未完成FR越沿，无PPO恢复/完整成功/稳定性优越。

根代理同配置2048自然P01块已自然封存，run `train/20260915T0633266376940Z_g4a58c0190ef7_d19a81607e6044a281a010de29469635`，实际+2048/16/320，之后立即正式重载C174592已完成。当前无活动Isaac、无新生产变化，不以权限或资源耗尽解释停止。[training_P01_fixed2048_after_wheel_audit_actual.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_P01_fixed2048_after_wheel_audit_actual.json)、[RECOVERY.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/RECOVERY.md)记录最新恢复路径和实数；旧172544计划仅历史，不是下一次恢复命令。

## 区分 nominal 几何追逐与 actual 跟踪崩溃

旧B2/P09：source RR knee−37.8°，mapped nominal约−27.8°，geometry补偿约−32.2°，最终目标−60°、实际−60.000011°。这是N的局部几何路径，不是policy请求硬限。旧semantic sensor本身已每tick旋转collider-local顶点；旧nominal Jz却只平移COM Jacobian到link-origin，遗漏真正最低碰撞点的转动项。只控制RR两关节、边界随上一final漂移，在低高度/多点接触下可持续追逐不可达净空。

已实施 `contact_aware_bounded_rr_nominal_v3`：Jz使用当次真实最低collider顶点，Jx仍为wheel link-origin；真实TOP承载沿用−15mm接触带，不将接触轮当作必须永久腾空的AIR。保留2°工作余量、source-mapped±10°补偿范围、1.25°/tick final slew，actual q附近2°局部信赖域；无局部解时给有界候选及whole-body reconfiguration标志，不保证全身可达。不改物理、验收或硬限。232项定向/相邻CPU覆盖最低点Rodrigues差分、120次固定source不累积到硬限、旧−60ACK按原slew恢复、TOP/AIR/ground正反例及其余10通道不变；CPU通过不是放置成功。

相反，旧C168192/P02 target≈−5.434°却actual≈−60.017°；历史C172544 target−4.347183765°而actual−60.025439882°。P02当时RR局部几何未激活，不能把final−N、actual−final或controller bias统一叫HISTORY residual，更不能说policy直接请求−60°。

## 高度候选：真实身体/安装点，不用base原点替代

旧B2 P08准备5360→5408，真实body collider最低点下降32.075mm；RR髋由URDF安装点变换估算下降8.019mm，与upper-link origin吻合，但旧日志无同次USD joint frame读回，不能冒称直接髋传感器。原A对应base原点下降21.364mm而同类RR安装点升1.219mm，已经说明base与mount不是同一个量。

固定旧末态body的小扰动显示FL/RL hip每减少1°，相应轮link-origin约下降3mm；倒推body升高只能依赖简化单足固定/纯平移假设。FL当时AIR不承载，RL有真实GROUND载荷，减RL准备也可能削弱卸载，不能靠“高一点”选成功。真正wheel几何center未独立获得的字段保持null，不把link-origin伪装center。

| 实际候选与版本 | 实际结果 | 有效解释边界 |
| --- | --- | --- |
| FL准备−4°/RL0，新geometry，3fc |8844ticks/73.7s，P09 blocked，无RR C/P|RR Q5442→5529撤销、5764→6154撤销；未再追膝硬限，仍未完成。geometry与FL幅值同时改变，非FL单因素归因。|
| FL0/RL准备−3°+late-entry，a9c |8857ticks，RR/RL均放置，P13未完成|旧recovery脉冲干扰固定终点；非完整success。RL幅值与entry时序一同改变。|
| 同RL3+P13 final-stop owner，4a |8857ticks/73.808333s，P13完整success|验收/物理/固定1s未改；这是真实zero nominal成功，无PPO credit。|

FL4首次pre5160减量33.6°，下一native收到33.6°，无逐tick累加。6120→6160，base降7.209mm、真实RR USD mount降26.541mm、轮底净空降35.845mm；geometry在6136起明确tracking-exceeds-trust/envelope/needs-whole-body。终态RR target−27.8°/actual−28.893°，31.107°余量，却GROUND、front−64.778mm/gap−51.133mm、无C/P。bodycollidermin129.728mm、USDmount189.876mm不等于成功摆动路径。[B_HEIGHT_FL4.aggregate.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/B_HEIGHT_FL4.aggregate.json)保留全部事件。

late-P09 guard保留source640四轮stop，仅延迟source648的FL/RL+FL-wheel原子组，等当前合法over-top入口（AIR bottom≥0或原TOP接触带+真实承载，另有两只实际支撑）；不要求先capture，不停物理/mapper/residual，允许原AIR/TOP carry形成入口。a9c及新4a RR I5428/Q5434/C6155/P6155；该组在source pre6155释放、实际post6156派发、decision6160可见。因此**C/P先于新组实际作用，不能称晚组创造了capture**。RL Q6251/C6658/P6727，P13入口6728。

## 已成功的 P13 owner 与旧结果区别

旧a9c：firststop native8737，recovery servo/wheel脉冲8745，固定终点8857时region/support真但controlled假；原finalstop8865未执行。原termination_source名POST_COMPLETION_LOSS，但post_completion_loss_observed=false，不能写成已确认区域丢失。保留 [p13_RL3_completion_interruption.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p13_RL3_completion_interruption.json) 和旧失败视频。

当前4a：owner观察pre8737、下一持有dispatchpost8738、首完整decision8744；N8737–8857保持，actual native四轮目标全0，旧8745脉冲未派发；source时钟、mapper与真实反馈继续。8857固定1s结束时region/support/current-control/task-completed均真，loss=false。[P13_final_stop_owner_success.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/P13_final_stop_owner_success.json) 对照 [B_HEIGHT_FINAL_STOP.aggregate.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/B_HEIGHT_FINAL_STOP.aggregate.json) 完整8857行native/physical同tick、raw/projected/directeffect全0。

**真实停止终态**：body collider minimum172.999229mm，USD RR mount241.153101mm，RRwheel link-origin100.192405mm/collider minimum49.940715mm，RRkneeactual−27.371109715°/lowerheadroom32.628890285°。不是旧a9c moving终态190.85/255.65mm。完整 [zero success视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_final_stop_latest.mp4)1108帧已验收/发布。parent [31×80 CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/height_transfer_candidate_comparison.csv) 的v2 payload区分link-origin/未知center，完整数据typed/roundtrip/preview通过。

## 最新 C174592：同版学习后仍未恢复

[最新四轮审计](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/four_wheel_P02_checkpoint174592.md)核对C174592/C172544/成功zero实际初态完全相等；新旧C共同723tick、new/zero共同812tick sourceN Full12完全一致。后续物理状态不同，不称同状态反事实。tick400四轮canonical N=.3、mappedN=.3、bias0：

| 通道 | projected residual canonical | final target canonical | direct actual qd native | actual qd canonical | ground normal N |
| --- | ---: | ---: | ---: | ---: | ---: |
| FL |−.503215|−.203215|+.014348|−.014348|12.3886|
| FR |−.278786|+.021214|+.021147|+.021147|0，AIR|
| RL |−.262423|+.037577|−.053483|+.053483|2.6842|
| RR |+.490672|+.790672|+.517062|+.517062|14.4494|

单位rad/s，native signs[-1,+1,-1,+1]，实测来自同tick joint_vel，不是目标或扭矩。FR不承载；rotation不等于承载推进。812tick执行链核验通过，projected误差≤1.11e−16、native组合≤2.92e−8，103height nativeqd符号校验0误差。没有N漏传/单位/错误mask证据。source_partial_order.layers为空，逐tick唯一owner未知；只确认完整N保持，不武断归给单一来源。

772个正N tick中FL反向756、FR近停644、RL近停620、RR增强772；.05仍仅描述，不是任务门槛或唯一轮牵引因果。RR连续12tick误差1°起26（旧27），2°起46（旧43），5°起159（旧63）；局部延后不等于恢复。tick400RR target/actual−4.284/−22.691°，较旧实际−28.388°局部减小；最终812仍target−4.246801342°/actual−60.022796599°。FR Q24、gap−7.551mm/front−171.007mm、C/Pfalse，bodycollision=false；不能把C172544的+.320mm套给本次。

[新正式证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p02_policy_recovery_checkpoint174592.json)与 [canonical证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p02_policy_recovery_evidence.json)均绑定官方reload、naturalP01/all12/no-mask/no-teacher/0updates和真实812终点。原片102帧和 [实测四轮原速视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/PPO_checkpoint174592_full12_wheel_qd.mp4)完整发布，root视检首末与panel101清楚可读；旧所有C与诊断不覆盖。

最新2048实际P01=10/P02=726/P03=35/P04=59/P05=1218，P06+0。五入口FR C/P都出现，但4个完整episode是P05 FALL×2/BLOCKED×2，373决策尾非terminal。**P02 terminal=0**，不是采到正式mean硬限结果；状态邻域接近与否薄统计未知。16update真实actor改变/有限非零梯度，Adam LR范围1e-5..3.375e-5末1e-5，norm不变；[training_fixed2048_distribution.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_fixed2048_distribution.json)实际2048全12mask、16372native tick验证通过。[training_fixed2048_reward_credit.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_fixed2048_reward_credit.md)验证普通20切换非done、4真终止无bootstrap，不能将随机动作回报归给未执行mean。

## 历史 C172544 四轮：N 没丢，但残差持续抵消

[C172544历史四轮审计](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/four_wheel_P02_checkpoint172544.md) 全读、root核验通过。C172544与同版成功zero的tick0实际q/qd/base/CoM/contact/geometry完全相同，共同723tick的source N Full12完全相同。旧C171520的693tick共同N也相同，但它是eec6旧policy版本，新C经过τ=.5训练8次update，不能视为仅改变温度的因果试验；正式评估均用条件均值、不采sigma。

tick400，单位rad/s、顺序FL/FR/RL/RR：

| 数据层 | FL | FR | RL | RR |
| --- | ---: | ---: | ---: | ---: |
| source N = mapped N，canonical |.300000|.300000|.300000|.300000|
| projected residual，canonical |−.524233|−.286199|−.259328|+.498092|
| final target，canonical |−.224233|+.013801|+.040672|+.798092|
| final target，native |+.224233|+.013801|−.040672|+.798092|
| direct measured qd，native |+.045131|+.012796|−.044501|+.691982|
| measured qd，canonical |−.045131|+.012796|+.044501|+.691982|
| true GROUND normal force N |11.5157|0，AIR|3.4489|13.8061|

canonical前向正→native signs[-1,+1,-1,+1]，实际joint names为front_left/front_right/rear_left/rear_right_ankle，ids8..11；对应*_wheel contact body。四轮N不是mask/scale对象：P01/P02 residual cap=.6，仅tanh×cap后受mask/slew，120Hz每tick≤.015rad/s。tick16→17保留原projected一tick，不清零。723tick投影复算最大差1.11e−16；native还原与N+residual最大差2.90e−8（float32派发）。全部723tick actual setter/mapping/counterfactual通过；92个真实height native qd符号校验0误差。没有漏写、单位或符号错误证据。完整372输入未保存，所以独立重建HISTORY tensor仍unknown，不额外forward。

tick41–723的683个正N tick：FL final反向682tick，FR绝对target<.05为579、RL为515，RR高于N683tick。阈值仅描述，不是门禁；不是“all12=1所以协同正确”，也不是只有RR提供实际牵引的证明。FR在tick400 AIR、真实承载0；转轮不能虚称负载推进。

## 历史 C172544 时序，不倒置因果或混用终态

相对zero，C172544在tick1已有all12 final及actual/contact/base分歧；不能只挑wheel作唯一原因。新旧C的四轮tick1 target反而完全相同[-.015,−.015,−.015,+.015]，但servo目标与实际已不同，四轮目标首次差异5/2/2/6。不能声称轮目标先变导致所有首分歧。

新C RR≥1°误差连续12tick起于27，base下降3mm起于38，四轮N+.3开始41，RR≥5°起于63；FR top-gap于97达101.662mm，191起低峰5mm，CoM下降3mm持续起于379。RR始终真实ground，FR自2为空中。成功zero base下降3mm更早于23，却723前缀没有RR≥1°连续误差，CoM也未持续下降3mm；base原点下降不是CoM下降或独立失败判据。

新C终态723tick：FRgap仍**+0.320mm**、front−181.033mm、无C/P；RR target−4.347184°/actual−60.025440°，bodycollision=false，firstunfinished FRfrontedge。旧C171520终态gap−5.570mm，不得套用新C。新C完整 [91帧视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/PPO_checkpoint172544_full12.mp4) 和 [formal证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p02_policy_recovery_checkpoint172544.json)保留真实安全终点。

## 诊断干预、测量和学习边界

历史CP168192/3fc wheels4-off从第一决策raw8..11置0、保留N和其余8策略/HISTORY闭环，16s/1920tick到P05，FR Q27/C1672/P1686；0update/0onpolicy、不是formalC。它改变后续HISTORY及servo轨迹，旧CP/runtime不同，不能隔离共同轮量/差动或唯一原因。[p02_wheels4_completed_comparison.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p02_wheels4_completed_comparison.json)。本次不永久mask，不强制同轮速，不改gain/reward/rho。

真实startup getter servoK600/D60/约2.7Nm/5rad/s、wheelK0/D20/1Nm，与配置一致；computed/applied effort只是implicit PD估计，不是独立PhysX实测电机力矩。旧run无同次getter不能倒填；未知null+reason。只读height stream按true tick0/every8/terminal，读取actual transformed collider bounds和USD joint localPos0/liveparent mount；不step/reset/update/write，不改372维与共用sensor判据。

最新四轮原速证据 [zero连续P01–P03](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_P01_P03_wheel_qd.mp4)、[C174592全失败过程](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/PPO_checkpoint174592_full12_wheel_qd.mp4)、[历史mask诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/p02_wheels4_mask_diagnostic_16s_wheel_qd.mp4)严格区分，15fps。每帧target/actual精确ledger，不将平移作旋转，不forwardfill载荷；firstencodedtick8非tick0。旧C172544的 [独立像素检查](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/C172544_overlay_independent_pixel_audit.json)只属于旧片。成功zero固定机位的末段机器人部分出画，这个视觉覆盖边界保留，不重跑相机、也不推翻日志和物理成功。

最新 [严格same-version pair](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_vs_PPO_checkpoint174592.mp4)通过runtime/evaluation/camera/seed及真实初态验证。[B_FINAL_STOP_C174592_matched_height_quality.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/B_FINAL_STOP_C174592_matched_height_quality.json)只用共同真实0..812/813样本，独立heightB102/C103，不伪造B812高度；完整B8857/C812另列。sharedrollRMS B/C=.227216/.116905rad、pitch=.179952/.117051rad，C却有更低bodycollidermin63.278mm/USDmount122.578mm和硬限失败，不证明稳定性更优。C后1006RUN ENDED帧不入统计。

139 CPU四轮/相邻测试（19新增）与温度384通过/1显式deselected/0skips只支持代码语义，套件重叠不求和，不计物理信用。上一τ1024有1个P02硬限终态，本次2048没有；两次正式C都失败，不认定sigma或GAE根因。本批已完成、无活动仿真，不预填新候选或成功。
