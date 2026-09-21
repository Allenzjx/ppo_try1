# CP182528随机评估：FL捕获后RR未完成，低高度安全中止

封存source `20260918T0810262817782Z_g3a50657a96c9_63b77f8309b24cd4a750cb8bbf4e3013/source`；同CP182528，training_style_conditional_gaussian、seed4101，自然P01，非optimizer数据、非prefix课程。本次只摘代表行，不重做全量概率/控制审计。

## 真实事件与第一个未完成任务

FR placed=1564；FL crossed=2483、placed=2761。P06首request tick2768，随后P07/P08/P09首request分别4888/4896/4904。**RR从未crossed或placed，P09的RR连续抬升/前送、越沿与放置未完成**；不能把出现过初始抬升当成RR成功。

| RR事件 | 精确物理tick | 独立当前资格核验 |
| --- | ---: | --- |
| 首次qualified lift，连续无支撑上升9.273mm | 5167 | 后续returned evaluator5168：AIR、current_lift_valid=true |
| 越沿前落地，qualification及current lift撤销 | 5394 | evaluator5400：active_lift/current_lift_valid=false；虽已再AIR，不能自动继承旧资格 |
| 再次qualified lift，连续无支撑上升8.170mm | 6335 | evaluator6336：AIR、current_lift_valid=true |
| 再次越沿前落地撤销 | 6381 | evaluator6384：GROUND、资格false |

末个正常返回的任务评估为tick6696：RR current_lift_valid、lift_established、active_attempt及history.active_lift均false，crossed/placed也false；contact_mode=GROUND_AND_OBSTACLE、bearing_verified=false、front−49.252mm。`event_ticks.active_lift.RR=5167`保留的是**首次历史事件时间**，不是当前资格，也不会覆盖6335的再资格事件。事件发生拍与后续15Hz评估拍已分开，不伪造同拍状态。

## 安全终止来源

最后decision6696→6699在 **tick6699 / 55.825s** SAFETY_ABORT，environment_step_returned=false；这三拍有真实native/physical记录，但没有终拍semantic step_info，不能将6696评估冒充6699。

实际base_link z=**.014881544m < .015m**，命中isaac_fsm_backend.py:6551的FALL高度条件。gravity_z=−.998662，不命中`>−.30`的姿态条件；线速度范数.127997m/s、角速度范数.516702rad/s，均远低于5/20的explosion条件。base_link/obstacle精确碰撞detected=false。故这是低高度FALL，不是本次机身障碍碰撞或速度数值爆炸；任务仍失败。

## 请求交接与代表性全身响应

P05→P06 raw HISTORY与上一决策完全一致；首物理拍全12 REQUEST继承最大数值尾差4.44e−16，final servo8最大差也仅4.44e−16°。四轮final则各增加.3rad/s，与P06 nominal前送接管相符，不能把跨单位max=.3写成关节跳变，也不能称全部final动作都完全不变。未发生全局清零。新kernel首拍10个cap扩大通道gate=true、RL/RR wheel=false。FR knee前filtered REQUEST+1.731778°，raw历史+.072283重表达center+.015464；base−.002260、conditional+.013691、sigma.073150，真实sample−.007754；首P06末tick2776 REQUEST−.868455°。这只确认实际入口处理，不证明后续全身风险已解决。

下表base/raw无量纲，其余关节量为canonical度。REQUEST含已有滤波/限速历史，不把它强行等同当拍cap×tanh(raw)。

| tick / 事件 | body z mm | FR knee base / raw | FR knee REQUEST / final / actual | RR hip REQUEST / final / actual | RR knee REQUEST / final / actual |
| --- | ---: | --- | --- | --- | --- |
| 2776 / 首P06末 | 84.232 | −.002260 / −.007754 | −.8685 / 47.3034 / 51.0511 | +8.4520 / 8.4520 / 8.1847 | −5.8349 / −5.8349 / −5.5799 |
| 5167 / RR首资格 | 59.667 | +.047861 / −.201865 | −22.3067 / 15.6342 / 17.2944 | +13.5121 / 27.0606 / 18.0218 | −8.5300 / −8.5300 / −8.9756 |
| 5394 / 落地撤销 | 58.570 | −.025750 / −.186770 | +2.6611 / 40.6020 / 45.6934 | +11.7095 / 12.3095 / 13.7963 | −8.3396 / −38.6396 / −39.7350 |
| 6335 / 再资格 | 32.485 | +.019354 / −.024155 | −7.6854 / 30.2555 / 26.9929 | +12.8388 / 3.4388 / 4.0962 | −7.2921 / −47.5921 / −46.9368 |
| 6381 / 再撤销 | 22.589 | +.034212 / −.104343 | −16.6854 / 21.2555 / 23.2266 | +11.2705 / 1.8705 / 3.2984 | −6.6742 / −46.9742 / −46.6003 |
| 6699 / FALL | 14.882 | −.029784 / +.017421 | +2.0738 / 40.0147 / 44.4878 | +2.5793 / −6.8207 / −4.7986 | −9.5176 / −49.8176 / −49.7662 |

末拍RR hip/knee base分别+.439898/−.240282、raw+.107886/−.270811。FR knee mapped N在P06代表拍为48.171859°，P09代表拍为37.940937°；RR还受独立nominal几何层影响。例如5167 RR hip mapped N23.8°被几何层修为13.8°，加REQUEST13.512056°得到candidate27.312056°，最终slew为27.060562°；**不能将final−原mapped N全部当作PPO作用或输出丢失**。该拍记录tracking超出局部线性可信范围、nominal操作包络已达、需要全身高度/支撑重配置；这不是新发现的mask错误，也不保证几何层一定创造真实净空。

代表行均派发校验通过且没有headroom clipping，但有实际跟踪滞后和接触/载荷变化，不据ACK推断物理必然正确。FL捕获后并未持续承载：tick2768已经AIR（gap.394mm），tick2848/4888/4904分别AIR约35.30/40.36/61.38mm；RR两次资格时相对台面净空仍−40.84/−41.93mm，随后重新接地。终态FL AIR约13.64mm，FR障碍接触、RL地面、RR地面加障碍混合接触。不能把FL朝向/历史placed虚构成稳定承载，更不能用FR knee某一时刻正负号孤立解释所有机身下降。

结论：本次随机策略真实完成FR/FL捕获并到达RR任务，但没保持可用RR抬升或越沿/放置，最终低高度中止；既不是完整成功，也不据单次观测宣称新实现bug或单关节根因。当前自然P01续训改变采样覆盖，不是已改变reward/caps/temperature；P05 offset200仍只是下一课程候选，未在本run执行。

证据：[CP182528_stochastic_handoff.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/CP182528_stochastic_handoff.json)。额外模型前向/优化/仿真=0，未用CUDA、未改生产或DELIVERY/RECOVERY，CPU解析已自然退出。
