# P07 课程首回合：连续后腿运动与观测诊断

## 固定窗口与结论

仅分析 run `20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5` 的已完成 episode 0：global **73089–73166**，78 个真实 PPO 决策，P07=1、P08=1、P09=76。它们属于已保存 rollout_000537；不扩展到同 rollout 后 50 条下一回合观测，也不重复 checkpoint/hash/训练累计账本。

信用起点为真实 prefix 后 tick5952 / 49.6s / P07；随后运行 622 个 physics ticks（77×8+末决策6），到 **tick6574 / 54.783333s 的 P09 BODY_COLLISION 真终止**。本段 PPO 物理时长5.183333s；54.783333s包含无 PPO credit 的真实前缀，不能当作78决策的训练时长。

- 连续到达 P07→P08→P09；普通切换没有 done/terminal/reset。
- RR 的 initial6037、initial6233、hard qualified6374 均早于 collision6574；**RR 从未 cross/placed**。最接近决策末前沿距离仍 −25.574mm，末 −32.144mm。
- 前腿已有 Q/C/P 是 prefix 历史；不记作该78决策新取得。FL 在信用起点有接触承载，但首 P07 决策中变 AIR，其后又有真实短 TOP 承载，末再次 AIR。历史 placed 不能替代当前支持。
- CoM 字段经源码追溯确是13刚体质量加权全机器人 CoM，不是 root/base 单刚体 COM。然而其 relative_base 是 **world轴差向量**；world-y 变化不能直接当成向实际 FL 支点移动，更不能据此认定碰撞主因。
- 这是随机策略的训练后缀片段，不是固定 mean 全程评估，也不是完整或后缀成功。新 capture shaping 需要 RR hard Q+C；本段 C=false，RR 新 capture 分支始终未激活。

## 证据时间分辨率

输入为前78行 `residual_and_projection_audit.jsonl`、首条 completed episode 和 `rollouts/rollout_000537.pt`。本训练没有完整120Hz raw observation/contact trace：

- 每决策保留末 tick 的 physical evaluator、nominal/residual/actual drive/native target，以及逐 tick 的 native/own-request/hold 标记。
- hard event history 记录真实 event tick，不能用日志写入的8tick边界冒充事件 tick。
- 保存的324维 rollout observations 是 **动作发出前** 的观测。第0条对应tick5952；第77条对应tick6568，随后最后动作运行6tick才在6574终止。它不是6574 terminal observation；第78条属于下一回合，刻意排除。
- CPU只读一次 torch.load，线程1、interop1、map_location=cpu、禁CUDA可见设备，无模型构造/优化器/仿真。验证 source schema、编码器和 command order 的小文件hash与保存runtime一致；128维外形中的前78条 raw/old_value/done逐行绑定。policy/critic观察一致，全部324组在此78条中均未触及clip±20，逆缩放没有饱和歧义。解码的float32数值仍有正常舍入误差。
- 同目录 `p07_continuous_first_episode_decode.py/.json` 保存脚本与唯一一次输出receipt，执行exit0/2.763196s。没有写真实权重、动作、训练计数或历史manifest。

## P07→P08→P09 与动作连续性

| 事件 | physics tick / 时间 | 当前完成证据 | PPO terminal |
|---|---|---|---|
| P06→P07、信用起点 | 5952 /49.600000s | rear_approach=1，真实prefix提供 | 尚未发首动作 |
| P07→P08 | 5960 /49.666667s | workspace_RR=1、workspace_RL=1、support_RR=1 | false |
| P08→P09 | 5968 /49.733333s | workspace_RR=1、load_ready_RR=1 | false |
| 首P09完整决策结束 | 5976 /49.800000s | 已执行新阶段raw | false |
| 最后P09决策结束 | 6574 /54.783333s | BODY_COLLISION | true |

P07/P08 各一决策是上述物理条件在8tick交接边界已经满足，并非两阶段完整 authored motion 都执行完毕。名义执行器保留各层，旧层继续按自己的 touched-channel ownership 运行；后层拥有的通道优先，阶段交接不清空全12通道。

逐tick native audit 对本段最初交接给出：

- P07 g73089：5953–5960共8tick均 own request effect=true；prefix→首raw的bridge记录 hold=false，首tick逻辑动作最大变化servo0.5°/wheel0.015rad/s，非强制清零。
- P08 g73090：**5961**仅首tick handoff_hold=true/own=false，5962–5968七tick own=true；交接 applied jump为0，残差保持差只有约1e−16浮点舍入。
- P09 g73091：**5969**仅首tickhold，5970–5976七tick own=true； applied jump为0。
- g73092：5977–5984无hold，8tick own=true。因此不能称一决策阶段raw全部丢弃。

名义、残差和实际dispatch是不同层，不能把 nominal 归零或阶段短等同于 residual 被关掉：

| 决策末tick | 名义四轮（FL,FR,RL,RR；rad/s） | 逻辑实际dispatch四轮（rad/s） |
|---|---|---|
| 5960 | [.3,.3,.3,.3] | [.410311,.298502,.237809,.254222] |
| 5968 | [.3,.3,.3,.3] | [.305311,.193502,.342809,.359222] |
| 5976 | [.3,.3,.3,.3] | [.298118,.199930,.400625,.464222] |
| 6040 | [.3,−.63,.3,.3] | [.430148,−.980358,−.073099,.212900] |
| 6144 | [.3,0,.3,.3] | [.371295,−.286706,−.004875,.298444] |
| 6216 | [.3,.3,.3,.3] | [.521268,−.029471,−.187840,.301777] |
| 6574 | [.3,.3,.3,.3] | [.514707,−.082901,.003651,.150584] |

P07 的 FR 调整没有因进入P09而马上消失：决策末tick6000开始nominal FR由.3降至.125，6008 −.075、6016 −.275、6024 −.475、6032达到−.63；6120以后回升，6144为0，6216回到.3。这里报告的是可见调度与输出，不将各关节正负号直接等同世界升降。

同样，全身servo建议/实际响应持续存在。5960→5968→5976，名义 FL hip=31.55→37.6→46.1°、RL hip=6.9→14.3→18.5°、RR hip=0→0→1.6°；6040 RR nominal hip/knee=[52.4,0]，到6240及后段已是[−6.9,−37.8]。残差、mapper compensation 与最终slew后的servo drive不同于这些建议，详见JSON，不能单用建议解释实际几何。

本P07 from_handoff 的名义诊断明确 **没有P06 layer**，p06 tail/retirement均not_applicable；四轮.3是接管后的名义内容，不是该片段新触发P06尾层。

native符号按真实映射：FL/RL轮轴与canonical向前符号相反。例如6574 PhysX float32 wheel targets为[−.514707446,−.082900815,−.003650532,+.150584266]，对应上表逻辑正向dispatch。不能混用符号。七个predecision观测中mapper tracking compensation始终有保留的非零分量；与逐tick零handoff jump一起支持状态延续，而不是相切时重建零mapper。没有完整每tick滤波数值，故不宣称重建了所有内部buffer。

## RR真实过程：initial、短接触、qualified、未cross

| 时刻 | 真实已记录事件或决策末状态 | 不能混淆的界限 |
|---|---|---|
| 6016 /50.133333s | 首个可见RR AIR边界，air count=1；6024已GROUND | 无完整raw，不能排除更早子tick短AIR；不是Q/C/P |
| 6032 | AIR count7（当前连续AIR后缀起于6026），initial仍false | 不是“轮子离地就qualified” |
| **6037 /50.308333s** | first whole_body_initial_clearance，gain3.115311mm；own motion6.872299°，whole-body20.880188°，command40.051889° | 本PPO片段新事件；不是cross/placed |
| 6144–6312若干边界 | 有GROUND与obstacle-active但non-TOP交替，见JSON分段 | “OBSTACLE_OTHER”仅表示当前几何非TOP，缺接触点不能擅称特定墙面接触 |
| **6233 /51.941667s** | 第二initial，gain3.784910mm，own2.090229°/whole-body6.083135° | 经实测过程获得；不据此新加RR自身2°要求 |
| **6374 /53.116667s** | hard qualified，clearance above top +0.186214mm，记录upward excursion36.955312mm、jointmotion16.873695° | Q真实早于碰撞200tick/1.666667s；C/P仍false |
| 6376 | RR AIR，front−47.430mm，clear+3.016mm，air count33 | 该连续AIR后缀起于6344 |
| **6400** | 本78条决策末中最接近前沿：front−25.574mm，clear+27.099mm，AIR/load0 | 仍未越过中心front plane |
| **6416** | 本78条决策末峰值clear+29.149mm，front−27.279mm，AIR/load0 | 非120Hz全轨迹峰值保证 |
| **6574** | RR AIR count231，front−32.144mm，clear+1.845mm，Q=true、C/P=false | 当前within_top_xy=false，outside XY27.144mm；后退且降高，但不能将其说成Q未获得 |

RR在78个决策末的接触类别：AIR53、GROUND14、obstacle-active非TOP11；不能用相邻AIR边界推断中间每tick都无接触。末air count231独立支持6344–6574的当前连续AIR段。RR末仍有净空但水平未到区域，first unfinished在发生BODY_COLLISION前仍是RR越沿/放置链；不是RL末端任务或旧入口守卫。

本段FR/FL历史：FR Q71/C1665/P1695；FL Q2461/C3115/P3583，均早于信用起点5952，全部明确排除为本PPO新QCP。RR硬Q6374则为信用段中新增。

## FL当前承载与全身载荷

**起点5952不是FL AIR**：真实predecision pair force为FL obstacle5.700586N、FR obstacle7.144557N、RL ground8.689030N、RR ground7.094827N，载荷份额[.199119,.249557,.303504,.247820]。随后5960末FL已AIR/count7，说明5954–5960连续AIR；不能把教师历史placed或起点接触沿用为整个片段FL支持。

决策末FL AIR61/78、TOP17/78：5960–6424的59个末边界AIR；6432TOP、6440AIR；6448–6568的16个末边界TOP；6574末AIR。该分组是边界采样，不声称59×8全部持续AIR。

| tick | FL 当前 | FR 当前 | RL 当前 | RR 当前 | 载荷份额 FL / FR / RL / RR |
|---|---|---|---|---|---|
| 5960 | AIR | TOP | GROUND | GROUND | 0 / .412467 / .532688 / .054845 |
| 6032 | AIR | TOP | GROUND | AIR | 0 / .488219 / .511781 / 0 |
| 6240 | AIR | TOP | GROUND | obstacle非TOP | 0 / .355160 / .330865 / .313975 |
| 6376 | AIR | TOP | GROUND | AIR | 0 / .495949 / .504051 / 0 |
| 6480 | TOP | TOP | GROUND | AIR | .067023 / .480726 / .452251 / 0 |
| 6568（最后动作前） | TOP | TOP | GROUND | AIR | .071262 / .480923 / .447815 / 0 |
| 6574（终止） | AIR | TOP | GROUND | AIR | 0 / .533386 / .466614 / 0 |

终止FL clear+3.348mm、front+.394742m、load0、support=false；FR TOP、RL GROUND确有当前支撑，RR AIR无载荷，support_count=2。load fraction采用既有pair force标量聚合，不是独立测量的竖直支撑力分解，更不是静力平衡证明。

## 实际关节、轮速、重力姿态与速度解码

关节顺序始终 FL hip/knee、FR hip/knee、RL hip/knee、RR hip/knee。以下 q/qd 是 **实际传感器读回的canonical logical deg/deg/s**：位置从adapter actual_full12来，速度由native rad/s转换deg/s并乘SERVO_COMMAND_SIGN；不是nominal目标，也不是直接USD关节rad。后腿sign为−1，恢复native绝对角还需要standing offset，不能省略它。

所有行是决策前tick，不与同global的末tick混配：

| pre tick | 实测 q（deg，8通道） | 实测 qd（deg/s，8通道） |
|---|---|---|
| 5952 | [21.83, -12.17, -0.28, 45.62, 6.31, 0.47, -1.03, 0.76] | [-3.53, 2.26, -2.04, 1.02, -3.66, 3.59, -3.97, 3.71] |
| 5960 | [24.29, -13.35, 0.94, 44.34, 5.61, 1.34, -0.21, 2.00] | [66.33, -28.00, 13.18, -27.72, -15.30, 27.46, 37.29, 47.56] |
| 5968 | [30.41, -15.66, 1.01, 43.20, 6.46, 1.59, 0.28, 2.46] | [103.56, -40.36, -18.48, -6.17, 26.98, -5.50, -10.38, -11.85] |
| 6032 | [46.62, -17.12, 0.74, 19.67, 30.77, 1.23, 33.22, -13.84] | [-12.72, 1.58, -36.17, -20.26, 41.90, 7.69, 51.04, -17.69] |
| 6368 | [38.68, -20.93, 16.73, 14.39, 30.29, -3.11, -44.10, -59.26] | [-30.76, 13.01, 30.63, -9.04, -24.39, 2.30, -29.34, -7.85] |
| 6376 | [35.99, -21.32, 14.39, 14.97, 30.80, -2.08, -45.69, -59.20] | [-28.61, -11.02, -56.15, 8.31, -2.37, 36.98, -13.78, 4.69] |
| 6568 | [35.79, -23.52, -2.89, 13.40, 50.02, 0.77, -19.51, -55.79] | [-11.09, 30.29, 20.02, -20.98, 74.49, -49.92, 5.17, -10.72] |

| pre tick | 实测四轮canonical速度（rad/s） | base_link线速度 world（m/s，旋回所得） | 全机器人CoM速度 world（m/s） |
|---|---|---|---|
| 5952 | [0.2753, 0.2141, 0.3262, 0.3180] | [0.0168, -0.0005, 0.0091] | [0.0165, -0.0009, 0.0045] |
| 5960 | [0.4117, 0.2607, 0.1609, 0.4810] | [0.0472, -0.0211, 0.0444] | [0.0201, -0.0392, 0.0372] |
| 5968 | [0.3048, 0.0013, 0.3057, 0.3081] | [0.0769, 0.0143, -0.0047] | [0.0599, 0.0271, 0.0380] |
| 6032 | [0.4095, -1.0002, 0.2583, 0.2769] | [0.1176, 0.0035, -0.0016] | [0.0739, 0.0118, 0.0048] |
| 6368 | [0.4487, -0.0315, -0.1064, 0.1927] | [-0.0358, 0.0088, -0.0569] | [0.0062, 0.0296, -0.0205] |
| 6376 | [0.5249, -0.0601, 0.0087, 0.1671] | [0.1142, 0.0027, 0.0452] | [0.0976, 0.0379, 0.0646] |
| 6568 | [0.5018, 0.0007, 0.2081, 0.2416] | [-0.0429, -0.0145, -0.0916] | [-0.0532, -0.0312, -0.0947] |

编码原始body_linear_velocity属于body坐标；上表用同tick保存的normalized quaternion旋回world。body_angular_velocity同样是body轴，JSON保留；raw/chassis quaternion、gravity、Euler导数也来自同一predecision观测，未按每回合重新level。

例如5952 projected gravity=[−.031879,+.041788,−.998618]；6368=[−.128255,−.009911,−.991692]；6568=[−.195650,+.207828,−.958399]。最后动作前6568仍有向下速度：base Vz=−.091558m/s、全机器人CoM Vz=−.094717m/s；最后动作6tick后的6574日志body speed降至.007641m/s，**这是两个不同时间**，不能用6574末速率代替碰撞前下降率或断言接触造成了全部速度变化。

## CoM是否向FL侧：可定义量与不能推出的结论

| pre tick | CoM−base_link，world轴（mm） | 同差向量旋到body轴（mm） | chassis RPY（deg） |
|---|---|---|---|
| 5952 | [-18.006, -118.163, 70.199] | [-15.710, -121.024, 65.733] | [-2.396, -1.827, -0.024] |
| 5960 | [-18.267, -118.666, 70.399] | [-16.140, -121.024, 66.813] | [-1.956, -1.688, -0.022] |
| 5968 | [-19.167, -118.413, 72.660] | [-16.701, -121.024, 68.867] | [-2.101, -1.911, -0.016] |
| 6032 | [-40.603, -121.300, 82.251] | [-22.694, -121.024, 89.250] | [0.164, -11.987, 0.029] |
| 6368 | [-21.876, -121.137, 80.695] | [-6.966, -121.024, 83.481] | [0.573, -7.369, -2.082] |
| 6376 | [-21.918, -118.902, 82.536] | [-7.128, -121.024, 82.053] | [-0.987, -7.305, -1.994] |
| 6568 | [-34.066, -100.672, 102.915] | [-13.818, -121.024, 83.960] | [-12.235, -11.283, 0.317] |

78条predecision中world相对base y范围−128.980至−99.563mm，起末−118.163→−100.672mm；world CoM Vy范围−.068549至+.066378m/s，起点−.000910、末动作前−.031198m/s。因此不是持续朝+worldY的平移。七个明确展示的点旋到body轴后y都约−121.024mm（差约2e−5mm），同时roll/pitch明显变化；这个事实不足以解释其内部动力学原因，更不能把world相对y的变化直接归因于主动将质量横移到FL。

现有324组只保留轮的前/后距离、顶面gap及速度，没有四个轮center的world-Y坐标或真实接触点。故无法从本78条存储观测精确构造“CoM到实际FL触点”的水平向量、绕FR/RL支持线的稳定力矩、完整支持多边形随tick变化；也不能仅用“FL名为left”推断CoM向该支点移动。6568 pre-observation support diagnostics记录有效三支持且margin+.032002m，但6574已两支持；不能跨时复用该margin来宣称终止时稳定。

### 该CoM字段的实际生产来源（有限源码追溯）

这不是base-only COM代理：

1. `sensing/sensor_reader.py:_body_state` 约392行严格要求 live body_names 等于锁定的13个SENSED_BODIES（base_link、8腿link、4wheel）。约400–402行读取 **body_com_pos_w、body_com_lin_vel_w、default_mass**，按每个body名称建映射；没有用root_com_pos_w替代。
2. 同文件read约275–278行把这些13-body数据传入 `compute_full_body_com`。
3. `sensing/com_diagnostics.py:18–61` 默认required_bodies=SENSED_BODIES；缺任意body位置/速度/质量、质量非正/非有限则invalid。计算 `Σm_i p_i / Σm_i` 和 `Σm_i v_i / Σm_i`，返回included_bodies及明确source，无base-only fallback。
4. `semantic_observation.py` 约262行的 `com_position_relative_base` 是该全机器人world CoM减base_link world位置；约263行的 `com_velocity_world` 直接取全机器人CoM world速度。仅前一个做平移减法，**没有旋到body轴**。本报告的body轴列是离线用同tickq显式旋转所得。
5. 因此字段的生产定义可以称13刚体全机器人CoM；但本rollout没有逐link质量/位置矩阵，不能独立在此重算13项或把近常量y归咎于传感器/质量错误。该近常量不构成全身载荷因果解释，不扩展新实验。

## 终止判定、原生执行与局限

最后记录 physical evaluator valid=true、reason=central body/obstacle collision，task termination=TASK_FAILURE_BODY_COLLISION，env terminal=BODY_COLLISION，非纯阶段deadline。已有collision detector契约为exact base_link/obstacle pair加持久/穿透证据；**本训练精简记录没有完整原始body pair force向量、active history、contact point或penetration值，无法重新逐tick审验6574的exact pair细节**。报告既不捏造“已读到2tick原始接触”，也不据缺档反向否定detector。

本首回合所有逐tick native审计verified、所有no-in-episode-state-write标志true；末6tick都存在真实own-request effect。说明真实命令有执行且未靠中途root/力/重力写入制造状态，不能自动证明任务安全成功。

已可证实的控制/几何链是：合法前缀到P07 → 真raw参与P07/P08和持续P09 → FL当前载荷改变与多次RR接触/离地 → RR硬Q → 前沿尚未跨越、净空回落 → BODY_COLLISION真终止。哪些动作或载荷变化是碰撞原因，还不能由单个随机训练后缀和稀疏physical snapshots唯一识别；不添加新姿态、支撑或RR自身动作门槛，也不将本诊断设为训练门禁。

## 附加假设核验：Q后15mm carry切换是否重新暴露FR负向owner

仅复核相同78行，无新增CPU加载/实验。代码 `semantic_supervisor.py` 约982–988行的carry覆盖确需：P09/P12、无terminal、hard active_lift、当前clearance≥现有15mm、front<−5mm；满足时四轮proposed覆为[.3,.3,.3,.3]。这是建议分支，不是硬任务完成条件。

本样本硬Q后26个决策末边界的结果：

- 6376 clear3.016mm、6384 clear14.941mm，低于15mm；6392–6552共21个边界均≥15mm；6560 clear12.726mm、6568 clear4.377mm、6574 clear1.845mm重新低于。边界证据是一段上穿、后续下穿，不是反复开关。
- **26个末dispatch nominal四轮全部是[.3,.3,.3,.3]**。FR negative nominal在hardQ之前就已结束：6216回到+.3，而Q事件6374。因此这里没有观察到“关carry即恢复P07 FR−.63建议”的效应。
- 即使条件在内部切换，覆盖的+.3与当前合成建议可能数值相同；日志未保留逐tick的owner选择/覆写前向量，不能宣称已逐tick证明该分支从未执行。也不能从末边界nominal推导隐藏的每个中间tick，但没有可见证据支持假设中的反复负向建议复现。

| 末tick | RR gap（mm） | 末几何是否≥15mm | nominal FR | residual FR | 实际逻辑dispatch FR（rad/s） |
|---|---:|---|---:|---:|---:|
| 6384 | 14.941 | false | +.3 | −.314222 | −.014222 |
| 6392 | 23.237 | true | +.3 | −.367156 | −.067156 |
| 6400 | 27.099 | true | +.3 | −.247156 | +.052844 |
| 6408 | 28.974 | true | +.3 | −.328154 | −.028154 |
| 6552 | 16.297 | true | +.3 | −.208729 | +.091271 |
| 6560 | 12.726 | false | +.3 | −.278790 | +.021210 |
| 6568 | 4.377 | false | +.3 | −.292901 | +.007099 |
| 6574 | 1.845 | false，且已terminal | +.3 | −.382901 | −.082901 |

FR实际目标确有正负变化，但表中是相同+.3 nominal上residual跨越−.3的可见结果；6552→6560下降穿越15mm时FR实际目标仍为正，没有同期方向翻转。6574末反向则伴随residual变为−.382901，不是nominal恢复−.63。不可把这一分解直接当作策略误动作/碰撞因果结论。

严格时间边界：每行gap是物理步后的current state，nominal/residual是该决策最后一次dispatch所用的source state动作（先于此末观测一个physics tick）。无120Hz完整几何/owner流，不能把“末gap跨阈值”精确贴到同一行最后dispatch的分支时刻。

报告与JSON保存后停止；没有改变运行版本或增加任何训练/实验更新。
