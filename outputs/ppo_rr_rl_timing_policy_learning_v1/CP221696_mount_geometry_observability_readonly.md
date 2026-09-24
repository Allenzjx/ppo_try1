# CP221696：安装点、接收方向与准备空间（只读）

范围：仅同一封存 DET `20260923T2245371502769Z_gd7e97ee7b7e4_083d82482a6541f8a129462a60dd3dd0/source` 的已存几何／状态，以及静态资产、当前传感与观察代码。未运行 Torch、Isaac、模型或新物理；未修改生产。

## 1. 真正安装点与当前可验证边界

正确对象是 **USD hip Joint 的 body0/base_link 上 `localPos0`**，经该刚体的 `body_link_pos_w/body_link_quat_w` 变换：`mount_world = base_link_origin_world + R(base_link_quat_wxyz) * localPos0`。不是上腿活动刚体原点、轮心、轮底、base COM 或接触点。

本 run `height_diagnostics_startup.json` 实际解析并保存了 RR：

- Joint `/World/WLRRobot/joints/rear_right_hip`；body0 `/World/WLRRobot/base_link`。
- localPos0 米：**[−0.17017756402492523, −0.24299190938472748, 0.1100122332572937]**。
- tick9656 已验证 world 米：**[0.640854725433370, −0.218626886698343, 0.205077632949541]**。
- 同 tick 父 link 原点 `[0.816249191761017, 0.006649315357208, 0.068712033331394]`，quat wxyz `[0.9975147247314453, −0.06842344999313354, −0.016646742820739746, −0.002354523865506053]`；raw `bodies.base_link` 与独立 height diagnostic 相符。

静态 URDF `C:/robotics_sim/wlr_robot/sw2urdf_output/wlr_robot_isaac/urdf/wlr_robot_isaac.urdf` 的四个 joint 都以 base_link 为 parent：

| canonical leg / hip joint | URDF parent-local xyz（m） |
|---|---|
| FL / front_left_hip | [0.125605412969816, 0.001308093025761, 0.111477729785037] |
| FR / front_right_hip | [0.125613921776569, −0.241332100772366, 0.111477729785037] |
| RL / rear_left_hip | [−0.170186078223461, 0, 0.110012233831204] |
| RR / rear_right_hip | [−0.170177569416710, −0.242991906974238, 0.110012233831204] |

**验证限制：**RR URDF 与本次 live USD float32 localPos0 一致；本 run 没有保存另外三点的 USD Joint 解析结果。当前资产 `C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd` 是二进制 USDC，锁定 SHA `e8a2a2b1485a32a50e851a07b9dd8ac4945b78ec49b7fada2b61c3eeb1e18892`。本次未用 Isaac/PXR 再开 stage，所以 **FL/FR/RL 表内是确切 URDF 值，不冒充已在当前 USD 验证的 localPos0**，不据它们发布“实际四安装点平均高度”。

后续一次只读 runtime 解析即可补齐：复用 `semantic_height_diagnostics.resolve_rr_mount` 的相同 USD API，按四个明确 joint 名各取唯一 Joint、body0、localPos0；要求四点同一 base_link、资产 hash 一致，然后用已有同 tick 刚体姿态。不要假设坐标关于 y=0 对称：该资产 base_link 原点明显不在左右中心。

定义 `Δh_LR=(z_FL+z_RL−z_FR−z_RR)/2`（正=左侧高），`Δh_RF=(z_RL+z_RR−z_FL−z_FR)/2`（正=后侧高）。直接对世界重力 z 取高度，yaw 不改变高度意义；不要从未经校验的 Euler 符号猜左右。URDF 前后 local z 本身差1.465496 mm，故“原始差为0”也不应被当成固定姿态奖励目标。tick9656 raw base roll/pitch 为 −7.845666°/−1.921657°，仅作坐标诊断，不作角度目标。

## 2. FR 接收方向 CoM：本次已有真实短窗证据

`sensor_reader._body_state` 使用全部13刚体的 `body_com_pos_w/body_com_lin_vel_w/default_mass`；`compute_full_body_com` 质量加权且缺一刚体就失效，不退化成 base-only。raw 保存总质量 **2.930858523 kg**、来源、included_bodies、valid，但没有逐体质量数组，不能仅从该 JSON 独立重算质量分配。

本次 tick9656 的 RL→FR transfer role 已保存固定短窗 **9596→9656 / 0.5 s**：方向 world `[0.765845402, −0.643024743, 0]`，定义为窗口起点 CoM→当时 FR 轮心的水平单位向量（不是持续追着运动目标重定向，也不是安装点方向）。真实 CoM 位移投影 **−0.094833 mm**，末端速度投影 **−2.661144 mm/s**。因此此短窗没有向 FR 接收方向前进；更不能把方向性运动当成 FR 已承载或卸载。

此拍 FR 实际 TOP bearing **13.475726 N**，FL TOP **1.756256 N**；RL 是 GROUND_AND_OBSTACLE、force **10.612465 N** 但 `bearing_verified=false`；RR AIR/0 N。四腿 `load_fraction_valid=false`，已有0.521/0.068等数值不能被说成可靠的归一化载荷转移证明。

## 3. 此拍空间／余量到底是什么

单位°，负余量按实际关节距物理硬下界计算（hip −135，knee −60）；不是根据名义角或末端高度推断。

| leg | hip FINAL / actual | knee FINAL / actual | actual hip / knee 负余量 |
|---|---:|---:|---:|
| FL | 34.346 / 34.160 | −44.830 / −45.120 | 169.160 / **14.880** |
| FR | 8.726 / 9.422 | −21.414 / −21.356 | 144.422 / **38.644** |
| RL | 17.565 / 18.903 | .207 / .167 | 153.903 / **60.167** |
| RR | 9.373 / 9.105 | −56.821 / −56.593 | 144.105 / **3.407** |

所以 **此拍并非 FL 已到负向关节边缘**；FL knee 离执行投影下界−58也还有12.880°。RR膝才更接近负边缘。这些余量不证明任一方向必有碰撞安全工作空间。

此拍 RR AIR、Q/cross=true、合法XY、gap44.220 mm，`rr_top_reachable=true` 只是当前两条其他 verified 支撑+XY/余量代理，不是 FK/IK 可达性证明。RL→FR `workspace_progress=.684399`，由已饱和 `joint_margin_proxy=1` 和仅 **0.368799 mm** 的 FR wheel-body 径向收缩组成；未直接表达 FR 安装高度准备。RL wheel gap=−50.525 mm、front=−49.369 mm、仍 ground；这不是 RL 膝/上腿的 collision-free clearance 测量。

## 4. 原 request sigma，不据 raw 尺度盲加探索

从 decision **1207 / input9648→end9656** 的 `video_policy_decisions.jsonl.policy_request` 原值读取；DET `sampling_draws=0`，rho=.9，temperature=.25。这些是策略记录的 raw Gaussian sigma，**不是经过 tanh、HISTORY、投影后的实际角度/轮速标准差**。

| 通道 / index | 原 effective raw σ | learned raw σ | conditional raw mean | tanh(mean) | cap |
|---|---:|---:|---:|---:|---:|
| FL knee / 1 | 0.11675259470939636 | .700515509 | −1.514697552 | −.907769144 | 36° |
| FR knee / 3 | 0.01902173087000847 | .473429799 | −.509196758 | −.469319135 | 112° |
| RL hip / 4 | 0.08210519701242447 | .328420758 | −.476135880 | −.443143666 | 24° |
| RL knee / 5 | 0.005988160613924265 | .035928968 | .005743652 | .005743588 | 36° |
| RR hip / 6 | 1.1308242082595825 | 1.130824208 | .929029644 | .730141163 | 24° |
| FL wheel / 8 | 0.20696085691452026 | .827843487 | −1.245439172 | −.846999764 | 1.2 rad/s |

RR hip ×4已经启用；不建议继续加倍。此 RR carry 状态未开启现有 RL-prep 的 FR/FL knee ×2门，且 FR knee 原有 physical-equivalent 比例18/112已进入effective sigma；FL knee 的负侧 tanh压缩和RL knee很小的learned sigma须与各自cap、实际跟踪一起看，不能只比较raw数值定新倍率。现有422的actual q/quat/CoM/wheel/contact/carry/reachable可用于透明的纯当前状态准备许可，不必为诊断安装点新增actor维度。

## 5. 最小可观察补强与测试建议（不实施）

- **先补只读证据，尽量复用已有量：**四安装点 world z 和上述两高度差；现成 transfer role 的 signed CoM→FR 位移/速度及 reference tick；逐腿真实 pair/contact/verified bearing 与归一化载荷 validity 分列。现有422已含 base quat/gravity、joint位置速度、mass-CoM、wheel几何/接触，以及48维角色聚合；并未直接含四安装点高度差或 signed 短窗 CoM位移。几何高度差可由已验证静态安装点+现有姿态确定，不需要新状态计时器。
- **若用于新的重叠准备许可／势能：**只公开有限 preparation 的物理许可及必要连续量，不释放整段强 late group、不生成目标角。FL方向余量可从既有 actual joint/固定limits重算，避免再用早已饱和的margin proxy冒充空间；RL的独立碰撞自由空间目前缺真实上腿/小腿 collider clearance，应标未知，或先增同姿态 collider→障碍物的保守距离诊断。轮底净空、关节角和机身AABB距离都不能冒充整条腿轨迹安全。
- **最小测试：**四Joint同父和唯一性；姿态恒等/纯roll/纯pitch/yaw/共同平移的高度差符号；URDF≠USD或缺点时fail closed；RR world点与现有独立诊断相符；冻结FR窗向量不随CoM移动重建；AIR不得获得承载信用；混合接触load-invalid不造比例；FL上述14.88°余量反例不误报顶限；任何新增行动许可输入必须明确可观察、版本化并清旧rollout，不引入精确角度奖励。

现有 reward 的 task-space 项只有机身 collider AABB 对障碍物的保守距离，不是 FR接收高度／FL关节余量／RL腿部净空的专项项。RL在RR未placed或缺当前支撑时目前仅获既有0.1 workspace准备份额；可以研究复用这项表达有限准备，但不能把本页几何代理当动作成功证明。
