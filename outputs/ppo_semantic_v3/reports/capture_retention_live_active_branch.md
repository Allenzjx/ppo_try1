# Capture retention：真实主动分支的固定窗口核验

**范围仅 g61340–61440：101 条已优化决策、808 physics ticks。真实出现 R<1；曾短暂恢复区域保持度，随后退回 GROUND，窗口末未恢复。**

来源：[run 20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189)，episode 7，自然 P01 训练。实际 update 日志确认 **global 61440 / update 445** 已完成，finite gradient 与 actor changed 均 true。本报告不延伸到后续采集或优化。g61339 仅作为首行 shaping 的前状态边界，不计入 101 条样本。

PowerShell 只读重算；没有运行 Python/Isaac，没有修改生产、先前报告或训练。没有重做全训练报告。

## 1. 放置事件与实际 TOP 的证据边界

该回合 RR 的硬事件为 **qualified t5346 / crossed t5396 / placed t5397**。窗口开头 g61340/t5352 当前 RR AIR、clearance +4.330383 mm、front −21.053728 mm，已 qualified 但尚未 crossed/placed。

- g61345/t5392：front −1.544055 mm、clearance −2.368620 mm，within-top-XY=true、top-geometry=true、obstacle pair=true，但 **top-contact=false**、top-count=0，尚无 crossed/placed。进入 5 mm 容差区没有冒充放置。
- g61346/t5400：front **+2.514723 mm**、clearance **−2.389780 mm**，within-top-XY/top-geometry/top-contact/obstacle-pair 均 true；GROUND=false、AIR=false；**连续 TOP 样本 5**、load fraction **.1454026062**。此时真实历史已记录 crossed5396/placed5397，并从 P09 交接 P10。

这里的直接几何观测是决策结束 **t5400**，不是伪装成 t5397 的独立原始传感器样本。事件计数和五个连续 TOP 样本支持既有判定的放置事件；本训练流的每物理 tick 紧凑审计只有 native 请求/写入核验，没有逐 tick 接触点/法向几何。声明仍是 `verified_body_pair_and_live_wheel_geometry_no_contact_point_classification`：可核实当前既有 TOP+XY 判定证据，不可额外宣称 t5397 的接触斑块已被独立证明为纯台面法向，也不据随后回退重分类原事件。

## 2. 独立 phi / shaping 重算

每条已 placed 腿按实际 `top_xy_outside_distance_m` 和 `clearance_m` 计算：

`Rxy=clip(1−outside/.25)`，`Rz=.008/(.008+max(0,−.015−clearance))`，`R=min(Rxy,Rz)`，单腿新值 `.8+.2R`。

独立重算原未 placed 腿的 workspace/load/lift/carry/capture 与前驱条件，再合成完整新 phi；旧反事实仅把已 placed 腿恢复成 1，所有其他实际状态与公式不变。分别计算 `S=5(.995 phi_next−phi_prev)`。

101 条的完整新 phi、相邻 before、shaping、task-progress family（含原时间成本）与日志相比，最大绝对差均为 **0**。全部 101 条 native `all_ticks_verified` 与 no-in-episode-state-writes 核验均 true，无终止；阶段样本为 P09=7、P10=1、P11=2、P12=91。

RR 已 placed 的行数为 **95**，其中 **89 行 R<1**；FR/FL 始终 R=1，RL 尚未 placed。故本窗口新旧 phi 的直接差精确来自 RR：

`delta_phi = .85/4 × .2 × (R_RR−1) = .0425 × (R_RR−1)`。

`delta_S = 5(.995 delta_phi_next−delta_phi_prev)`；这不是每 tick 固定惩罚。即使当前保持度低，折扣项及局部改善也可能让新旧 shaping 差为正，不能把总体 reward 的正负直接当成保持度好坏。

## 3. 一段真实下降→几何恢复序列

单位：几何 mm；phi 和 shaping 为实际状态上的新公式及旧反事实。除 t5400 外，下表恢复过程并没有新的 TOP 放置事件。

| g / tick | RR 当前状态 | outside / clearance | R | 新 phi / 旧 phi | 新 S / 旧 S | delta S |
|---|---|---:|---:|---:|---:|---:|
| 61346 / 5400 | TOP，load .145403 | 0 / −2.389780 | 1 | .676662979 / .676662979 | +.332954878 / +.332954878 | 0 |
| 61348 / 5416 | AIR，load 0 | 0 / +.169480 | 1 | .674929531 / .674929531 | −.021817237 / −.021817237 | 0 |
| 61349 / 5424 | obstacle pair，非 TOP/非 GROUND | 7.205717 / −3.278329 | .971177134 | .678775028 / .68 | +.002258110 / +.008352345 | −.006094235 |
| 61350 / 5432 | obstacle pair，非 TOP/非 GROUND | 8.313826 / −2.960666 | .966744694 | .678586649 / .68 | −.017906560 / −.017 | −.000906560 |
| 61351 / 5440 | obstacle pair，非 TOP/非 GROUND | 5.881242 / −2.038880 | .976475033 | .679000189 / .68 | −.014907308 / −.017 | +.002092692 |
| 61352 / 5448 | obstacle pair，非 TOP/非 GROUND | 3.969244 / −1.566729 | .984123024 | .698535813 / .699210585 | +.080214728 / +.078572660 | +.001642067 |
| 61353 / 5456 | obstacle pair，非 TOP/非 GROUND | 2.175383 / −1.397817 | .991298470 | .699336494 / .699706309 | −.013480008 / −.015014036 | +.001534028 |
| 61354 / 5464 | obstacle pair，非 TOP/非 GROUND | .913254 / −1.492633 | .996346983 | .698930478 / .699085731 | −.019503343 / −.020580033 | +.001076690 |
| 61355 / 5472 | obstacle pair，非 TOP/非 GROUND | 0 / −1.141689 | 1 | .68 / .68 | −.111652391 / −.112428657 | +.000776266 |

这是真实保持度从 1 降至 .966744694 后连续恢复为 1；不是编造的恢复，也不是实际重新放置。t5472 的 front **−3.535863 mm**，处于原 XY 测量容差内；top-contact=false、load=.371177073。恢复的是允许容差的当前区域进度，没有补写新的 crossing/placed 历史、没有要求固定载荷。随后 g61357/t5488 为 AIR/load0 且 R=1；g61358/t5496 AIR/load0 但已退出区域，R=.977856430，说明接触状态本身并不决定保持信用。

## 4. 后续退回地面，未恢复当前台面位置

| g / tick | RR 当前状态 | front / clearance (mm) | R | 新 phi / 旧 phi | 新 S / 旧 S |
|---|---|---:|---:|---:|---:|
| 61400 / 5832 | obstacle pair，非 TOP | −32.671567 / −17.591147 | .755347807 | .669602282 / .68 | −.048126566 / −.017 |
| 61405 / 5872 | 窗口内首次 GROUND，load .571800 | −49.542811 / −50.709533 | .183026434 | .645278623 / .68 | +.065690725 / +.08925 |
| 61407 / 5888 | GROUND，load .454287；最低 R | −49.309131 / −50.746525 | .182871666 | .645272046 / .68 | −.016373352 / −.017 |
| 61421 / 6000 | GROUND，load .086257 | −102.293011 / −49.613441 | .187734194 | .638689666 / .673210962 | +.055628674 / +.055474538 |
| 61440 / 6152 | GROUND，load .067042 | −115.084526 / −49.535676 | .188077417 | .636344447 / .670851156 | −.015814288 / −.016788718 |

t6000 的 Rxy=.610827956、Rz=.187734194，当前下降净空项取较小值；新 RR 单腿值约 **.837546839**，不再固定 1，直接 phi 比旧反事实低 **.0345212968**。窗口末差为 **−.0345067098**。在这些实测状态中，新保持项确实看到了退回；但它没有撤销历史 placed，也没有保证训练会避免此行为。

首次 GROUND 之后到 g61440 未回到 R=1，未恢复当前 TOP；只有台下几何的小幅变化，不能包装成再次完成。截止 t6152，FR TOP/load .388531，FL AIR/load0，RL GROUND/load .544427、未 qualified/cross/placed；仍是 P12 非终止尾段，不是全任务成功，也不是后续结果预测。

结论只限于真实信号链：有效放置记录后，当前区域退回引起新旧 phi 与 shaping 的可核算差异；也有一段短暂的区域进度恢复。不能从这种公式响应证明策略已学会保持，不能把后腿所有失败归因于 RR 回退。报告到 61440 停止。
