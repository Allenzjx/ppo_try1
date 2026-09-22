# CP216448 deterministic：P05 pre-edge 停滞，固定 28–38 s

结论：**先发生的是 P05 源 nominal 的合法四轮 stop；不是 residual mask 丢失。** 策略在 stop 前已反转悬空 FL wheel、减小 RR 前送；stop 后仍下发非零轮残差，但没有形成有效的整体前送。FL 约停在前缘前 34 mm，现有 post-cross capture continuation 与 assist 都尚未具备启动条件。此为固定窗口诊断，不是最终录像判定。

范围：tick **3360–4560**，另取 3359 校验一拍交接。初查时 raw/native 文件尚未刷出；收尾时它们已经写出，因此只补取同一窗口的 10 个对应样本，**raw 不再列为缺测**。未扩展到后续状态，未运行 forward／物理／优化。

## 第一处停止与有限源时序

- 28.0 s：四轮 N 和 mapper N 都为 **+.3 rad/s**。
- **tick 3488 / 29.0667 s**：after-frame N 首次四轮归零，该拍真实 dispatch 仍用前一状态的 +.3。
- **tick 3489 / 29.075 s**：实际 mapper／dispatch nominal 才变成四个 0，此时 FL front=−66.484 mm，尚未 cross。
- 3569 首次记录 source_endpoint_issued=true，3570 记录 source_unfold_dispatched=true；至 38 s 不再有新的 P05 名义前送。
- 固定 P05 源合同明确写有 `wheel all 0.300`（相对 .8667 s）和 `wheel stop`（9.0667 s），随后膝／hip 展开到 [22.8°,−13.4°] 的有限 endpoint（9.7333 s）。**不能将作者明确的 stop 改称接线 bug。** 当前 live atomic-group ID 未存于本次选取 trace，未伪造某个 source batch ACK。
- 所有 1201 拍满足 **dispatch N = 前一 after-frame N**、**final wheel = 同一 ACK 的 mapped N + 已投影 REQUEST**，误差均 0。不是拿重算 N 相减冒充策略残差。

## 四轮逐通道：原始 policy 到执行与实测

单位：raw 为无量纲 Gaussian latent；其他轮速 rad/s，按 canonical 方向。cap=[1,.6,1,.6]，实际 residual permission 四轮均 1；以下 REQUEST 是实际 HISTORY／tanh／投影后的记录。诊断为 deterministic，sampling_draws=0。

| 时刻／腿 | 源／mapped N | raw policy | 已投影 REQUEST = 有效附加量 | 最终 target | 实测 canonical |
|---|---:|---:|---:|---:|---:|
| 28 s FL | +.300 | −.902982 | −.717747 | −.417747 | −.417564 |
| 28 s FR | +.300 | +.014162 | +.008497 | +.308497 | +.292403 |
| 28 s RL | +.300 | +.032670 | +.032659 | +.332659 | +.433561 |
| 28 s RR | +.300 | −.311128 | −.180878 | +.119122 | +.088577 |
| 38 s FL | 0 | −.873603 | −.703200 | −.703200 | −.703704 |
| 38 s FR | 0 | +.009983 | +.005989 | +.005989 | −.258784 |
| 38 s RL | 0 | +.028913 | +.028905 | +.028905 | +.154630 |
| 38 s RR | 0 | −.290859 | −.169755 | −.169755 | −.102310 |

38 s native setter-buffer wheel targets 为 **[+.703200,+.005989,−.028905,−.169755]**，与 canonical 的 FL/RL 轴符号翻转相符；不能直接把两套符号混用。选取的 native audit 均 verified=true、setter_dispatch_targets_equal=true、actual_mapping_matches_dispatch=true；最后写入方记录为 `robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim`。**具体 native joint IDs、native joint 实测角速度未在这些选取字段中保存，JSON 保持 null。**

FL 当时悬空，虽然旋转接近目标，却不能提供地面牵引。FR／RL 在接触中实测明显偏离小正目标；这是实际跟踪／负载耦合现象，**不等于被 mask，也不足以单独确定驱动饱和、摩擦或某个电机是唯一原因**。RR 目标和实测均负。不是“只有 RR 轮子有指令／旋转”。

## 支撑、几何与 continuation 边界

30.883–38 s 的 855 个真实 tick：

- FL 全程 AIR、侧向合法、current physical valid；front 在 **−34.319 至 −33.454 mm**，gap **+2.133 至 +8.113 mm**。
- FR TOP、RL/RR GROUND 三个其他支撑持续存在；记录的支撑投影裕量 **+26.278 至 +32.061 mm**。FL 本身不承载。
- 机身 x 净变化仅 **−0.542 mm**；不是轮端看起来静止就假定轮电机没动。
- 38 s FL nominal hip/knee=[22.8,−13.4]°，真实 final=[19.039,−31.797]°，actual=[18.493,−31.785]°；策略保留了明显更负的膝残差，真实构型不同于 source endpoint。但本窗口不证明单个关节是停滞的唯一原因。

这些数据排除了“当前没有其他支撑／完全无离地净空所以必须禁所有前送”的解释；**不证明任意前送必然安全或能捕获**。FL 尚缺水平越沿，不能只做垂直落脚就授予 placed。

全部窗口 fl_capture_pending=false、allow_capture_continuation=false、assist=WAIT/correction=0。生产 `_capture_continuation_status` 的 pending 要求历史 FL cross，legal_path 又要求 within_top_xy；assist WAIT 也要求 crossed_FL + within_top_xy。当前 cross=false、withinXY=false，故这两个既有 post-cross 机制没有处理这个 pre-edge 停滞。实际 owner 诊断同时表明 P06 layer 不存在、final-stop owner 未接管、policy_residual_restricted=false；**不是 P13 stop 或 assist 二次写入覆盖四轮**。局部 P05 warning 本身不是产生前送动作的机制。

本次仅定位事实：有限源 stop 后，当前策略的轮腿组合没有补足越沿进展，pre-edge 不在既有 capture continuation 的适用范围内。未提出／实施新 nominal、approach controller、reward 或安全放宽。

[JSON 逐拍选取证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_p05_hip_only_continuation_v1/CP216448_P05_preedge_stall_live_readonly.json)；固定 capture 1202 行 SHA256：`c75f2ce35096e193316c43bf2f4ea865a56b537b041def4369dbab49518b167f`。

