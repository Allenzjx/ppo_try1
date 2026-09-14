# Block13 episode1：P09 FALL，连续抬升未完成越沿

固定核查2026-09-10 15:11:29 UTC；run `20260910T1451368221058Z_g7db0d17f398d_19bb6607a9384f08b00d341ba21060b7`。只读episode1：global150338–150629，共292决策/2335ticks/19.458333s；最后动作7ticks，只有第292条为terminal。未重做episode0或读取episode2。

## FALL 的实际分支

terminal_observation policy/critic均为372维、逐值相同且全部finite，`terminal_observation_finite_fallback=false`，相关观测未触及schema clip。按实际schema回解gravity_z=−0.937394142；机身高度由锁定bottom0−relative_bottom得 **0.014778882265m**，由top0.05−relative_top交叉得0.014778881520m，误差7.45e−10m。scene_factory实际SHA与该run runtime contract一致。

生产判据是`base_z<0.015 or gravity_z>−0.30`（[isaac_fsm_backend.py:6520](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/isaac_fsm_backend.py:6520)）。本例低于高度线约0.221118mm，**触发低机身高度分支，不是gravity姿态分支**。这是已保存float32特征回解，不冒充原始双精度物理流，也不笼统声称机器人已翻转。body-frame线速度[+0.051791,−0.010382,−0.094162]m/s，仅作终态测量。

监督器为`SAFETY_ABORT/PHYSICAL_SAFETY`，具体环境原因FALL；它是独立安全中止，不重分类为BODY碰撞或wheel-only任务违规。timeouts=false、bootstrap_allowed=false。任务未成功。

## FL 放置与 RR 连续性

FL I/Q=tick948/7.9s，C=1105/9.208333s，P=1820/15.166667s；P05→P06端点1824。FALL末端FL仍TOP/support=true，实际承载1.125431N，不是只靠历史假定它在承载。

RR有两次不同尝试，必须区分：早期P05 I925/Q931于tick984真实地面接触撤销；当前尝试在 **P06 I2017/16.808333s、Q2040/17s**，向上位移9.038195mm。历史`event_ticks.active_lift.RR=931`是保留的早期首次记录，不能误当作当前尝试的Q时间。

当前RR在P06→P07@2040、P07→P08@2048、P08→P09@2056均AIR/current_lift_valid=true，三个普通交接均非terminal。桥接审计的previous residual与carried residual精确相同、无禁用通道丢弃/phase-scale裁剪；residual跳变仅浮点舍入（servo≤1.42e−14deg），wheel交接跳变0。实际nominal/applied servo最大跳变分别10.6/7.4/8.5deg，因此不宣称全部动作零跳变，但没有证据是residual清零或重新抬腿造成。

RR current-valid在决策端点2040后持续true，直到终态2335才因`physical_safety_abort`变false；AIR仍true，I/Q建立历史仍true，C/P均false。末front−282.454368mm、顶面gap+52.410751mm（地面相对抬升102.641239mm），连续AIR327samples/2.725s。足端离地很高并不等于已越前缘；本回合在P09安全中止，后续RR越沿/放置尚未发生，不能称完整成功，也不因此把C/P额外设为P09门槛。

## 实际样本与更新

292条分阶段：P01=2/P02=79/P03=7/P04=9/P05=131/P06=27/P07=1/P08=1/P09=35，P10–P13=0。native证据连续1–2335，2335条全verified且有实际target effect；own-phase2327，八个交接tick单独区分。

限定读取的第六个optimizer记录确为update1142/global150656（20 optimizer steps）；episode1末global150629已被该完成边界覆盖，故292条都已优化。这不是checkpoint重加载/完整评估证明，本报告未读取或修改official pointer/CP。未启动仿真、加载Torch、创建新分析工具或更改生产/主报告；没有据此新增训练门禁。

[固定JSON证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/natural_p01_block13_episode1_p09_fall.json)
