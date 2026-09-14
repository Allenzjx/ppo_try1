# P10 block8 episode1：真实 RR 膝硬限中止

结论：实际越限关节为 **rear_right_knee（RR knee）**，不是 FR knee 或 RL knee。终态实测−60.0258155425°低于−60°硬限，最终目标−41.5848034400°；不是命令舍入误差。RL 本回合没有新 I/Q/C/P。保留原 HARD_JOINT_LIMIT 结果，不称为后缀成功。

## 固定证据与计数

Run `runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541`，生产69aeaca。只解析 audit 第457–512行（episode1），global145865–145920；固定 bytes `45360335–50619998`（含端点，5259664 bytes），SHA256 `dc86a133703660a2f23e8757fe8d36fbf12aaabd3f1ce1a4342ef998d1795e30`。另核对 completed_episodes 第二行，episode_index=1/seed1001/56decisions/task_success=false。没有读取第三教师前缀或后续策略。

P10=1/P11=1/P12=54，真实策略物理tick446，native effect446/446 verified，无 forbidden state writes；own-phase request444，差额2为普通交接hold。教师实到7584/63.2s后开始策略，终态8030/66.916666667s，策略物理后缀仅3.716666667s；不能将含教师的66.9167s当策略执行时长。

global145920是采样计数，不是已保存checkpoint的证明。按此次主控快照仍只有145792/1104/22080已优化保存；本回合56与上回合末72合为待更新的128。此报告不读取或预支后续reset完成后的第4次更新。

## 实测与目标分开核对

实际值来自同tick `transfer_roles.FL.receiver_workspace_state.joint_range_margin_deg.rear_right_knee`：FL角色的对角接收侧为RR，position=negative_margin−60。终态 negative_margin=−0.0258155425269493°。对照同tick终态观测 `actual_joint_position_deg` offset38、RR knee index7、scale90，policy/critic均372维且完全相同，finite_fallback=false；float32回解−60.0258153677°、速度−36.2718585134°/s，与margin反算吻合。这不是将命令当成实测，float32误差远小于实际越界量。

| audit row / tick | RR knee 实测° | RR knee final target° |
|---|---:|---:|
| 507 / 7992 | −52.450036 | −46.800457 |
| 508 / 8000 | −54.843112 | −47.373830 |
| 509 / 8008 | −57.243538 | −46.764319 |
| 510 / 8016 | −58.335958 | −44.338448 |
| 511 / 8024 | −58.787618 | −44.584803 |
| 512 / 8030 | **−60.025816** | −41.584803 |

末端 RR knee nominal−27.2°、projected residual−14.384803°，最终目标−41.584803°；实际仍向负方向运动并越限。FR knee末端headroom裁剪index3，target−57.99999999999999°，但FR knee实测−57.0063548894°仍在硬限内；不能将“被命令裁剪的关节”误当实际安全中止关节。其余七个实测关节均未越界。

生产 `sensing/guard_state.py:365` 对 live logical joint.position_deg逐关节比较闭区间，`semantic_backend.py:290`保留该实测安全信号。这次独立实测证据支持原判定，未发现错误下发或舍入导致假硬限。受载、耦合与跟踪动态均可能影响实际响应，仅凭该片段不能确定唯一物理成因；不因此放宽硬限。

## RL 任务与终态

RL的仅有I记录为教师tick7/1722，本episode策略credit-start之后没有新I，也没有Q/C/P或撤销事件。RR Q6912/C7109/P7579同样是教师历史，不计策略成绩。

末端 RL GROUND + FRONT_WALL，front−49.683mm、台面净空−49.993mm、active_attempt=false；physical_evidence_status=CONTACT_BEARING_UNVERIFIED，load_fraction_valid=false，不能将数值load比例当已验证承载。接触载荷未知不会否定独立实测关节越限。

terminal source/reason均HARD_JOINT_LIMIT；gravity_z float32回解−.990079，非翻转证据。body speed .060653m/s、angular speed .390640rad/s；没有用这些速度推造不存在的成功判据。首个未完成任务仍为P12 RL有效抬起、越沿、受控放置，安全中止独立成立。terminal bootstrap=false，reward breakdown total−43.3859528334。

本次仅新增报告，无生产、配置、checkpoint、主manifest或进程写操作，未启动Isaac或测试。
