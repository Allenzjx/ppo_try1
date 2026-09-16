# 固定 τ=.5 自然 P01 2048：封存实数

Run `20260915T0633266376940Z_g4a58c0190ef7_d19a81607e6044a281a010de29469635` 已自然封存 SUCCEEDED（训练进程完成，不是越障成功）。HEAD4a58不变，CP172544/1313/26260 → **CP174592/1329/26580**，实际新增 **2048 decisions / 16 PPO updates / 320 optimizer steps**。

实际 request-phase 样本：P01=10，P02=726，P03=35，P04=59，P05=1218，P06–P13=0；16372实际physics ticks。教师前驱0，不把缺省frozen_fsm参数误称实际teacher执行。5次自然入口，4个已结束episode如下；无P02 terminal，无完整成功。

| Episode | 决策数 | 终态 / tick | FR I / Q / C / P事件tick |
|---|---:|---|---|
| 0 | 247 | P05 FALL / 1975 | 13 / 19 / 1202 / 1254 |
| 1 | 608 | P05 BLOCKED / 4864 | 10 / 18 / 1006 / 1046 |
| 2 | 619 | P05 BLOCKED / 4948 | 24 / 31 / 1195 / 1234 |
| 3 | 201 | P05 FALL / 1601 | 11 / 17 / 997 / 1048 |

四个episode都曾取得FR I/Q/C/P；这不是终态接触持续有效的证明，更不是完整越障。第五episode的373决策尾段在P05/tick2984，非terminal、bootstrap=true，不称结束episode。已结束四次terminal均bootstrap=false。

16次更新均actor changed且有限非零梯度。**实际Adam LR范围1e-5..3.3750000000000014e-5，最终1e-5**，不能写成始终1e-5；配置默认LR3e-5也不等于全程实际LR。原网络、Adam与RNG从CP172544正式恢复，normalizer hash保持不变；Adam/RNG在训练后合法改变。当前同HEAD同版本普通续训，无新迁移因子，合法P01 reset与fresh storage，不继承旧物理状态。

最新不可变checkpoint：`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000174592.pt`，同名manifest记录SHA `1da3e970d4be999efe5d26a9bb09f52126ea85bbafdab0ad59b9bb6dcecf1881`，save_load_round_trip=true。这是读取已有sidecar/load证明，未额外加载模型、重哈希checkpoint或扫描新C。根代理已启动自然P01正式C174592；其结果不在本训练receipt中预填。

完整计数、保存链、逐update LR与原始路径见 `training_P01_fixed2048_after_wheel_audit_actual.json`。本次先复用既有thin helper，再只读4条已封存completed_episodes补充FR事件；未改生产或干扰活动Isaac。
