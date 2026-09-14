# P07 教师入口首回合：事件与控制链核查

结论：本回合是 BODY_COLLISION 终止的教师初始化策略后缀，不是完整 PPO 成功。切换符合当前准备／减载规则；P09 入口不表示 RR 已抬起。没有发现明确断指令、切换清零、单位错或非法状态写入，亦不能据此排除全部软件问题或把失败唯一归因于策略。

## 固定证据范围

生产版本 `06716a88bcc9`；运行 `runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0647306226372Z_g06716a88bcc9_c6cfcadd350b418a9969f7dfc35406a5`。
仅读 residual_and_projection_audit.jsonl 第1–12行（g141569–141580）、completed_episodes.jsonl 首行、prefix_evidence.jsonl 至首次 credit-start 第742行，及相关判据。没有读取后续策略回合、启动仿真或修改生产。

seed1001/episode0；12策略决策、93物理tick、后缀0.775s；P07=1/P08=1/P09=10。前缀5912tick/49.266667s不计策略数据；总时长50.041667s。此处12条不代表已更新／保存。

## 入口与普通交接

| 实际 tick / 时间 | 事件及当时依据 |
|---|---|
| 5912 / 49.266667s | 教师到 P07；P06 rear_approach=1，FR/FL 已有 Q/C/P。四腿当前支撑，FL/RR载荷0.16623/0.19747。无BODY接触，观测有限。 |
| 5920 / 49.333333s | P07→P08：RR/RL 接近前缘、role_prepared_RR=1；FR/RL支撑，FL已AIR、不承载。 |
| 5928 / 49.400000s | P08→P09：接近前缘、transfer_ready_RR=1；FR/RL支撑、RR载荷已验证为零，FL仍AIR。 |
| 6005 / 50.041667s | P09 中 BODY_CONTACT 终止；未进入 P10。 |

两次切换均 continuous_takeover=false、terminal=false；不是 RR 抬升快捷通过，也未切断 episode。历史源 P07 tick6648 与这里 semantic 入口不同，时间差本身不证明故障。

入口0.5s窗口 CoM 向FL位移+6.117mm、当前投影速度+13.014mm/s、RR减载0.02187、其他支撑短窗连续率1。5920/5928对应位移+3.576/+0.720mm，当前速度已为−51.144/−5.649mm/s；通过的是独立减载路线（0.23350/0.20103），不是“CoM一直向FL”或“FL已承载”。全身实际运动与FR/RL当前支撑也有记录。

semantic_transfer_roles.py:185 使用当前方向／减载／真实抬升之一，加实际响应与当前至少两个其他支撑；减载尺度0.02、卸载上限0.20。28.7s response-duration 不单独令 transfer 满分：5952失去当前方向／减载证据后指标与计时清零；6000短暂满足速度路线，6005又未ready。不是永久资格，也不是动态稳定证明。

## RR 实际进展与失败

12个决策末端RR初抬I／成立Q／当前有效／越沿C／放置P全false；所存事件无RR I/Q/C/P。最初AIR原地面增高仅1.786mm，5928为0.024mm；5944–6000末端GROUND，终点AIR增高0.835mm。没有EDGE/TOP进展；无接触不等于可控抬起。

入口→终点：RR前缘距离−209.619→−322.117mm，退远112.499mm；body净后退27.253mm。FL最终台面净空207.239mm但载荷零，不能证明全身转移成功。第一个未完成任务为P09有效RR抬升、继而越沿／放置。

终点 evaluator 为 VALID/VERIFIED、TASK_FAILURE_BODY_COLLISION、BODY_CONTACT，不是超时或录像问题；motion_allowed/body_control均false，原因为physical_safety_abort。role残存0.879减载诊断不覆盖安全终止。reward含−40终止事件、下一潜势零、无bootstrap。

BODY判据仅接受精确base_link/Obstacle pair加持续或实时穿透佐证（body_collision_detector.py:49）。缺终点原始force/point/persistence全量流，不能独立量化冲量／具体撞点。已存BODY包围盒最低z38.454mm低于50mm台面，但包围盒不能替代接触证据。

## 指令、连续性和真实响应

93tick均native-effect verified；12末拍mask全开、mapping/setter一致、previous-ACK独立核对通过；headroom无裁剪。末拍final slew额外约束FL hip三次、RR hip一次，不是整体屏蔽。

P07→08→09 residual原值继承、wheel交接零跳变；新owner请求最大关节跳变7.4°/4.2°，最终仍走成熟mapper1.25°/tick。两次首tick继承旧residual，故own-new-phase-effect各7/8，而实际effect仍8/8。bootstrap180保持、ACK连续；无mapper重启/filter清空证据。episode/command tick相差179，有同拍记录，不可混接。

实测入口→终点：FL hip21.770→53.282°、knee−12.108→18.560°；FR knee45.572→68.581°；RL hip6.538→21.655°；RR hip−1.172→52.791°。有全身响应，却未取得合格RR抬升。终点RR hip nominal/residual/target/实际为55.6/−1.374/55.476/52.791°，不能把target当实测。

FR wheel nominal后段由+0.3变−0.63，是源owner请求；终点FL/FR/RL/RR目标+0.087/−0.767/−0.112/+0.618，实测+0.087/−0.702/−0.178/+0.617rad/s，非清零。12末端实测关节在原硬限内；缺全量中间关节存档，不能称逐tick离线重验。

保留失败供固定版本PPO更新，不重设zero5/5门禁或热改参数／范围。无依据把正常owner跃迁判作软件缺陷、恢复双重pre-slew；也不能凭“有下发、有运动”声称已学会。
