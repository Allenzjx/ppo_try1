# Run10：FL 放置→后腿准备的两个固定窗口

只读范围：已完成 `block10_summary/training_block_summary.json`，以及 Run10 `20260908T0358211077438Z_gdb991aa68103_df532577a9954407b67295896fa2af9f` 的 **12 条**决策：g133302–133306、133353；g133549–133553、133640。前者 ep0 最后停在 P06，**没有进入 P09**；后者 ep2 贡献本块全部 55 条 P09。全块 6 次 FL 首次放置属于 PPO 信用，不能视为全任务成功。未读 Run11、PT 或完整120Hz传感流。

## 已测交接与后果

| 固定回合 | FL 首次放置；P05→P06 | 交接末 FL 当前状态（前缘/净空 mm，载荷份额） | 随后与终止 |
|---|---|---|---|
| ep0：P04/05/06=5/51/49 | tick2139；g133304/t2144（17.8667s） | TOP、support=true；+5.864/+0.00260，0.140944 | g133305 仍TOP；g133306 已AIR、load=0。105决策后 g133353/t2532，21.1s，P06/FALL；RR只有initial，无硬Q/C/P |
| ep2：P04/05/06/07/08/09=6/46/32/1/1/55 | tick2108；g133551/t2112（17.6s） | TOP、support=true；+57.189/+0.983，0.148305 | g133552 已AIR、load=0。RR Q2365→GROUND撤销2615，再Q2688→撤销2744，无C/P；141决策后 g133640/t2824，23.5333s，P09/FALL |

两次进入P06时 RR/RL 前缘分别为 **−629.396/−611.641mm**、**−586.918/−612.396mm**，rear_approach=0。ep0终止 FL仍AIR、不承载，实际支持FR/RL；body线/角速率0.639565m/s、2.032955rad/s。ep2终止FL已重新TOP承载0.295488，四腿均有当前支持，但RR/RL仍在−186.123/−421.511mm，RR在GROUND、净空−53.065mm；body速率0.323335m/s、0.761120rad/s。**有接触不等于未跌倒或静态稳定。** 本窗口无精确跌倒姿态/原始力矢量，不能反推FALL传感错误。

## 控制依赖：没有发现截断或重复叠加

g133305、133552 的交接记录中，`previous_projected_residual_full12` 与前行相同，`carried_projected_residual_full12` 逐值保留；禁用通道丢弃、相位范围裁剪均为0。首tick hold的投影动作跳变仅浮点误差（最大servo≤1.78e−15°），随后 **7/8 tick** 有本次phase request的native effect，不是整决策丢弃新动作。非终止bootstrap=true；历史FL placed未清。两窗口ACK/mapper计数各决策连续+8，无state write；终止分别4/8tick且bootstrap=false。

各±2窗口 nominal12完全相同：servo `[48.2,−36.7,0,45.9,17.5,0,0,0]`°，四wheel均+0.3rad/s；没有阶段新加一遍nominal。源码 `semantic_supervisor.py:1001–1054` 为各层拥有通道的**赋值**、后层优先，并非delta累加。但投影首tick近零跳变**不证明8tick实际运动平滑**：ep2 RR hip实际canonical drive由1.366856→7.366856°。两首P06决策四轮残差分别为 `[-.050020,.471547,-.205714,.248580]`、`[-.261601,.179229,.092832,.314883]`；真实float32 native wheel目标分别为 `[-.249980,.771547,-.094286,.548580]`、`[-.038399,.479229,-.392832,.614883]`rad/s（native左侧符号与canonical不同）。两交接后残差的headroom effective均等于request，未见此处权限截断。

## 支撑、奖励与结论边界

ep2 g133552 的RR角色虽 `transfer_ready=true/motion_fraction=1`，却明确记录 receiver FL AIR、`receiver_current_support=false`，支持仅FR/RL；0.5s窗口CoM沿接收方向位移+38.740mm、连续响应2.1167s。`semantic_transfer_roles.py:146–184,203`允许实测转移/减载与其他动态支撑，不把“ready”定义为FL必须承载，也不证明两接触稳定。故这是**当前支撑已丢失但角色运动证据仍成立**，不是伪造FL支持。

交接两行task family为+0.236422/+0.265906，总reward+0.233118/+0.262089；ep2下一行task−0.021944、总−0.025795，再下一行+0.051631/+0.048296。Phi前后连续衔接，代码仅全局 `5*(.9985*Phi_next−Phi_before)`，阶段本身无重复bonus；FL放置会解锁前驱门控后的RR进度，但不代表RR此刻已在workspace。合法FL AIR不自动扣capture信用，运动证据也降低body成本：这是既定协同取舍，**现有数值不足以证明奖励依赖冲突或其导致跌倒**。两个真实终止均Phi_next=0并含−40，total分别−42.365418/−42.518725。

**结论：本次没有证实需要修复的交接截断、重复nominal、伪支撑或PBRS接线缺陷；证实了放置之后仍会丢失承载/后腿资格并失败。** 未以失败相关性选噪声、范围或奖励修改；有限决策端点不能辨识唯一动力学原因。首次提取曾因PowerShell序列化前置Warning失败，经授权仅重读同12行完成；这不是训练异常。
