# #38 首个自然 P01 随机回合：FL 已放置，P06 FALL

固定范围：run `20260907T0800095559611Z_ge8462f066908_1e4552a1742445da9fee08d329f8b8fb`，episode0，g119041–119233；仅读首193条 action audit 与首个 completed episode。该回合已包含于已发布 g119296 完整更新边界，不计后续回合。无 teacher；HEAD e8462f、history Gaussian 随机采样，不是固定均值评估。

## 结果与真实事件

193决策、1541物理tick、12.841667s；末决策只有5tick（差3tick）。正式结果 **P06 FALL**，physical evaluator valid=true / SAFETY_ABORT，task_success=false，return −41.476261。阶段样本 P01/P02/P03/P04/P05/P06 = **1/134/4/1/48/5**；P07–P13未访问，不评价其质量。

FL 硬 **Q1180 / C1481 / P1497** 成立。t1496 为首次连续TOP样本1：front +4.402mm、clearance +2.915mm、真实TOP/support、归一化load .181855。t1504仍TOP，front +6.958mm、gap +2.917mm、load .036796，history已记P1497，P05→P06。这里没有独立逐tick原始力/接触点流；不把t1496或1504的载荷冒充t1497精确力值。正式事件和两侧当前物理证据支持capture，并非仅阶段标签。

FR Q37/C1083/P1109。RR仅initial1279、1396；RL仅initial7，两后腿均无硬Q/C/P。五次阶段转移均非terminal、允许bootstrap。首未完成任务为P06后腿workspace；终止时RR/RL front −564.956/−601.062mm。FL此时已AIR/load0、front −7.265mm，不能将历史P当持续台面支撑，也不据此断言FALL原因。

## 控制范围与未覆盖窗口

P05 t1120进入、t1504退出，仅 **3.2s**；原P05有限源长9.733333s。至本回合终止，该连续源仍未到endpoint，FL nominal仍[48.2,−36.7]°。因此 **没有“nominal尾段结束后”的实测窗口**；以下只是P05实际48个决策末点，requested为filtered请求，actual为canonical最终目标。

| 通道 | nominal范围 | requested范围 | actual范围 |
|---|---:|---:|---:|
| FL hip ° | 0…48.2 | +6.632…+16.040 | +9.343…+65.490 |
| FL knee ° | −36.7…−26.1 | −3.160…+14.136 | −41.110…−23.814 |
| FL wheel rad/s | −.125….300 | −.236…+.231 | +.064…+.531 |
| FR wheel rad/s | .300 | −.232…+.135 | +.068…+.435 |
| RL wheel rad/s | .300 | −.163…+.167 | +.137…+.467 |
| RR wheel rad/s | .300 | +.035…+.239 | +.335…+.539 |

P05 FL hip raw全部正、knee raw有正负；实际目标相邻决策增/减次数分别31/16与25/22，角度符号不等于运动方向。实测控制前一tick FL q范围hip11.420…64.567°、knee−39.016…−24.516°，不等同目标。48末点headroom均未裁减，requested=effective。

四轮P05 canonical实际命令全正；native float32因左轴反号，FL/RL全负，FR/RR全正。随后5个P06末点FL轮实际−.108…+.207（3正2负），FR+.243…+.663；FL hip65.991…71.426°、knee−43.721…−37.729°。这是已观测控制，不是尾段后保持能力或因果模型。

1541/1541 native tick记录verified，四类状态写入均0。范围统计主要是决策末点，未重演全部GPU缓冲区或中间目标。该随机回合证明当前路径曾取得FL真实capture；不能据此推出固定均值会成功、名义断口是唯一原因，或全任务已有改善。
