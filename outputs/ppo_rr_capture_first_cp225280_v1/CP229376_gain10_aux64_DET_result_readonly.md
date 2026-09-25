# CP229376 gain10 / AUX64 确定性结果

封存91.166667s /10940tick /1368帧，error=null；local/full均false，INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED。这是有限RR AUX64训练谱系的DET，后腿helpers OFF、原FL assist ON；本次评估不增加PPO或AUX。

前缀直接引用 `CP229376_DET_RR_activation_live.json`：998拍raw/FINAL对CP225280最大差0；FRplaced2761、FLplaced5165、RRqualified6995/cross7979完全一致，不再重扫历史参考。

## RR实际变化（370个active决策端点）

gate FINAL=[+7.485144,−58.000000]°；首gate可见7984 actual=[+7.223447,−57.760433]°。

| RR | FINAL范围° | actual范围° | actual相对7984范围° | actual末尾相对7984° |
|---|---|---|---|---|
| hip | [+2.490, +7.026] | [+2.233, +7.001] | [-4.990, -0.222] | -4.368377 |
| knee | [-58.000, -54.338] | [-57.760, -54.104] | [+0.000, +3.656] | +3.376893 |

| tick / 秒 | FINAL / actual RR° | gap mm | conditional raw / applied local raw mean |
|---|---|---|---|
| 7992 / 66.600000 | [+7.026, -58.000] / [+7.001, -57.760] | 58.749197 | [+0.745, -0.594] / [-0.327, +0.142] |
| 8224 / 68.533333 | [+2.672, -54.605] / [+2.440, -54.390] | 47.140810 | [+0.486, -0.462] / [-0.313, +0.170] |
| 10940 / 91.166667 | [+3.119, -54.635] / [+2.855, -54.384] | 51.159650 | [+0.510, -0.463] / [-0.277, +0.146] |

最小gap 47.140810mm（tick8224），末gap 51.159650mm；FINAL knee仍等于−58仅1/370。实际关节确有改变，但仍未取得真实台面接触。

完整native布尔扫描10940行、tick1…10940连续：RR TOP=true 0，字段缺失0；gate起2962行TOP=0、GROUND=0。rear_task_assist_disabled为true 10940行，RR assist active=0。所以无TOP并非只从15Hz端点推测；此扫描未计算120Hz关节/间隙极值。

active source RR=[−6.9,−37.8]、mappedN=[−8.15,−39.05]恒定，generic RR bias=0，actual tracking列表空，P09 late未消费。

FL末尾 nominal=+0.000000、FINAL=-1.012844、实测canonical=-1.024421rad/s；370拍中FINAL负速370拍、实测负速370拍。源864stop首次端点8200，composite N四轮全0首次8896；两者不可混为同一停止。

相对旧CP228864：最小gap 55.230760→47.140810mm；末gap 57.408820→51.159650mm。间隙数值改善，但两者均无TOP。这是两个已评估checkpoint的观测差异，不证明gain10、某次更新或单一关节是唯一原因。

第一未完成任务仍是RR真实顶部捕获和同次有效承载保持0.5s。没有完成RR捕获，更不能称RL或整段成功。视频由主线程交付；本报告未导出视频、未改生产、未执行训练。
