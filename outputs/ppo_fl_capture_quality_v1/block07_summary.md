# block07：768 个真实更新样本，3 次 FL 捕获及后续结果

Run `20260918T0650463688089Z_g3a50657a96c9_630e9161d7094af18eb29dbbf6df35b9` 已自然封存，lifecycle=SUCCEEDED 指**训练预算与保存流程完成**，不是完整越障成功。现有 `training_receipt.py` 已生成 `block07_768_receipt.json`。

## 真实优化计数与覆盖

block07 连续 global181249–182016，共 **768 decisions、6 PPO updates（1382–1387）、120 optimizer steps**。六次更新均记录有限非零梯度、LR=1e−5。9 次普通阶段变化均不置terminal；只有2次BODY_COLLISION安全/任务终止。prefix不计入学习覆盖。

| 范围 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10–P13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block07 | 0 | 0 | 0 | 0 | 175 | 518 | 2 | 2 | 71 | 各0 |
| block01–07累计 | 6 | 488 | 9 | 21 | 1557 | 1045 | 10 | 17 | 431 | 各0 |

七块为512+896+384+128+128+768+768=**3584 decisions / 28 updates / 560 optimizer steps**，从178432/1359/27180到182016/1387/27740，区间连续。block01–05的2048属于旧quarter kernel；block06–07的1536/12/240才属于显式迁移后的REQUEST-history kernel，不能把前五块追认为新kernel训练。累计10个真实终止episode：BODY_COLLISION=5、FALL=5；无完整任务成功。P01/P02非零质量成本的证据来自前块；block07没有这些阶段的学习样本，不以prefix前段替代。

## 三次 FL 捕获不是三个完整成功

三次前缀均由块内冻结CP181248确定性策略实际生成，P05/offset150、354个prefix决策、tick2832/23.6s入口；accepted=true、miss=null，prefix credit=false。三次都从未placed的入口继续真实随机PPO，不是N前缀、fallback或已触地快照。

| suffix episode | FL placed tick | P05→P06 global / tick | 交接拍 FL bearing | 后续已观察结果 |
| --- | ---: | --- | ---: | --- |
| 1 | 3010 | 181271 / 3016 | 2.709520N | global181549 / tick5239 / 43.658333s，P09 BODY_COLLISION；RR qualified lift5106，无crossed/placed。终止时FL仍TOP/support，0.358727N。 |
| 2 | 3236 | 181600 / 3240 | 4.856021N | global181884 / tick5509 / 45.908333s，P09 BODY_COLLISION；RR qualified lift5261，无crossed/placed。终止时FL AIR/support=false，gap143.391mm、0N。 |
| 3 | 3634 | 181985 / 3640 | 5.289245N | 预算末global182016 / tick3888 / 32.4s仍P06、无terminal；FL已再次AIR/support=false，gap24.014mm、0N，RR尚无qualified lift/crossed/placed。 |

上述三次交接拍FL均TOP且support=true，gap分别−0.663354/−0.002445/−0.203973mm；捕获真实存在。但历史placed不等于后续一直承载，特别是episode2与3已出现FL离开顶面。第三段没有自然任务终止，不称成功、失败终局或可跨run继续的已保存物理状态（manifest physical_env_state_saved=false）。这里只确认真实训练事件与后果，不能证明最新固定CP自然P01确定性越障能力，更不能将改善归因于单一kernel因素。

## 最新封存 checkpoint

`outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000182016.pt`

实际文件SHA256与manifest一致：`6de7e8c192251303fdec6b431493188b3b7ca048586ca97aedc10865dd139799`。manifest累计182016 decisions / 1387 updates / 27740 optimizer steps，FL分支累计3584/28/560，save_load_round_trip=true，Identity normalizer、LR1e−5；未额外actor前向或优化。

来源：block01–07各自封存receipt、block07封存prefix_evidence与residual_and_projection_audit、CP182016 manifest及文件hash。本次只新增block07 receipt与本摘要，未改DELIVERY/RECOVERY或生产文件，未启动Isaac/CUDA；CPU汇总进程已自然退出。

## 后续前段/P06维护（封存 block08–09）

block08自然P01新增256 decisions / 2 updates / 40 optimizer steps，真实请求P01=2、P02=210、P03=4、P04=1、P05=39。block09采用明确标注的P06/successful_nominal/offset0前缀，新增128/1/20；**实际128个学习样本全部P06**，不能声称本块学习覆盖RR摆动或P07–P13。两块均没有真实terminal；预算完成不等于完整任务成功。

block08按真实120Hz物理子状态统计如下，全部1695个质量样本β和加权成本均非零：

| 实际物理子状态 | 样本数 | β范围/s | 加权质量成本 |
| --- | ---: | ---: | ---: |
| P01 未资格准备 | 15 | .015–.030 | .000067941165 |
| P02 未资格准备 | 387 | .015–.030 | .000712294711 |
| P02 有资格抬升/转移 | 305 | .015 | .001903423392 |
| P02 功能性携带 | 988 | .015 | .011880983828 |

逐物理成本合计 **+.0145646430967**；真实reward中body_stability合计 **−.0145646430967**，逐决策二者负和最大误差5.42e−20。按请求phase聚合的body项为P01 −.000073299842、P02 −.014491343255；请求阶段和该决策内实际物理子状态口径不同，不能逐阶段硬等同。其余阶段没有新body成本。

一次CPU只读核验完整256条已封存记录及rollout1388/1389：256/256实际总reward转换float32后与PPO storage **完全相同**，raw、旧μ/σ/logp亦逐张量相同；各20个真实optimizer minibatch、每样本实际用5次，hook oldlogp与实际使用advantage逐样本对应保存张量、误差均0。第一/第二更新分别有128/84个包含前段质量的学习决策。分项float64总和与记录float32 reward最大差1.31e−8；两个首minibatch的ratio偏离1最大3.81e−6/1.97e−6。两次正式更新已有有限非零梯度、LR1e−5。本检查没有重新运行actor、GAE、optimizer或CUDA。

这证明非零前段质量成本进入了实际总reward与真实学习链路；**未计算该成本项的独立梯度，也未隔离它对总advantage/更新的贡献**。PPO使用含任务进展与后续结果的总回报，并进行advantage标准化，不能由“成本非零”“存在总梯度”推出单项一定产生指定方向更新，更不能据此声称姿态、稳定性或完整越障能力已经改善。

| block01–09累计实际请求 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10–P13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 已优化样本 | 8 | 698 | 13 | 22 | 1596 | 1173 | 10 | 17 | 431 | 各0 |

截至封存block09共 **3968 decisions / 31 updates / 620 optimizer steps**，CP182400累计182400/1390/27800，manifest分支计数一致并记录save_load_round_trip=true。REQUEST-history迁移后实际新增1920/15/300；前五块2048仍属旧kernel。此次未重新哈希全部checkpoint，未读取活动block10，未改生产、DELIVERY或RECOVERY；一次CPU核验已自然退出。
