# P07 真实后缀：首两次更新与首次 RR TOP

只读范围：`train/20260923T1918580156414Z_gfa4b98ed506e_73c2c47d6e63471ebbcd876e45516dd5`。源码 `fa4b98ed506e`；未改生产、未另启 Isaac/CUDA、未载模型。JSON 原文件不改。

## 结论

首次无后腿任务辅助的 RR **真实短时 TOP/placed** 已发生：decision **221466**，70.333333 s，episode tick **8440**，TOP/support=true，实测承载 **0.29630 N**，gap **−0.1002 mm**，TOP 连续样本2。随后立即掉载；不是已经形成可持续支撑，更不是确定性或自然 P01 成功。

前256决策为 **221057–221312**，完整完成 PPO1693/1694（+256决策、+2更新、+40 optimizer steps）；P07=1、P08=1、P09=254。补查首次 TOP 窗口221441–221568时，PPO1696/CP221568也已写出，因此该窗口已归入完成更新。没有把更晚 live 样本算入本报告。

本次是43 s真实成功 nominal 前缀后开始的 **stochastic PPO suffix**；teacher prefix不在PPO storage。不是当前policy自然P01。

## 为什么 P07/P08 各只有一次采样

43.0667/43.1333 s的两次过渡都有 FR/FL历史placed、真实物理有效、准备/转移任务值1；当时RR仍GROUND，RR lift/cross/placed历史全部false。因此不能称“前缀已完成RR抬升”。固定FL接收方向短窗CoM进展约6.54/7.07 mm，FR+RL桥接接触与动作响应满足当前准备定义；FL当时AIR，不被算作实测支撑。

这是调度目标已满足后的连续接管，不代表P07/P08源动作已执行完：P08/P09源仍pending，P07层继续。P09 source到45.0 s/tick5400、实测RR qualified AIR后才开始；47.3333 s首次决策采到RR已越沿。两次普通phase转换均 `terminal=false`。因此短phase采样不能误判为虚假抬升，但任务分桶数量也不能代替前驱准备的真实覆盖量。

## RR/FL实际控制权与第一处分歧

256条均rear assists=false、旧RR14 WAIT零、RR通道无capture owner、无RR wheel投影证据、controller drive bias全0；12维mask全1，原子实际派发核验通过。每拍一次Gaussian、无额外forward/draw，原raw Gaussian logp用stdlib独立复算误差见JSON。

P09 late从 **50.4 s / decision221167** 起停在source tick648，原因 `current_RR_bearing_before_FL_RL_transfer`。等待146条的源RR目标固定hip−6.9/knee−37.8°（mapped −8.15/−39.05°），没有发旧FL/RL late组；身体并未freeze，policy继续产生真实动作：

| 等待窗口量 | 最小…最大 |
| --- | --- |
| RR hip raw / conditional mean | −7.459…6.899 / −6.589…6.284 |
| RR hip effective σ | 0.0828…2.2624（状态门会变化） |
| RR hip residual / final / actual | −24…24° / −32.15…15.85° / −32.23…15.43° |
| RR knee residual after headroom / final | −18.95…−9.478° / −58…−48.528° |
| RR gap；TOP样本 | 11.18…51.93 mm；0 |
| FL wheel nominal / final / measured | 0或+0.3 / −1.200…−0.0134 / −1.327…+0.00321 rad/s |

RR hip探索已经实际到负方向，不能再归因为“σ极小、负目标从未执行”。但raw经HISTORY/filter/slew，负raw不等于同拍负最终目标，例如50.4 s raw−2.106而有效请求仍+6.656°；必须使用同拍mapped N与有效请求。

FL反转也不是nominal遗漏或native符号假象：47.3333 s源+0.3、policy有效修正−0.76987、最终−0.46987、实测−0.34225 rad/s；60.0667 s源0，最终−1.01271、实测−1.03507，FL TOP承载2.82746 N。源stop正确，policy仍负向；已有零nominal后的反转可追至43.4667 s，但当时FL AIR，不能算牵引。

暂停nominal late不等于policy完全没有早期全身偏移。等待期间FRknee最终−39.14…−15.00°而源+31.1°；FLknee−48.00…−32.53°而源−13.4°；RLhip11.60…23.57°而源28.2°。这是持续policy作用，不能伪称late source偷跑，也不足以单独证明已经进入RL卸载。

## 首次接触 → late释放 → 掉载

| decision / time / episode tick | RR传感器结果 | RR hip final/actual；knee final/actual（°） | 源接续与FL wheel |
| --- | --- | --- | --- |
| 221465 /70.2667 /8432 | AIR，gap+3.722 mm，0 N | 3.353/8.546；−56.061/−55.757 | late仍648；FL源0，final−0.9445 |
| 221466 /70.3333 /8440 | TOP，gap−0.100 mm，0.2963 N；placed事件8440 | 7.353/6.887；−55.620/−55.594 | late start诊断8439，源650；FL源−1.07，final−1.8945 |
| 221467 /70.4000 /8448 | AIR，gap+0.366 mm，0 N；旧placed仍true | 10.853/7.611；−48.043/−53.564 | late继续658；P10 knee源−29.3，FL仍−1.07 |
| 221468 /70.4667 /8456 | AIR，gap+3.990 mm，0 N | 11.345/9.268；−44.582/−49.571 | late666；P10 knee源−27.2 |
| 221568 /77.1333 /9256 | P11；RR AIR，gap38.778 mm，0 N | 7.294/7.994；−42.572/−42.483 | late1466；FL源0，final−0.0147 |

实际接触释放了late组，随后P10的RRknee展开与FL/RL组一起发生，RR接触未保持。读取的源码明确late当前承载许可仅作用于首次source648准入；启动后并不逐拍重锁该组。不能把这个旧placed继续视作当前支撑。接触拍采样仍carry、RRhip σ1.3857；下一拍状态已RLprep，σ0.2510（局部carry×4退出），尽管拍末RR又AIR。此为下一处实际接续障碍，而非本报告已经证明某个单关节是唯一原因。

FL **源**−1.07反向脉冲是在新late许可后出现；此前持续反转主要来自policy。RL在所查窗口仍GROUND，77.1333 s bearing12.17585 N，历史无RL合格lift/cross/placed。不能报告RL已卸载成功。

数据限度：TOP/力数据是决策末传感器，8个物理步日志另证实派发；没有完整120 Hz力日志，所以不虚构8438/8439逐拍力。RR actual取最后派发前真实读回，mapper ACK时钟与episode时钟有bootstrap偏置（例8618 vs8439），不是精确同刻的派发后响应。RR髋负向变化、全身姿态及下一轮raw同时变化，尚不能把触地因果全部归给hip。

复核文件：`P07_suffix_first256_readonly.json`、`P07_suffix_first_RR_TOP_readonly.json`、只读stdlib分析器 `audit_P07_first256_readonly.py`。本报告不提议热改或重置当前rollout。

## 补充：成功 N 并非第一次轻触就立即执行 late

定向复用成功 N_ref `prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source`。只用二分seek读取late前后约0.3 s窗口，并为直接确认TOP序列起点向前补9拍，最终6110–6191共82拍physical/native及10条decision evaluator；没有全文件顺扫、重跑或新评估。旧N与本轮C不是同HEAD配对。

重要补正：**6155是N的越沿＋placed事件，不是第一次传感器TOP。** 6112/50.9333 s的evaluator已TOP/support、bearing4.09746 N、连续TOP=1；6120/6128/6136/6144/6152计数依次9/17/25/33/41。6155后才执行late6156，P10 knee首次新源目标在6161。raw障碍接触还早于6110，但不能将raw OBSTACLE直接称为严格TOP；有限接触history的`consecutive_active_ticks=3`也不能替代evaluator的连续TOP计数。

| 成功N窗口 | RR实测接触/载荷 | FL实际接触/力；RL实际接触/力 | RR knee源/最终/实际，° |
| --- | --- | --- | --- |
| 6112，TOP序列计数1 | TOP，4.097 N，gap−1.205 mm，尚未placed | FL TOP4.512 N；RL ground11.353 N | −37.8/−39.05/−39.257 |
| 6152，TOP计数41 | TOP，6.484 N，gap−0.301 mm，尚未placed | FL TOP5.928 N；RL ground9.591 N | −37.8/−39.05/−39.435 |
| 6155，cross+placed | raw障碍有效、垂向6.215 N，gap−0.269 mm | FL障碍5.413 N；RL ground8.802 N | −37.8/−39.05/−39.447 |
| 6156，late首拍 | raw障碍有效、垂向8.006 N，gap−0.275 mm | FL障碍8.192 N；RL ground9.993 N | −37.8/−39.05/−39.435 |
| 6160，TOP计数49/P10入口 | TOP，9.688 N，gap−0.267 mm | FL TOP10.138 N；RL ground7.857 N | −37.8/−39.05/−39.357 |
| 6161，P10膝首变 | raw障碍有效、垂向13.547 N，gap−0.262 mm | FL障碍11.585 N；RL ground4.577 N | −34.6/−37.8/−39.282 |
| 6168，TOP计数57 | TOP，14.347 N，gap−0.096 mm | FL TOP11.022 N；RL ground2.985 N | −29.3/−31.35/−37.189 |
| 6176，TOP计数65 | TOP，垂向11.112 N，gap−0.016 mm | FL TOP11.846 N；RL ground2.333 N | −27.2/−27.2/−32.972 |

6155–6191全部37拍RR raw障碍接触有效，力模6.215–15.823 N，gap−0.517…+0.005 mm；同期决策evaluator持续TOP。这里的微正gap并不推翻真实接触。力模和垂向承载不混用：例如6176模11.136 N、evaluator承载11.112 N。

相对地，本轮C221466只有RR **0.296 N、连续TOP2**；同拍FL TOP3.030 N、FR TOP13.810 N、RL ground16.001 N（RL另有`OBSTACLE_AMBIGUOUS`表面标记，不能改写成RL TOP）。下一拍221467，**RR与FL同时AIR/0 N**，FR仍TOP14.590 N、RL ground4.147 N；再下一拍FL重获TOP2.322 N，RR仍AIR。N在P10膝展开时已有FL＋RR实际承载，随后RL负载下降；C当时没有持续保持这套支撑。

这给出的物理证据是：一次合法轻触及历史placed不足以证明接续后仍可用；必须区分当前支撑保持与掉载后的recapture。它不证明“必须等固定44拍/一秒”或“必须达到N同角/同力值”，也不构造新的固定相似度门。接触保持失败仍需结合当前动作/载荷趋势处理，不能将接触历史重写或把AIR算作支撑。

补充原始精简证据：`N_ref_first_RR_contact_window_readonly.json`；分析器`read_N_first_RR_contact_window.py`只读JSON/stdout，不载项目运行时。保留N原文件与结果。

## 补充：已完成1697–1699更新中的FR/FL knee探索宽度

只读已完成决策221569–221952，共384条都为P11/P12且公共`rl_prep_transfer=1`；均匀取5态。两膝局部sigma multiplier均为2。**μ是实际条件Gaussian均值，raw是本拍抽样，二者不是同一量**。下表σ单位为raw，不是角度；final/actual为canonical°，actual仍是末次派发前读回。

所有5态固定：FR knee scale **112°**，源N **+31.1°**，mapped N **+33.036758°**，controller bias0，当前残差headroom **[−91.036758,+174.963242]°**。FL knee scale **36°**，源/mapped N **−31.4°**，bias0，残差headroom **[−26.6,+239.4]°**。

| decision /time /phase | FR raw / μ / effectiveσ | FR final /actual | FL raw / μ / effectiveσ | FL final /actual |
| --- | --- | --- | --- | --- |
| 221569 /77.200 /P11 | −.57838 /−.58411 /.03361 | −25.370 /−24.770 | −2.14876 /−1.68866 /.25946 | −58.000 /−58.423 |
| 221665 /83.600 /P12 | −.51293 /−.51331 /.03849 | −19.852 /−19.452 | −.89703 /−.65889 /.23790 | −56.236 /−55.496 |
| 221761 /90.000 /P12 | −.59518 /−.57296 /.04070 | −26.727 /−27.748 | −.87867 /−1.19389 /.30499 | −57.062 /−58.291 |
| 221856 /96.333 /P12 | −.70481 /−.70838 /.05357 | −34.993 /−29.003 | −2.33531 /−2.04839 /.36599 | −58.000 /−58.417 |
| 221952 /102.733 /P12 | −.70208 /−.66934 /.04320 | −34.800 /−33.075 | −1.21735 /−1.42609 /.28367 | −58.000 /−58.481 |

FR绝对 **+30°仅作为用户候选**：用上述同拍mapped N算，所需残差 `30−33.036758=−3.036758°`，静态raw `atanh(−3.036758/112)=−0.02712056`。它在tanh可达范围及真实headroom内；距离5态条件μ分别 **16.57、12.63、13.41、12.72、14.86σ**。因此“FR×2已经足够探索该候选”没有数值支持；当前分布集中在负残差，障碍不是+30°超出执行器范围。现μ±σ映成静态残差，相对中心仅约2.68–3.89°单侧变化。该分析既不要求达到+30°，也不证明该角度适合当前浮动机身。

FL不是同一种“σ极小”：rawσ .238–.366，但均值经常位于负向饱和区。221569/221856/221952的有效负请求分别由−35.034/−35.332/−30.199°裁为**−26.6°**，final均−58°。静态tanh请求若要脱离此负边界，需raw大于`atanh(−26.6/36)≈−.948`；几个不同raw落到相同行为确实存在。另两条样本已有未饱和final−56.236/−57.062°，所以不能说FL完全无正向恢复自由度，也不能只凭再扩大σ认定探索改善。

HISTORY与slew分开：这5态μ均来自当前0.9历史条件分布，而非把sample当均值；例如221569 FR `previous_raw=−.579896`、`base_mu=−.622060`，混成μ−.584112。静态raw反算并未包含后续filter、速率限制或原子目标slew，不能一拍直接把final跳到+30°；现σ距离是**当前一拍条件分布**诊断，不是多步更新永远不可到达的证明。未改reward、sigma、草稿或生产。
