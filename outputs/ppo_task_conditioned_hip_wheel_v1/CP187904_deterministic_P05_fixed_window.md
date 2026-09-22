# CP187904 确定性 P05：固定途中窗口与独立封存终态补记

Source：`runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T0627142354953Z_gee5a9651591d_91d8d06aaced49ebb0ad9d5a5c067b31/source`。

范围严格截止 **tick3600/30.0s**：`video_policy_decisions.jsonl`中endpoint2800–3600的101条（首条包含2792→2800的边界上下文），以及 native/physical t2800–3600各801条。只读已写完窗口，没有读取活动终态、额外forward、随机采样、仿真或改生产；本诊断新增训练量0。

## 直接结论

FL下放相关动作真实存在于执行链，但**不是已形成几度级负向hip捕获**：hip REQUEST在−.061095°到+.041647°之间，后半段回到微小正值；knee REQUEST为−6.677177°到−6.620261°，实际knee确实到约−18.8°。负knee是否单独令轮端下降不能凭符号推断，它与source hip变化、mapper和全身状态共同作用。

当前比旧CP183552的已知悬空末态更接近台面，但本窗口仍无接触、无放置。不能称“下放动作被mask”或“FL已经承载但判定器拒绝”，也不能由途中接近推断最终成功。

## 同拍链路（角度统一为canonical servo °）

R为日志的REQUEST，所有选定native tick逐值等于headroom有效残差；N为mapped/controller后的基线，不能与source nominal混称。final/actual来自同tick的120Hz物理记录，actual为步进后读回。

| tick | FL hip R / mapped N / final / actual ° | FL knee R / mapped N / final / actual ° | FL gap mm |
|---|---|---|---:|
| 2800 | −.052205 / 22.400000 / 22.347795 / 29.239304 | −6.638346 / −12.150000 / −18.788346 / −19.117561 | 20.064677 |
| 3000 | −.054135 / 22.210518 / 22.156383 / 22.637605 | −6.660999 / −12.150000 / −18.810999 / −18.810254 | 7.640995 |
| 3200 | −.007462 / 22.210518 / 22.203056 / 22.678581 | −6.664236 / −12.150000 / −18.814236 / −18.812731 | 9.453357 |
| 3400 | +.035677 / 22.210518 / 22.246195 / 22.726904 | −6.670707 / −12.150000 / −18.820707 / −18.818212 | 9.306176 |
| 3600 | +.041647 / 22.210518 / 22.252165 / 22.733583 | −6.672276 / −12.150000 / −18.822276 / −18.821875 | 8.050933 |

101个decision端点全为P05、FL crossed=true/placed=false、AIR、bearing0。801个120Hz物理样本中，FL wheel的ground/obstacle两个verified contact pair均0次active、最大normal force均0N，故未漏掉夹在15Hz采样之间的一次真实触地。窗口最小端点gap为 **2.775512mm/t2824**，仍没有接触，随后回到约8–9mm；不能把最小gap标为captured。

## Policy / HISTORY / sigma

101条均为 `deterministic_conditional_mean`，raw逐值等于conditional mean、sampling_draws=0；不是随机探索视频。μ=.1network+.9H。t3600的FL hip/knee分别为：

- network `[+.002351, −.285282]`；H `[+.002310, −.285553]`；conditional/raw `[+.002314, −.285526]`；当前effective σ `[.102999, .086142]`。
- 全窗口hip network范围[−.003997,+.002575]、conditional[−.003394,+.002314]；knee network[−.286903,−.279249]、conditional[−.285747,−.283178]。hip不是“网络明显负向，但只被旧正历史拖住”；此时网络和历史都近零，knee负向则持续存在。
- σ范围hip[.097788,.104167]、knee[.083326,.086861]。这些σ是同一策略当前分布参数，本确定性模式并不采它们；σ较大不能被称为本片动作噪声。
- `FL_crossed_air_pending`权重100/101端点为1，t2832为.925171（基于前拍实际输入的连续proxy权重），其余状态权重为0；未伪造接触或把sigma proxy当成真实TOP判据。

所有801个native tick的phase mask全12为1，REQUEST=effective，setter/索引派发校验通过。绝非只以mask全1作结论：上述实际角度也有真实读回。最终slew仍是独立层；全窗口hip实际final与headroom候选的最大差.007491°，例如t3201新R刚变化，final受既有限速微调，不能把它误称采样被屏蔽。

## Mapper周期：确定性策略下仍可测见

晚段t3000–3600，source hip恒为22.8°；mapped N在 **22.210518/23.460518°** 两级之间切换，峰峰1.25°，每4个physics tick（33.33ms）反向、8tick（66.67ms）精确重复，频率 **15Hz**。N序列lag4的RMS差1.25°、lag8/16差精确0。同期hip REQUEST仅−.054135→+.041647°的小幅慢变，不能将final的1.25°周期当成PPO的同幅度输出。

例t3201 mapper反馈采样将N从22.210518升到23.460518，t3205降回；tracking compensation在−.589482/+ .660518°间切换，与4tick反馈节拍同步。actual hip在t3200–3208从22.678581升到22.884412再回22.683499°，约.206°的同频物理响应真实存在。整个晚窗口actual hip为22.637605–22.936699°，knee仅−18.826422至−18.775570°。这定位到现有mapper/执行闭环的周期，不是额外注入随机噪声的证据，也不在运行中热改它。

## 与旧CP183552的严格有限对照

已读取旧 `outputs/ppo_fl_capture_quality_v1/CP183552_P05_diagnosis.md`：它是**另一个checkpoint/版本且已结束**的确定性run，t5796/48.3s末态gap25.992981mm，hip R+4.309496°、final27.778330°、actual27.194547°；knee R−1.609089°、actual−13.755439°。当前CP187904 **途中t3600** 为gap8.050933mm、hip R+.041647°、knee R−6.672276°，仍AIR/0N。

可报告两份实值差异：原来约+4.3°的hip偏移已不出现在本窗口，knee构型也明显不同，当前轮端更接近台面。不能把旧末态和新途中状态当成同条件完整终态对比、仅归因于某个单关节/新reward，或据此宣布稳定性和可靠捕获已经解决。后续训练选择应关注真实近接触后的捕获及保持分布；此窗口只说明尚缺哪段行为，不构成视频门禁或现在改reward的依据。

## 有限源码定位与最小可测候选（仅提案，未实施）

实际链路为 `semantic_residual_adapter.py:180` 单次调用 mapper；`semantic_tracking_reference.py:202 build_tracking_reference` 仅构造该拍计算用的上一ACK REQUEST参考，不写物理状态。核心为 `infrastructure/servo_target_mapper.py:194 ServoTargetMapper.advance` 的反馈分支和 `:243 tracking_correction_step`：

```text
e0 = (standing_deg + sign*N_source - measured_physical_deg)/sign
c_desired = clip(gain*(e0 + previous_REQUEST), -10, +10)  # 当前参考层
c_new = clip(c_prev + clip(c_desired-c_prev, -1.25, +1.25), -10, +10)
N_mapped = clip(N_source + c_new, physical_joint_limits)
```

FL sign=+1，角度在mapper内部统一为deg；measured入参rad由advance转deg。gain=8、反馈间隔4tick；1.25=150deg/s×1/120s（构造函数:78），只在sample_feedback时更新补偿（:180），非采样tick保持。这里没有额外的PPO随机噪声，也不是“每tick直接把8倍误差叠加到角度”的积分器。

两个真实样本精确解释往复：

- t3201：`e0=.121419°`，上一R=−.007462°，desired=.911655°；旧c=−.589482°，差+1.501137°，slew截到+1.25°，新c=+.660518°，mapped N=23.460518°。
- t3205：`e0=−.084412°`，上一R=−.002924°，desired=−.698685°；旧c=+.660518°，差−1.359203°，slew截到−1.25°，新c=−.589482°，mapped N=22.210518°。

对晚窗口150次实际feedback样本，只用日志数值重构上述递推，mapped N最大误差 **0**，150次全部恰为1.25°反向步长。e0仅[−.136699,+.162395]°，desired仅[−.767049,+.916615]°：±10°补偿饱和、硬关节限和参考reserve clip均不是触发层（reserve clip 0次）。801tick中的晚窗口actual q与**上一已步进物理tick**读回逐值一致，previous_REQUEST也与上一native ACK逐值一致，误差均0；未发现符号/单位、额外旧拍延迟或混入当前尚未执行R的实现错误。

因此可确认的是：**现有高增益、4tick采样保持与1.25°slew在这个实际闭环形成了两点极限环**，不是已证明某个单位/索引bug。`SERVO_TRACKING_CONVERGENCE_BAND_DEG=.75`(:24)只用于结束tracking时的大陈旧bias退役判断，不是正常feedback分支的deadband；不能因这里误差小于.75°便声称生产漏执行了既定deadband。旧CP183552报告的末段mapped hip单次1.25°变化与本结果一致，但本次才在限定窗口逐tick证实8tick周期，不借旧报告宣称所有P05都一样。

**最小候选：只在隔离的新控制版本中，先测试现有构造参数 `tracking_gain` 从8降到2；其他代码、4tick采样、1.25°/tick最终slew、±10°补偿边界、previous-ACK REQUEST语义及所有硬限不改。** 这是单个反馈增益候选，不添加滤波状态，不改HISTORY、ρ、σ、reward，不按PPO是否零选择不同控制器。2只是待测候选：按同样固定输入，两例desired降到+.227914/−.174671°，与原c的差不再触发1.25°截断；这不保证新闭环稳定或捕获，因为下一步物理输入会变。

可先用上述真实日志作无物理写入的递推正反例：单位/后腿sign对称、上一ACK参考不吞残差、source改变时正常slew、负载大误差保留有限补偿、tracking结束保持既定退役语义。然后在本次录像自然结束后，才考虑一次有限同入口P05真实对比，看mapper/actual周期、gap、实际接触保持、后继接续；不能仅看final抖动下降。旧成功zero源码/配置/视频继续冻结；若候选以后采用，B/C应共用新mapper并显式按环境转移变化迁移，不能悄悄替换当前CP录像或称旧zero已覆盖新控制版本。不建议未经实测增加`.75°内直接停补偿`，那可能冻结有偏target并妨碍最后几毫米捕获。

本次只完成源码及记录算术定位，没有运行该候选，也没有将它设为视频、训练或zero继续执行的门禁。

## 后续封存补记：blocked 的明确条件，不是碰撞或录像损坏

本节在根任务通知本次det视频自然结束后才读取：最终run/source/video manifest、最后一条policy decision及physical observation、相关predicate/timeout源码和配置。**没有把后来的终态回填成前面t3600窗口当时已知的结果**，也未重扫整条轨迹或读取新的stochastic run。

实际终态 **t6099/50.825s/P05**，最后policy interval只执行6096→6099三个physics tick。逐拍semantic任务结果为 `INCOMPLETE_CONTROLLER_BLOCKED`，直接来源 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。具体约束是：**P05所需的真实 `placed_FL` 尚未达到1，且本阶段有限恢复时间已用完。**

时间完全由现行公开规则得到，不是录像预算或硬关节姿态门槛：

```text
P05进入：t1632 / 13.600s
当前completion placed_FL = .35*lift(1) + .35*crossed(1)
                         + .30*(.5*top_geometry(1) + .5*top_samples(0/2))
                         = .85
有效期限 = 30 + min(10, 30*.5) * .85² = 37.225s
终止时刻 = 13.600 + 37.225 = 50.825s = tick6099
```

源码接点：`semantic_supervisor.py:1100` 的placed进度仅在历史真实placed成立时返回1；未placed时几何接近不能替代真实top样本。`:1398–1403` 在 `age >= local_limit+allowance` 才设置本次blocked。配置P05 `completion_predicates=[placed_FL]`、基础30s，`history.minimum_top_samples=2`。末态 `stall_diagnostic=true` 也是真的，但它只报告6s窗口进度变化不足；配置明确 `terminates:false`，不是这次直接终止条件。

末态 FL：AIR、gap **8.077880mm**、bearing0N、consecutive_top_samples0、top_contact=false、support=false、within_top_xy=true、top_geometry=true、placed=false。即“已越沿、悬在顶部上方，未获得放置”，不是已接触却被接受规则误判。independent physical evaluator仍valid、termination_reason/source均null；最后physical记录all_finite=true，base/obstacle `detected/real_pair_active/persistent`均false，penetration=0。故应分类为 **有限任务期限内未完成FL捕获**，不应改称真实碰撞、数值/硬限安全中止或执行器故障。末次派发verified及无episode状态写入证据也正常；未发现造成此次停止的控制派发异常。

外层run lifecycle=`DIAGNOSTIC_FAILURE`，`source_acceptance_error="SemanticVideoError: episode did not meet common physical task"` 是对未完成任务的正确拒收，不是物理evaluator另报了一次碰撞。媒体manifest记录763帧、encoder正常finalize、完整decode763帧、frame_count匹配、error为空、black-like帧0；**媒体有效与任务失败应分开报告**。

截至这次完整det评估，第一未完成任务确为P05 FL真实放置，未进入P06，尚无完整PPO成功。原先窗口“更接近台面”的观察仍成立，但现在可明确补充：它最终没有转为接触；不能仅延长期限、放宽placed或引用zero成功改写本次结果。后续stochastic录像及训练由根任务继续，本节没有修改任何运行或验收条件。
