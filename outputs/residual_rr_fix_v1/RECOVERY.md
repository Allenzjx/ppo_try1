# RR 修复真实续训记录

两个真实训练块均已封存；新分支从 CP177792（1354 updates/27080 optimizer steps）实际新增 **640 decisions / 5 PPO updates / 100 optimizer steps**，最新 **CP178432 / 1359 / 27180**。训练 lifecycle `SUCCEEDED` 仅表示更新操作完成，两个块均无完整 episode，不是越障成功。

- checkpoint：`outputs/ppo_residual_rr_fix_v1/checkpoints/history/checkpoint_step_000178432.pt`
- SHA256：`19acb91f64a2b728e54f02d50dcaf187e11fd432ff68901ba91ff0af58f6a665`
- manifest SHA256：`78dc71b80f897fa843f365c0ae3ba7150ade970eaf5ff57325141e465a91453a`
- runtime：`6c2121b68654eaaa2afa7c19e9d02ae770493d2f` / `cfad9c3664f4e479fd16a9902178cc4e84327ab06aa03dd3db83e3d96a89aed1`

| 实际阶段 | 自然 P01 块128 | P06课程块512 | 分支合计 |
| --- | ---: | ---: | ---: |
| P01 | 2 | 0 | 2 |
| P02 | 126 | 0 | 126 |
| P03–P05 | 0 | 0 | 0 |
| P06 | 0 | 166 | 166 |
| P07 | 0 | 1 | 1 |
| P08 | 0 | 1 | 1 |
| P09 | 0 | 214 | 214 |
| P10 | 0 | 1 | 1 |
| P11 | 0 | 19 | 19 |
| P12 | 0 | 110 | 110 |
| P13 | 0 | 0 | 0 |

首块128/1/20，1024物理ticks、teacher=0；末尾tick1024/P02非终态、FR尚未越沿/放置。第二块512/4/80，4096 PPO物理ticks；一次成功nominal前缀335 decisions/2680ticks明确排除，FR/FL前缀事件也不算PPO能力。两块共5120 PPOticks，普通阶段切换1+6次，错误done=0。

第二块末尾tick6776（56.466667秒）/P12是非终态rollout边界：RL未越沿/放置；RR虽历史越沿5726、放置5731，但当前GROUND、front_distance=-0.130729m、clearance=-0.050347m、current_lift_valid=false、rr_placed_currently_usable=false；P12 entry_valid=false，原因placed_RR。因此不能把历史placed或后缀进度称为成功。RR首次资格4001随后被撤销，真正对应此次越沿的最后资格是5434；6053再次落地撤销当前抬升，但跨越历史保留。

实际640/640 decisions全12许可、epsilon0，5120/5120ticks native下发已验证并有真实residual目标影响；这不等于每拍每个通道都非零。quarter .25/rho .9、原容量、有效Adam LR1e-5、Identity均保留。首块通过新factor迁移，第二块同runtime exact-resume，无第二次迁移；fresh rollout、合法reset，首update actor-before分别精确匹配CP177792/CP177920。640步后Actor/critic/Adam/RNG按真实学习演化，不能声称最终仍等于源。

第二块4次update均actor改变、有限非零梯度、LR1e-5；KL分别.024876/.026014/.017906/.022732，value loss分别.541666/.940205/7.451954/1.195867。P09任务势能项合计+.056756，P12为−.877219；观察到的非终态后缀折扣奖励−1.369575不是完整episode回报。正GAE也不等于任务成功，实际存在bootstrap及归一化影响。reward、HISTORY、温度和容量没有因此改动。

同目录 `training_P01_177920_{actual,signal}.json`、`training_P06_178432_{actual,signal}.json` 保存核算；`recovery_branch_and_training_manifest.json` 保存独立分支账本。checkpoint_last指针与CP178432及manifest一致。

正式自然P01视频评估已封存 `DIAGNOSTIC_FAILURE`：CP178432官方真实重载验证通过，5907ticks/**49.225秒**，739次正式全12决策，teacher=0，无mask干预，所有5907ticks native验证与实际residual目标影响。未新增optimizer update或训练信用。

第一项未完成任务是 **P05 FL越沿后的受控放置**：FR Q23/C1406/P1425；FL Q1484/C2106、未placed，末尾AIR、间隙+.003470128m、承载0。P05 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，stage age37.225秒=30秒基础期限+7.225秒实际进展延长；这是有限任务终止，不是外部截断，bootstrap=false。

物理evaluator valid/VERIFIED且无安全终止；5908条含tick0观测中机身碰撞0、非有限0、实际关节硬限违规0，最小硬限余量21.47384°。wrapper exit1因未满足共同任务而拒绝成功发布，不是Isaac崩溃。正式P06–P13均未到达，不能以训练后缀RR事件替代自然P01全程能力。当前新严格判据完整评估0/1成功，没有创建成功checkpoint指针。

质量只作本失败轨迹描述：roll RMS=6.954382°、pitch RMS=5.749541°、roll/pitch rate P95=.018862/.017192rad/s；固定全程quality score=null、all_phases_sampled=false。长期P05驻留与缺失后半程会影响指标，49.225秒失败轨迹不能与73.808333秒成功zero排序稳定性。具体在 `evaluation_results.json`、`evaluation_CP178432_actual.json`、`evaluation_CP178432_safety.json`。

zero v2成功结果保留，且不称learned PPO；历史CP177152保留 `RR_ACCEPTANCE_UNDER_REVIEW`。下一优先是补P05受控放置能力，再检查进入RR后的carry与放置保持；质量epsilon仍0，未启用.05，无稳定性优越、泛化或部署结论。
