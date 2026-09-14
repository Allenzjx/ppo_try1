# N+0：前 864 决策的 P09 连续性快照（不是终态）

来源：`runs/ppo_fsm_reference_p09_stable_v2/diagnostics/20260910T1427462398013Z_g7db0d17f398d_6d154f2f0ebc4d2a9cf08176a804299b/residual_and_projection_audit.jsonl`，flat-root 第 1–864 行，字节 1–61,398,292。固定内容 SHA256：`440e74328b3662ef76ae0f56e76e6131f59d25e6b6a11f27c407cdf2def6144a`；捕获时间 2026-09-10T14:47:10.735Z。首次 schema 提取进程退出后，获主控允许重读同一固定范围；后续统计只操作保留内存，没有追新尾部或读取物理大流。

截止 tick 6912 / 57.6s，P09，termination_reason=null、terminal_bootstrap_allowed=true、无任务成功；不能把本截止称为诊断终态。864 个评估决策、0 训练信用、0 PPO 更新。源阶段计数：P01=2、P02=182、P03=4、P04=1、P05=146、P06=310、P07=1、P08=1、P09=217。

## 实际入口与承载

| 交接 | tick / 秒 | 当前物理证据 |
|---|---:|---|
| P05→P06 | 2680 / 22.333333 | FL Q1559、C2468、P2677；此刻 FL TOP、8.530743N、有效载荷份额0.296900 |
| P06→P07 | 5160 / 43.0 | rear_approach=1；FL TOP、7.213744N、份额0.249151；RR GROUND，前缘距离−219.138mm |
| P07→P08 | 5168 / 43.066667 | role_prepared_RR=1；FL 已 AIR、实测承载0N；当前支撑 FR/RL/RR，不能用 FL 历史P声称它承载 |
| P08→P09 | 5176 / 43.133333 | transfer_ready_RR=1；FL 仍 AIR/0N，RR仍GROUND、未I/Q，前缘−216.193mm |

5168/5176 的独立角色事实 valid、normalized_load_valid、load_change_valid 均真；CoM 向 FL 侧的当前窗口位移分别 +6.438/+8.885mm，whole-body 实测关节变化 4.674/13.694°，actuated_response=true，short_support_continuity_fraction=1。这支持当前运动准备/转移入口，不等于 FL 接触承载，不证明 RR 已抬起，更不是静态稳定性证明。

## RR 真实事件链：曾成立 Q，后来回地撤销

- I5205 / 43.375s：初始抬升3.051mm、全身关节响应20.673°；首次尝试未Q，决策末态5232已回地。
- I5487 / 45.725s：第二次初始抬升4.625mm；Q5509 / 45.908333s：实测抬升8.060mm、记录的RR自身关节响应0.729°。**两次I及Q均发生在P09，不是P08已抬起再落地的交接截断。**
- Q所在决策末态5512：RR `OBSTACLE_AMBIGUOUS`、非AIR，current_lift_valid=true，body_control_evidence=true，其他实测支撑FR/RL；FL AIR/0N。全身实际变化2.810°，该窗口nominal关节变化0°。这是持续受驱系统的全身/接触响应，不可把泛化事件文本 `active_air_or_early_top` 当成精确AIR分类；当时load_fraction_valid=false，不用载荷份额推论减载。
- P09共有68个决策末态 current_lift_valid=true；最大地面相对抬升44.521mm（5832），最近前缘仍−21.605mm（5840）。35个末态的接触模式为TOP，但例如5832/5840的 `within_top_xy=false`、`top_contact=false`；没有任何RR C/P，接触标签不能代替越沿放置。
- 6243 / 52.025s 明确 `qualification_revoked_ground_before_cross` 及 `current_lift_revoked_ground`。之后不能继续把旧Q当当前有效抬升；event_ticks仍保留5509只是历史事件。
- 截止6912：RR GROUND、I/Q/current_valid/C/P全false、地面抬升0，前缘−48.438mm、台面净空−50.475mm；motion_continuation_allowed=true表示仍允许有界调整，不是任务已完成。FL此时重新TOP，实测13.839568N、有效份额0.481325；FR TOP0.674101N、RL AIR0N、RR地面承载14.239377N。历史FL放置与当前真实承载在此刻均有各自证据。

## 源 P09 序列已生成并实际下发，不是 owner 停钟

P06→07→08→09边界四轮N/final均持续+0.3rad/s，未清零。FL hip建议24.9→37.6→37.6→46.1°，RL hip6.9→6.9→14.3→18.5°；这些是新owner请求跃迁，最终mapper仍限速，不要求再加一次pre-slew。全部6912个native tick连续且verified，864个末tick setter/映射匹配；末tick可审计servo增量最大1.25°，handoff_hold=0。零raw与零projected residual贯穿864条，counterfactual residual effect=0正是N+0预期，不代表执行器没运动。零探针本身不能单独验证非零residual的继承。

源RR建议在决策末态实际经历 hip1.6°(5184)→55.6°(5256)，knee0→−37.8°(5304)，hip再回至−6.9°(5408)。5832已发出最后全身servo owner端点 `[-18.5,-31.4,0,31.1,15.4,19.4,-6.9,-37.8]`，同时FL wheel−1.07；6048收到最终四轮0，至6912保持。它覆盖源P09的5.4s全身事件和7.2s最终wheel事件（这里给的是决策末态分辨率），不是仅首段生成。生产 `semantic_supervisor.py` 的 `_continuous_advisory` 每tick推进全部layer，RR源关节无Q/高度停钟；本快照capture抑制仅记录已捕获FR/FL，没有RR新swing被抑制。

截止时P06 rolling contribution因当前几何gain=0而retired；RR carry诊断为 `no_extra_wall_push_source_and_residual_adjustment_continue`，没有固定抬升计时门槛或15mm动作门槛。N的RR为[-6.9,−37.8]°，实际最终target为[-7.534519,−27.8]°；mapper保留补偿[-0.634519,+10]°，不是PPO residual或源owner漏发。实测RR约[-7.472540,−29.147507]°，由同tick角色实测关节余量按现行下限−135/−60°回解；不能把N或target冒充actual。

## 有界结论

前864条没有新增执行/停钟/阶段reset缺陷证据：源P09端点序列已下发，物理评价始终VALID，无安全或BODY终态。派生N确实完成了FL越沿放置，并曾使RR成立功能抬升，但没有将RR带过前缘并放置，随后回地；截止第一未完成任务仍是P09的RR再建立可用抬升、越沿及放置。末态body线速度0.015096m/s、角速度0.039669rad/s，低速也不等于越障完成。保留运行后续真实终态；本快照不证明完整N+0成功、FSM/PPO优劣或唯一失败原因，也不构成续训门禁。
