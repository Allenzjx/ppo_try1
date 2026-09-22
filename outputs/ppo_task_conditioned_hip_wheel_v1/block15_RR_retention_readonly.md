# Block15：P10–P12 的 RR 当前可用性与奖励

结论：**RR 退离平台/掉回地面已有几何势函数惩罚，并非漏掉 RR；但“当前可用”硬谓词与柔性势函数不是同一条件。** 具体缺口是当前可用性没有直接门控后续 RL 的势函数信用，历史已放置保留80%份额。不能据此笼统认定奖励反号，更不能直接增加权重。

范围：生产649ccd9；仅真实封存 rollout1522，128个P12 learner样本，末拍6184–7200，无terminal。[逐拍选样/系数/rollout SHA](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/block15_RR_retention_readonly.json)。RR placed6155发生在真实 successful_nominal 前缀，**前缀0 PPO信用**；RL lift6528发生在本次 learner。没有新 forward、优化、仿真或生产变更。

## 1. 已存在的 RR 信号

[physical_potential](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:1150) 对历史placed腿使用：

`Phi_RR = (.85/4) * (.8 + .2*r)`

`r = min(clamp(1-outside_xy/.25), .008/(.008+max(0,-.015-clearance)))`

因此 RR 当前保持份额的最大 Phi摆幅 .0425，乘potential_weight5对应未折扣 .2125；**.8历史份额不退回，不等于虚构它仍承载**。奖励为 `5*(.9985*Phi_after-Phi_before) - .02*dt`，这里dt=8/120秒。没有独立“RR失载事件 −bonus”或逐秒接触惩罚。[_current_capture_retention](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:1288) 的新增contact/positive-gap混合仅限FR/FL、且RR尚未placed的前驱窗口；不作用于RR或本次P12。

| 真实末拍 | RR状态 / usable | retention | RR分量PBRS | 全reward |
|---|---|---:|---:|---:|
| 6328 | 受控AIR，gap +6.06mm / true | 1 | −.001594 | −.006274 |
| 6336 | AIR，gap −.242mm / false，之后曾恢复TOP | 1 | −.001594 | +.005776 |
| 6528 | TOP / true；RL新lift | 1 | −.001594 | +.414566 |
| 6760 | AIR、平台外，gap −45.695mm / false | .206746 | −.086673 | −.091313 |
| 6768 | GROUND，gap −50.351mm / false | .184539 | −.006053 | −.010693 |
| 6776 | GROUND、front −66.894mm / false | .184631 | −.001314 | −.005955 |

生产纯函数对128快照的Phi与真实日志全部一致。几何退化首先在6728–6768持续扣分；已经处于低retention的静态后续拍没有重复一次性大扣分。P12所有quality family实际为0（当前几何quality阶段不含P12，contact/smoothness权重为0），所以本窗没有姿态质量成本压过保持收益的证据。

## 2. usable分别在哪里生效

[_current_rr_placement_usable](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:102) 要求：有效且未终止；RR历史placed/cross；lift_established；非GROUND；within_top_xy；当前TOP，或者 current_lift_valid + AIR + clearance≥0。它**不要求恒定TOP承载**，允许合格AIR接续。

- **阶段验收：有效。** `predicate(placed_RR)` 只有usable时可等于1，否则最多.99。P10/P11/P12入口都要求placed_RR，故失去可用性会阻止当前阶段完成/前进；本128拍64次entry_invalid，原因均placed_RR。历史事件不被抹除，阶段不自动退回P09，也不清零动作。
- **Phi：不直接消费该bool。** RR分量只用上式geometry；RL的前驱许可仍检查历史placed_RR。因此6336 usable=false而retention=1是真实的容差差别（soft允许下降至−15mm），其他腿进度使全reward为正；此窗64个unusable拍有18个reward>0。这证明有局部“可继续获得别腿进度但验收入口无效”的信用差异，**不证明故意掉RR的长期回报更高**。
- **actor observation：没有名为rr_placed_currently_usable的新增独立位。** 该字段在task日志；372已有Phi/阶段进度、RR几何、ground/obstacle接触及力、历史placed和当前RR lift-valid替代位，role48也含当前支撑连续性。它不是用历史placed伪装全部当前状态，但P12的phase_progress主要是placed_RL，entry_valid/reasons本身不直接进encoder。
- **receiving x3：有意看历史placed，不看usable。** P10–P12 × RR历史placed时FR/RR wheel sigma×3，回落后仍探索恢复；本128请求全开，其他10维不变。它不是当前支撑证明，也不是失载mask。

## 3. 终态与解释边界

仅RR失去当前可用性不会立即done：允许有限恢复，真实碰撞/安全失败或局部/全局任务期限才终止。期限耗尽标为INCOMPLETE_CONTROLLER_BLOCKED，不是SUCCESS；届时Phi_after=0、非成功终态成本−40、无bootstrap。本128拍没有终态，不能预先写入将来失败信用。

RL6528即时reward+.414566但实际标准化A仅+.003953；RR6768 reground reward−.010693而A仍+.259183，7200 A才−1.387106。实际信用包含后续回报、critic和整rollout标准化，不能由单拍符号宣布目标冲突或修复完成。已明确：几何回落有信号；直接current-usability credit门控缺位/soft容差不同是现有设计事实，不是mask或丢日志。当前课程继续，不建议盲增reward权重。

