# CP226304正式确定性录像：已封存，首次未完成仍为RR捕获

同次自然P01单回合，seed4001，runtime `892385cba8a7089b52567bb7f558c018fac82a77`；CP226304正式重载与hash校验通过。控制方法：**PPO＋已声明原FL capture assist；后腿任务辅助OFF**，继承有限AUX谱系不隐瞒。本次eval没有PPO/AUX更新。训练后缀中两次RR捕获不是这条确定性录像的成功。

## 结果与物理分类

- 1381条issued/completed决策、11044物理ticks、92.033333s，最终P09。
- 终止 `INCOMPLETE_CONTROLLER_BLOCKED`，具体来源 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；局部30s＋当前进展许可4.9s共34.9s耗尽。是普通任务未完成，非外部数据truncation，terminal bootstrap=false。
- run lifecycle `DIAGNOSTIC_FAILURE` / process exit1由 `episode did not meet common physical task` 拒收触发，**不是viewport错误**。
- physical valid=true/runVALID，physical termination_reason/source均null，实际末拍body_collision=false、all_finite=true；没有记录为BODY/HARD_LIMIT/FALL/NaN或纯轮爬升终止。
- 必须保留末拍 `physical_evidence_status=CONTACT_BEARING_UNVERIFIED`：RL同时GROUND与障碍不明确接触，bearing_verified=false；不能将其当已验证顶部支撑。

| 腿 | 有效抬升 tick(s) | 越沿 tick(s) | 放置 tick(s) | 末拍真实状态 |
| --- | --- | --- | --- | --- |
| FR | 24(.200) | 2783(23.191667) | 2793(23.275) | 合法TOP，12.981N，bearing verified |
| FL | 2805(23.375) | 4319(35.991667) | 5256(43.8) | 合法TOP，2.174N，bearing verified |
| RR | 7108(59.233333) | 8152(67.933333) | **无** | 有效AIR，XY内，gap55.335mm/front+99.685mm，0N |
| RL | 无 | 无 | 无 | GROUND_AND_OBSTACLE，front−50.215mm；bearing未验证 |

RR有效抬升和越沿保留，但没有形成可继续利用的顶部接触/承载；这就是第一处未完成任务。RL没有有效卸载、越沿或放置；未完成向FR侧转移验收、整机停车或home。RL数值bearing_force=9.372638N、总反力19.448065N，因bearing_verified=false，不作已确认支撑力解释，更不是TOP或新资格。

前段事件证据：FR放置后+6s FLgap58.739mm、CoMz164.394mm，接近CP225280已接受前驱；FL真实placed并进入P06，FR→FL放置20.525s，原FLassist有实质贡献。tick5264 FL短暂掉载，随后恢复，不能只凭历史placed称全程承载。详见 [前段六秒对齐](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_rl_recovery_learning_v1/CP226304_DET_FRplaced_front_space_6s.md) 与 [FL→P06](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_rl_recovery_learning_v1/CP226304_DET_FLplaced_P06_evidence.md)。

## 末拍控制缺口与CP225792旧窄窗口

下表每格是 **same-tick mappedN / filtered REQUEST / FINAL / actual**，单位canonical度。旧点为CP225792 tick9432/78.6s；新点为本次终态11044/92.033333s。它们都属于RR已越沿但未承载窗口，**不是相同物理状态、同时间或单变量干预**，只用于核对相同作用类型是否仍存在。

| 通道 | CP225792旧窄窗 | CP226304末拍 |
| --- | --- | --- |
| FL knee | -12.150 / -32.887 / -45.037 / -45.368 | -12.150 / -32.876 / -45.026 / -45.311 |
| FR knee | 31.290 / -51.473 / -20.183 / -20.475 | 31.413 / -51.467 / -20.054 / -19.383 |
| RL hip | 28.200 / -12.424 / 15.776 / 16.822 | 28.200 / -12.586 / 15.614 / 17.230 |
| RL knee | 0.000 / 0.781 / 0.781 / 0.416 | 0.000 / 0.690 / 0.690 / 0.887 |
| RR hip | -8.150 / 15.922 / 7.772 / 7.508 | -8.150 / 16.085 / 7.935 / 7.672 |
| RR knee | -39.050 / -19.481 / -58.000 / -57.766 | -39.050 / -19.381 / -58.000 / -57.759 |

source RR hip仍−6.9°，mappedN−8.15°，当前正REQUEST+16.085°令FINAL+7.935°；RR knee REQUEST−19.381°经headroom裁为−18.95°，FINAL仍−58°。FL knee未形成正向准备，FR knee约−51.467°残差抵消source+31.1°，RL调整仍小。它们支持“均值动作尚未形成捕获/协同准备”的具体缺口，不证明某一个关节方向必然能单独解决问题。

末拍12 mask全1，cooperative_prep_allowed=true、rearowner17全0、native mapping/dispatch verified；唯一headroom clip为RR knee通道7。未据此把准备问题归成所有动作被RR承载门锁死，也未自动释放受依赖的late source组。没有重开RR下降辅助、修改reward或更改硬限。

四轮按FL/FR/RL/RR，canonical forward-positive，rad/s：

| 层 | FL | FR | RL | RR |
| --- | ---: | ---: | ---: | ---: |
| source N | .300000 | .300000 | .300000 | .300000 |
| REQUEST | -1.022232 | -0.070186 | -0.007573 | -0.103726 |
| FINAL | -0.722232 | 0.229814 | 0.292427 | 0.196274 |
| actual | -0.756532 | 0.070773 | -0.033105 | 0.195821 |

FL持续反向来自policy抵消N，不是mask输出丢失；当前FL确有TOP接触。RL FINAL正但actual小负，且接触证据有歧义，不能将它归成policy反向或未写指令。RR AIR轮转动不算牵引。没有凭这一末拍推断完整因果。

## 封存录像与质量

viewport capture PASS，1381帧、15fps，完整decode/帧数/PTS连续/非黑屏均PASS；主episode正常速度、无拼接。最后动作仅4物理ticks，物理92.033333s，编码92.066667s，0.033333s差为末帧显示量化，没有额外物理tick。

原生容器header duration异常（286331153s），但解码时间92.066667s、1381帧/PTS有效；保留原始manifest此项，不将header异常说成物理失败。发布成片由artifact任务按既有导出流程处理；本报告没有运行编码器。

本回合已观测质量：roll/pitch RMS 8.002°/5.528°；p95 10.751°/9.880°；roll/pitch rate RMS 0.053430/0.049618rad/s。all_phases_sampled=false，fixed_quality_score=null；缺P10–P13不能评分为完美，未证明稳定性优于N。

Checkpoint SHA `c0f4727bcc4e1963b5e9d079158dde1f5f662390397b61049ff893c40e1910f6`；原生video SHA `60ef50ba1a4fcc9ba33a92810ac707c945575021da633fbf86165b46a6f08756`。首次未完成任务、完整失败尾段和辅助标签均应随交付保留。全部分析为标准库只读，未导入Torch/PXR、未更改生产或另开仿真。
