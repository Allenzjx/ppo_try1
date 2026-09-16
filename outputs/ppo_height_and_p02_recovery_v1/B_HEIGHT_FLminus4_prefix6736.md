# 首个 FL−4 zero：固定前缀诊断（非回合最终结果）

仅检查 post tick≤6736，重点5400–6600，未追活动尾、未读/修改最终状态。主 physical/native 文件当时仍在缓冲，使用已落盘的每8tick height 与决策端点；事件 tick 来自真实 evaluator event history。详细数值见 [B_HEIGHT_FLminus4_prefix6736.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/B_HEIGHT_FLminus4_prefix6736.json)。

RR 先后在5442、5764获得Q，真实ground接触于5529、6154撤销Q。第一轮回落时 wheel 一直为0，没有新的stop先发生。5480→5528，base origin反而58.061→81.488mm，RR mount却172.511→128.177mm；不能用单一base高度代替RR安装点/可达空间。

第二轮：四轮+.3在6048→6056端点间停止；6056→6064进入源step18（FL/RL姿态组合与FL wheel−1.07）。此组合后的FR支撑消失、RL载荷变小，RR仍顶角承载。6120时RR TOP=12.326N、gap−14.710mm；随后6120→6160 mount下降26.541mm、gap下降35.845mm，base只下降7.209mm。6160时RR已GROUND15.919N，FR/RL悬空，其他实测支撑仅FL。

**最终全轮stop在6272→6280，晚于6154落地至少118tick**，不能归因为这个最终stop触发回落。更早的源stop与step18确实先于失稳，但同一时间改变多腿姿态和轮速，尚未隔离“wheel单因果”。原step18仍是有限源动作，未擅自删停轮或改幅度。

新geometry在6120 trust有效且未到envelope；6136开始出现trust=false/envelope=true/needs_body=true，knee目标−37.8、实际−39.964，6144暂失当前Q，6152短暂恢复，6154真ground撤销。5400–6600端点最负knee目标−40.196、实际−43.058，未再追到−60；但有界模型明确报告局部约束不足，不能把数学边界通过等同于真实净空保证。

下一个独立RL−3有信息价值：5408 RL确实GROUND承载16.381N（55.53%），FL则AIR0N，之后RL实际约32.6°而RR安装点显著下降。但减少RL正向段也可能破坏减载，方向只是可检验假设。若优先保留初期载荷转移，可选现有“current RR lift后小幅RL恢复”入口；5442后和5764后均有真实可用窗口。当前FL−4 run的post-lift offsets为0。两候选都不能宣称已奏效；独立RL实验须取消FL减量，组合不能伪称单变量。

6736端点RR仍在GROUND、C/P未完成，额外TOP/AIR continuation不满足条件；这不是回合最终失败报告。继续等待自然真实terminal。未修改生产、验收、限位或reward；没有新增仿真。

