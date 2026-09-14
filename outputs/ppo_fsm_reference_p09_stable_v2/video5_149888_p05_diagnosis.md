# Video5：149888 自然 P01，FL 越沿后仍未放置

固定 run `20260910T1411052557957Z_g7db0d17f398d_fda5b92927ef45d7aa2b048e98015a8e`，2026-09-10 14:26:43.677313Z finalized。800次策略决策、6396 physics ticks、53.3s；`DIAGNOSTIC_FAILURE`（主控launcher exit1），P05 `INCOMPLETE_CONTROLLER_BLOCKED`，0 optimizer updates。不是完整成功。

实际source已证实官方加载149888，checkpoint-loaded/verified=true、migration=null、372维、seed4001、deterministic conditional_mean。自然P01入口tick0/decision0/done=false，reset_count1/options{}/prime0。记录的源checkpoint SHA `1706e50255c0c46e10f47847d80283efaa5e3b67f36691a59a007e5e8b20b236` 是**加载来源证据，不是结束时模型哈希**；本次未加载/重哈希模型，不能把initial load或0更新扩张成评估末全部模型状态bitwise一致证明。

## FL 与第一未完成任务

FL I=tick1989/16.575s（向上4.046096mm），Q=tick1992/16.6s（向上8.211614mm），C=tick2587/21.558333s；P从未成立。第一未完成是P05受控放置，而非没有抬升或没有越沿。C后首次决策端点是tick2592；之后477个端点全部AIR、TOP0、最大承载力0N。整回合端点GROUND248/AIR552。

后C净空范围（**不同时间，不合成同一个最佳状态**）：最大101.499263mm@tick3064/25.533333s；最小4.388608mm@tick3120/26s；末端11.249324mm。末front distance+99.897448mm、within_top_xy=true、top_geometry=true，但TOP/contact/support均false、surface=NONE、bearing0N、load_fraction0且valid；consecutive_air4386、consecutive_top0。几何上接近顶面不等于真实接触和承载。

P05入口tick1928/16.066667s；本次期限为30s + **7.225s**当前进展额度 +0固定post额度 = **37.225s**，不是上一回合34.9s。实际stage_age37.233333s时（比当拍有效限额多1/120s）`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；placed_FL进度0.85、stall=true。总时长53.3s不是200秒窗口截断。独立physical evaluator为VALID/VERIFIED、success=false、termination_reason=null；监督器的局部任务未完成与它一致，不应改称BODY/WHEEL_ONLY违规或FALL/HARD/数值安全终止。

## 控制与完整性

末FL nominal hip/knee=22.8/−13.4deg；residual=+6.743639/+22.170331deg；实际最终目标=28.988702/10.020331deg。末nominal四轮均+0.3rad/s；residual轮FL/FR/RL/RR=−0.525752/−0.242005/−0.071607/+0.450338，实际轮目标=−0.225752/+0.057995/+0.228393/+0.750338rad/s。不是沿用video4的零nominal轮命令。

800个决策端点mask均12个1；P05全部559端点的residual等于`tanh(raw)×phase_cap`（最大误差4.44e−15）。末post-mapper基准+residual与实际目标精确一致，未见末端residual被mask或裁剪吃掉。**范围限制：**全P05简单基准+residual与最终目标曾有最大标量差0.143315；该聚合跨servo-deg/wheel-rad/s，未保留最大点通道及controller/native组合明细，不能给它指派一个物理单位或断言是裁剪，也不能宣称全阶段所有后级投影为零。此观察不建立训练门禁。

阶段样本P01=2/P02=198/P03=6/P04=35/P05=559，P06–P13=0。policy内逐tick native证据连续1–6396，全部verified且有native effect；own-phase6392，四个交接tick单独区分。state-write审计全通过。末决策6392→6396为真实4ticks并正常返回；没有补成8ticks。

仅53.3s未完成窗口的质量：roll RMS0.10178598rad、pitch RMS0.07447134rad；roll/pitch-rate RMS0.06523335/0.05084786rad/s。all_phases_sampled=false；不据此宣称全程稳定性优于FSM。

原capture录制时全解码记录PASS、800unique帧/15fps、black0、ledger完整、encoder在关闭前finalize；媒体53.333333s，末帧量化0.033333s。**容器duration仍286331153s、container_valid=false。** 原MP4为76,051,817bytes，记录SHA `9888c649fbb23b1a0c5502b630991bbd18ded5f34c2ba82595246ba6a8bd6115`；本次不重解码、不remux、不改源manifest或发布成功文件。

本次仅一次固定policy ledger扫描及小元数据读取，未扫physical/native大流、未读active N+0、未改生产/四主报告/CSV/训练。原始依据为该run/source内的`semantic_video_source_manifest.json`、`video_policy_decisions.jsonl`及录制manifest；[固定标量摘要](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/diagnostics/video5_149888_fixed_summary.json)保留本次核验范围。
