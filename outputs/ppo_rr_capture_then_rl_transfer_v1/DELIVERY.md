# RR 捕获→RL 转移：当前交付

## 当前权威影片：CP220544 / 97c4367 / v6 独立前段验证候选

`video_review/CP220544_ancestor_contact_onset_v6_review/` 已完成并封存同一自然P01确定性episode的三项交付：

- `CP220544_DET_v6_full_attempt_INCOMPLETE.mp4`：1825帧、15fps、121.666667秒；完整正常速度且保留失败尾段。物理终点为tick14600/P09，结果`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。
- `CP220544_DET_v6_RR_window_RL_not_reached_detail.mp4`：同run连续帧区间[1036,1825)，789帧/52.600000秒；只呈现真实RR窗口并明确P10/RL NOT REACHED。
- `historical_N_vs_CP220544_DET_v6.mp4`：1825帧/121.666667秒；历史成功N_ref在自身73.808秒结束后明确冻结，不是当前版本的新B、同控制器比较或新增物理时间。
- `export_receipt.json`、`CP220544_v6_selected_fourwheel_evidence.json`与`QA_SUMMARY.md`：封存来源、checkpoint、完整decode/连续PTS/黑帧检查、四轮及RR实测证据。

真实终点RR间隙为−0.003537004毫米，当前AIR/0N，无TOP、无placed；P10与RL均未到达。近零或轻微负几何间隙不是接触、承载或成功。assist仍为`DESCEND_PROGRESS`，累计travel 52.041667°，未达到53°上限；本次失败不是搜索预算耗尽。本视频checkpoint为独立前段已验证候选`checkpoint_rr_contact_onset_v6_ancestor_step_000220544_g97c4367ee293.pt`（SHA256 `0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01`），不是最新已学习CP221184。v6迁移和本次评估均新增0学习；此前真实新增的640 policy decisions /5 PPO /100 Adam只属于较早e24a训练并包含在最新已学习CP221184中，不能记到这个CP220544 v6候选上。

## 历史记录（以下内容均不是当前权威交付）

## 最新物理结果（57ae41e，headless，无对应新视频）

保留前段候选 CP220544 的独立 v4 控制评估于109.625秒自然结束：RR真实抬升、越沿，最后顶部间隙13.428596毫米，当前AIR/0N，仍未放置，RL未到达。最后2秒仍下降2.6382毫米，终止原因是已声明捕获辅助的40度累计搜索预算耗尽，不是硬限、安全失败或成功。四轮最终/实测速、固定CoM投影和逐拍来源见 `headless_57ae41e_fixed_COM_review/ACTUAL_SUMMARY.md`；现有影片不能代表这次无录像run。

该控制评估新增0PPO/0AUX，不继承较新CP221184的640决策信用。正在合法保存边界实现v5末段有进展捕获及捕获目标连续继承；尚未发布或取得v5物理成功。成功N_ref不变。

## 最新影片：CP221184 /632295b/v4，P05未完成

`video_review/CP221184_v4_deterministic/` 已实际导出：

- `CP221184_DET_v4_full_attempt.mp4`：911帧、60.733334秒，最新保存重载权重，自然P01确定性完整回合；保留全部失败尾段。
- `CP221184_DET_v4_predecessor_failure_tail_detail.mp4`：同次episode最后60秒，明确RR/RL NOT REACHED。
- `historical_N_vs_CP221184_DET_v4.mp4`：与已保留历史成功N比较；C先结束后的定格明确标示，不记为新物理时间，也不是本版同构B。
- `export_receipt.json`：完整来源/权重/运行版本和正常速度、PTS、full-decode证据。新版本的wheel-v4控制已声明，但此episode未到其激活阶段。

真实结果：60.733333秒/7288ticks，P05年龄40秒时`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。FR真实placed/currentTOP，FL未捕获（终点front−67.508mm/gap−49.999mm），RR/RL未到达；不是机身碰撞、数值/硬限安全中止或RR方向验证。新轮速辅助未激活，不能将此结果归因为RR修复已失败或已成功。

CP路径`checkpoints/history/checkpoint_rr_carry_handoff_v4_step_000221184_g632295bc49cf.pt`，SHA`2fed192f9c5141a40f769893d0266ec3b95c91d55a4b3091acc411cd99672f90`。累计221184/1693PPO/33860Adam；本轮真实新增640/5/100全部发生于此前e24a/v3训练，v4迁移及本次评估新增0。P07/P08/P09=3/2/635，P10–P13=0；没有完整越障或稳定性优势结论。

最新CP与此前前段更可用的CP220544均保留。后者若用于独立v4控制评估，将明确标作历史兼容候选，不继承最新640学习信用，不覆盖latest指针。`sampling_target`目前只是未被读取的比例声明，后续需用真实P01 learner run与后腿run在完整update边界交替，不能宣称已自动按30%混采。

## 以下为较早已封存进度（不是当前运行状态）

## 最新实际结果（覆盖下方较早进度）

e24a/v3真实训练已正常封存：新增640个有效决策、5PPO更新、100Adam step；累计221184/1693/33860。最新已学习CP为`checkpoints/history/checkpoint_step_000221184.pt`，SHA`e721e9b52d6f05e8024a397363ca65e1b151a5a19efd2c5c805b59d53e1b0d6b`，官方保存重载和独立训练审计PASS。P07/P08/P09样本=3/2/635，后续阶段0；真实前缀2068决策不计学习。新AUX0，LR1e-5、Identity和原完整Adam/谱系保留。

第一学习回合RR仍未捕获：107.183秒/P09，有限hip20°+knee20°搜索耗尽；最终gap34.788mm/front+1.379mm。第二回合在完整update后非终止截断，不判成功或失败。原计划剩1408决策未运行、不计账。没有完整RR/RL或自然P01成功。

CP221184暂无确定性自然P01视频。下列已交付影片全部属于旧CP220544/26db控制v2，不能冒充最新权重。当前在合法停训边界实施v4支撑轮前送和接触交接修订，尚待保存重载及真实验证；这属于公开控制辅助，不是网络学习收益。

正在运行的e24a/v3训练已完成首个真实更新并保存重载：`checkpoints/history/checkpoint_step_000220672.pt`，SHA256`168cfb484a6508c6e8fa0bd60741c54b7ee207af982bc13e25a6db0e90d0cd12`。新增128决策/1PPO/20Adam，累计220672/1689/33780；actual LR1e-5、Identity不变。后续rollout仍在采集。**CP220672尚无确定性完整评估或视频，不能用下面CP220544/v2录像代表它。**

最新已交付标注视频位于`video_review/CP220544_RR_feedback_v2_g26db2a1946e8_review_corrected_v2/`：

- `CP220544_DET_full_RR_RL_attempt.mp4`：1561帧/104.066667秒，正常速度、完整同次episode。SHA256 `a22e2f21169366824fe92d82d93f7ef388c6d642027a1cc232c99d5ac57e38be`。
- `CP220544_DET_RR_capture_to_RL_detail.mp4`：同run尾段35秒，明确标注RL NOT REACHED，未冒充RL进展。
- `N_vs_CP220544_DET_same_camera.mp4`：历史N于73.808秒终止后定格，不是新B或后续稳定物理时间。
- `export_receipt.json`与`rr_capture_selected_response.json/.md`：source/CP/控制绑定、完整PTS/decode验证、8个实测里程碑及四轮source/target/actual。

root已检查末帧与配对图；正确HUD明确`RR WINDOW REACHED | RL NOT REACHED`。未带`corrected_v2`的前一导出目录保留用于追溯，其中旧HUD有含糊RR/RL窗口字样，不作为交付版本。上述都是训练前26db/v2视频，不是e24a/v3或新学习权重的评估。

e24a/v3已保存重载且真实2048决策续训正在单Isaac运行；尚在真实前缀，不提前填写PPO完成数。见RECOVERY顶部。

## 最新已完成真实评估：26db / CP220544

- 完整自然P01、确定性、12通道，已声明FL+RR capture assist；不是纯网络。
- run：`runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0356402230589Z_g26db2a1946e8_c1a7c95ca1db41d69882389e1e97cd8d`
- 104.033333秒、12484物理ticks、1561评估decisions；P09普通任务未完成，非碰撞/NaN/跌倒成功替代。原结果`DIAGNOSTIC_FAILURE`保留。
- 原始视频：同run的`source/actual_viewport_video.mp4`，SHA256 `7d92835b9381b641eb59e3d9c060c2bb083dd651710f3dc9dd935106dd519af8`；完整失败尾段保留。标注版与历史N对比正在独立导出。
- checkpoint：`checkpoints/history/checkpoint_rr_capture_feedback_v2_step_000220544_g26db2a1946e8.pt`，SHA256 `a32a4f01afbfbcda06ba59f7e439bb5ab9aa1dbefb3d8e3093e0dd86c827e85e`。已保存并严格重载。不是新学习权重。

| 项目 | 实测结果 |
|---|---|
| 前驱 | FR/FL均有真实placed历史；终点FL仍为TOP接触 |
| RR | 有效抬升、真实越沿；无TOP捕获/承载/placed |
| RR终点 | front+43.208mm，gap45.703mm；目标hip/knee−1.786/−45.361°，actual−2.048/−45.120° |
| 本版控制贡献 | XY处保留前拍FINAL膝目标；负向hip实际20°，未完成落脚 |
| 第一处未完成 | RR捕获；BLOCKED仍持有6/7，膝不能靠policy残差自行展开 |
| RL/向FR转移 | 尚未到达，不能声明完成 |
| FL持续反转 | source stop后policy负修正仍在；本版wheel整形关闭，未声称已修复 |

## 学习与控制分账

最新已验证学习源仍220544 policy decisions /1688 PPO /33760 Adam。RR新分支截至本评估新增0真实PPO、0AUX；CPU合成更新测试不计真实学习。旧block11的+2048/+16/+320已经包含在220544中，不能再重复相加。

原成功N_ref、接受的CP218496视频、旧checkpoint与用户工作树修改保留。历史N对比不是当前控制实现的新B；N在73.808秒结束后必须标注定格。

## 下一冻结修订

正在实现有界hip→knee搜索：保留原hip20°/12秒额度，之后保持到达的hip并尝试knee正向1°/秒；初始2秒，只有实测下降反馈才续行，knee总额度20°。接触、跟踪、XY/支撑及200秒全局限制不变。不重复执行P10源节点，不增加隐状态，不把候选方向写成已证明的因果。显式same410迁移保留权重/Adam/LR/Identity/RNG/谱系，清空旧rollout；测试/发布完成后再真实采样。当前未声称v3成功或已增加PPO更新。

恢复与精确边界见`RECOVERY.md`。
