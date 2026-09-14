# Video4：148352 自然 P01，FL 已越沿但未受控放置

固定完成 run：`20260910T1310293487954Z_g7db0d17f398d_9f7a4dce1be54856b99818002f74a91d`。749 次实际策略决策、5988 physics ticks、49.9 秒；2026-09-10 13:24:55.071516Z finalized。任务 `INCOMPLETE_CONTROLLER_BLOCKED`，视频 lifecycle `DIAGNOSTIC_FAILURE`（主控 launcher exit1）。不是成功，也不是新增训练：optimizer updates=0。

加载证据来自真实 source manifest：`official_load_semantic_checkpoint=true`、`checkpoint_loaded_and_verified=true`、saved global=148352、migration=null，372 维 `diagonal_transfer_state_v1`；训练 seed1001、视频 seed4001。官方来源是 `checkpoints/history/checkpoint_step_000148352.pt`；manifest 记录的 checkpoint SHA 为 `62ade3e4a8b916b770a4ad2de351cf955423be665ac3cf76a46b818cd57af00c`，本次未再哈希模型。自然起点 proof 为 P01/tick0/decision0/done=false、reset_count1/options{}、prime0、`semantic_natural_P01`；不是 teacher 后缀、历史姿态快照或迁移加载。

## 第一未完成任务与 FL 证据

P01–P04 已完成，第一未完成是 **P05 的 FL 受控放置**，不是未抬起或未越沿。实际事件：I=tick1852（15.433333s），Q=tick1856（15.466667s，测得向上位移9.757391mm），C=tick1976（16.466667s），P始终false。Q/I均发生于已经进入P05后的全身运动，不能把阶段标签或先前准备完成当成FL已放置。

末端 FL：front distance **+109.436754mm**，相对平台顶面净空 **+33.599999mm**，within_top_xy=true / outside_xy=0；但 AIR=true、TOP=false、surface=NONE、support=false、bearing=0N、load_fraction=0（有效测量）。因此几何越沿并不等于顶面接触或承载。749个决策端点分类为GROUND231/AIR517/OTHER1/TOP0；C被记录后的503个端点全部AIR。终点 evaluator 的 consecutive_air_samples=4124、consecutive_top_samples=0，也支持末段连续悬空，而非已放置后被误判。

P05从tick1800（15.0s）持续34.9s，终止来源 **LOCAL_BOUNDED_RECOVERY_EXHAUSTED**：nominal_limit30s + current_progress_allowance4.9s，固定post allowance0；实际stage_age=effective_limit=34.9s，placed_FL进度0.7、stall=true。不是全局200秒或max-decisions窗口耗尽。账本唯一termination reason是任务未完成；没有BODY_COLLISION/WHEEL_ONLY_CLIMB任务违规，也没有FALL/HARD_JOINT_LIMIT/数值异常独立安全终止。独立物理evaluator仍为valid=true/run_validity=VALID/success=false、termination_reason=null；它与任务监督器的局部期限未完成分类并不矛盾。

## 实际末端控制，没有把 residual 屏蔽掉

| FL 通道 | nominal | 实际 residual | 实际最终目标 |
|---|---:|---:|---:|
| hip（deg） |22.8|+8.005723|30.267624|
| knee（deg） |−13.4|+21.675706|9.525706|
| wheel（rad/s） |0|−0.492226|−0.492226|

末端mask为12个1，12通道均有已验证native target effect。全部524个P05决策端点的residual等于对应 `tanh(raw) × phase_cap`（误差<1e−10）；末端全12通道误差3.55e−15。末端的post-mapper基准+residual等于最终目标，因此不是末端mask、residual裁剪或后级目标限幅吃掉FL请求；nominal与post-mapper基准不同，不能拿两者差额误报为裁剪。其他wheel目标FR−0.225239/RL+0.095062/RR+0.485036rad/s。这里不外推为每个120Hz瞬间都不存在slew/headroom投影；全程瞬时`tanh×cap`与决策端点residual最大差0.044193（跨阶段/瞬态不能直接判成故障）。

实际阶段样本：P01=2/P02=189/P03=7/P04=27/P05=524，P06–P13=0。普通阶段切换tick16/1528/1584/1800。policy ledger内逐physics-tick native证据恰好连续1–5988，5988条verified且都有target effect；own-phase effect5984（四个普通交接tick不混称新owner效应）。全部决策的in-episode state-write审计通过。最后决策5984→5988仅4ticks，environment_step正常返回终止；source的`interrupted_final_decision_ticks=0`表示未被物理观察器异常截断，不代表末动作执行了8ticks。

部分轨迹质量（仅该49.9s失败窗口）：roll RMS0.09567495rad，pitch RMS0.06230576rad，roll-rate RMS0.06043063rad/s，pitch-rate RMS0.04630216rad/s；all_phases_sampled=false。不能用这些数宣称全程稳定性优于FSM。

录像元数据：原capture全解码PASS、749unique帧/15fps、black0，749帧账本完整；媒体49.933333s，与物理49.9s差0.033333s为末4tick帧的显示量化。原容器duration仍为286331153s、container_valid=false；本轮仅说明问题，**未remux、未发布成功视频**。

依据：[实际source manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1310293487954Z_g7db0d17f398d_9f7a4dce1be54856b99818002f74a91d/source/semantic_video_source_manifest.json) 与同目录 `video_policy_decisions.jsonl`。仅一次完整决策账本扫描，随后定向读取最后262144字节核对末条命令；未扫描physical大文件或另读native原始流，native完整性以上述policy内逐tick证据为界。未改生产、四主报告、官方pointer、checkpoint或训练进程。
