# 当前交付入口：CP225280前段保留分支已恢复FL→P06

最新已封存正式视频属于独立 `cp225280_front_preserved_v1` 分支的 **CP227328**，不是下方历史草稿中的CP230144。保存重载后从自然P01执行92.433333s，FR/FL实际placed、43.6s FL→P06保留，RR有效抬升/越沿但未捕获，RL未完成；后腿补全器OFF、原FL辅助ON、已声明owner投影ON。保护分支实际新增2048真实决策/4PPO/80Adam，当前没有活动训练或评估。

当前三项简明回答与完整计账见 [CP227328_URGENT_STEER_DELIVERY.md](CP227328_URGENT_STEER_DELIVERY.md)，正式三片视频与receipt在 `video_review/CP227328_deterministic_front_preserved_v1_review/`；准确续作点见 [RECOVERY.md](RECOVERY.md)。CP225792/CP226304视频、原成功N、CP225280与旧分支CP231680等模型/数据全部保留。下面内容仅为原旧分支的历史交付快照，不代表当前最新视频或当前选择的稳定起点。

## 历史草稿：CP231168训练封存、CP230144正式视频P05未完成

**状态：最新训练模型CP231168已保存重载，但尚未正式确定性评估；最后正式自然P01 DET及已QA三片视频仍属于CP230144。本稿不是完整越障成功或稳定性优于FSM的声明。** 当前完整模型为CP231168 / PPO1762 / Adam35240，runtime `59e868f3e223c589e7645a0f5d63f91fa6119fb6`。本轮从CP225280起真实新增 **5888 policy decisions /37 PPO /740 Adam**；另有 **AUX32 accepted /32 attempted**，独立计账。

最新CP230144正式自然P01 DET重试 `20260924T1206270979619Z_g59e868f3e223_6daab84a9021442ca9d5a14aea51c95e` 已封存：**56.933333s／854 decisions／P05未完成**，FR placed、FL/RR/RL未placed，`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，物理判定器无安全失败。新full/detail/N对照位于 `video_review/CP230144_deterministic_collection512_review/`：

- `CP230144_DET_full_RL_completion_attempt_INCOMPLETE.mp4`
- `CP230144_DET_FL_to_FR_RL_recovery_detail_INCOMPLETE.mp4`：真实内容标记 `RR_NOT_REACHED_PREDECESSOR_FAILURE`，不是RR/RL进展。
- `N_vs_CP230144_DET_same_camera.mp4`：历史N，不是新同构B；已结束画面明确标为冻结，不计新物理时间。

full/detail均854帧／56.933334s，pair为1108帧／73.866667s；全量decode通过、PTS连续单调、black-like帧0。主任务已目视末帧和对照末帧。绑定信息见同目录 `export_receipt.json`。当前独立FR knee方向诊断另存；它不修改权重、不计PPO/AUX，不是这些正式视频的一部分。

首次录像尝试 `20260924T1200016881929Z_g59e868f3e223_682c728eb6c54cdfa895ae0c3a080c98` 因 `VIDEO_OR_ARTIFACT_ERROR: viewport callback_count=0 after 3 waits, expected 1` 中止：物理执行936 ticks／7.8s、发出117个决策／录像请求，但实际完成视频只有116帧／7.733334s。原视频、日志与manifest保留。这是录像回调／产物错误，**不是物理任务失败**；与上述完整重试结果分开。

最新FL链：qualified2048（17.066667s）；净空峰值2170为+17.050mm；2602（21.683333s）尚在前缘前142.710mm时首次跌回台面以下；3196（26.633333s）真实GROUND撤销资格，此后无新qualified。末帧虽AIR0N，但active_attempt=false、gap−51.185mm，不是新抬升。P05的600决策辅助全WAIT/未初始化，XY/TOP各0，故不是辅助下降预算耗尽。source stop在post-step3120出现，实际dispatch3121四轮native为0；policy轮残差继续存在，不是stop漏发。详见`CP230144_DET_P05_failure.md/json`。

上一条完整正式自然P01 DET为CP229120/72e：57.866667s停在P05，`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；该视频身份与失败保留。成功N_ref和原FSM未覆盖。

## 实际训练与物理覆盖

| 已封存课程 | 新决策 | PPO / Adam | nominal前缀（全部零credit） |
|---|---:|---:|---:|
|f6d P07后缀|768|6 /120|2580|
|72e P10后缀|512|4 /80|3068|
|72e P12+8后缀|1024|8 /160|2331|
|72e自然P01|1536|12 /240|0|
|65a AUX32后P07后缀|512|4 /80|645|
|59e P12+8，512 collector|512|1 /20|1554|
|59e自然P01，CP230144起|1024|2 /40|0|
|合计|**5888**|**37 /740**|**10178，零credit**|

按实际请求阶段计数：P01=6、P02=777、P03=8、P04=11、P05=1007、P06=458、P07=6、P08=6、P09=704、P10=12、P11=16、P12=2877、P13=0。普通阶段转换未作为done；教师前缀未混入上述数量。

新增59e自然P01的1024决策仅覆盖P01=4、P02=485、P03=4、P04=1、P05=530。首回合243决策在16.166667s/P02因局部有界恢复耗尽而未完成，物理VALID；第二回合781决策在52.066667s/P05是非terminal预算尾并bootstrap，不是第二次已终止失败。后者FR在tick1997真实placed，FL16次新资格全部落地撤销，无FL/RR/RL cross/place；没有后腿物理窗口或教师前缀。详见`59e_natural_P01_1024_coverage.md/json`。这些是随机训练轨迹，不是CP231168正式DET评估或已证明的学习增益。

| 同口径物理决策端点窗口（非互斥） | 本轮总数 |
|---|---:|
|RR合法可落脚AIR协同准备|372|
|RR当前真实承重＋前腿准备|481|
|固定FR轴CoM/body正投影且RL载荷比例下降|145|
|当前qualified RL边缘恢复|17|
|qualified RL AIR到达顶部捕获区|0|
|RL合法TOP承重／placed／P13／全任务完成|0／0／0／0|

第三类是诊断代理，不等于侧向FR转移或绝对力卸载；部分短窗包含前缀运动。各计数不是成功次数或持续秒数。

**资格和捕获归属：** P10/P12前缀中的RR placed6133，以及P12前缀中的RL qualified6212，均不能称为网络新成果。整个已封存训练中RL共有14段learner新资格、129个相应qualified端点；另23个端点继承前缀资格，总152。RL均未cross/place。f6d P07首回合RR qualified5401/cross5618/placed5850发生在交接5160之后；65a P07则为5432/6185/6270，亦属learner真实放置，但后来RR/FL支撑丢失。它们是后缀训练证据，不是自然P01确定性成功，也不是孤立证明AUX有效。

自然P01训练1536中RR出现两次短资格（5657→5702、10211→10301），未cross/place。72e P12地面后再离地有3个端点但没有TOP恢复；59e首回合RR落地后也没有合法TOP恢复。因此“临时接触恢复”“重新进入合法XY”“地面后重新抬升”和“稳定捕获”仍严格分开。

## 用户六问：封存事实与贡献边界

1. **相对成功N还缺什么？** 不是某个固定角度，而是保持RR/FL真实支撑、形成有效FR侧接收及RL连续越沿捕获。成功N可提供并发顺序和实际响应参考；本轮learner已出现RR放置及RL短卸载，但没有RL越沿。不能只把FR knee设为+30°当作解决。
2. **RL碰边是否直接终止或冻结，修了什么？** 没有发现“腿一碰边就done”的通用规则。edge-v4允许同次当前qualified、非GROUND的RL边缘接触继续恢复；掉地仍撤销资格。已发出的依赖RR支撑的FL/RL servo owner在掉载时按公开FINAL/request anchor有界暂停，独立恢复及显式wheel stop保留。它是控制接口改动，不是网络已经学会；机身碰障、纯轮爬升及硬安全未放宽。
3. **FL作用是否形成前右运动和RL卸载？** 部分learner窗口确有前送及RL实测力降到0，但CoM横向分量不总朝固定FR方向。例如59e 6816→6872 CoM=(+30.61,+13.09,−11.99)mm、RL14.608→0N，而FR方向y为负。不能将前送投影说成完成侧向转移，也不能单因果归给FL。
4. **RL knee正向／短退是否改善碰边并保住RR？** 成功N支持这条候选协同；本轮17个qualified edge端点证明恢复路径有实际覆盖，未证明可靠越沿。RR/FL仍会掉载，RL仍会落地。独立FL wheel方向probe只准备未执行，不能宣称其改善或新增AUX成果。
5. **无后腿实时补全的确定性policy到哪里，各贡献是什么？** 最新CP230144完整DET在56.933333s停P05：FR放置，FL资格因真实落地撤销，未进入后腿。原FL assist仍声明ON但本次P05始终WAIT，后腿自动下降/几何完成/强制前向assist OFF；公开owner projection不隐瞒。P07后缀RR放置是随机训练过程事实，不能冒充DET。控制接口修复、72e reward-only、独立AUX32及真实PPO更新分别记账，未把冻结模型换reward说成学会动作。
6. **最新模型、训练量与下一处真实阻碍？** CP231168是最新完整更新，尚未正式评估、不是已验证最佳；最后正式视频仍是CP230144。总5888决策／37 PPO／740 Adam，独立AUX32。最后P12后缀RL未越沿；最新1024自然P01训练的两回合分别止于P02未完成和P05非terminal预算尾。最后正式DET的首个未完成任务是FL保持净空前送到合法捕获区域并真实放置，不能用后缀P12覆盖替代验收。独立FR knee−4°/3s probe已封存为P05未完成：有限窗口FR确有响应，但FL净空更差；它是负面方向证据，新增PPO/AUX均0，不作成功标签（`CP230144_FRk_probe_SEALED.md/json`）。

## 具体改动与验证边界

- 49eb→f6d：可观察owner439、当前RL边缘恢复许可、已发owner有界暂停／投影、FL行程与RL空间proxy修正，以及零credit前缀schema接收。没有改资产、碰撞、重力或执行器能力。
- 72e：在现有retention份额内修RR已placed后GROUND/outside几何潜势；合法AIR/TOP旧值保留。共享观测task_progress_potential、actor、HISTORY、控制不变；terminal安全分支保留。不针对首次RR lift冒称有效。
- 65a：有限front-retention439 AUX身份和账本，32/32单独计数，保存重载；未部署teacher，不重置网络。
- 59e：仅显式当前实验collection512及其受约束publication/loader接线，γ=.9985、λ=.99、实际LR、网络、控制、reward不变。旧128 receipts原样保留；fresh512 buffer。512 collector/GAE定向CPU测试通过，真实块20 minibatch、每个raw/logp样本5曝、张量日志一致。

首个512块把第443样本真实terminal纳入同一rollout；RR GROUND组raw GAE250个全负，归一化后仍100正/150负，故不能只看归一化符号判断任务收益。新旧策略/critic不同，128与512对比不是独立因果消融；γ/λ不变，GAE半衰期仍约4秒。

最新训练checkpoint（**未正式评估**）：`outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000231168.pt`，实际SHA `a859f84973e3fced3c79b03af7042bedfc68e96996da83b60791494bae2b39c2`。sidecar SHA `fd6dc56a0968ffdc333df89d3175d7809fc7d81494318f6476ec08029e8572b0`，`save_load_round_trip=true`。封存指针和sidecar匹配；原CP230144及其正式视频身份保留，没有用旧视频冒充新权重。

汇总复用本目录七份sealed coverage、`first_owner_episode118_readonly.md`、`59e_collection512_vs72e_sealed.json`、`DELIVERY_DRAFT.md`、`RECOVERY.md`，并对CP230144作有限P05窗口核查。CP230144三片视频路径与QA已由artifact流程封存，但不代表CP231168。本次仅stdlib核验最新封存1024决策，没有重扫其他训练历史、读取下一活跃课程或运行Torch/PXR。
