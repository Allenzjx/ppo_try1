# 59e：自然P01的1024决策封存覆盖

本块从CP230144自然P01随机采样，seed1001，runtime `59e868f3e223c589e7645a0f5d63f91fa6119fb6`。**1024有效决策／2次完整PPO／40 Adam** 已保存重载到 **CP231168／PPO1762／Adam35240**。训练生命周期SUCCEEDED不表示任务成功。无教师前缀、零前缀credit、没有后腿物理窗口。

## 实际覆盖与终态

| 回合 | 全局决策／样本 | 终点 | 实际任务结果 |
|---|---|---|---|
|0|230145–230387／243|16.166667s，P02，terminal|INCOMPLETE_CONTROLLER_BLOCKED／LOCAL_BOUNDED_RECOVERY_EXHAUSTED；物理VALID、无安全终止；FR未越沿或放置|
|1|230388–231168／781|52.066667s，P05，**非terminal预算尾**|FR placed；FL/RR/RL未placed；bootstrap=true，不称第二次完成失败或成功|

实际request阶段：P01=4，P02=485，P03=4，P04=1，P05=530；P06–P13均0。RR可落脚AIR准备、RR承重前腿准备、固定FR轴运动＋RL载荷比例下降代理、qualified RL边缘恢复、RL AIR捕获区、RL合法TOP承重、RL placed、P13及完整成功这九个统一窗口均0。

第一回合FR新资格24→139、548→625、1283→1940，均因GROUND撤销，无cross/place。第二回合FR新资格tick21（0.175s），cross1981（16.508333s），placed1997（16.641667s），全部来自learner且没有教师前缀。FL有16次新资格：2011、2873、3023、3233、3446、3903、4099、4209、4466、4607、5147、5263、5625、5744、6054、6188；全部在越沿前落地撤销，末次6213（51.775s）。完整成对事件见JSON。RR/RL新资格、越沿和放置均0；早期RL的AIR或whole-body初始净空不是qualified lift。

| 回合1预算尾腿 | 实测接触／力 N | 台面gap mm | 前缘距离 mm | 当前资格 |
|---|---|---:|---:|---|
|FL|AIR／0|−49.515|−159.478|已撤销，active_attempt=false|
|FR|合法TOP／12.043|−0.330|+46.242|有效、已placed|
|RL|GROUND／13.813|−51.401|−686.715|false|
|RR|GROUND／2.534|−50.091|−657.398|false|

接触端点计数：FR合法TOP承重532、GROUND84、AIR340；FL GROUND615、AIR406、合法TOP承重0；RR GROUND1022、AIR2；RL GROUND759、AIR265但qualified为0。这些类别不强制穷尽所有边界接触模式。

首个未完成任务分别为回合0的FR前送越沿捕获、回合1的FL保持净空并前送捕获。**未进入后腿，不以P05反复短暂资格当作后腿进展；CP231168尚未正式确定性评估。** 最后正式视频仍绑定CP230144，56.933333s/P05未完成，不能替最新权重背书。

## 更新与checkpoint

两次完整512 rollout更新分别是1761/global230656、1762/global231168；各20 Adam，均有非零有限梯度且actor权重改变，LR=1e−5。没有新增AUX。普通phase变化不作为done；第243样本真实terminal，最后预算尾继续bootstrap。

- Checkpoint：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000231168.pt`
- 实际SHA256：`a859f84973e3fced3c79b03af7042bedfc68e96996da83b60791494bae2b39c2`
- Sidecar：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000231168_manifest.json`
- Sidecar SHA256：`fd6dc56a0968ffdc333df89d3175d7809fc7d81494318f6476ec08029e8572b0`
- sidecar `save_load_round_trip=true`；文件hash与sidecar匹配。
- 本轮从225280累计：**5888决策／37 PPO／740 Adam**；AUX32 accepted／32 attempted独立计账。

来源仅本sealed run的1024条完整decision端点、completed_episodes、optimizer更新记录、final manifest和checkpoint指针/文件hash：`C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260924T1257554297942Z_g59e868f3e223_8a6cf8fd89554916abdc8a6b9f6fe1ac`。只读stdlib核验；不加载模型、Torch/PXR或读取下一活跃课程。

