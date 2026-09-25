# 59e首个512采集块：封存覆盖与信用传播

run `20260924T1121437392178Z_g59e868f3e223_63b8d42e71504d8197a9228aabc8625b` 已自然封存，生命周期SUCCEEDED仅指训练工作流成功，**不是越障成功**。

真实新增512决策／1 PPO（1760）／20 Adam；累计230144／1760／35200。从本轮225280起累计4864／35／700，另有独立AUX接受32／尝试32；此块新增AUX0。

CP `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000230144.pt`。
SHA256 `376ab4b6fe6f2a6a82bedd1190ba0814dc5e5b396127c48c34cf7b1d4a8265f1`；sidecar `30dc90ece977823762eb7f19b79b944f41b20e14c5231c45014ab58d17db3ba9`。两文件实际哈希与最新指针一致。

## 真实采样覆盖

两次连续successful_nominal前缀各777决策，共1554，PPO credit均0；均在P12 tick6216交接，无fallback P01。learner仅P12=512：首回合443真实terminal，第二回合69非terminal尾（tick6768 / 56.4s）。首回合81.291667s因LOCAL_BOUNDED_RECOVERY_EXHAUSTED结束，非安全／wheel-only失败；见首回合独立记录。

| 物理端点窗口 | 全512样本 |
|---|---:|
|RR合法TOP承重／实际前侧准备|111|
|RR可落脚AIR准备|74|
|RR当前GROUND|250|
|FL合法TOP／AIR|144／367|
|FR合法TOP／AIR|485／27|
|RL当前有效资格|35（前缀继承7，learner新事件28）|
|RL qualified边缘恢复／合法capture区／TOP承重／placed|0／0／0／0|
|正FR轴投影且RL比例下降|31，非侧向转移证明|

第一回合learner RL资格6302、6861均落地撤销；第二回合仅继承6212，6250落地撤销，无新资格。两次前缀RR cross/placed均6133，不计作learner新捕获。首回合RR在7756落地后无合法TOP恢复。

第二回合尾：FR TOP18.262N；FL AIR gap1.507mm；RR当前有效AIR、在XY内、gap47.086mm；RL GROUND27.418N、前缘前101.253mm。此非terminal尾正确bootstrap，不能称为失败或成功。

## 官方512收益／raw likelihood核验

CPU只读取saved rollout tensor，没有checkpoint模型加载或forward。20个minibatch；每个raw／old log-prob样本出现5次；actions/reward/old value/logp/normalized advantage与日志逐样本相同。当前条件Gaussian重算density最大误差3.082e−6。γ=.9985、λ=.99不变；value loss85.8765，mean KL.018906，clip fraction.221875。

| 首回合对比 | 旧72e、128采集 | 新59e、512采集 |
|---|---:|---:|
|真实terminal decision|452，在第4个rollout|443，在第1个rollout|
|RR GROUND样本|295|250|
|该组raw GAE均值|−4.5444|−10.1572|
|该组raw GAE正／负|207／88|0／250|
|该组normalized advantage正／负|100／195|100／150|
|首回合RL当前有效资格样本|28|31|
|qualified RL edge recovery|2|0|
|RL cross／placed|0／0|0／0|

新terminal：reward−43.061016、old V−10.295301、raw GAE−32.765717、normalized advantage−3.503326；done切断第二回合回报。长采集确实把这次失败纳入此前同一rollout；但γ/λ不变，GAE半衰期仍约4秒，并非30秒影响等权回传。

归一化会改变符号：新RR GROUND组raw GAE全负，仍有100个normalized advantage为正。不能仅看归一化符号宣称策略学会恢复或学会后退。旧新权重、critic及学习谱系不同，这不是隔离rollout长度的因果实验；官方旧128 receipt保持原样，没有离线拼成伪512。

完整逐窗口r/V/raw GAE/normalized advantage及20批次一致性结果：`59e_collection512_vs72e_sealed.json`。最新CP的自然P01确定性验收由主任务另行运行，不能使用这些教师前缀后缀样本冒充完整PPO成功。
