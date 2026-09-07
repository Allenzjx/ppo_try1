# Critic 终止目标拟合：value clipping 与共享 adaptive Adam 的只读核验

2026-09-06。仅 PowerShell/本地源码/完成日志/标量数学，未运行 Python、Isaac、tensor load、测试或优化器。没有修改任何参数、生产或现有报告。此报告是 `learning_signal_horizon_readonly.md` 的独立后续，不选择修复，不设训练门禁。

## 固定范围

运行：`runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189`，runtime `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`。

只统计 **global≤60,080** 的 episode 0–5 终止 audit，以及同一范围内完整 optimizer records。六个 episode 均实际结束，不是新回合的计划；但 cutoff 不是128边界：范围内最后完整更新为 **434 / global60,032**，episode5的终止60,080位于下一批。本报告不读取/计入60,080之后的更新，也不称该尾段已在这个cutoff内完成优化。

读取实际 `checkpoint_step_000059776_manifest.json` 的 source_run/runner_config（未load tensor，未重复hash）：128 steps、`use_clipped_value_loss=true`、`clip_param=.2`、value_loss_coef=1、max_grad_norm=1、5epochs×4minibatches、Adam、adaptive KL=.01、normalization false/per-minibatch advantage normalization false。manifest配置初始LR=3e−5，**实际 optimizer_learning_rate=1e−5**；二者不同是调度状态，不是配置失配。

## 1. 六个真实终止样本的 value/目标误差

下表用每条audit实际old_value和float32 reward，不拿double环境reward替换已传给PPO的值。真终止无bootstrap，因此该槽return target=reward、TD=reward−old_value。

| Episode | 决策数 / 时间s | global / phase | 真终止原因 | old_value | 实际reward=terminal target | TD residual |
|---|---|---|---|---:|---:|---:|
| 0 | 939 / 62.6 | 55595 / P06 | INCOMPLETE_CONTROLLER_BLOCKED | −3.450804 | −42.274586 | −38.823782 |
| 1 | 677 / 45.075 | 56272 / P09 | BODY_COLLISION | −4.501774 | −42.445522 | −37.943748 |
| 2 | 931 / 62.066667 | 57203 / P06 | INCOMPLETE_CONTROLLER_BLOCKED | −4.025687 | −42.352142 | −38.326455 |
| 3 | 923 / 61.533333 | 58126 / P06 | INCOMPLETE_CONTROLLER_BLOCKED | −3.583601 | −42.236500 | −38.652899 |
| 4 | 935 / 62.333333 | 59061 / P06 | INCOMPLETE_CONTROLLER_BLOCKED | −3.903664 | −42.266937 | −38.363273 |
| 5 | 1019 / 67.933333 | 60080 / P09 | INCOMPLETE_CONTROLLER_BLOCKED | −4.937240 | −42.814350 | −37.877110 |

都包含−40失败事件及terminal phi归零/成本，不是“−40被错误放大”。Episode1为真正碰撞，且最后动作仅1/120s；不能把它当deadline。其余为阶段未完成终止。六行是不同真实状态、不同采样时刻/网络版本，**不是同一状态的纵向拟合曲线**；只能确认反复出现较大的on-sample终止预测误差。

## 2. Value clipping 实际公式及条件性平区

本地 RSL-RL5.0.1 `algorithms/ppo.py:304–309`：

`v_clip = v_old + clamp(v_new−v_old, −.2, +.2)`

`L_v = mean(max((v_new−return)^2, (v_clip−return)^2))`。

`.2` 是原始 value/return 单位的绝对差，不是20%目标误差，也不是把return限制在某个区间。它与policy ratio复用同一clip_param；value_loss_coef=1。value网络前向本身不clamp，rollout记录的old_value也没有被限制为[−.2,.2]。

以真实episode0 terminal的 `v_old=−3.450803756713867`、`y=−42.27458572387695` 为数值基点，只改变假设的更新后预测v，得到以下**公式反例，不是已记录minibatch轨迹**：

| 假设v_new | v_clip | unclipped平方误差 | clipped平方误差 | 对本样本v_new的梯度 |
|---|---:|---:|---:|---|
| −3.550804（向目标改进.1） | −3.550804 | 1499.531290 | 1499.531290 | clip内，两条路径导数相同，仍可改进 |
| −3.850804（向目标改进.4） | −3.650804 | 1476.387021 | 1491.796533 | max选择常数clipped项，严格内部梯度0 |
| −42.274586（恰到目标） | −3.650804 | 0 | 1491.796533 | 此样本仍选择常数clipped项，梯度0 |
| −3.050804（反向变差.4） | −3.250804 | 1538.505072 | 1522.855559 | max选择unclipped项，梯度约+78.447564 |

对这组固定old value和target，严格平区为约 `−80.898368 < v_new < −3.650804`；边界处的tie/导数约定不用于上述结论。正向拟合超出.2后该样本可以失去继续拉向远处target的梯度；反向恶化或严重过冲则不必平。这是 `max + clamp` 代码可证明的分支，不是所有critic梯度恒零。

### 不能推出“每轮只能改变.2”

- clamp只在**这个样本的loss分支**，不是参数投影或全局输出约束。一旦某步更新把预测推到平区，代码不会把真实v拉回old±.2。
- 同一个critic的其它样本共享参数；其它状态的梯度可以改变本状态预测。Adam已有moment也可能产生更新；当前batch的old_value在多个epoch内固定，而下个rollout重新取old_value。
- 128×1、4minibatches意味着每mini batch32样本；每个样本通常每epoch出现一次，共5次，不是该terminal独占20步拟合。更新后storage清空，不会自动持续重放同一terminal直到拟合完成。
- 因此不能据 `.2` 推出需要 `(42−3)/.2` 次更新的必然下界，也不能证明已经发生该平区：现日志没有保存每次minibatch重算的 `v_new−v_old` 和分支选择比例。

可证伪的表述是：若真实minibatch中该terminal的预测始终没有向target跨出.2，则这次拟合不足不能归因于“该样本已经进入clipped flat”；若确实跨出且clipped误差占优，则该分支会抑制其自身继续拟合梯度。当前只有代码机制与终止误差，缺实际分支占用数据。

## 3. Actor KL 如何约束同一 Adam 的 critic LR

官方 `ppo.py:115–118` 用 `chain(actor.parameters(), critic.parameters())` 构造**同一个Adam**。actor与critic是独立MLP、参数/Adam moments各自存在；“共享optimizer/LR”不等于共享神经网络权重。

`ppo.py:268–294` 每个minibatch根据**actor分布**的KL调整LR：KL>.02则除1.5且不低于1e−5；0<KL<.005则乘1.5且不高于1e−2；随后给该Adam的所有param groups统一赋值。判断不读取critic误差，所以critic即使有上述约38的terminal误差，也没有独立提高LR的路径。

`ppo.py:361–376` 总loss backward后，对actor、critic**分别**作max_grad_norm=1，再同一optimizer.step。不是actor大梯度挤占同一个联合norm预算；但共享LR仍受actor KL影响。梯度范数限制不是输出value变化的硬Lipschitz界，Adam更新也不能仅用 `1e−5×误差` 推算。

新MDP边界记录initial LR=3e−5、reset_all_moments。当前cutoff内42次完整更新（393–434，global54784–60032，共840 optimizer steps）：

- 第393更新末LR=6.75e−5；394–434的**41次更新末**均1e−5。
- 整块记录的平均KL范围 .013904703–.038417591；它不是每minibatch KL，不能用平均值倒推出所有20次调度分支。
- 所以“长期处于floor”有更新末证据；**没有证据证明全部840个minibatch LR均固定1e−5**，中间可以升高再降低。
- 1e−5非零，Adam/critic没有停止。相对某一相同optimizer状态下更高LR，它缩小参数步长尺度；实际value变化仍受网络Jacobian、moment、其它样本及clipping影响，不能由LR单独证明失败主因。

## 4. 真实terminal批次loss，不被普通小loss掩盖

已完成且在cutoff内覆盖episode0–4终止的批次：

| 包含的terminal episode | 完整update / global | 更新末LR | 已记录平均value_loss |
|---|---|---:|---:|
| 0 | 400 / 55680 | 1e−5 | 107.946147 |
| 1 | 405 / 56320 | 1e−5 | 103.928418 |
| 2 | 412 / 57216 | 1e−5 | 105.821385 |
| 3 | 420 / 58240 | 1e−5 | 85.981531 |
| 4 | 427 / 59136 | 1e−5 | 105.620197 |

相邻许多非terminal批次value_loss约.001–.02；所有42批最小 .000747659、最大107.946147。大的终止批次误差确实存在，不能只摘非terminal的小loss宣布价值已校准。反之，该平均loss由整批/多个epoch产生，不等于terminal单样本loss或clipped-flat占用率，无法单凭它给clipping定责。

episode5对应下一批的完成更新在60,080 cutoff外，故本表不读取或预填。

## 审查结论与边界

已确认三个事实同时存在：真实terminal value持续严重低估负结果幅度；value loss有“朝远目标改进超.2后可能对本样本平坦”的分支；critic LR由actor KL共同约束且当前41次更新末处在1e−5。三者构成具体、可验证的拟合限制候选，但**不是已经证明的单因果链**，更不能推出“每轮只能改.2”或“Adam卡死”。

后续若另行授权只读诊断，已有saved terminal observations及后续checkpoint可以检查同一终止前状态的critic预测是否改善；要判断实际clipping平区占用还需对应minibatch重算值/分支信息，现有摘要不足。此处只指出证据缺口，不新增启动门禁，也不要求当前训练停下补诊断。

不直接选择关闭value clipping、分离LR、改变gamma/lambda或任何其它修复。主规范的global PBRS/真终止吸收态不改，不增加阶段/入口/AIR奖励，不动entropy/std。仅新增本报告，写完停止。
