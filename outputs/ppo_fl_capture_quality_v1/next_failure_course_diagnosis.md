# block06 / block07：下一短课程的有限证据

结论：最早重复的是 FL 真捕获后 P06 的**接触／载荷重分配**，不是可单独认定的失败。它伴随真实 RR 减载和后续前送，可能正是合法准备；不能因 FL 暂不承载而否定任务进展。已确认的控制失败在后续 P09 FALL／机身碰撞。数据支持训练“捕获→连续前送／后腿准备→RR可完成入口”的连接段；尚不能确认一个共同的最早有害动作或单通道机械原因，不能称为新执行 bug。

来源仅封存 block06 `062710…92db59…`、block07 `065046…630e916…`，同 HEAD `3a50657`；共1536个已优化 learner decisions，不含 prefix 信用。只解析两份已有 decision audit；无 actor forward、GAE 重算、CUDA、Isaac或生产修改。配套 JSON 保留事件窗口的真实 Full12 各层数据，不复制全部原始日志。

## 重复的先后关系

tick 均为各 episode 的真实120Hz时钟。FL placed 是原事件记录；“首失支撑”是15Hz decision-end 最早观测，且从该点连续至少3个 decision 无 FL support，不冒称精确接触断开瞬间。

|块/episode|FL placed tick|P06首失支撑 tick；距placed|P06有FL支撑样本|最终已封存结果|
|---|---:|---|---:|---|
|06/0|2818|2880；0.517s|14/75|tick3495 P09 FALL|
|06/1|2806|2816；0.083s|21/226|tick5099 P09 BODY_COLLISION|
|07/0|3010|3048；0.317s|56/252|tick5239 P09 BODY_COLLISION|
|07/1|3236|3352；0.967s|74/235|tick5509 P09 BODY_COLLISION|

block07尾段还有一次真实 placed3634，3664首失支撑；截至3888仍P06、非终态，不能计失败或完整成功。block06尾段仍P05、未placed。四个终止 episode 的 RR 均未记录 cross/place；两腿已placed的历史不能替代当前承载。部分 P09 后段 FL 再接触，但这不抹去最终失败。

上述“失支撑”计数不是失败分数：06/0首P06至首失FL支撑，RR bearing从7.30降到0.258N；07/0从5.44降到1.86N；07/1从11.92降到0N。四条episode由首P06至首P09，RR向前缘净推进分别0.138/0.282/0.300/0.278m，body forward分别增加0.158/0.298/0.301/0.279m。首失FL支撑时body omega仅0.072–0.217rad/s且未失败，证据明确不支持“P06没有下游能力”。当前placed后势能使用 `.8+.2*current_capture_retention`，保留XY/低于top下界约束、允许上方AIR为后续准备，是既定任务语义，不据此建议新增固定两前腿承载奖或门禁。

## 策略／参考／真实响应分层

1. **新首拍 kernel 生效，不是旧的cap首拍放大。** 每个已到P06的首个 learner decision，实际 gate=10通道。06/0的FR knee：上一P05 raw=−0.283，首P06 center=−0.059、μ=−0.080、raw=−0.077；物理REQUEST由−6.606°到−8.660°。没有把旧raw直接乘112°。后续stage age>0按既定设计回到raw HISTORY。
2. **nominal没有恢复旧捕获入口。** 这几个早期失支撑窗口，N的FL hip/knee保持22.8/−13.4°，FR保持0/45.9°；capture owner明确hold current nominal、退休既有swing层。P06四轮N均0.3rad/s，最终四轮依各自residual形成差速，不是只一轮有target。post-mapper N随已有反馈状态可与N不同；JSON分别保留，未用独立重算N冒充当拍policy作用。
3. **持续偏置不是巨大raw饱和。** 两块全部1536×12实际raw中 `abs(tanh(raw))≥.95` 为0，最大abs(raw)=1.22425；P06最大片段σ为0.292(FL hip)，非早期旧块的多倍σ尖峰。因此这两块不支持再以“raw饱和”为由改温度。小到中等latent经现有较大cap仍可产生几十度REQUEST。
4. **06/0的早期物理链有实测响应。** 2832仍FL TOP/5.61N，2880 FL bearing=0；FL final hip/knee 30.116/−4.171→27.081/−3.411°，最后派发前actual为27.720/−2.702°。同时FR knee REQUEST −8.66→−21.85°，final36.66→23.47°，actual39.54→23.40°；2936只剩FR/RL支撑，FR knee REQUEST已−33.21°。动作确实影响了关节，不是仅ACK；但全身载荷耦合不能归因只FL或只FR。
5. **07/1说明不是固定方向的FL hip问题。** 3240捕获后，FL hip final到3336已0.695°，actual5.272°，此时REQUEST−19.536°。3352失FL及RR支撑时，FL raw−.334、μ−.361、当拍innovation+.028；其μ主要延续前拍center−.425（base_mean+.212只占.1）。这不是该拍突发极端采样，也不是mapper把nominal清零。FL离地后gap3.64→9.0mm，CoM z约.169→.170m；不能把这段称为先机身下沉。
6. **P09已发生全身控制失败，具体因果仍需保留边界。** 07/0首P09只有FR/RL支撑，FL gap51.77mm、FR knee final6.76°（native N46.59°、REQUEST−39.83°）；这些入口事实自身仍不等于失败。终止前FR knee raw−.741、μ−.678、innovation−.063，REQUEST−70.56°，native N31.1°、final−39.46°、actual−35.29°；CoM z .168→.141m并机身碰撞。07/1终止时FL仍无支撑且gap143.39mm，RR回GROUND。可确认带策略偏置的真实target/关节/身体变化与失败先后存在，不能仅据相关性断言取消某个通道即可消除碰撞，亦不能将全部失败化约为相同RR抬腿不足。

所有记录的same-tick mapping/setter校验通过，首失支撑点REQUEST与headroom-effective一致；这只排除了这些日志中的组合/派发矛盾，不证明理想跟踪。某些FR hip target与actual差可很大（例如06/0 tick2880约1.34°对15.65°），需要按实际负载理解，不可借ACK宣称物理跟踪完美。actual servo来自最后一次派发前已有tracking测量，腿接触/CoM是decision-end，二者相差至多末拍而非同一时间点；JSON保留各自tick。

## 下一短课程建议（等自然P01评估后选择）

优先保留自然P01；可继续一个短的真实checkpoint-prefix P05后段→P06及后续P09块，覆盖捕获前驱和捕获后最初1–2秒，并让后续实际越障结果影响准备动作，而不是只从RR已起腿的P09入口训练。目标是完成有效减载／前送后仍有可完成的RR入口，不是强制FL持续承载。当前正在进行的N-prefix P06是有用的另一种真实入口，但其身体／关节／载荷和raw HISTORY不同，不能替代上述连接段，也不能与此两块作严格单因素因果对照。P10样本用于后续任务覆盖，不能据后缀通过宣称完整成功。

短块检查实际bearing／支撑分配、P06前送进展、FR knee请求/实际响应及后续RR进展的联合结果，不固定膝角、固定FL承载或永久屏蔽wheel。不建议仅凭本比较改N、reward、cap、温度或物理标准；尚无同步数据矛盾证明新执行链损坏。两个块的源权重、优化过程、RNG与prefix offset均不同，因此不能据结果变化宣布新kernel更好／更坏。最终课程选择应以随后CP182528自然P01第一个未完成任务为主，这里只是已封存训练分布的补充证据。
