# 有限 AUX 备用：旧成功短段的数据可用性（只读）

结论：**有可信的短段物理证据，但不能直接把这条旧控制轨迹蒸馏为当前 deferred 版本的成功示范。** 未实施 AUX、未生成 teacher/训练数据包，新计 PPO/AUX 均为0。fresh PPO 主线优先。

## 精确窗口

只选 episode1 的 **17 个决策**（旧 local173–189/global225453–225469），状态起点9360，动作终点9368–9496，总计1.1333秒；不把前172个长 AIR 决策标成功。

|片段|决策数|实际证据|跨版本限制|
|---|---:|---|---|
|9360→9432，下降|9|每个决策端点 gap 均下降：77.304→4.937mm；9432实际RR hip/knee=+6.939/−31.765°|旧source仍等待648；新carrier时钟/stop/公开等待状态已经不同，不能断言执行历史等价|
|9432→9440，首次触地|1|原生计数一致指向9436首TOP，9437 placed；9440承载12.072N、hold0.0333s|**9436同时放行旧P09 late组**；这行跨越控制差异，不能当新source的动作结果|
|9440→9496，保持|7|终点61个连续TOP原生样本、hold0.5s、承载13.261N；真实局部成功|旧late组参与全身运动；当前版本不发这组，不能直接复制保持轨迹|

TOP起点来自已核验原生观察器计数/hold记录，不是本次独立读取逐120Hz接触日志。最终load_fraction_valid=false；13.261N是真实力读数，不等于可靠归一化载荷比例。此成功发生在首次optimizer更新之前，local均值仍零；不是已学会的确定性行为、完整越障或视频。

## 数据可得性

17/17行均有有限的真实447维actor输入、12维原始Gaussian sample、conditional mean/std、匹配的old_logp、原REQUEST/FINAL及实测结果；selected raw=实际issued raw，12通道mask全1、HISTORY恰好一次，未发现RR capture-assist owner。old_logp范围10.285509–19.885571；连续密度logp可为正。

输入末8列与before_local的差异最大2.3842e−8，是float32记录舍入；不是缺少输入。448新增资格可从**前一条同episode物理记录**重建，本窗口17行均true（established、非GROUND、continuation）；不能仅追加1就认定新旧数据已兼容。当前TOP仍保留谱系而current_lift_valid=false的合法情况也保留。

## 为什么不能直接复用成功标签

旧late在9436将 N 的 [FLh,FLk,RLh,RLk,FLwheel] 从 [38.6,−13.4,28.2,0,+0.3] 改为 [−18.5,−31.4,15.4,19.4,−1.07]；新版本整段递延四个关节目标和FL反向脉冲。P10 RR knee、P11 FR hip虽仍合法，其旧实测效果与旧FL/RL支撑变化耦合；9448还包含原P10 knee +0.75°正常控制修正。不能把这些全归给policy。

可保留9行下降为候选方向证据，17行用于旧行为审计；不可直接拟合全17行“成功动作”，不可将FINAL当raw head、再次叠HISTORY，也不可把旧logp/旧样本加入新on-policy rollout。若fresh PPO不足，优先取得**当前source版本下**可持续下降→接触→保持的短段，再单独决定和计账有限AUX。

来源：[旧成功归因](first_stochastic_RR_capture_attribution_v2.md)、封存 train_first2048_ecf205e/decisions.jsonl；明细及每行证据见同名JSON。本报告不是可执行AUX数据包。
