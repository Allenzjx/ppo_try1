# 当前448成功短段：有限AUX备用审查（未实施）

结论：已有**当前控制/448接口兼容**的真实成功短段，可保留为有限AUX候选。
先完成正在运行的512/PPO并检查真实学习实效；已有CP227840固定控制DET失败证据。
不把每512后必须重跑完整DET设为AUX门槛；新CP未评估时必须明确标未评估。
此成功发生在update6之前，不是update6收益，离线分析也不等于新CP物理失败。
本审查只用标准库读取已确定的7,315,514字节，不追读live尾部；未导入Torch/PXR、
未训练、未改生产、未创建成功teacher训练集。

## 最短连续窗口与排除范围

|用途|真实范围|决策数|证据/限制|
|---|---|---:|---|
|严格最短“首次TOP动作+0.5s保持”|8248→8312，global227874–227881|8|入口gap仅2.784mm，不能单独代表接近任务|
|建议最短连续接近→接触→保持候选|8224→8312，global227871–227881|11|最后一次下降从15.739mm开始，接续到真实TOP/hold0.5s；全部11行无headroom裁剪|

11行中的前3条请求前仍在topXY外，实际入区是该动作链的一部分，不能改写为已在顶部。
全部同attempt资格有效；首次TOP action后真实bearing，最后12.6055N、
load_fraction_valid=true、61连续TOP样本/0.5s、非GROUND。源控制与当前版本一致：
P09 late保持未消费，新的P12卸载仍pending；P10合法knee动作允许执行。

不把998条冻结前缀纳入AUX/PPO信用。前30条保留为上下文/对照，其中有长期负边界裁剪
和8184→8224回升，不能因整回合后来成功而全标为好动作。它们仍是合法当前on-policy
数据，由原PPO照常使用；本建议**不删除或改写正在收集的rollout**。

**15.7mm入口的11行不足以证明59mm激活入口的确定性闭环能力。**
最短8行更局限，不能通过换成接触边缘入口来冒充自然P01或完整RR接近能力。

## 精确监督对象：实际conditional raw，不是FINAL

41/41 selected raw等于issued raw、old_logp匹配；原观测448已记录。
rho=0.9，原生产数学是：

    mu_cond(o) = 0.1 * (prior_head(o) + local_head(o)) + 0.9 * history_center(o)
    teacher_conditional_target = actual_issued_raw
    equivalent_local_head_target = (actual_issued_raw - 0.9 * history_center) / 0.1 - prior_head

条件均值代数复算误差≤1.11e−7，反解再组合raw误差≤5.56e−17。
当前样本的HISTORY来自真实输入，不用teacher虚拟历史替换，不再滤一次。
反解local目标会把随机创新放大10倍，例如首下降行RR目标约[-5.040,+4.501]，
而旧local head约[-.0154,-.0509]；因此不宜直接对大local目标做无约束MSE。
优先在实际conditional raw空间拟合，固定sigma用于尺度化/约束，不训练sigma。

下表按**请求前**是否AIR分组；首次TOP动作归AIR。raw无量纲：

|组/关节|selected范围|selected均值|old conditional μ范围|old μ均值|sample−μ均值|
|---|---|---:|---|---:|---:|
|AIR下降4条 hip|[-.506192,.070361]|-.108525|[.040501,.544304]|.207510|-.316035|
|AIR下降4条 knee|[-.056804,.374359]|.190862|[-.123786,.265466]|.018080|.172781|
|TOP保持7条 hip|[-.583902,1.574768]|.635942|[-.450585,1.045075]|.374081|.261861|
|TOP保持7条 knee|[.074439,.803091]|.389378|[.091253,.659385]|.321683|.067695|

成功并非“hip永远负/knee永远正”。TOP后hip有明显反向均值需求；不能只取接触前
负hip片段再推广到保持。其他10通道亦有真实联合动作，最大创新约2.55sigma，
不能声称RR两个通道单独充分。

归因：首次TOP时N仍[-6.9,-37.8]、mappedN[-8.15,-39.05]、generic RR bias=0。
接触后P10使N knee到-29.3再-27.2；8264有一次generic knee +0.75deg。
这仍兼容当前控制，但后段不能全称网络贡献；也不能把这部分N/偏置减掉后伪造raw标签。

## 若PPO均值仍失败：最小可审计AUX建议

1. 在当前512完整保存后，优先检查固定输入均值、GAE及实际投影是否改善。可结合已
   存在的CP227840真实DET缺口决定是否启用，不要求每个新CP先完整DET；若新CP未
   物理评估，明确标为未评估，不能以离线核对断言其物理失败。冻结本11行的原始
   source字节/行范围、SHA、观测/动作版本、CP与runtime；不拼旧v1成功段，不改reward、
   GAE/old logp，不把离线短段塞进PPO storage，不以拟合误差充当物理成功。
2. **仅local mean最终层RR6/7两行及对应bias**是合理的首个受限试验，不复制全12随机
   动作作为强制监督。保留全部12通道实际记录作联合归因；若两个RR行不足，不自动放开
   整网或全12。首先检查闭环失败是否来自teacher窗口外/其他支撑状态。
3. 固定trunk、std输出、其他10均值行、prior、critic、Identity及原PPO Adam全部状态。
   使用单独AUX优化器，只接受两个RR行的独立leaf参数副本；通过功能式前向计算原有
   conditional mean，结束才copy回那两行。不能把整块最终层交给另一个Adam再以零梯度
   冒充冻结，因为动量/weight decay可能改到其他行。原PPO Adam不step、不重建、不清空，
   原param IDs/顺序、moments、step、effective LR全部hash保持。
4. 冷边界预声明固定小LR和1…64步预算（默认8仅为备选，最终值由数值诊断封存），用conditional raw的尺度化
   Huber/MSE，仅RR两通道；同时限制固定参考输入上的条件均值偏移/KL，超限停止，
   不无限追逐单次样本。约束拒绝/失败也保留每次attempt及候选参数/指标，不能静默
   回退后将attempts计为0。冻结参数对同一观测的其他10均值/std必须逐值不变。
   成功保持段也纳入，防止只学会下降后持续同方向动作。拟合改善只是数值检查。
5. 单独记录AUX minibatch/step/所用行/前后哈希；PPO决策数、PPO update、PPO Adam步数
   不虚增，原prior/critic/Adam哈希相同。保存AUX优化器状态/recipe以供审计，但不把它
   装入原PPO optimizer。AUX后续原Adam动量属于旧PPO历史，明示保留而不冒称已重估。
6. 在合法空rollout边界发布**唯一新checkpoint**，保留父CP且文件名含AUX序号/新版本，
   不能以相同CP计数覆盖旧文件。AUX训练方法与receipt必须进入版本/manifest；既有
   save/load必须保留该谱系。若为此修改生产路由/配置，在新runtime显式迁移：
   校验父checkpoint/sidecar/embedded hashes与确切允许差异，保持448/控制/物理/
   HISTORY/分布语义不变；普通load仍精确匹配，不借用旧5f→1e10修复特例绕过校验。
7. 重载AUX候选，重新采集fresh on-policy rollout再做PPO；自然P01同一个学生评估，
   后腿无隐藏接管。短后缀/单次拟合不等于完整能力。若59mm入口仍不能到此窗口，
   该教学最多说明末段局部修正，不得据局部loss宣布RR已解决。

只建议这项受限试验作为备选，尚不足以保证达到目标；新PPO真实成功回报可能已经提供
有效学习信号，应先读完成后的GAE/固定输入均值再决定是否需要AUX。

完整41行代数/物理索引在current448_success_aux_readonly.json；
脚本analyze_current_success_aux_readonly.py为stdlib只读审查，不是训练器或teacher发布器。
