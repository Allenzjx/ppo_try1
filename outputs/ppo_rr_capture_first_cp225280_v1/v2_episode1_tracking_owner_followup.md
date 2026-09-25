# 更正：pending late 与 RR tracking 的真实关系

**旧 pending648 的 RR tracking 本来为空。** 旧v1按该事件状态核验（7992/8192/9432），mapped RR N为[−8.15,−39.05]°；不能表述成“旧RR跟踪一直保留，后来被新stop截断”。

|source状态 / 新v2端点|tracking|RR mappedN°|
|---|---|---|
|原sample647，late待发|空|旧pending −8.15/−39.05|
|原未消费late648–863|全8servo|由原full12组声明|
|新7992，carrier next661|RR hip/knee|−4.4/−35.3|
|新8192，carrier next861|RR hip/knee|−7.2407/−42.8|
|原stop864 / 新8200，next869|空|−8.4907/−44.05|

机制：新carrier滤掉FL/RL四个tracking名字，却继承了被延期full12组的FR/RR名字；FR被已有capture owner退休，RR获得原pending没有的反馈许可。原wheel-stop864本来就不声明servo tracking，所以后来回到空列表。当前P09的source-tracking eligibility不依赖endpoint是否为假；强制endpoint_issued=False不是空tracking原因。

确证的是**意外启动tracking，然后正常stop结束它**。新stop后knee mapper偏置为−6.25°，旧pending为−1.25°；但实际策略、机身和反馈输入也不同，不能把全部偏置差视为隔离因果结果。

最小gap后10312–10352的source/mappedN固定，policy knee REQUEST转负、FINAL回到−58°，同时FL失载/RL前壁接触；不能把该局部回升直接归因于17秒前的tracking切换。

本核对只用原纯标准库MotionExecutor和既有封存行，无Torch/PXR、无生产修改。JSON保留精确source样本及事件对齐旧/新行。
