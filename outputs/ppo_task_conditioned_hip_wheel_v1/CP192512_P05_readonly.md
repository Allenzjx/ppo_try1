# CP192512 det：P05间隙改善，但仍未捕获

封存run `78f9ffd75d504f0e8447f44e61d134ce`结束于6220 tick/51.8333 s，`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。独立物理evaluator valid=true、termination=null、success=false；FL lift1778/cross2785/placed=null，末AIR、within_top_xy、gap16.259794 mm、bearing0 N。不是碰撞，也不是成功。

## 源入口与同拍实际链

P05首个实际源派发1753，源FL终值22.8/−13.4°第一次出现在**2921**，相差1168，与旧已验证源长度一致；没有重新搜索Recording或重放mapper。其前一个decision frame2920为推导的预动作入口，**不是直接记录的probe trigger**。第一完整动作端2928可与旧CP189952的2984对齐；两个结束点6220/6276同样差56tick，比较的是相同P05源龄区段。

数值顺序FL hip/knee，deg；三个点的源N均22.8/−13.4。

|节点|2928源端后完整动作|2944最小gap|6220终点|
|---|---:|---:|---:|
|真实同拍mapped baseline|21.550/−12.150|21.550/−12.150|22.221/−12.150|
|filtered REQUEST=effective|+1.740/−6.495|+1.729/−6.485|+1.888/−6.477|
|最终target|23.290/−18.645|23.279/−18.635|24.108/−18.627|
|实测actual|27.511/−18.823|23.618/−18.687|24.587/−18.626|
|FL gap mm|23.876|14.299|16.260|
|body collider最低世界z mm|136.910|137.017|136.999|

2928–6220的3293物理采样：hip REQUEST始终+1.706680至+1.888031°，负值0；h/k headroom clip0，REQUEST与effective最大差0。FL ground/obstacle接触均0。源下放使actual hip在入口有负速度（2928为−42.750°/s），**不等于PPO发出了负hip残差**。所有映射/派发核验通过；native command clock独立为episode+179。最后一次动作6216→6220只有4个物理tick，未丢弃这个真实终止尾。

## 对照旧CP189952：哪些变化确实可见

复用已保存`CP189952_P05_readonly.json`，没有再跑旧run或探针。相同源龄区段：

- FL gap最小/均值/末值由18.483/21.190/21.165 mm变成14.299/16.466/16.260 mm；末间隙减少4.905 mm，但仍未接触。
- hip REQUEST均值由+2.502489降至+1.800162°；终点由+2.578326降至+1.887838°。knee终点REQUEST由−8.063660变为−6.476714°。
- 终点actual hip由25.257739降至24.587313°（−.670426°），knee由−20.213970变为−18.626498°（+1.587473°）。两关节及全身真实响应都发生了改变，不能把gap差全归因于一个hip。
- 源N未变。终点mapped hip仅由22.199620变为22.220538°（+.020918°），knee mapped同为−12.15°；这是mapper/反馈基线，不是PPO的当拍修正。最终hip target差−.669570°可分为mapped+.020918与有效残差−.690488，不能把N→mapped也算进PPO作用。
- 新末body collider最低z136.999 mm，旧134.566 mm，反而高2.433 mm；本次末gap改善不是伴随更低的body collider。AABB保守分离下界新86.999 mm，不是精确mesh净空或base z。

## 网络与HISTORY的原始记录

仅使用该动作实际记录的无量纲网络/条件分布字段，未新跑network forward。终点hip：network base mean **+.105433859**，原观测previous raw H **+.105248466**，conditional/实际selected raw **+.105267003**，rho=.9；actor解码的上一filtered REQUEST为+1.887508°。knee对应−.276783019/−.276708722/−.276716143，上一REQUEST−6.476549°。入口2920→2928动作的hip base/H/conditional为+.095144518/+.097175427/+.096972331。

即网络自身与H都维持正hip方向，不能解释成“网络已有负向意图，单被H拦住”。同时conditional不是network base原值，source、mapper、H与当拍有效残差也不是同一量。该视频日志没有完整原372观测及全部6组H，JSON明确为null，只保留实际存在的network/raw/actor解码H字段，不从后状态重建前观测。

本结论只是新det更接近但仍未完成FL捕获；没有为有限AIR进展生成capture标签，没有优化、运行探针或触碰活动stoch。JSON保留实际源端2921、最小gap及末段所有同拍链和旧数据引用。
