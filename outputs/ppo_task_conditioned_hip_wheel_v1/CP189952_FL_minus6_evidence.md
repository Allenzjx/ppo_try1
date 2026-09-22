# CP189952：FL−6确实缩小AIR间隙，但仍未接触

独立诊断`FL_minus6_CP189952_20260921_01`自然封存4800 tick/40 s，P05未捕获，physical termination=null、success=false、FL placed=false。600诊断动作，0 PPO decisions/updates、0训练信用、learned_state_unchanged=true。这是预算到达而非完整成功；没有持续接触正标签。比较对象为同CP −3诊断和正式det `a6d92def5d74491580cccd936a1edf18`，runtime contract及参数hash均一致。

## 输入与执行核验

直接`probe_entry`为2976/24.8 s，与−3相同；此前实测full12、base、FL h/k target/actual逐点相同，首差2977。锚点是前一ACK的hip REQUEST **+2.440408°**，−6平台为 **−3.559592°**，不是每拍再减6°。12-decision quintic ramp=2977–3072；entry累计75 decisions包含ramp，纯hold=3073–3576（63 decisions）；release3577–3672、follow3673–3912，之后实时策略继续至4800。不是全approach恒定−6°。

600对pre/post receipt hash对应；实际许可全1；其他11维manual raw delta严格0，仍来自同拍实时det策略。600个ramp/hold物理tick的hip REQUEST=effective，hip headroom clip=0，映射与最后派发核验通过。没有清零HISTORY或叠加手工knee动作。

|节点tick|同拍mapped N hip|hip REQUEST/effective|hip final /actual °|FL gap mm|
|---|---:|---:|---:|---:|
|3072 ramp末|22.643|−3.560|19.083 /19.200|6.301|
|3080最小gap|22.391|−3.560|18.831 /19.152|6.118|
|3576 hold末|22.193|−3.560|18.633 /19.119|9.969|
|3672 release末|24.067|−.903|23.165 /22.085|16.848|
|3912 follow末|23.479|+2.396|25.875 /25.298|23.533|
|4800封存|23.450|+2.546|25.996 /25.429|21.934|

源N hip/knee在这些节点为22.8/−13.4°，mapped是实际同次派发数据，不是独立重算nominal。所有h/k同拍链保留在JSON。撤销后接回的是受干预真实状态下的旧策略，不是强制恢复未干预轨迹；release末REQUEST仍−.903°，follow末才回到正修正。

## 同时间窗口：−6 /−3 /正式det

|窗口|gap最小值 mm|gap均值 mm|hip actual均值 °|
|---|---|---|---|
|ramp2977–3072|6.301 /12.644 /18.483|14.801 /18.052 /21.534|22.591 /24.074 /25.598|
|hold3073–3576|6.118 /12.274 /18.928|10.252 /15.863 /20.479|19.223 /22.224 /25.215|
|release3577–3672|10.021 /14.655 /20.435|12.070 /15.568 /21.071|20.031 /22.645 /25.263|
|follow3673–3912|16.923 /17.835 /20.783|21.746 /20.670 /21.977|24.382 /24.796 /25.280|
|之后3913–4800|20.286 /20.276 /19.478|21.655 /21.494 /20.874|25.287 /25.297 /25.275|

−6与−3的最小gap都在3080：6.118与12.274 mm，同tick正式det为19.220 mm。hold实测hip相比正式det低5.992°；不是请求未执行。**正作用是有限窗口gap接近；反面是仍留6.118 mm间隙、没有捕获，且释放后这一优势不持续。** 不线性外推下一幅度必成功。

三个run的每个上述窗口，FL ground/obstacle原始接触计数均0；没有可声称的FL承载或捕获保持。FR obstacle、RR/RL ground的pair contact在每个窗口全部物理tick都active且已验证；这仅描述接触组合，不能据此把每个接触都说成TOP bearing或证明稳定支撑质量。封存FL仍within_top_xy、AIR、bearing0 N，gap21.934 mm。

hold body collider最低世界z的均值为134.678/134.711/134.371 mm，最小134.230/134.099/133.934 mm；没有靠整机明显塌低获取这次gap下降。−6四髋安装点从USD真实frame、当前body pose读取且时钟对齐，hold均值FL218.312、FR212.743、RR197.691、RL203.269 mm。已封存−3报告的FL均值217.517、RR197.784 mm；正式det只有RR安装点记录，不能从base/CoM伪造其他髋对照。JSON中的三方窗口只重新聚合−6四髋，−3髋量可追溯原`CP189952_FL_minus3_evidence.json`，不是缺失值为零。

手工只改hip不等于静态单关节隔离：hold knee actual均值为−19.987/−20.123/−20.203°，其他在线策略、mapper、机身及接触反馈仍有响应。因此同期gap改善不能全部归因于纯hip几何，更不能宣称轮驱因果或正式PPO修复。

结论限定为：**−6具有实测局部AIR gap-approach进展，但没有真实FL接触/保持。** 释放后旧策略没保持这个状态，不自动否定有限AIR approach作为后续候选；也不能将其升级为成功capture标签或训练信用。本次诊断与正式全通道PPO、zero严格区分。仅输出离线报告/JSON，无生产或活动harness修改，无新仿真。
