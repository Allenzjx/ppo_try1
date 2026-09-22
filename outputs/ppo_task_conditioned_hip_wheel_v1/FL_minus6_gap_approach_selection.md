# 真实FL AIR gap-approach候选索引（未训练）

来源：已封存`FL_minus6_CP189952_20260921_01`，CP189952。索引JSON：`FL_minus6_gap_approach_selection.json`；辅助提取脚本只读已有文件，输出写入outputs，无生产/训练/Isaac操作。

完整候选窗为原始decision **372–446**，入口tick2976、结束3576，共5 s：12 ramp +63 hold。75个pre/post全部保留，未挑选局部下降样本。标签范围仅`FL_AIR_gap_approach_not_capture`；选中动作通道仅FL hip=0，其余11通道不以零目标监督。诊断训练信用0，所有capture标签false。

## 每个真实样本的核验

75/75前后evaluator均valid、P05、FL历史已crossed且当前AIR/within_top_xy；完整600物理tick内FL无ground/obstacle接触、轮心均在真实障碍XY范围，未发生body collision。FR/obstacle及RR/RL/ground每8tick接触均pair_verified且active。所有8tick派发验证通过、无任何servo headroom clip。这里验证的是本次真实接触条件，不是为以后学习硬编码永久固定支撑组合。

每个终点gap数值上均低于入口及同期正式det，但不是每步都下降：

|分段|局部gap下降动作|局部gap增大动作|保留数量|
|---|---:|---:|---:|
|12 ramp|11|1|12|
|63 hold|28|35|63|

ramp index3（decision375，tick3000→3008）gap由17.674644增到17.767924 mm（+0.093279 mm），仍完整保留；hold的35个反向变化也全部保留。最早动作对同期det的优势仅0.032313 mm，这是数值符号证据，不是宣称每个动作都有稳健独立因果收益。入口36.760291→窗末9.969020 mm，窗内最小6.117986 mm；入口后的源N本来也在下放，因此全部入口gap变化不能归因于单hip。

`eligible_for_gap_approach=true`的75项表示**整个有限AIR接近窗口下的物理及出处条件符合**，不表示75个单步均正进展、捕获成功、策略已经保持或自动获得训练授权。整窗body collider最低z134.229827 mm，没有以碰撞为代价；没有FL接触就没有接触保持/捕获正标签。

## 原始数据绑定，没有重造观测

JSON提供run_manifest、probe_decisions、physical_observations、native_tick_audit、probe_entry的绝对路径与SHA256；每个样本提供原始pre/post一基行号、字节offset/length、receipt hash及actor input hash。核验了原372维有限obs转float32的1488字节与真实actor输入一致，SHA256一致；原6组H经原schema缩放/clip的float32编码逐值一致，previous REQUEST与真实ACK一致，previous applied与ACK final一致，raw issued与随后step_info相同。600对日志未改写，只建立索引。

字段位置在JSON的`field_locations`中，包括原observation、actor输入、真实H、raw issued、nominal、filtered REQUEST、effective及final。所有75项其他11维manual delta精确0；没有把历史状态替换为基线状态或清零H。

另保留同run干预前24个**非辅助标签**holdout：decision0–15、75、147、220–225，覆盖P01/P02/P03/P04。仅供比较actor mean/σ改变，不监督其它通道为0。0号显式标记原始reset观测：其启动ACK没有independent-policy字段，但原日志REQUEST历史确实为0，原372编码/字节照样核验；不是人工造零。其余均为真实持续H。

## 撤销与后继证据未删除

decision447–458 release、459–488 follow、489–599后继继续保留并绑定同一原文件，不作为上述辅助目标。release末gap16.847792 mm，follow末23.532518 mm，封存21.934067 mm；终态P05、AIR、bearing0、未placed、无物理termination。旧策略接回后未保持优势，不能伪称成功；也不能因此否认已实测的有限AIR接近进展。输出未启动任何辅助或PPO更新。
