# CP230144续训：自然P01首回合已结束，P02未完成

仅核查run `20260924T1257554297942Z_g59e868f3e223_8a6cf8fd89554916abdc8a6b9f6fe1ac` 的首个completed episode与其243条完整残差端点，global230145–230387。不读取第二回合、120Hz大日志，不加载模型／helper／Torch/PXR，不改生产。

**真实结果：** seed1001的随机训练回合在tick1940／16.166667s／P02结束：`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。物理判定器VALID、无独立termination或安全reason，不是机身碰撞、wheel-only或硬限失败。真实terminal不bootstrap，`time_outs=false`，非外部截断。P01=2、P02=241，teacher/checkpoint前缀均无。

本报告记243个真实learner采样；首个512 collector块尚未在本报告中封存，**不预支新增PPO/Adam计数**。它也不是新的正式确定性视频结论。

## FR实际资格与当前状态

| FR真实qualification | 时间(s) | 真实GROUND撤销 | 时间(s) |
|---:|---:|---:|---:|
|24|0.200000|139|1.158333|
|548|4.566667|625|5.208333|
|1283|10.691667|1940|16.166667|

没有FR cross/placed，其他三腿也均未cross/placed。末拍FR已GROUND、实测1.245N、gap−50.003mm、前缘前50.447mm，`air=false / active_attempt=false / top_contact=false / within_top_xy=false`。历史event_ticks中的最早FR24不能代替当前有效lift；第三次资格在终拍精确撤销。

P02完成量 `lifted_FR=0, clear_FR=0, approach_FR=0.838210`。当前progress credit=0、current_eligible=false、continuation_allowed=false；`current_FR_AIR=false / same_air_qualified_lift=false / existing_lift_complete=false / existing_clear_complete=false`。已有最佳剩余距离15.884mm，当前剩余40.447mm，未达到捕获入口。

P02 age16.033333s，nominal limit15s、当前进度余量0.585497s、有效limit15.585497s；旧局部预算已超过且当前资格/credit丢失，所以记录有限恢复耗尽。不能将它改称传感器故障或以phase已P02冒充前送完成。

| 末拍腿 | 当前实测接触 | gap(mm) | 前缘距离(mm) |
|---|---|---:|---:|
|FR|GROUND1.245N|−50.003|−50.447|
|FL|GROUND13.749N|−50.710|−204.395|
|RR|GROUND13.921N|−50.520|−705.262|
|RL|AIR0N，current_lift_valid=false|+85.507|−790.414|

**第一未完成任务：FR保持有效抬升净空、继续前送进入合法捕获区域，随后越沿和放置。** 当前RL虽AIR但未取得当前有效资格、离前缘仍远，不能称为提前完成后腿任务。后续课程不在本次有界核查范围。
