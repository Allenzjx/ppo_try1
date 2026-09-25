# CP230144 FR knee −4°独立探针：已封存，候选未改善任务

Run：`runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe/CP230144_FRk_entry_minus4_v1`；HEAD59e、CP230144、seed4001、自然P01学生前缀，无nominal teacher前缀／快照。2026-09-24T12:52:36.072952Z封存，manifest状态`DIAGNOSTIC_SEALED`。主任务报告进程exit0只表示脚本正常完成清理和封存，**不表示物理试验成功**。

实际854个独立诊断决策／6832ticks／56.933333s：P01=2、P02=247、P03=4、P04=1、P05=600。P05 age40s，原生任务terminal为`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，非诊断预算截断。物理判定器VALID、无独立安全终止；仅FR placed，FL/RR/RL未placed。

## 权重和计数

- 正式loader已加载CP230144，封存前和finally均执行原有learned-state不变检查；manifest记录`frozen_learned_state_unchanged=true`，无integrity error。此只读复核不重新加载模型。
- checkpoint实际文件SHA再次核对为`376ab4b6fe6f2a6a82bedd1190ba0814dc5e5b396127c48c34cf7b1d4a8265f1`，与运行前绑定一致。actor/critic/optimizer/normalizer绑定哈希保存在同名JSON。
- **新增PPO sample/update/Adam=0/0/0，新增AUX accepted/attempted=0/0，teacher credit=0。** 854行逐条验证为零credit、未进入on-policy storage、`applied_log_probability_for_PPO=null`。原Gaussian density仅只读诊断，不给干预样本伪造PPO资格。
- 恰45个决策修改raw channel3，起始tick2096…2448；其他11通道均与实际干预HISTORY上的当前student raw相同，不是与不存在的无干预历史相同。正常3秒窗口到2456释放，非支撑丢失强制退出；没有后续自动teacher部署。

## 实际方向效果：复用已封存through2560窗口

原先REQUEST entry−4.369525°，hold时为−8.369525°；FINAL从entry40.947977°到36.947977°，确实完成相对entry−4°。由于未干预student同时自然变化，tick2336与正式基线的同拍FINAL差是−2.606811°，actual差−2.606830°，不能把同拍差写成−4°。源N及mapper N在五个配对端点相同。

| 同拍对比：probe减formal | tick2336（hold） | tick2560（释放后） |
|---|---:|---:|
|FR knee FINAL差(°)|−2.606811|−0.392555|
|FL净空差(mm)|−6.974|−1.803|
|CoM高度差(mm)|−2.892|−0.778|
|FR hip跟踪误差增量(°)|+3.213|约+0.734|
|FL前缘进度|约多前进1.189mm|反而多落后0.021mm|

through2560报告已核465个120Hz物理点（2096–2560）：FR始终合法TOP承重、RL始终实测承重，最低力分别11.814N／12.488N；不是因为入口支撑缺失使探针无效。干预确实到达实际关节，但降低FL净空／CoM并放大FR hip跟踪偏差，微小前进差没有形成捕获。**这个具体−4°候选不受证据支持，不作为AUX成功标签**；不据此否定所有FR调整方向。

## 最终物理结果

FL qualified2048／17.066667s；3195／26.625s真实GROUND撤销资格，未cross/place。末态FL GROUND1.461N、gap−50.940mm、前缘前106.285mm，active_attempt=false；FR TOP10.576N，RL/RR GROUND。与正式基线同为P05未完成，但不是同一run或正式policy评估，不互换视频身份。

证据：本run封存manifest（SHA`f61b6a485ef47276155ba7c127649e125bbabab38796e0b53638bb4d81385953`）、854条诊断记录零credit/通道检查、`CP230144_FRk_probe_through2560.md/json`。本次未重扫120Hz大文件，无Torch/PXR/Isaac／模型／生产修改；下一真实PPO块由主任务独立运行和计数。
