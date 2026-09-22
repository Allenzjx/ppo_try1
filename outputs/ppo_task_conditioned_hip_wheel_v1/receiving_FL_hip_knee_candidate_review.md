# Receiving 双通道候选：仅准备，未物理运行

独立文件 `direction_probe_receiving_FL_hip_minus1_knee_plus1_candidate.py`；旧双通道/−6/−3脚本、共享helpers和生产 `src/scripts/configs` 未改。不是当前训练的门槛；仅供同一个新checkpoint完成det/stoch评估后，若P05仍失败，由主代理决定是否进行一次有限人工诊断。CP197248只用于本轮兼容性实证，不指定它为最终诊断checkpoint。

## 精确差异

- 新候选28行：只把 `POLICY` 绑定到官方 `RECEIVING_WHEEL_POLICY`。EXPERIMENT、六配置目录、372布局及branch lineage保持原值，未伪造新工程或旧profile别名。
- 31–58行：本地短metadata validator保留原路径/roundtrip/runtime精确相等/六配置/branch计数检查，另外要求完整 `receiving_wheel_policy_contract` 精确相等。旧profile、改动gate/actor、旧runtime或不匹配计数都拒绝；不提供migration参数或隐式迁移。
- 61–83行：FiniteFLHipKnee、trigger、ACK锚点函数与旧候选源码逐字相同。原有限状态机数值未改：只选hip−1°/knee+1°，其余10维同拍live baseline；12-ramp，entry累计75（包含ramp）或capture+24后release12/follow30；真实H/mapper/owner/硬限不清零、不替换。
- 86–171行receipt的原控制/372/H/hash/两通道绑定校验AST不变；仅新增人工动作policy likelihood=null、无PPO/AUX信用的明确字段。原策略request原样保存在同拍receipt中；人工issued raw不是策略采样，不能使用原baseline的概率充当其likelihood。
- 175行main的物理执行AST与旧候选相同；仅manifest schema及4项说明/来源hash字段变化。248行仍经官方loader，但传入新POLICY。natural P01、seed4001、唯一资源锁、异常封存、状态不变验证均保留。无生产适配。

新profile在P05的receiving gate不激活；同权重/同状态/H时det路径与父策略相同。**新训练后权重可能改变**，须用实际新CP同条件det作对照，不能把CP194560当新策略反事实。若后来进入P10–P12，receipt将原样保存官方receiving审计；det仍不抽样。

## CPU验证与封存

12项新portable测试PASS（0.057s）：旧文件hash、共享globals未变；3个控制函数源码相同；整段ramp/hold/捕获提前release/尾部逐步与旧实现相同；其他10维不动；容量拒绝；严格新metadata正反例；新request与两通道receipt/hash/零信用；receipt/main AST只含声明性差异。没有重跑旧历史物理、训练或模型forward。

另只读核验真实 `checkpoint_step_000197248.pt` 及其manifest：新profile、197248 decisions / 1506 PPO updates / 30120 optimizer steps、roundtrip=true；checkpoint hash正确；runtime与HEAD `649ccd906421d06c8b5c699f28101730910e885c` 精确一致。完整hash见同名seal JSON。此验证没有加载权重进行动作推理，更不是物理成功。

复跑CPU测试（在本outputs目录，运行后恢复原CUDA变量）：

```powershell
$taskPriorCuda=$env:CUDA_VISIBLE_DEVICES
try {
  $env:CUDA_VISIBLE_DEVICES='-1'
  & C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe -m unittest -v test_direction_probe_receiving_FL_hip_minus1_knee_plus1_candidate
} finally { $env:CUDA_VISIBLE_DEVICES=$taskPriorCuda }
```

未提供/执行物理启动命令；所有CPU helper已退出。原CP194560负hip已执行但gap约16.253mm的证据，仅支持有限耦合方向假设，不证明能捕获或保持，不发放正aux标签。
