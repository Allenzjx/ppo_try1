# FL hip−1 / knee+1：隔离候选，未运行物理

仅新增候选`direction_probe_FL_hip_minus1_knee_plus1_candidate.py`与测试`test_direction_probe_FL_hip_minus1_knee_plus1_candidate.py`。旧三份harness字节不变；`git diff --stat -- src configs`为空。没有改生产、全局CASES、配置、HISTORY、reward或策略参数。

入口复用原真实谓词：P05 source endpoint + FL已cross + AIR + within_top_xy + 未placed + evaluator valid且无终止；不是固定tick。锚点直接读实际previous ACK的`independent_policy_residual_requested_full12`，并要求与live `previous_residual_full12`全12维完全一致；不是最终drive或nominal。随后只选索引0/1，锚点hip−1°、knee+1°，不是每拍累加。

继承原`FiniteDirection.start/apply/complete`：12-decision quintic ramp；entry累计75（包含12-ramp、最多63纯hold），或capture-age+24先触发release；release12、follow30。其余10维raw每拍来自同状态在线det baseline、逐样本保持不变；H真实连续。跨阶段caps改变不重设物理锚点，既有容量拒绝、正式backend/mapper/物理硬限不变。双通道只是一项待测几何假设，不是名义角模仿或永久bias。

新receipt使用真实case `FL_hip_minus1_knee_plus1`、selector `(1,1,0…0)`；拒绝第3通道变化、少一个选中通道、错case/offset或注入raw与所报物理REQUEST不符。保留原372观测、六组H、原策略request、pre-action先flush/hash及实际ACK/后继执行审计。control分支明确channels=()、offsets={}、live_channels=12、anchor_source=null、无干预；selector全0。

runner保留旧已审结构，避免通过修改旧模块globals来替换回调。AST测试限制差异为显式case/类名/manifest、双通道receipt校验与入口ACK=H核验；正常backend、loader、单进程锁、物理observer、异常封存、learned-state unchanged检查全部继承。所有人工诊断的PPO decisions/updates/optimizer steps为0；training_data_eligible=false、auxiliary_loss=0，不自动生成capture或保持标签。

## CPU验证

**48/48 PASS**：新17项 + 原31项回归；CUDA_VISIBLE_DEVICES=-1，测试断言没有torch/Isaac导入。涵盖真实锚点来源、固定锚点非递减、其他10维在线、时序、跨阶段caps、两通道容量/非有限拒绝、trigger正反例、receipt/hash/H绑定、mask语义、control标签及旧文件哈希。

单条真实证据只读`runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/FL_minus6_CP189952_20260921_01/probe_entry.json`，未扫raw：entry tick2976的ACK独立REQUEST与live H全12维完全相等，前两维 **+2.4404078965400906 / −8.14630390733378°**，且不等于drive target。该旧record仍标FL_minus6，没有重标成双通道物理结果。

可复核CPU命令（在本outputs目录执行；不调用main）：

```powershell
$taskPriorCuda = $env:CUDA_VISIBLE_DEVICES
try {
    $env:CUDA_VISIBLE_DEVICES = '-1'
    & C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe -m unittest -v test_direction_probe_FL_hip_minus1_knee_plus1_candidate test_direction_probe_FL_minus6_candidate test_direction_probe_preaction_candidate
} finally {
    $env:CUDA_VISIBLE_DEVICES = $taskPriorCuda
}
```

未来物理CLI只供主代理审阅，**本轮没有执行**：新候选接受`--case FL_hip_minus1_knee_plus1 --checkpoint <审定CP绝对路径> --run-dir <全新目录> --max-seconds <15至85> --expected-head <当时HEAD> --enable-physical-diagnostic`。没有显式enable、正确checkpoint/hash和精确当前runtime/六配置绑定时拒绝，不自动迁移或从旧checkpoint覆盖最新权重。具体checkpoint、物理资源与启动时机由主代理在现行课程自然结束后决定，训练无需等待本候选。

最终SHA256：

- candidate：`86a890b2b73029d432dede04356edf70ac571bc98522138e35dbaf4f84119a17`
- tests：`9d6219b28e437f84a86731bc1c3072b8cfc586bf96e2ef1bcfe065f2fe9dfdce`
- 旧真实probe entry：`c5f72507857d551982c8714ef46e5ed558ad1f315ffdfb821a53323c38fdcfb5`

CPU helper已退出，无后台分析进程；未启动Isaac、优化器或新物理试验。
