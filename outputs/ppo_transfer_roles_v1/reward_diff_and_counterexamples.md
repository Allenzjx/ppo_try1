# 角色转移期间的最小奖励变更

状态：已实现、针对CPU回归通过；未运行Isaac、未提交、未宣称物理改善。此次只编辑 `src/wlr50_clean/ppo/semantic_reward.py`、`configs/ppo_semantic_v3/reward_config.yaml`，新增 `tests/unit/test_semantic_transfer_role_reward.py`。角色监督器、nominal、策略、迁移和历史记录不属于本次修改。

## 精确差异

仅当当前任务明确携带 `transfer_roles_version='diagonal_transfer_roles_v1'`，奖励器消费监督器提供的实测角色活动系数 `physical_transfer_fraction=f∈[0,1]`。缺失、非有限或越界会拒绝，不退回阶段标签；此处不依据目标低载自行判定转移活动。

复用已有 `transfer_attitude_weight=0.2`，故 `w=1−0.8f`：

| 成本分量 | 旧路径 | 明确新角色版本 |
|---|---|---|
| 姿态偏离 | w×attitude | 相同 |
| Euler角速 | rates | w×rates |
| 身体角加速度 | acceleration | w×acceleration |

f=1时三者保留20%成本；f=.5为60%；capture/settle活动系数回到0时恢复100%。每个实际120Hz tick单独加权并按dt积分，不以决策末值代替全间隔。不新增体姿硬阈值、阶段固定姿态或接收腿承载门。

奖励配置仅更新revision和说明，所有数值权重、工程尺度、gamma=.9985、return profile/lambda及五family顺序不变。未改rho或网络。无角色版本或其他版本的任务保留旧奖励结果。

这**不是修复原本不存在的通用AIR罚**。原实现已有全身actuation抬升、post-cross加载不丢历史信用、主动发力/被动反弹的保守区分，以及applied-only平滑；均予保留。新奖励器只信任上游版本化活动系数，不把其存在冒充动力学因果证明。

## 针对反例与结果

新增37项合成反例基于现有真实观测构建器及奖励数据类型：

- f=0/.25/.5/.75/1的三身体分量精确公式；除body_stability外各family与旧路径相等。
- 明确版本才生效；缺标记、旧/未知标记保持完整旧输出。明确版本缺f、NaN/Inf、负数或>1拒绝。
- 同实测活动仅改变phase标签不改变奖励；TRANSFER标签但f=0也恢复全部成本。
- 活动1→.5→0连续恢复，不reset历史；八tick不同活动值逐tick积分。
- 无接触向上运动无通用反弹罚；连续静止接触只增大力不产生大力奖励；已有主动command变化仍排除被动反弹分类。
- nominal/residual相互抵消而实际命令不变，平滑仍为0；残差幅度正则仍关闭。
- SUCCESS/FALL/BODY_COLLISION/INCOMPLETE及非终止的PBRS、终态事件、bootstrap标记与旧路径相同。

另运行38项既有观测/奖励/env及反弹、applied-only、失败边界回归。最终 **75 passed /0 failed /0 errors /0 skipped**，pytest输出1.27s，JUnit suite时间1.209s；证据：`C:/robotics_sim/wlr_robot/transfer_role_reward_cpu.xml`。首次命令因未设PYTHONPATH产生收集错误，设置仓库src路径后重跑全述范围通过；未修改测试预期来掩盖错误。

这些CPU结果证明代数、版本兼容和已有奖励边界，不证明角色系数的物理有效性或策略成功。新角色上下文的正确性由其专属观测/监督器测试及后续真实采样验证。未增加第六family、大力奖、非零动作奖、历史模仿项或优化器成功门禁。Python测试进程已退出，文件写入结束。
