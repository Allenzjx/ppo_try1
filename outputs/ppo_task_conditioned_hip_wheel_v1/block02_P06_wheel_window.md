# 正式训练 block 02：P06 四轮与前送只读窗口

结论：这个窗口不是“只有 RR 有轮指令/转动”。四轮源 N 和同次 mapper baseline 均保持 +0.300 rad/s，四轮最终目标与物理角速度均存在。实际机身前移 196.989 mm；但 FL 大部分时间悬空，连杆仍明显往复，尚不能凭这些共存观测证明轮驱是主要推进力。

## 范围与口径

- run：`runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0509408527373Z_gee5a9651591d_35cb60975d944752a3227f41558c2fca`。
- 固定 P06 窗口 tick 2800–4000，23.333–33.333 s；151 个 15 Hz 物理端点，150 个动作区间，1200 个逐物理步派发/几何记录。仅读取已完整写入的行，到 tick 4000 停止；未分析后续结果。
- global decisions 186530–186680；窗口内 update 1423 在 global 186624 后真实改变 actor。这是学习中的随机 rollout，不是固定 checkpoint 评估，也不是因果牵引实验。

## 四轮完整通路

单位 rad/s；表中“均值 [最小, 最大]”是 150 个相同动作区间末端。四轮 source/N/mapped baseline 都为 +0.300；residual mask 都为 1，作用于 residual，不乘掉 N。本窗口 REQUEST 与同次派发 effective residual 相同；并非使用独立重算 N 作差。

|轮|有效 residual|最终 canonical target|实测 canonical joint qd|target/actual 负向样本数|
|---|---|---|---|---|
|FL|+0.0757 [−0.4090,+0.6237]|+0.3757 [−0.1090,+0.9237]|+0.3745 [−0.1091,+0.9233]|8 / 8|
|FR|+0.1114 [−0.0764,+0.2298]|+0.4114 [+0.2236,+0.5298]|+0.3313 [+0.1129,+0.6237]|0 / 0|
|RL|+0.1308 [−0.1690,+0.3996]|+0.4308 [+0.1310,+0.6996]|+0.4760 [+0.1093,+0.7789]|0 / 0|
|RR|−0.1496 [−0.2138,−0.0522]|+0.1504 [+0.0862,+0.2478]|+0.1696 [+0.0309,+0.4299]|0 / 0|

首个保留端点 tick 2800 已有策略差异：FL/FR/RL/RR final 为 +0.67185/+0.43716/+0.41470/+0.15721，实际为 +0.67213/+0.46248/+0.42248/+0.24701。这不是首次 P06 分歧时间，只是本分析窗口首点。相对 N 的差异在 residual 层已有明确记录；不能归为 mask 丢失。RR 在整个窗口受负 residual 减速；FL 有 8 个采样端点被策略抵消到反转。四轮均无恰好零 target，只有 FL 的 2 个实测端点 |qd|<0.01。

canonical 顺序 FL/FR/RL/RR 对应 full12 索引 8/9/10/11；原生关节轴符号为 −/+ /−/+。原生 target buffer 与该符号映射的最大误差 2.93e−8 rad/s，符合 float32。1200/1200 派发 verified；150/150 最后 buffer 读回与 setter dispatch 一致；无 handoff hold、无 servo clipping。读取源为现有 `write_data_to_sim` 后的 `robot._joint_vel_target_sim`。这只是目标派发证明，**该 buffer 不是实测 qd**。

实测 canonical qd 来自 `robot.data.joint_vel[:, wheel_ids]` 经既定轴符号映射后的 `WheelObservation.velocity_rad_s`，由 evaluator 保存。原生未映射 actual qd 和数值 joint IDs 本日志未单独保存，JSON 明确为 null；不拿 native target 填充。FR/RL/RR 的端点实际与目标 RMS 差为 0.1121/0.0683/0.0474 rad/s，明显大于悬空 FL 的 0.0131，说明不能把 ACK 通过当作完美物理跟踪。

owner 证据：P06 source layer 在场，source endpoint 尚未发出，tail 状态 `finite_source_before_endpoint`，rolling retirement gain 恒为 1；final stop owner 未取得所有权。捕获的 FR/FL nominal servo owner 被保留，不等于阻止后续 residual。完整逐源事件 owner ledger 未落盘，保留 null，不臆造额外写入方。native command tick 与 episode post-step tick 相差 179，含 reset/settle 时钟偏移及先派发后读回，不把两种 tick 当成同一时间轴。

## 真实接触、轮端与机身运动

151 个端点中 FR 为 TOP 支撑 151/151；RL、RR 是真实 GROUND 支撑 151/151。`contact_surface=NONE` 对后轮表示没有障碍接触，**不是没有 ground 接触**。FL 只有 6 个 TOP/verified-bearing 端点、145 个 AIR；其历史 placed 在全部 151 点仍为真。采样到的 FL TOP 簇为 3576、3640–3648、3664–3680，不能外推成连续 120 Hz 接触时长。

四轮中心在世界前向分别移动 FL 189.317、FR 193.145、RL 196.117、RR 199.980 mm；这是轮端平移，不是轮轴转角。近前缘后轮中心距离由 −476.243 变为 −277.246 mm。机身前移 196.989 mm，真实质量加权 CoM 位移为 [+195.335,−5.409,+3.130] mm；CoM 运动不意味着 FL 悬空时承载。

120 Hz 机身 collider 最低世界 z 最小 113.751 mm；body/obstacle AABB 保守欧式分离下界最小 63.751 mm。两者不是 base z，也不是精确 mesh 净空；该窗口几何质量项代价为 0（均高于 20 mm 工程 margin）。实际 hip 安装点高度未记录，留 null。

## 连杆往复与可支持的结论

N 的 8 个关节建议整个窗口固定为 `[22.8,−13.4,0,45.9,6.9,0,0,0]` 度，但 residual 下实测关节有往复。示例 FL knee 范围 −28.46 至 −13.12°，15 Hz 采样总变差 97.11°、净变化 −1.59°；FR hip 总变差 74.43°、净变化 −0.36°；RR hip 42.31°、净 +2.59°。这些实际位置由保存的 measured joint margin 与已核验下限相加恢复，并与上限余量交叉校验，不是 target 冒充 actual。采样变差不等于精确路径、机械功或周期步态证明。

因此：前送确实发生，四轮控制链未丢失；仍应关注 FL 捕获保持不足和无明显净构型改善的关节往复。本窗口没有扭矩/机械功、完整 slip 或固定 checkpoint 的轮驱干预对照，**主动力占比尚未证明**，不因这些数据直接改 reward、冻结 wheel residual 或新增训练门禁。

完整逐通道数据：`block02_P06_wheel_window.json`；重现脚本：`training_p06_readonly.py`。本工作仅新建/修正 output 诊断文件，未修改生产、运行配置或活动训练。
