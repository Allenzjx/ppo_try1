# 封存视频的物理事件窗口提取

`video_event_windows_readonly.py` 是现有 `rr_probe_readonly.py` 的小入口适配器。它不创建仿真、不重放 evaluator、不修改任何生产文件，也不要求新增视频成功门禁。新自然 P01 的 deterministic C、stochastic C 和同版本 B 可在各自自然封存后分别运行。

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' `
  'outputs/ppo_task_conditioned_hip_wheel_v1/video_event_windows_readonly.py' `
  --run '<已封存视频的 source 目录>' `
  --output 'outputs/ppo_task_conditioned_hip_wheel_v1/<唯一名称>_event_windows.json'
```

输出采用 exclusive-create，不覆盖历史分析。必需文件是视频 source manifest、physical observations、stage transition evidence、video policy decisions；height diagnostics 可缺省并留下 null。已封存 direction probe 也可用于提取逻辑自检，不因此变成正式 PPO。

## 相同任务窗口

|窗口|起点|目标终点|
|---|---|---|
|FR 准备→捕获|自然 P01 tick 0|独立物理 FR placed|
|FR 抬起→越沿|该记录保存的 FR qualified lift|FR crossed|
|FL 越沿→捕获|FL crossed|FL placed|
|FL 捕获后保持|FL placed|P07 进入；后腿准备开始，不把其后有目的的 FL 打开自动判错|
|P06 后轮接近|P06 进入|P07 进入|
|RR 准备→放置|P07 进入，包含 P07/P08 准备|RR placed|
|RR 抬起→放置|该记录保存的 RR qualified lift|RR placed|

进入 P07 本身不能代替 RR 放置。目标未达到时，窗口保留到真实终点并标为 INCOMPLETE；起点未达到则 NOT_REACHED。不能用更短的失败窗口/停滞窗口的低 RMS 击败完整任务。

每窗少量主指标为真实时长、wrapped 相邻 120 Hz 欧拉 roll/pitch 导数 RMS、峰值合成倾角、body collider 最低世界 z、body/障碍 AABB 欧式保守分离下界、轮端净空、body/CoM/四轮中心位移。四轮轴角速度与目标另列，绝不把轮端平移或轮速均值称为牵引。当前 TOP bearing/support 使用已保存的 evaluator 端点，明确采样 tick，不伪称 120 Hz 连续载荷时长。

RMS 公式为 `sqrt(∫(roll_rate²+pitch_rate²)/2 dt / T)`，单位 rad/s；不再除以 T，不称为速度归一化。真实时长和净前送距离并列展示。当前脚本没有增加“稳定性改善”的自动结论。

body collider 最低点不是 base z，AABB 下界不是精确 mesh 净空。只读取已独立记录的 RR USD hip 安装点；其他 hip 若此日志未记录则 null，不推算。任一物理样本缺失/非有限，该窗物理汇总为 null + 原因；不补 0、不截掉安全终止的坏样本美化指标。原物理 success/termination、外部 budget stop 和 source/video acceptance error 分开保留。

权威终态来自 source manifest 的独立 physical evaluation，包含最后不足 8 tick 的真实 partial decision，不能用最后一个完整决策代替。比较前仍须核对 root 已保存的 checkpoint、运行配置和 sampling mode；历史 B 仅用于提取逻辑自检，不代替本轮同版本 B。

## 已完成自检

- `event_windows_selfcheck_success_B.json`：旧成功 B 的终态保留 8857 tick / 73.808333 s，而非最后完整决策 8856 tick。RR qualified→placed RMS 为 0.019444716773352126 rad/s，与原 RR 工具完全一致。
- `event_windows_selfcheck_incomplete_RRA.json`：旧有限 RL−3 诊断到 7200 tick，原 termination null、外部 budget stop true；正确保持 INCOMPLETE，RR placed 目标 tick 为 null。
- 两条记录干预前 FR、FL、P06 窗口物理指标完全一致。
- 7 项纯 CPU 测试通过，包括终态 partial tick、真实失败/未完成口径、缺失/NaN 不补零，以及恒定角速度窗口延长 10 倍时 RMS 不额外除时长。

注意完整 FR 0→1502 的 RMS 为 0.1178893992 rad/s；它与先前只统计 P01/P02 的质量项窗口并非同一窗口，不能混列或据此声称回归。
