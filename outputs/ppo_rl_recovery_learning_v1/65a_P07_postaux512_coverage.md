# 65a P07 post-AUX：512 个真实优化样本

封存 run：`20260924T1046432416199Z_g65a9255be6d9_bff8c2f6d5574a42925e6b9d8f382a94`；runtime `65a9255be6d9`。
生命周期 `SUCCEEDED` 表示训练块完成，不是越障成功。**本块仍为128步rollout，+512决策 / +4 PPO(1756–1759) / +80 Adam**；不是正在准备的512步收集版本。

最新实际 CP229632：229632 / 1759 / 35180，LR=1e−5、save/load roundtrip=true。
checkpoint 与 sidecar 已独立读取校验SHA：
`74f6d2753e14f64b64af11321e10ece0c8658a0af4e34ffdfd60fe524f535762`；
`a882a6c3150dd180aa81255fcbffa0272914d267b0edbf972920c20e63028da7`。
本轮从225280起累计 **4352决策 / 34 PPO / 680 Adam**；另有独立 **AUX32 accepted /32 attempted**（本P07块新增AUX0）。

## 接管、事件与真实结果

连续 successful_nominal 前缀645个决策，在tick5160 /43.0s /P07交接，全部0 PPO credit；所有512学生行都明确无prefix数据入storage，没有fallbackP01。

| 事实 | tick / 仿真秒 | 归属 |
| --- | --- | --- |
| RR实测有效抬升资格 | 5432 /45.2667 | 学生接管后；在本块第一次优化之前 |
| RR越前缘 | 6185 /51.5417 | 学生 |
| RR实际顶部placed | 6270 /52.25 | 学生 |
| 首个决策端点RR失去TOP转AIR | 6328 /52.7333 | 此后未恢复合法TOP承载 |
| RR首次实际回GROUND、撤销当前资格 | 6543 /54.525 | 精确事件；首次端点6544 |
| 后续RR再次回GROUND | 7791 /64.925 | 历史placed保留，但不代表支撑 |

RR在6272–6320七个连续决策端点具备合法TOP承载，RR力约1.005–6.449N；FL同时提供实测支撑。随后RR AIR、FL也出现AIR，未把这段捕获保持为可持续的下一次转移支撑。真实捕获已经出现，**不能归因为prefix，也不能从单条后缀轨迹分离AUX与PPO各自因果贡献**；仍不是自然P01完整确定性成功。

RL全512行：GROUND511 / AIR1；当前合格卸载0、qualified edge recovery0、合法AIR落脚区0、cross0、TOP bearing0、placed0。历史只有tick7的前缀初始净空，不存在学生RL资格事件。第一项未完成后续任务仍是保住/恢复RR支撑并形成可用RL卸载条件。

## 样本覆盖（非互斥物理窗口）

| P07 | P08 | P09 | P10 | P11 | P12 | P13 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 137 | 1 | 2 | 370 | 0 |

RR可达AIR准备17；RR实际承载前腿准备7；RR当前qualified412；RR历史placed后GROUND34、当前qualified AIR289，**首次掉地后合法TOP重捕获0**。FR合法TOP承载512、FL315；RR placed后FL AIR129。
局部固定FR轴的正向CoM+body投影且RL载荷比例下降7行，仅为该投影代理，**不等于真实向右侧转移或RL已有效卸载**。计数是15Hz决策端点，非120Hz接触持续时长。

整块单一学生episode，terminal行0、completed_episode0；77.1333s /tick9256 /P12 age24.6667 为正常完整update的**非终止尾部**，允许bootstrap，没有任务失败或安全终止事件，也没有完成任务。

尾部当前：FR TOP13.944N；FL AIR0N/gap+12.478mm；RR当前合格AIR但落脚区外、gap−32.074mm/front−115.064mm；RL GROUND14.240N/front−215.022mm。RR与FL均不承担当前载荷，不能用历史placed冒充支撑。

没有该训练episode录像；该checkpoint尚无正式自然P01确定性评估。本报告只读上述封存run和最新指针，不启动仿真或加载模型。
