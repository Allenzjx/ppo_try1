# 已封存 τ=.5 训练块：实际分布

基于 `training_P01_temperature1024_actual.json`，封存后只做一次实际 decision audit扫描；不加载模型、不额外forward。CP171520→172544，1024决策/8 PPO updates。

| 实际阶段 | 样本 | effective σ最小 / 平均 / 最大 | derived learned σ平均 | pooled z平均 / population std |
|---|---:|---:|---:|---:|
| P01 | 6 | .04128 / .08873 / .15595 | .17747 | −.01738 / .87825 |
| P02 | 484 | .01819 / .08588 / .41122 | .17175 | .00479 / 1.00209 |
| P03 | 19 | .03109 / .10546 / .28029 | .21093 | −.03339 / .99143 |
| P04 | 19 | .02842 / .11264 / .29592 | .22527 | .07318 / .93655 |
| P05 | 496 | .01673 / .11405 / .85121 | .22809 | .01559 / 1.00408 |

P06–P13本块0样本。均值按这些阶段全部12通道的实际样本描述；不是同状态对比或独立正态性检验。

effective σ来自保存的实际 old_distribution_std_full12；learned σ**仅由effective/.5反推**，不是独立读取head，`exp(logσ+logτ)`带来的浮点差仍存在。z使用同决策 `(raw−recorded_mean)/effectiveσ`；不能把温度缩放当成网络学会稳定的证据。

实际终止2次：首episode第158决策，P02 HARD_JOINT_LIMIT，终tick1264/10.533333s，最后实际interval8tick；第二episode第617决策，P05 INCOMPLETE_CONTROLLER_BLOCKED，终tick4929，最后interval1tick。两者bootstrap均false、任务成功false。最后249决策的P05尾段非terminal，不记作完成episode。

1024个决策都验证all12 enabled及真实native audit；所有12通道raw均非零。P02四轮各484个endpoint的native policy target effect均非零；这只证明实际目标控制作用，不证明合理四轮协作、跟踪或任务成功。逐phase/通道完整统计见 `training_temperature1024_distribution.json`。
