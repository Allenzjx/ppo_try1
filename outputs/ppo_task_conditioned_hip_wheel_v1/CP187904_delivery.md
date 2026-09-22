# CP187904：首轮固定模型视频与真实训练交付

这是首个2048新版本决策后的检查点，不是任务完成声明。后续2048自然P01真实训练已于2026-09-21 07:27 UTC启动，以下固定模型结果不会随新训练改名或覆盖。

## 完整原速视频

- [CP187904 deterministic，从P01完整尝试](CP187904_deterministic_review/CP187904_deterministic_P01_full.mp4)：50.825s，P05未完成，FL仍AIR/gap8.078mm。
- [同CP stochastic seed4101，从P01完整尝试](CP187904_stochastic_seed4101_review/CP187904_stochastic_P01_full_seed4101.mp4)：42.050s，完成FL捕获但保持不足，P09机身碰撞。
- [本轮同版本N与deterministic对比](CP187904_N_pair/N_vs_CP187904_deterministic.mp4)：左N完整成功73.808s；右完整失败尝试，结束后的定格明确标注，不增加物理证据。
- [本轮B=N+0完整成功](B_current_review/B_current_Nplus0_P01_full.mp4)：自然P01到P13受控停车，8857tick；与历史N_ref分别保留。

全部是真实Isaac单次episode、15fps/1x、没有teacher、切zero或人工诊断动作。已完整解码并检查首尾预览。B、det、stoch共用ee5a965源码、同六配置、scene4001和相机。两条C绑定同一CP和actor hash，评估0次优化。视频中的native wheel qd是电机轴转速，不是轮端平移或牵引力。现有视角保留全身，但远侧轮端仍有局部遮挡，不声称每拍所有接触均肉眼可见。

## 对用户四项物理建议的实际回答

1. **FR与RL hip**：独立有限RL hip−3°诊断在FR仍可用时使角速率RMS低约0.82%、body collider最低点高0.277mm，FR捕获慢0.067s；仅一对小变化。正式CP187904 det对当前B的完整FR捕获窗RMS低11.586%、峰值倾角低5.982%、body collider最低点高1.241mm，但捕获慢0.983s、FR AIR期间平均gap小约2.636mm。这是多通道闭环结果，不能全部归因于RL hip，或声称无速度代价/可重复收敛。stoch的body最低点反而低1.545mm。
2. **FL下放**：本轮有限hip−3°实测使gap变化，但没有达到可靠捕获；更早的−2°接触诊断不等于本轮学成。正式det hip残差已接近0、knee仍约−6.6°，最终FL悬空8mm。stoch真实捕获后至P07只有37/240 evaluator端点TOP承载，203/240 AIR，不能把历史placed当当前载荷。
3. **P06推进**：stoch实际body净前送295.279mm/15.933s；B为307.597mm/20.667s。四轮有真实运动，但没有接触功率分解证据证明主推进能量来自wheels；不把更早进入P07称为同距离速度改善。stoch保持更差、RMS更大、body空间更小，尚未证明无益linkage循环减少。新的rolling状态sigma减少无关腿部扰动，但未加入固定姿态/统一反向轮惩罚。
4. **RR空间与时机**：独立N诊断中准备期RL回收使已取得资格的RR重新落地；合格抬升后有限FL/RL回收则保持抬起并完成carry/放置，RR安装点最低高度+1.677mm、body最低几何点+3.963mm，放置慢0.183s。它是人工有标签诊断，非PPO收益。正式stoch在5001tick曾合格，5016当前资格已失效、5018回地、5046碰障；没有RR越沿或放置。FL AIR期间从未算作承载。

确定性仍未学会可靠FL落脚；随机探索的一次捕获不能替代这个目标。当前没有合格的“最佳完整成功PPO”，最新模型也不自动是最佳。

## 实际代码、训练与迁移

生产commit `ee5a9651591d20bea48be8ceba075fa66594cb36`：12个Python模块、2个wrapper、6个隔离配置及1个新定向测试。主要修改为actor/distribution的可观测状态sigma，training的实际梯度审计，migration/loader/CLI的兼容加载，reward/supervisor/observation及新task_quality的软接触保持和有限空间成本，video同分布评估。TaskEvaluator、NominalMotionProvider、SemanticControllerAdapter及mapper/物理/资产/硬限没有修改。

版本`task_conditioned_hip_wheel_sigma_v1`：372输入/12动作，rho=.9，容量及mean结构保留；sigma=exp(learned_log_sigma)*.25*B(state)/capacity，全部通道正值，训练采样/old-current likelihood/load/stoch共用真实内核。B表和状态门控详见[训练设计](TRAINING_DESIGN.md)。没有gSDE/新动作表示/额外外部负bias/辅助loss。

实际质量epsilon=.06，前段有效系数.015–.03/s，保守20mm机身-障碍几何不足成本最多.03/s；不奖励无限高度。P06复用原前腿保持potential份额，按真实后腿位置在RR准备前淡出；无永久双前腿接触条件。γ=.9985、λ=.99、Phi权重5、成功/非成功终态±40、时间成本.02/s保留。

从实际CP185856迁移，完整actor/critic/Adam/LR1e−5/Identity normalizer/RNG和历史计数保留；清空旧未完成rollout、重新真实reset采样。软potential的观测语义发生变化，不能声称同物理状态网络输入数值完全不变。前缀行为和诊断均0学习信用。

本首块实际新增**2048 decisions /16 PPO updates /320 optimizer steps**，最新固定评估CP187904/1433updates/28660steps，已保存并官方重载核验。P01–P13实际请求量依次为：

`4, 406, 8, 4, 549, 830, 2, 5, 240, 0, 0, 0, 0`

前缀2984decisions另列，未加入训练量。4个P09碰撞episode、5个非终态采样预算结束；22次普通阶段切换均非done。没有全程或后缀成功。实际系数/损失/更新链见[first2048摘要](first2048_training_summary.md)。新定向生产测试13/13通过；旧保护测试270/272，两个324维断言在未改生产同样失败（实际372维），未伪称全旧测试通过。

## P02四轮与P05失败层次

[逐轮证据](CP187904_P01_P03_four_wheel_evidence.md)：P02四轮N均+.3rad/s（canonical方向）；194个前送decision端点四轮最终target及实测角速度都非零，不是只有RR。四轮平均final为FL.32076/FR.35572/RL.30710/RR.17309，实际canonical为.19503/.35630/.37020/.25644rad/s。RR是被policy负残差减速；不是其他三轮被mask。FR AIR轮转不算牵引；FL跟踪差未被唯一归因于负载。

P05 det的15Hz目标周期属于未捕获时持续tracking的离散补偿周期；同版B真实捕获后正常退役tracking，没有重现该周期。[只读对照](B_vs_CP187904_P05_mapper_readonly.md)尚未证明单位/符号错误或唯一根因，未修改gain、HISTORY或N。P05期限终止的done/Phi/GAE链核对正确，冻结评估的失败reward不冒充学习信用。

详细同事件窗口数据、分母和不可排名的失败窗口见[本轮B比较](CP187904_current_B_comparison.md)。下一块连续2048正在正式采样，完成量以新receipt为准；计划量不写成已完成量。
