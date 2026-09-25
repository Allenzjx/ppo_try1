# RR-first 当前交付

## 最新已交付确定性视频：CP229376（未完成）

- 完整原片：`video_review/CP229376_gain10_aux64_DET_0d0f894_review/CP229376_DET_P01_to_RR_placement_INCOMPLETE.mp4`
- 同次RR节选：`video_review/CP229376_gain10_aux64_DET_0d0f894_review/CP229376_DET_RR_capture_detail_INCOMPLETE.mp4`
- 接受基线对比：`video_review/CP229376_gain10_aux64_DET_0d0f894_review/CP225280_vs_CP229376_DET_same_camera.mp4`

保存后重载同一组合策略，自然P01；冻结CP225280先验＋局部全12通道RR模块。
控制版本0d0f8948999222f9b8710b0eb913a9419537be83，448观测、一次HISTORY。
原FL hip-only辅助ON，后腿任务辅助/几何接管/强制wheel方向OFF。
历史有限AUX64单独计账；本评估新增PPO/AUX均0。

视频为正常速度、同一连续episode，真实91.166667s/10940tick/1368帧；编码91.2s。
保留完整失败尾段，RR节选为同run最后371帧；对比旧参考定格明确标注，不算新增物理稳定时间。
三片均完整解码并目视检查。对应`export_receipt.json`保留哈希和来源。

## 实际能力与第一处缺口

998拍捕获前raw/FINAL与接受的CP225280差值均0，覆盖FR/FL放置、P06及RR有效抬升/越沿。
RRqualified6995/cross7979，非只保护P01–P06。没有实时切教师或动作回放。

RR actual末尾相对入口：hip−4.368377°、knee+3.376893°；最大实际knee展开+3.656320°。
active最小gap47.140810mm，末51.159650mm/0N，完整10940行native记录RR TOP=0。
对比上一已评估CP228864，最小/末gap为55.230760/57.408820mm；有下降改进，仍没有捕获。
FL active最终与实测轮速仍反向，末−1.012844/−1.024421rad/s。没有用abs或mask掩盖。
第一未完成任务仍是RR真实顶部接触与同次有效承载保持；不具备转入RL专项的验收条件。
结果INCOMPLETE_CONTROLLER_BLOCKED，非仿真错误、成功或完整越障。

## 已完成真实学习

局部模块累计4096有效决策、8次PPO、160次Adam；冻结先验225280历史决策另列。
其中task-v2累计2048决策/4PPO。13963前缀决策不计PPO，AUX64不计PPO。
最近块新增512/1PPO/20Adam、1995前缀0信用；phase P09/P10/P11/P12=504/1/1/6，目标全为RR。
143样本机会有随机TOP/0.541667s保持/11.394116N；369样本为正确bootstrap的非终止预算尾。
随机成功发生在该次更新前，不冒充新checkpoint的确定性成功。

实际storage、Gaussian logp、GAE、20个minibatch各5次曝光核验通过，冻结prior不变。
485合法AIR同输入RR条件均值变化：hip−0.017138386/knee+0.008015373 raw，非实测角度。
本次冷迁移仅改变RR均值参数坐标，初始动作/σ保持、真实Adam矩映射、原LR继承；迁移学习计数0。
未改物理资产、源前驱、HISTORY、reward或关闭后腿学习通道。详细见UPDATE8_LEARNING_ATTRIBUTION.md。

## 恢复点和正在进行的工作

已保存/重载并有上述视频的checkpoint：
`checkpoints/history/checkpoint_CP229376_local004096_aux000064_lineage448_v2_g0d0f89489992.pt`
SHA `cd6d724af6901d03ee235f3b5d3160b4f9db86195d0cf423b3e1e159625368da`。

同版本连续2048有效RR样本/4个完整update已实际启动：
`runs/ppo_rr_capture_first_cp225280_v1/train_gain10_continuous2048_from_CP229376_0d0f894`。
每完整update保存，未终止的物理episode/历史连续继承。后续计数以实际保存为准，尚未取得的新更新不预先计账。
不中断正常episode，不热改rollout，不新增RL预算或后腿实时老师。

截至update9的新增完整保存：CP229888（local4608／PPO9／Adam180，AUX仍64），
本块实际新增512／1／20；尚未评估，不能用上面的CP229376视频代表它。
checkpoint SHA `e60c82d31267049a8f00ebb64c0e117e250c82485d4013355334ca9d20a518bf`。
同一第4episode跨update继续，剩余请求更新尚未计账。详见CURRENT_STATUS。

随后update10也已完整保存：CP230400（local5120／PPO10／Adam200，AUX64），
本连续块累计新增1024／2／40，尚未确定性评估。
checkpoint SHA `804d990df6c7aa21e0acfe427066b068aace35495924eb10cfa97f6f3b2e777c`。
第7episode跨update连续采样，当前唯一Isaac运行不间断。
实时状态及最新完整保存点见CURRENT_STATUS.md、RECOVERY.md和checkpoint_last_pointer.json。
