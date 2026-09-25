# CP226048 视频封存后的有界核对计划

当前视频仍在运行；本文件不要求中止、不改源码／配置、不加载模型。先按原输出链完成并封存自然 P01 完整视频，保留失败尾部和源 manifest。

已收到的只读运行观察不是最终结论：t66.53/P09 尚无当前 RR qualified lift；RR ground约3.03N、gap约−50.28mm、front约−248mm，FL AIR gap约181.6mm。P09 source clock80 等待 current free lift，owner17 全0，尚无 late owner干预。t65.93 最终目标含 RR hip约70.87°、FL wheel canonical约−1.0267rad/s。**重捕获 retention-only 修订对尚未首次 placed 的 RR 返回零 delta，不能修复或解释成已修复这段首次抬升问题。**

封存后只读本次 source、manifest 与已有 CP225280 formal natural-P01 对照；如使用独立方向探针，只准引用其干预前窗口，明确标诊断，不能混为正式评估。无需重扫 Recording／全部历史。

1. 固定本次 checkpoint/source/controller/确定性/FL assist与rear辅助标签、自然P01完整时长、真实终止原因；未到达 RR capture 或 RL的窗口明确为0／未覆盖，不用阶段标签替代。
2. 窗口一：P05真实接触到P06接续，核对前段是否仍保持。只列首次FL接触/placed、进入P06及接触是否保持。
3. 窗口二：P07开始到本次首次RR当前有效lift，若始终没有则截止真正终止。用物理事件对齐，不把不同模型相同秒数／P09标签当相同构型。已有旧参考的约58.29s RR qualification是对齐候选，不是当前成功证明。
4. 仅取窗口二的起点、首处分歧、峰值／终点几组相邻真实拍：FR/FL/RL当前支撑及RR载荷、gap/front、CoM固定接收方向与世界xyz；RR hip/knee raw mean/request、源N、mapper、最终target、actual；四轮canonical target/actual；owner17和source等待理由。与当前快照对应的原时间戳/单位保留。
5. 区分：source等待是在RR未取得物理条件后的结果，还是先有不当指令；实际all12与owner状态；policy反向／角度差、饱和、跟踪误差。owner全部inactive只排除该时段owner投影实际接管，不证明其它控制/反馈等价，更不单凭权重计数宣称退化原因。
6. 只有本回合真实取得RR lift/capture才追加后续窗口；否则第一未完成任务写为RR首次有效卸载／抬升（按终态具体证据），RR recapture、RL转移／放置不计覆盖。后续P10课程和自然P01训练的更新/样本量分别报告。

交付仅一张关键拍小表、当前视频与同镜头对照及明确首个未完成任务。先呈现真实新视频，不以旧CP视频顶替，不把即将应用的reward候选写成当前策略已经学到的改进。
