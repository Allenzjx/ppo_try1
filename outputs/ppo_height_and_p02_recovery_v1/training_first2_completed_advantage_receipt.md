# 首两次真实 PPO 更新：GAE 可见性

固定截止 global168448 / update1281：新增256 policy decisions、2 PPO updates、40 optimizer steps。只读取两条完成更新、两条对应GAE审计及前256实际决策。未读取后续rollout、checkpoint或张量，不代表整个2048块完成，也不宣称168448已单独保存checkpoint。

实际样本：P01 4、P02 100、P03 5、P04 1、P05 146；P06–P13在本固定前缀为0。首决策从P01 tick0执行到8，没有教师roll-in。四次普通阶段切换tick32/832/872/880均done=false、bootstrap=true，未切断跨阶段回报。FR在P02有100个Q历史样本，其中92个具当前AIR、横向ROI及≥15mm净空；此桶不是完整functional-carry判定。P03/P04观察到FR crossing/placement，训练继续进入P05。

新审计显示真正的区别：

- 第一rollout：P02 rawGAE100/100负，均−0.767940；但全rollout标准化后50正/50负。不能说100个P02动作都按负优势更新。
- 第二rollout：P05 rawGAE128/128正，均+1.326482；即时reward均值却−0.007585。即时reward、oldV、rawGAE与标准化优势不能混用。
- 两条审计的rawGAE均值严格等于returns均值−oldV均值。全局及各phase标准化min/max/mean仿射关系误差≤1.44e−7；stored mean≈0，报告population std≈0.996086，符合128样本用torch无偏std归一化。未新增critic forward或独立重算未记录的last_values。
- 本前缀没有terminal；真实短终态验算尚无样本，不虚构通过。两个rollout尾部均非terminal，用原有官方bootstrap。

256决策的12个raw通道均实际非零；runtime mask全12为1。每通道endpoint direct-target-effect非零计数为[255,254,256,252,255,256,256,256,256,256,256,256]。全部2048实际physics ticks验证target链，均有真实native目标效应，无episode内姿态/力写入。own-phase-effect为2044而非2048是现有动作请求/阶段来源标记的范围，不是丢弃4个PPO样本。

实际更新1280/1281均actorchanged、finitegrad，LR1e−5；KL .016050/.017203，clip fraction .196875/.248438，value loss .591807/1.531075。未发现需要新增门禁的实现问题；这些是采样训练前缀，不是重新加载后的完整PPO评估。

详细证据见 training_first2_completed_advantage_receipt.json。按根代理要求等待完整块结束通知后才做全量聚合。

