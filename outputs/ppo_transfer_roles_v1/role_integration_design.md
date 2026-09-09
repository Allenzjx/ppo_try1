# 四腿转移角色：最小接线设计（只读）

已完整读新附件；基于当前源码，不运行回放/Isaac，不改生产。以下是可实现方案，不是动力学证明。

## 已确认不足

`semantic_supervisor.py:638–673`：workspace只是目标轮前缘距离[-.22,.06]及横向ROI；support只是其它腿“力≥.2N”数量≥2；load_ready=clip((1−目标载荷)/(1−.2))。因此瞬时掉载/AIR可与受控卸载同得1，未检查接收侧或连续转移。725–739的新reciprocal只改善距离梯度，未补角色。400/521/533的载荷来自接触向量模长，不是竖直承载；墙接触也可能贡献计数。845–846的transfer_fraction仍是低载代理。

P05须placed_FL才进P06；P06层只在首次进入该phase时创建（spec P05/P06；supervisor:965–972）。所以P06轮行建议在P05不可用，存在调度依赖缺口；“实际FL是否必须靠它捕获”须由根的Recording证据确认，不能由源码单独证明。

## 角色与当前事实分离

|目标|接收侧|优先桥接/上下文|
|---|---|---|
|FR|RL|其余真实接触；RL调节|
|FL|RR|新FR接触可用性；RL连续回归|
|RR|FL|FR/RL；允许FL开空间时AIR|
|RL|FR|FL/RR；检查RR当前接触|

这些是偏好，不是固定支撑组合门。以stage.active_leg+Q/C/P历史派生`role_context`：target、receiver、preferred_bridge、observed_contacts、receiver_workspace、direction_context、pending_capture。首次FL越障/历史placed后重新开空间可区分；“有意开空间”只能是上下文，掉载不能自动算完成。历史placed绝不写回current support。

## 已有数据与空间代理

backend:174–188、330–332传同tick raw与task。sensor_reader:266–278、385–424及com_diagnostics:18–60已用13刚体COM位置/速度、default_mass算全身质量加权CoM，无base替代。可用mΔv诊断线动量变化；现raw缺惯量/完整接触力矩，不能声称得到全身角动量。

raw有upper/bot/wheel实测link位姿、q/qd、轮端与障碍几何。最小接收侧collector给出body-frame轮端/连杆位置、关节硬限余量、实测收缩/伸展趋势和侧向空间代理，并逐项标valid/source；关节余量不等于Cartesian可行域。现nominal_geometry:55–99仅P09/P12且排除GROUND，不能直接作为四腿准备collector，否则制造AIR前提；只复用其standing/sign/DOF映射校验，保持只读。

contact_classifier:179–185保留force_w、point及history，但切向仅模长；点可None。support hull少于三点返回valid=false（com_diagnostics:109），仅表示静态面不可算，不能否决两点动态交接或补造第三点。

## 最小短时证据

在Evaluator同tick observe中增独立transfer记录，沿用现0.5s窗口：target/receiver、参考tick、固定世界方向、c0/v0、receiver0、当前CoM/接收侧各自位移、载荷差、真实接触变化、body姿态/ω趋势、validity。方向在窗口开始由接收侧机身方向定义，不每帧追逐移动轮心；分别报告ΔCoM·dir与Δreceiver·dir，轮端独动不能冒充CoM转移。

结合受控减载/保持低载时的继续运动、真实支撑响应与接收空间形成连续progress；瞬时低载不独立完成。无接触不给反作用冲量；力峰不当转移成功。缺向量/点则降级代理，缺核心时钟/有限数据沿原失败路径。目标不变跨phase保留窗口；目标改变重设参考，reset清空，Q/C/P原逻辑独立不动。不要求接收腿先承载、摆腿先AIR或CoM先到位。

优先保持324、17-key、rho/gamma及actor不变：角色由stage/history可恢复，q/速度/接触/CoM已有；但324轮几何无y/连杆位姿，短窗锚点也不能由单帧精确恢复。新增详细代理先放task snapshot诊断；若用于替换准备进展，复用现phase_progress/全局Phi而不叠加奖励，并明确非完整Markov证明；不能静默改编码或藏新增硬锁存。

针对反例：历史FR置地但当前AIR；FR/RL两点；只移接收轮；无接触下击；RR自己少动但全身抬升；跨phase不断历史；有效异构姿态；P05捕获动作可执行但AIR仍no placed。无新增启动门禁。
