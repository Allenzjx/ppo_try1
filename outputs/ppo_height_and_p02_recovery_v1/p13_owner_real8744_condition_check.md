# P13 owner：真实8744条件重放

只读取已封存 a9c RL−3 B 的8737–8745物理/native小窗口和其中唯一完整task endpoint8744；无全程扫描、模型/PhysX加载或生产修改。

使用真实8744 task evaluator、同tick observation、已派发且preceding的 N（轮全0），在最小provider状态fixture中直接调用新生产 `_observe_final_stop_owner`：

- active=true、current_evidence_fresh=true、current_entry_eligibility=true。
- current_four_leg_evidence_valid=true、top bearing supports=3、preceding-issued nominal stop=true。
- 真实FL为AIR、support=false、bearing_verified=true、bearing_force=0、load_fraction_valid=true、load_fraction=0；净空−0.340946 mm处于现有允许范围。没有虚构FL承载或改成合成正例。
- FR/RL/RR实测bearing为10.59648/14.06148/4.04140 N，均真实TOP支持。捕获当前非home servo N，tracking为空、normal bias全0，与该已派发记录一致。
- 输入task/observation前后完全未变。用同一真实输入仅改preceding N为滚动，或仅令observation时钟过期，两个CPU反例均正确拒绝。

没有schema mismatch。**直接实录重放证明最迟observation8744即可取得owner，在原dispatch8745恢复脉冲之前。** 8737是由真实post elapsed和120Hz物理数据推导的观察起点；该tick完整evaluator快照未保存，没有把8744 task前填冒充8737重放。

这是helper条件可达性验证，不是完整source执行或物理成功证明。CPU进程已正常退出0，没有后台进程或仿真会话。

[JSON receipt](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p13_owner_real8744_condition_check.json>)。

