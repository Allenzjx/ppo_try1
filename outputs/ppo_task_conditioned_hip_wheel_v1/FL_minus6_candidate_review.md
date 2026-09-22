# FL hip锚点−6：隔离候选，尚未运行

候选`direction_probe_FL_minus6_candidate.py`，CPU测试`test_direction_probe_FL_minus6_candidate.py`。只在outputs新增，不修改生产、共享配置、历史harness或正在执行的仿真。最终是否启动由主代理完整审阅并等待安全运行边界后决定。

有限幅值是上一ACK的FL hip REQUEST锚点−6°，不是绝对hip角、不是每tick继续减6°、不是宣称PPO已学会。与CP189952 −3对照若用同checkpoint/seed4001，应使用相同真实P05 endpoint+已越沿+AIR+未placed+within_top_xy触发；**不硬编码2976入口tick**。−3曾降低gap约7 mm却未接触，只支持进行一次有界下一幅值验证，不保证−6会线性再降7 mm或落脚成功。

时序严格继承旧`FiniteDirection.start/apply/complete`：12-decision quintic ramp；entry-age达到75或capture-age+24时开始release，以先到者为准；release12，随后follow30。**75包含12-ramp，原纯hold为63决策，不是12+75=87。** 完成后继续当前真实状态/HISTORY下的在线det策略；不恢复旧入口、不清HISTORY、没有knee叠加。

仅FL hip的raw可被手工覆盖，其他11维每次来自同拍在线det输出。源N、执行profile、mapper、安全许可和物理限位不变。anchor+2.44040789654°对应本次−6的plateau REQUEST−3.55959210346°；实际effective/target/actual必须由未来物理日志确认，CPU测试不充当物理成功。

纯receipt函数直接复制已审原函数，**唯一AST差异是显式预期case从FL_minus3改成FL_minus6**；没有别名伪装或事后改标签。主流程AST仅改变case/class/trigger名、manifest真实−6标签和零aux/来源hash字段。其余使用旧官方loader、372观察/HISTORY核验、pre-action先flush、actual执行链、独占资源锁、错误封存和learned-state unchanged检查。

14个新增定向测试加17个原测试，共31 PASS。覆盖锚点不递减、实时其余11维、跨阶段caps变更不重置锚点、相同时序、触发正反例、容量/非有限拒绝、错case/knee拒绝、mask仅作用residual、HISTORY不变、receipt重算hash、旧脚本字节不变及AST合法变化范围。未导入/启动Isaac，无policy forward、无更新。

原文件SHA256保持：

- `direction_probe.py`：`47d49535071af1b0f76cd13a249675a2c8cb93575b1b9751e3c702cdd16c1167`
- `direction_probe_preaction_candidate.py`：`c7c1eef885df4cccda650e81175fb73a30b6a327e5a3abbd76adaa682597690d`

manifest明确`training_data_eligible=false`、`auxiliary_updates=false`、`auxiliary_loss=0.0`、`positive_auxiliary_labels_emitted=0`、new PPO decisions/updates/optimizer steps均0。本候选完全不生成aux标签；即使将来出现短暂接触，也不能自动称持续捕获或输出正aux标签。
