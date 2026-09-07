# 修复后真实 workspace probe：>10°独立 residual 已下发

源 run：`runs/ppo_semantic_v3/interface_checks/20260906T0257098749051Z_g26fa541_workspace_repair`，HEAD `26fa541abd1b12665a3585aa39e0cf0f40f0e81c`。本次仅执行`segments=new_range`，不是重新执行三段。seed1001、P06 teacher suffix、offset0、raw magnitude .5，32决策/256物理ticks。本文只读PowerShell/.NET分析；训练正在进行，未启动Python/Isaac、未改runtime或首轮失败证据。

## 1. 实际通过的是接口响应，不是完整越障

- run生命周期 `SUCCEEDED`；32决策、256ticks全部P06；逻辑tick3585–3840，对应响应2.133333s；结束时`termination_reason=null`、`task_outcome_label=null`，无任务终止。
- 256/256 native审计 verified，actual reconstruction=setter=PhysX float32 targets；256ticks都有真实目标effect。32个decision的无禁写审计全部通过；optimizer credit/updates/steps均0。
- 八伺服实际canonical `drive_target − native_drive_target` 在236ticks均超过10°。第一次为第21响应tick：logical3605/native3784，八servo交替±10.5°；这正是首轮被接口拒绝的位置。
- 最终实际offset：hips +11.090811774°、knees −16.636217661°、wheels +.2772702944rad/s。最大真实servo offset16.636217661°。这是实际ACK/float32 target证据，不只是raw或projected动作非零。
- 不能扩大结论为“±24/36°全范围已验证”“所有13phase控制权已验证”或“任务/稳定性改善已验证”。这是固定raw=.5、P06短窗口的真实响应。

## 2. controller correction 与 policy residual 的字段含义

每tick native_response保留三个独立字段：

1. `native_audit.controller_drive_bias_full12`：本段256ticks始终全0。
2. `projected_residual_full12` / `native_audit.projected_residual_full12`：独立policy residual，确实逐tick升过10°。
3. `native_audit.combined_post_mapper_bias_full12`：total offset；256×12=3072个component均满足`combined = controller + projected residual`。

修复后的PPO专用adapter把原controller envelope校验只应用于controller correction；原`drive_feedback_bias_requested_full12`兼容字段仍表示两者之和，不应误读为controller补偿超过10°。源码ACK另有`bounded_controller_bias_requested_full12`、`independent_policy_residual_requested_full12`、`controller_bias_original_envelope_verified`、`drive_feedback_bias_requested_semantics`；prefix最后的真实ACK含这些字段。响应JSONL没有保存完整ACK，而是保存上面三个分开的audit字段及真实native buffer；不宣称日志含它没有记录的完整逐tickACK对象。

末tick3840真实目标（第一8 physical servo rad，后4 physical wheel rad/s）：

```text
[.571074605, -.501945734, .179497302, .503283560,
 -.295313865, .281994104, -.175648019, .277456939,
 -.577270269, .577270269, -.577270269, .577270269]
```

同tick零当前residual反事实的servo差peak只有约.03054330rad/1.75°，**不否定>10°已下发**：该反事实保留t−1实际target和原最终slew，只撤掉当前residual；不是把整段历史offset瞬间归零。

## 3. 与首轮的可比初态和窗口

比较基线来自首轮`20260906T0233067086105Z_g2248cd1_workspace`的nominal_zero、old_range各自**前256ticks**，不是其512tick末帧。两者与本次new_range的response初始`physical observation`完整字段/数组数值相同：logicaltick3584、29.8666667s，同base/joints/geometry/contact/CoM/commanded targets。同一P06 nominal `[22.8,-13.4,0,45.9,6.9,0,0,0,.3,.3,.3,.3]`，本次256ticks不变。

本次new_range与首轮new_range失败前的20ticks，**逐tick完整physical observation及actual native targets全部相同**。因此修复没有借改变起点或前20tick响应绕开旧接口。首轮new_range没有256tick结果，后236ticks是本次新增的真实证据；不能对不存在的旧样本填0。跨HEAD分析仍如实保留source版本，不把记录中的相同初态扩展为全部未记录内部状态的证明。

## 4. 相同256tick末帧：实际关节、身体和载荷

末帧均logicaltick3840/time32s；相对共同起点经过2.133333s。old_range是旧caps下`.5 raw`，new是修复接口的新caps，不是learned actor。

| 实际量 | zero（首轮前256） | old（首轮前256） | repaired new（本次256） |
|---|---:|---:|---:|
| body x位移 | +32.061mm | +38.254mm | +54.153mm |
| body z位移 | +3.142mm | −1.745mm | −24.264mm |
| base z末值 / 窗口最小 | 94.256 /91.289mm | 89.369 /88.759mm | 66.850 /65.338mm |
| body speed末值 / 峰值 | .017663 /.043024m/s | .020825 /.038884m/s | .031038 /.112280m/s |
| angular speed末值 / 峰值 | .007601 /.106284rad/s | .011615 /.081922rad/s | .042408 /.114249rad/s |
| 实际support count末值 / 最小 | 4 /4 | 4 /4 | 3 /3 |
| FL AIR ticks | 0 | 0 | 69 |
| FL / FR / RL / RR末帧实际载荷占比 | 20.99/25.36/29.10/24.55% | 19.02/27.15/30.98/22.85% | 0/41.44/49.99/8.57% |
| RR距前沿 | −462.985mm | −457.029mm | −424.139mm |
| RR bottom相对台面 | −50.197mm | −50.178mm | −50.322mm |

new的FL首次记录AIR在tick3726/time31.05s，末帧也为AIR、法向载荷0；不能称“载荷转移到FL”。其FL bottom净空仅+.705mm，应仅按真实contact/geometry描述，不在本分析发明新主动抬腿资格。new末帧CoM位置 `[.364249242,-.125145577,.150036600]m`，速度 `[.025092270,.000572634,.005800573]m/s；这与FL接触承载是不同量。

八关节末帧实测角度（FLH/FLK/FRH/FRK/RLH/RLK/RRH/RRK，°）：

| 段 | 实测角度 |
|---|---|
| zero | `[21.8563,-12.1512,-.3229,45.5877,6.3409,.5522,-.9330,.7842]` |
| old | `[23.7005,-13.9234,1.5495,43.7316,8.2695,-1.2978,.8954,-1.0230]` |
| new | `[32.6749,-28.7810,11.3397,29.2355,17.9412,-16.2803,9.7416,-15.6198]` |

new的8关节相对共同初态实际变化约 `[+8.829,-16.667,+11.096,-16.586,+10.913,-16.697,+11.162,-16.671]°`，不只是target变化。明显身体下沉、FL卸载、后腿载荷重分配说明新范围有更大的物理作用，也说明“作用大”不等于“更稳定”。三组共同窗口各256ticks均finite、无记录的body collision；new无physical termination。RR仍接地且远在前沿前，不是后腿跨越成功。

本报告不再新增CPU/static/hash验收门，也不把这个探针变成训练成功门。首轮接口失败报告保持原样，本次修复后的真实响应是独立新增证据；PPO可以继续在现有任务语义和真实失败机制下训练。
