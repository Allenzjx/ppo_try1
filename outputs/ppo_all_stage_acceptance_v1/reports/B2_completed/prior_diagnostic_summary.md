# Completed B diagnostic (not training)

Run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_all_stage_acceptance_v1\diagnostics\20260909T0246187023798Z_g2f942e824f82_a77db36275dd4bc4be422482922fde92`

900 diagnostic decisions / 7200 ticks / 60.000000s.
Task success: False; external window: True.
Termination: None; source: None.
First unfinished label: P09 (see actual goals in JSON).

这是900决策的外部诊断窗口结束，不是内部任务deadline或物理失败终止；原始 termination_reason/termination_source 保持 null。B 的零 residual 采样不计训练：新增训练decisions=0、PPO updates=0、teacher/policy prefix=0。

| Phase | Diagnostic decisions |
|---|---:|
| P01 | 2 |
| P02 | 182 |
| P03 | 4 |
| P04 | 1 |
| P05 | 146 |
| P06 | 330 |
| P07 | 1 |
| P08 | 1 |
| P09 | 233 |
| P10 | 0 |
| P11 | 0 |
| P12 | 0 |
| P13 | 0 |

Leg event counts (recorded I/Q/C/P/revoke):

- FR: `{"I": 1, "Q": 1, "C": 1, "P": 1}`
- FL: `{"I": 1, "Q": 1, "C": 1, "P": 1}`
- RR: `{"I": 1}`
- RL: `{"I": 1}`

| Leg | Initial I tick | Qualified Q tick | Cross C tick | Placed P tick |
|---|---:|---:|---:|---:|
| FR | 15 | 24 | 1487 | 1502 |
| FL | 1548 | 1556 | 2586 | 2677 |
| RR | 5909 | 未记录 | 未记录 | 未记录 |
| RL | 7 | 未记录 | 未记录 | 未记录 |

I 是初始离地事件，不等于硬Q；RL末态 initial_clearance=false。全窗口未记录 qualification-revoke 事件，不把未取得Q后的落地伪记为Q撤销。事件均来自新版共同 evaluator 的记录，不是本报告重新运行物理判定。

实际交接为 P01→P02@16、P02→P03@1472、P03→P04@1504、P04→P05@1512、P05→P06@2680、P06→P07@5320、P07→P08@5328、P08→P09@5336（120Hz tick）。8次均观测于非terminal决策，terminal_bootstrap_allowed=true，单episode决策与physical tick连续；本次摘要没有独立重建全部mapper/滤波内部状态。

窗口末当前 P09 的 `placed_RR` 进度为0.276737，而RR没有Q/C/P。RR front −41.865mm、clearance −27.760mm；当前nonground、非AIR、OBSTACLE_AMBIGUOUS、bearing_verified=false，障碍物pair反力2.35848N不能替代真实TOP承载。FL/FR当前TOP且bearing_verified=true（0.82794/12.61382N），RL当前GROUND（13.59243N）。保留共同physical valid=true/run_validity=VALID与physical_evidence_status=CONTACT_BEARING_UNVERIFIED的区别；各腿load_fraction_valid=false，因此不把显示的归一化载荷分数视为完整可靠分账。

Native audit: `{"physics_ticks": 7200, "native_effect_ticks": 0, "own_phase_effect_ticks": 0, "native_audit_anomaly_rows": 0, "state_write_anomaly_rows": 0, "nonzero_actual_canonical_drive_decisions": 900}`

900行raw与filtered request均为12维零，7200tick保存的原生下发验证通过且四类state-write异常为0。900行实际canonical命令含非零分量，仅说明目标/命令已下发，不能当作actual joint q或实体运动幅度。独立native/transition流为空，本报告使用决策内保存的每tick audit与事件历史，没有伪造独立流证据。

B diagnostic, never PPO training or a policy success denominator.
I/Q/C/P/revoke are logged evaluator events, not independent physical replay.
Native checks consume saved same-tick audit evidence; zero PPO effect is expected for B, not zero actuator motion.
Only raw first/last sampled; no full raw/contact rescan. Empty independent native/transition files are not imputed.
First unfinished label is not an independent physical failure classification; current goals and termination source remain separate.
Unvisited phases have zero diagnostic samples, not zero quality.
