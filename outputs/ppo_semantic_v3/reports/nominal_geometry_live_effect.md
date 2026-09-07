# Nominal geometry: bounded real-execution evidence, not task success

Runtime `4d268fc547b704c590c5f21563ea4b1970b021a1`; real N1/seed1001 P06 new-MDP run `runs/ppo_semantic_v3/train/20260906T1308472858273Z_g4d268fc547b7_dcefe730da37448599db863a5f261d47`, source52,608. This is an immutable bounded analysis of already logged data, not the final2,048-decision outcome. No Python or additional simulator was run for the PowerShell checks below.

## Completed-update subset

The checked subset is policy globals52,925–53,030, decision-end control ticks6,120–6,960. By inspection time update380/global53,120 had completed. Of these106 decision records,81 have a geometry-bearing decision-end actuator audit; these81 selected last-tick audits are not a claim to have separately analyzed all848 underlying120Hz ticks.

| Geometry status | Selected audit count |
|---|---:|
| identity_within_descent_allowance |42 |
| projected_exact_forward |8 |
| projected_relaxed_forward |16 |
| degraded_bypass_infeasible_box_downward |10 |
| degraded_bypass_infeasible_nonreversal |5 |

All81 selected audits have verified setter/dispatch equality, actual-target reconstruction and same-tick metadata. Context dispatch tick equals the audit physical tick; context source tick+1 equals decision-end control tick, and dispatch−source=180 due to existing reset priming. RR mapping is consistently body/row12, physical servo DOFs3/7, Jacobian columns9/13 and canonical pair6/7. Maximum recorded COM/link velocity identity errors are2.384e−7 and5.215e−8 respectively. These are live scene measurements, in addition to—not replaced by—the CPU ABI fixtures.

For all81, `g=corrected_native−raw_native` matches exactly. Reconstructing the two recorded float32 deltas, actual−geometry-zero-policy and geometry-zero-policy−raw-zero-policy, has zero discrepancy. The24 projected records have nonzero native adjustment; max absolute g14.4616degrees is a **native inverse-target adjustment**, not a one-tick physical angle change. Largest geometry target difference between same-state bounded branches is0.0433045rad; two branches can occupy different ends of the same existing per-tick bound. The15 degraded records provide no clearance guarantee and are not counted as successful limit-down projections.

## Concrete active dispatch

Global52,937: source control6,215 → dispatch6,395 → control readback6,216. Status `projected_exact_forward`. RR nominal adjustment is(+1.888875,−2.568827)canonical degrees; current PPO RR residual(+2.254575,−3.498607)degrees remains independently requested. The actual audit reports12 PPO-effect channels versus2 nominal-geometry-effect channels. The first-order nominal x preview remains0.011617964m, while z changes from−0.002270539m to approximately0. Real current clearance is nevertheless−0.047547663m. This establishes an exercised control interface and correct attribution—not physical lift completion or guaranteed subsequent response.

## Distinct later physical result

At root's later global53,111/control7,608 sample, RR had a real qualified lift event at6,999, then ground-before-cross revoked it at7,065. There was still no RR crossing or placement. Current RR was AIR with clearance−32.407432mm/front−99.327114mm; current FL was AIR/load0 despite historical front-leg placement. The same sample had a real projected-relaxed-forward context atsource7,607/dispatch7,787 with COM/link errors5.588e−9/3.725e−9. This later row was a live tail when read and is not included in the81-row completed-update statistics above.

Every geometry record explicitly states `current_policy_residual_used=false` and `physical_motion_guaranteed=false`. B's zero-PPO nominal path also changes under this runtime; an eventual A→C difference cannot all be attributed to PPO learning. Neither this report nor CPU841/0-failure tests assert completed RR/RL traversal, full-P01 success, comparative stability, or a successful video. The real training block continues without an A5/5 or full-success probe gate.
