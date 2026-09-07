# P06 checkpoint-prefix course — first two RR-entry outcomes only

Bounded contrast, **not a finalized block ledger or causal trial**. Source is the still-running `train/20260906T2326102309749Z_g2677995544c9_5826da63aa384e5d8f6e7ea4a025cfaf`, runtime2677995. This report reads only the first two completed episodes and one bounded policy-stream pass through **global89089–90307 /1,219 rows**, stopping there before subsequent rows. No optimizer file, checkpoint hash, new native full audit or later episode is read. The master training report is not changed.

## Credit boundary and actual outcomes

Both prefixes use the same frozen saved **C89088** actor receipt `733b1c1efaa2fddd8c5754f2ec94cb288f951fc0934503fb02acee51f33312bd`, with independent frozen storage and fixed deterministic inference. Each prefix is accepted with no miss/fallback after **350 decisions /2,800 ticks**, entering current P06 at23.333333s with176.666667s remaining. Prefix record `policy_credit=false`; the350+350 initialization actions are excluded from PPO. Credit begins on the next physical interval without resetting the episode clock or physical history. Fixed actor/source does not establish bitwise equal sensor trajectories: prefix FL crossing is2604 in ep0 and2605 in ep1, while other listed front Q/P stamps agree.

| Episode | Credited global range | PPO decisions / physics ticks | Full physical episode endpoint | Actual outcome |
|---|---|---|---|---|
| 0 | 89089–89449 | 361 /2882 | Tick5682 /47.35s /P09 | BODY_COLLISION; RR not qualified/crossed/placed |
| 1 | 89450–90307 | 858 /6864 | Tick9664 /80.533333s /P12 | INCOMPLETE; PPO RR Q5903/C6026/P6027, RL not placed |

The physical durations include the reset-only prefix. The first final decision is2ticks (361×8−6=2882); episode1 has858×8=6864ticks. Total prefix5600 plus policy9746 gives15346 physical ticks across these two episodes, not extra PPO credit. Both task-success values arefalse. Completing the RR subtask in episode1 is **not** suffix success, full-P01 success, a new saved-checkpoint evaluation or a guaranteed reproducible entry condition.

FR Q45/C1563/P1577 and FL Q1669/C2604-or-2605/P2793 belong to the frozen prefix, not to PPO's new achievements. RR Q5903/C6026/P6027 all occur after the2800 credit seam and are genuine policy-segment events. The learned sampling actor is not frozen for this contrast; only the reset-prefix actor is fixed. No controlled attribution to a particular optimizer update or Gaussian sample is made.

## Real transition geometry and contact

These are the measured evaluator snapshots at the end of the indicated outgoing-phase decision. The actual command receipt at that tick belongs to the **outgoing** phase; it is not mislabeled as a fully executed new-phase action. Each P07/P08 lasts8ticks in both episodes.

| Episode / phase just finished→new phase | Tick | RR front / clearance (mm) | RR contact / load | FL clearance (mm) / contact / load | FR / RL load | Body linear / angular speed |
|---|---:|---|---|---|---|---|
| 0 / P06→P07 | 5608 | -173.254 / -49.946 | GROUND / 0.1464 | 5.975 / AIR / 0.0000 | 0.3731 / 0.4806 | 0.0868m/s / 0.2907rad/s |
| 0 / P07→P08 | 5616 | -176.607 / -50.460 | AIR / 0.0000 | 3.847 / AIR / 0.0000 | 0.4050 / 0.5950 | 0.0674m/s / 0.2493rad/s |
| 0 / P08→P09 | 5624 | -177.907 / -50.872 | GROUND / 0.0484 | 11.849 / AIR / 0.0000 | 0.4074 / 0.5442 | 0.0532m/s / 0.2210rad/s |
| 1 / P06→P07 | 5280 | -210.033 / -50.705 | GROUND / 0.3282 | -0.529 / TOP / 0.3366 | 0.1560 / 0.1791 | 0.0611m/s / 0.1506rad/s |
| 1 / P07→P08 | 5288 | -201.729 / -50.859 | GROUND / 0.0410 | 0.325 / AIR / 0.0000 | 0.4134 / 0.5456 | 0.0825m/s / 0.2819rad/s |
| 1 / P08→P09 | 5296 | -187.742 / -50.936 | GROUND / 0.0323 | 11.237 / AIR / 0.0000 | 0.4106 / 0.5571 | 0.0656m/s / 0.2650rad/s |

At the P09 entry comparison, both RR wheels are **GROUND and unqualified**, with low but nonzero measured normalized load; both FL wheels are **AIR/load0**. The later-RR-placement episode starts with RR **9.835mm farther behind** the front plane (−187.742 versus−177.907mm), and essentially the same below-top clearance (difference−.065mm). The failed episode briefly has RR AIR/load0 at P07→P08 but returns GROUND at P08→P09; AIR at ground-level clearance is not an active-lift qualification. FL is TOP at ep1 P06 exit, but is AIR by both subsequent boundaries. These observations do not support inventing “RR must already be airborne,” “FL must be TOP at exactly P09 entry,” or a single fixed support template.

Both P06 exits record the real existing workspace goals fulfilled; P07/P08 then continue into the unfinished RR task. The compact evidence contains contact class/top/ground/obstacle flags and normalized leg loads, geometry and **speed magnitudes**. It does not contain a full measured joint-position vector, base/CoM world trajectory, velocity direction, complete quaternion or raw exact body-pair force/contact-point sequence. Therefore no CoM direction, actual single-joint pose, support-force magnitude or collision-mechanics cause is inferred here.

## Nominal, sampled request, projected residual and dispatched command

At the exact P08→P09 boundaries (ep0 tick5624, ep1 tick5296), the8 servo nominal suggestions are identical; inherited wheel nominal values differ slightly with current measured retirement state. The rows below retain all12 channels. Servo nominal/residual/drive are degrees; wheels rad/s; raw is the stochastic unsquashed policy request, **not the deterministic mean**. `actual drive` is the recorded canonical dispatch after mapper/geometry/headroom/slew processing, not measured joint position and not necessarily logical nominal+residual.

| Channel | Nominal ep0 / ep1 | Ep0 raw / residual / actual drive | Ep1 raw / residual / actual drive |
|---|---|---|---|
| FL hip | 37.60000 / 37.60000 | 0.32105 / 7.45102 / 38.24079 | 0.33054 / 7.65614 / 38.96884 |
| FL knee | -13.40000 / -13.40000 | 0.28656 / 5.52925 / -6.62075 | 0.35845 / 4.41820 / -7.73180 |
| FR hip | 0.00000 / 0.00000 | 0.14399 / 3.43219 / 3.43219 | 0.14805 / 3.52738 / 3.52738 |
| FR knee | 45.90000 / 45.90000 | 0.85884 / 23.35969 / 59.25969 | 0.29173 / 10.21427 / 46.11427 |
| RL hip | 14.30000 / 14.30000 | -0.63825 / -5.34274 / 8.95726 | -0.29084 / -5.77986 / 8.52014 |
| RL knee | 0.00000 / 0.00000 | -0.08040 / -2.88815 / -2.88815 | -0.20787 / -7.37728 / -7.37728 |
| RR hip | 0.00000 / 0.00000 | 0.28812 / 6.88029 / 6.88029 | 0.01630 / 0.39116 / 0.39116 |
| RR knee | 0.00000 / 0.00000 | 0.04882 / 3.86294 / 3.86294 | -0.16833 / -6.00313 / -6.00313 |
| FL wheel | 0.13015 / 0.11715 | -0.18132 / -0.21523 / -0.08508 | -0.08353 / -0.17832 / -0.06117 |
| FR wheel | 0.13015 / 0.11715 | 0.51698 / 0.39984 / 0.52999 | 0.05011 / 0.06008 / 0.17724 |
| RL wheel | 0.13015 / 0.11715 | 0.57419 / 0.31106 / 0.44121 | 0.07707 / 0.04615 / 0.16330 |
| RR wheel | 0.13015 / 0.11715 | 0.50952 / 0.28174 / 0.41189 | 0.05064 / 0.05522 / 0.17237 |

Thus FR knee dispatch is59.25969 versus46.11427deg; RR knee is+3.86294 versus−6.00313deg, and RR hip6.88029 versus.39116deg. These accompany differences throughout the action and measured body state, not an isolated joint intervention. FR/RL/RR canonical wheel dispatches are lower in the RR-placement episode at this entry, while FL is less negative (−.061172 versus−.085076); that correlation does not establish causality or prescribe a sign/template.

The first new-P09 decision ends at5632/5304 (entry+8). In ep0/ep1 respectively, RR is AIR/load0/clearance−51.264mm versus GROUND/load.04083/clearance−50.705mm; FL is AIR in both. Nominal RR hip is1.6deg in both, but actual RR hip10.73029/4.08082deg and knee+1.65768/−7.77653deg. Four-wheel actual vectors are[−.255944,.565925,.270340,.241025] versus[−.233718,.214689,.004353,.209825]. These distinguish continued real action/state trajectories without making the initial transient a new required gate.

## RR attempts, qualification, crossing and later support

Episode0 has a single RR initial-clearance hint at5648, with recorded upward excursion3.673743mm; at that same decision endpoint RR is still AIR **47.776mm below** obstacle top and183.889mm behind the front plane. It never qualifies. Its body collision at5682 is only58ticks after P09 entry5624; terminalRR is AIR/load0/front−207.503mm/clearance−46.434mm, and FL AIR/load0/+136.748mm. The compact termination retains BODY_COLLISION; absent raw body-pair data is not used to reclassify it.

Episode1 has eight RR initial hints:4178 (still P06),5423,5591,5617,5669,5712,5783,5842. It first earns **qualified_measured_upward_lift5903**, with recorded upward excursion42.300880mm, evidence-field `joint_motion_deg=17.742697` and actual above-top clearance+.397572mm. That motion summary is not proof of single-motor causation. At the next decision endpoint5904, RR is AIR/front−51.667985mm/clearance+1.500921mm/load0; FR is TOP/load.480362 and RL GROUND/load.519638, while FL remains AIR/load0/+109.876891mm.

The same already-retained tick5904 receipt records nominal four wheels+.3, FR projected residual **−.801808345**, and actual FR drive−.501808345rad/s. This selected row really uses residual beyond old−.6; unlike a nominal-plus-residual final target alone, it is direct recorded residual evidence. It occurs after entry during multi-channel exploration and cannot prove the new range caused qualification or prevented collision. No additional range scan is performed.

RR crossing6026 and placement6027 are recorded by the live history, respectively730/731ticks after P09 entry5296. At the first available decision endpoint6032, RR is genuinely TOP with obstacle-pair flagtrue/load.544311/front+6.212357mm/clearance−.593706mm; FL is also TOP/load.455689 while FR/RL are AIR. These geometry values are from6032, **not fabricated exact6026/6027 samples**. The outgoing P09 then transitions to P10, with new completion values describing the next task rather than invalidating the RR event.

The later episode still ends **P12 incomplete, placed_RL0**. At9664 RR has returned to GROUND/front−267.238mm/clearance−49.299mm/load.152664; RL is GROUND/front−312.712mm/clearance−50.552mm/load.532737, FR TOP/load.314599, FL AIR/load0. Historical RR placement is not persistent current support, and it does not satisfy the RL task or whole-task objective.

## Bounded interpretation

New information is a genuine contrast within the same version and frozen-prefix-source course: two superficially similar P09 entries lead to different RR subtask histories, with different live supports, stochastic actions and subsequent trajectories. RR completion requires continued measured exploration after entry; neither entry alone demonstrates qualification nor a particular nominal/servo/wheel difference is isolated as its cause. No hard entry threshold, static support template, cap adjustment, reward change or success gate is proposed. The main4096 block remains running and its final ledger is outside this fixed1219-row report.
