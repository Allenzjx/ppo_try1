# Closed episode 0: P07 control continuity (14 policy decisions only)

## Scope and result

Run: `runs/ppo_all_stage_acceptance_v1/train/20260909T0405165532990Z_ga8b148463115_14d21d83af52454f94eb0eabd4cc01ce`, runtime `a8b1484631156bad6e6ef0c96bd79a2fbf12e308`. Read only its first 14 policy audit rows, global **139905–139918**, and first completed episode record. No later active row, prefix reset, raw trajectory, checkpoint, or optimizer stream was read.

The initial selected-field extraction produced truncated tool output and no usable parsed result. A parent-authorized technical retry read the **same 14 rows and same completed record**, with a flat whitelist. This report uses that retained retry result; it is not a second physics run. Control arrays were rounded to eight decimal places for the compact extraction; source files were unchanged.

**No unconditional historical entry restoration or residual reset is demonstrated.** There is real changing actuation and measured joint/contact response. The episode is a recorded physical safety failure, not a demonstrated mapper/sensor execution fault. The exact FALL branch and its unique physical cause are not reconstructible from these compact fields.

## Phase and event chronology

The accepted credit record is `teacher_initialized_suffix`, requested/actual **P07**, tick **5912 / 49.266666667 s**, prefix attempt 0, `from_P01_current_policy=false`. The first post-action measurement is tick 5920; it is not the exact handoff pose. The episode has **P07=1, P08=6, P09=7** policy decisions, not two consecutive one-decision preparation stages.

| Episode tick | Observed phase boundary / RR state | Current support evidence |
|---:|---|---|
| 5920 | P07→P08; RR AIR, front −212.127 mm, gap −50.224 mm, load 0, Q/C/P false | FL AIR/load 0; FR TOP .44655, RL GROUND .55345 |
| 5928 | P08; RR GROUND/load .29012; `transfer_ready=false` | All four supports; FL TOP/load .32601 |
| 5952 | P08; RR GROUND/load .47758; preparation/transfer false | FL and RR support; FR/RL AIR |
| 5968 | P08→P09; RR GROUND/load .045657, front −161.724 mm, gap −50.788 mm; preparation/transfer true, Q false | FL AIR/load 0; FR TOP .45215, RL GROUND .50219 |
| 5976 | First P09 row; RR GROUND/load .078293, front −146.686 mm, gap −50.928 mm; Q false | FL AIR/load 0 |
| 5992 | RR AIR, closest sampled front −124.924 mm; gap −50.531 mm | FL AIR/load 0 |
| 5998 / 6003 | Recorded current-policy RR initial-lift / qualified-lift events | No RR crossing or placement event |
| 6017 | P09 FALL; RR Q true, C/P false, AIR/load 0, front −154.277 mm, gap −25.541 mm | FL AIR/load 0, gap +31.465 mm; FR TOP .50356, RL GROUND .49644 |

P08 readiness is not simply latched true from the first airborne sample: it returns false while RR reloads, then becomes true again before P09. Dynamic transition with RR still GROUND is observable; the short duration alone does not establish a predicate bug. RR later qualifies during the PPO segment and does **not** undergo a recorded post-Q ground revocation before this terminal. Its forward progress reverses after the sampled closest point; it never crosses. Inherited FR/FL placement history is not current support. RL initial events at ticks 7/1722 belong to the prefix, not these policy actions.

## Continuous control accounting

Order is `[FL hip, FL knee, FR hip, FR knee, RL hip, RL knee, RR hip, RR knee, FL wheel, FR wheel, RL wheel, RR wheel]`. N = logical nominal; R = filtered/projected residual REQUEST; F = final canonical target after mapping/final limiting. Servos are degrees relative to standing; wheels are rad/s. F is a commanded target, not measured joint position.

| Global / tick / policy phase | FL hip N / R / F (°) | FR wheel N / R / F (rad/s) |
|---|---|---|
| 139905 / 5920 / P07 | 26.30 / −4.00 / 22.30 | .300 / −.120 / .180 |
| 139906 / 5928 / P08 | 29.80 / −7.50 / 22.30 | .300 / −.225 / .075 |
| 139907 / 5936 / P08 | 33.80 / −11.50 / 22.30 | .300 / −.345 / −.045 |
| 139908 / 5944 / P08 | 37.80 / −15.50 / 22.30 | .300 / −.465 / −.165 |
| 139909 / 5952 / P08 | 41.80 / −19.50 / 22.30 | .300 / −.585 / −.285 |
| 139910 / 5960 / P08 | 45.80 / −16.9543 / 28.8457 | .125 / −.705 / −.580 |
| 139911 / 5968 / P08 | 49.20 / −12.9543 / 36.2457 | −.075 / −.825 / −.900 |
| 139912 / 5976 / P09 | 49.20 / −9.4543 / 42.2457 | −.250 / −.930 / −1.180 |
| 139913 / 5984 / P09 | 49.20 / −5.4543 / 48.7457 | −.450 / −.9884 / −1.4384 |
| 139914 / 5992 / P09 | 49.20 / −1.4543 / 55.2457 | −.630 / −.9245 / −1.5545 |
| 139915 / 6000 / P09 | 49.20 / +2.5457 / 59.2457 | −.630 / −.8045 / −1.4345 |
| 139916 / 6008 / P09 | 49.20 / −1.4543 / 52.7457 | −.630 / −.7558 / −1.3858 |
| 139917 / 6016 / P09 | 49.20 / +2.5457 / 54.2457 | −.630 / −.7454 / −1.3754 |
| 139918 / 6017 / P09 | 49.20 / +2.0457 / 52.9957 | −.630 / −.7304 / −1.3604 |

The first five FL-hip targets remain exactly 22.3° while negative residual cancels the increasing nominal. Later that opposition relaxes and the target rises. Thus cancellation is actually exercised here, not merely available in the cap. FR-wheel residual is negative in all 14 rows; the final target becomes negative at 5936, before its nominal turns negative. Its source then ramps through `.125, −.075, −.25, −.45, −.63`; P09 does not restore `.3`.

FR-knee nominal continues **45.9→37.15→31.1°**, retaining 31.1° through P09; its residual moves from +4° initially to −36.5° terminal. RR-hip nominal begins the first P09 row at **1.6°**, then rises to 44.65°, rather than jumping to a historical full target. Neither transition zeroes residual history: FL hip R is −4→−7.5 across P07/P08, and −12.9543→−9.4543 across P08/P09.

Raw samples, means and standard deviations are dimensionless Gaussian-kernel outputs, **not** these physical targets. Selected first→terminal `(mean, std, sampled raw)`:

| Channel | First P07 | Terminal P09 |
|---|---|---|
| FL hip | (.038825, .371194, −.721131) | (.332509, .654572, .050845) |
| FR knee | (−.079410, .450149, 1.003813) | (−1.120337, .803420, −.441517) |
| RR hip | (.005675, .347525, .493397) | (1.508096, .471098, 1.776256) |
| FR wheel | (−.040421, .175556, −.289638) | (−.690084, .225515, −.660406) |

All 14 controller-bias vectors are zero and endpoint headroom-clipped index lists are empty. Six recorded P09 geometry adjustments are identity/zero (`identity_within_descent_allowance`); unavailable earlier geometry diagnostics are not evidence of an extra adjustment. Final slew is active: at terminal RR hip the candidate is **51.06362161°**, previous final **41.41362161°**, and actual final **42.66362161°** (+1.25° for the final single tick), not the candidate. Absence of headroom clipping does not mean absence of slew limiting.

Terminal canonical wheel F is `[-.72333614, −1.36041656, −.55593057, +.08414208]`; dispatched native float32 wheel targets are `[+.72333616, −1.36041653, +.55593055, +.08414208]`. FL/RL sign reversal is the native mapping, not a mismatch. Terminal FL hip/FR knee/RR hip native servo targets are respectively **+.92633420 / −.15751268 / −.75025773 rad**.

## Ownership, response and limits

Source inspection: `semantic_supervisor.py`, `NominalMotionProvider.from_handoff`, `_continuous_advisory`, `evaluate`, and `SemanticControllerAdapter.from_live_prefix`. The handoff retains the current executed nominal/tracking state and measured/history supervisor, not the teacher's complete layer/command queue. New continuous layers start from current nominal, update their changed channels, and later owners take precedence. Capture ownership retires applicable existing layers; it is not a general historical-pose restore. On phase handoff the nominal holds for the designated tick.

The rows agree: one hold tick at episode **5921** entering P08 and one at **5969** entering P09. Every rolling/tail diagnostic explicitly says `layer_present=false`, `not_applicable_no_P06_layer`. This direct-P07 suffix is **not** evidence of an inherited live P06 layer or its retirement/tail behavior; seeded current nominal and newly created layers are the actual scope.

Measured physical joints are available only in stored pre-last-dispatch tracking context. First→last FL hip q is **.38421935→.93197358 rad**, FR knee **.80851543→−.08039179 rad**, RR hip **.00281950→−.49339479 rad**: real response accompanies the changing targets. These are not exact terminal post-action q. The recorded body-forward scalar rises **.09085477→.15469599 m** (+63.841 mm); linear-speed magnitude peaks **.25361236 m/s** at 5976 and ends .18830169; angular-speed magnitude rises **.29271996→.58430190 rad/s**. They do not reconstruct a CoM vector or signed root velocity. Current support varies between two and four, with FL absent at terminal despite its historical placement.

Audit conservation: **13×8+1=105** physics ticks, **105 verified / 105 actual native-effect / 103 own-phase**, the difference exactly the two hold ticks; all four recorded state-write categories are zero. Terminal is `FALL`, physical `SAFETY_ABORT`, source `PHYSICAL_SAFETY`, with physical-valid flag true and task/full-task success false. RPY, gravity, base height, raw CoM/base pose and the precise hard FALL threshold branch are **UNAVAILABLE** here. They must not be invented from speed/support trends.

The actionable observation is continuous nominal-plus-policy motion, actual opposition followed by changed opposition, support redistribution and a real recorded safety stop—not a demonstrated unconditional replay/reset fault. These 14 decisions neither identify a unique FALL cause nor justify changing entry gates, reward, caps or safety thresholds.
