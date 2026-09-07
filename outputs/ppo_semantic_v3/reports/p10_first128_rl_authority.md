# Block32: fixed first128 RL action/state observations

Scope is **only g108801–108928**, the first128 written credit rows in `runs/ppo_semantic_v3/train/20260907T0351509044004Z_gf1a9bbf650b1_f18d36a5a1bd43de83a723a754e0bc78/residual_and_projection_audit.jsonl`. No later row, optimizer/checkpoint/hash, whole-run result, raw/native full audit or subsequent episode is used. Only this report is written; no production/config/tests/master, Python/PT/GPU/Isaac or proposed parameter change.

## Bounded result and missing physical evidence

The128 globals are contiguous and contain1024 physics ticks: **P10=1, P11=1, P12=126**, with no terminal or physical failure. Credit starts atP10 tick7584/63.2s; P10→P11 at7592, P11→P12 at7600. The last credited point is tick8608/71.733333s, still P12, not an episode outcome.

RR Q/C/P6938/7109/7579 and FR/FL placement belong to the frozen-FSM teacher before the7584 takeover, not new PPO rear skill. RL has **no hard Q/C/P through108928**. It has one new whole-body initial-clearance event at7979:3.222477mm upward excursion, own-joint motion3.395731°, whole-body motion10.212175°, commanded motion20.277425°, gravity-direction change.012633246. That is an initial hint, not hard qualification. The current method permits whole-body actuation evidence; this report adds no RL-only joint-motion gate.

| Source phase | Samples | RL GROUND | RL obstacle pair active | RL AIR | RL true TOP | Current initial hint |
|---|---:|---:|---:|---:|---:|---:|
| P10 | 1 | 1 | 0 | 0 | 0 | 0 |
| P11 | 1 | 1 | 0 | 0 | 0 | 0 |
| P12 | 126 | 118 | 5 | 3 | 0 | 1 |

P12 sampled gap is **−51.271452 to−41.142119mm**, never above top at a decision endpoint. The highest endpoint,7632, is obstacle-pair-active rather than AIR, with front−50.433930mm/load.416071. The maximum recorded recent upward excursion is7.371731mm at7640, also pair-active, below the existing8mm qualification gain. These are endpoint extrema, not independently measured120Hz extrema. Hard qualification still requires the actual joint/whole-body, upward-gain, AIR and above-top conditions together; its event/history is absent.

The only AIR endpoints are:

| Tick / global | Gap mm | Front mm | Recent gain mm | AIR streak ticks | Current initial | FL / RR current loads |
|---|---:|---:|---:|---:|:---:|---|
| 7976 /108849 | −48.023471 | −123.762753 | 1.304276 | 2 | false | .455070 /.544930; both TOP |
| 7984 /108850 | −48.818631 | −128.043171 | 3.222477 | 10 | true | .506857 /.493143; both TOP |
| 7992 /108851 | −48.713283 | −126.329001 | .168104 | 5 | false | .525705 /.474295; both TOP |

The10→5 AIR-count reset means these three endpoints are not evidence of one uninterrupted lift across the entire interval. No unrecorded contact point or exact interruption tick is invented. At8608 RL is GROUND/load.558629, front−137.364623mm, gap−51.159343mm, initialfalse/QCPfalse; FL is AIR/load0 and RR has load.120241 but currentTOPfalse despite its teacher placed history. Current supports therefore differ across the window.

## Distribution and sampled sign coverage by phase

Means/std and raw samples below are the recorded **pre-tanh Gaussian coordinates**, not degrees, probabilities of task success, or filtered commands. Channels4/5 are RL hip/knee. Each P10/P11 row is only one observation, not a distribution estimate. Both means are negative throughout all128 points; std remains finite and positive.

| Phase / joint | Mean average [min,max] | Std average [min,max] | Raw sample range | Raw positive / negative / zero |
|---|---|---|---|---|
| P10 hip | −.154498 [same] | .137105 [same] | −.036193 | 0 /1 /0 |
| P10 knee | −.116308 [same] | .162324 [same] | +.074628 | 1 /0 /0 |
| P11 hip | −.173610 [same] | .124884 [same] | −.323452 | 0 /1 /0 |
| P11 knee | −.139374 [same] | .167902 [same] | −.277560 | 0 /1 /0 |
| P12 hip | −.357244 [−.441917,−.207823] | .088889 [.079586,.113806] | [−.599161,−.138735] | **0 /126 /0** |
| P12 knee | −.366647 [−.433490,−.194359] | .171913 [.161232,.185623] | [−.735090,−.031699] | **0 /126 /0** |

P12 raw sample means are−.360091/−.384514 for hip/knee. The absence of sampled positive RL raw actions in these126 points is an observed coverage fact, not a claim that the actor cannot produce them or that a positive canonical angle would necessarily lift RL. No independent-Gaussian probability is substituted for measured success.

## Expressed, filtered and actual RL targets

All target columns are canonical degrees. `REQUEST` denotes the stored projected/rate-filtered residual, distinct from raw action; `native` is mapped nominal before that post-mapper policy bias. Measured q is physical joint radians from `tracking_reference_evidence.actual_measured_physical_rad`, sampled before dispatch, not the canonical target or a guaranteed post-step pose. Standing offsets/sign conversion must not be ignored when comparing those different coordinate systems.

| P12 channel | Nominal min..max | REQUEST min..max (mean) | Native min..max | Actual drive min..max | Measured physical q min..max rad |
|---|---|---|---|---|---|
| RL hip | −10.1..31.2 | −12.443483..−3.415651 (−8.207868) | −14.634998..32.45 | −26.232670..27.661284 | −.536785..+.381956 |
| RL knee | −18.7..35.3 | −22.542112..−4.318376 (−13.016380) | −19.95..34.05 | −42.492112..29.557118 | −.580266..+.714237 |

Both P12 filtered REQUEST channels are negative in all126 samples. Measured q nevertheless spans both signs and substantial ranges: the joints were not motionless or disconnected. The history's `recent_joint_motion` often reads0 on current GROUND because its active-attempt window is reset, not because the measured joint never moved. Maximum recorded own/whole-body recent motion in P12 is17.532861°/86.457197° at7640, but that point is not AIR/above-top.

| Tick | Nominal hip/knee | REQUEST hip/knee | Native hip/knee | Actual drive hip/knee | Pre-dispatch physical q hip/knee rad |
|---:|---|---|---|---|---|
| 7592 P10 | 15.4 /19.4 | −.868246 /+2.681624 | 10.4 /24.4 | 9.531754 /27.081624 | −.271889 /−.347018 |
| 7600 P11 | 15.4 /19.4 | −4.368246 /−.818376 | 10.783462 /24.4 | 6.415215 /23.581624 | −.219319 /−.403300 |
| 7632 P12 | 15.4 /35.3 | −7.933799 /−6.680715 | 13.283462 /34.05 | 5.349663 /27.369285 | −.141155 /−.568963 |
| 7984 P12 | 31.2 /35.3 | −9.391964 /−18.287406 | 32.45 /34.05 | 23.058036 /15.762594 | −.353338 /−.292187 |
| 8608 P12 | −10.1 /−18.7 | −8.506202 /−16.167677 | −7.134998 /−19.95 | −15.641200 /−36.117677 | +.277549 /+.663360 |

Controller drive bias is zero on the two RL channels in this window. Hip actual equals native+REQUEST within1.78e−15 across128 endpoints. Knee has one non-additive endpoint, **8008/global108853**, which is explained by the existing final-target slew: native10 plus REQUEST−11.962564 would be−1.962564, while previous final=.889157 and actual=−.360843, exactly a−1.25° final step. This is not an unreported mapping mismatch. The other127 knee endpoints match the sum within floating-point rounding.

Nominal geometry context exists in8 endpoints:5 `degraded_bypass_infeasible_box_downward` (the initial obstacle-contact samples),3 `identity_within_descent_allowance` (the AIR samples); all recorded adjustments there are0. The other120 rows omit that context, which is **missing/not-applicable evidence**, not an invented zero-Jacobian measurement. Pre-dispatch measured RL q from the reference receipt is present for all128. Decision-end reference_used is false, but no claim of zero120Hz reference use is made without a substep scan.

## Front-wheel cancellation: what was actually sampled

Channels8/9 are FL/FR wheels. Their recorded raw means are negative at every point; raw sign exploration is broader than the RL joints, but filtering/nominal determine the actual command.

| Front wheel | Nominal range rad/s | Positive raw among all128 | Negative-nominal samples | Positive raw while nominal<0 | Positive REQUEST while nominal<0 | Actual>=0 while nominal<0 |
|---|---|---:|---:|---:|---:|---:|
| FL | −1.07..0 | 25 | 43 | 6 | 3 | **0** |
| FR | −.3..0 | 6 | 34 | 2 | 0 | **0** |

FL REQUEST spans−.706309..+.129165rad/s; actual spans−1.755233..−.011382. FR REQUEST spans−.968254..−.12; actual spans−1.041485..−.12. All128 actual commands for each front wheel are negative, including the85 FL/94 FR zero-nominal samples. The configured positive residual capacity does not establish that its full cancellation range was sampled.

- **Strong FL nominal−1.07 is genuinely present in the first9 points7592–7656.** All9 raw/request values are negative. At7592 raw−.212897, REQUEST−.12, native−1.07, actual−1.19; at7648 raw−.832439, REQUEST−.685233, actual−1.755233. Positive cancellation of this strong segment was not explored in these observed points.
- **Weak FL nominal−.3 receives limited positive filtered action** at7848/7856/7864: REQUEST+.009165/+.129165/+.009165, actual−.290835/−.170835/−.290835. At7856 raw+.260277 would map to unconstrained tanh*1.2=+.305465rad/s, but the request rises only+.12 from the previous+.009165 to+.129165. Thus a raw request that would statically cancel−.3 did not yet deliver full cancellation through the existing slew. At7864 raw is already negative; the residual positive request is filter history, not another positive raw sample.
- At7864 FR raw+.096350 also occurs with nominal−.3, but filtered REQUEST remains−.273131 and actual−.573131. Positive raw alone is not positive applied correction.
- At8608 both front nominals are0; requests/actual are FL−.303609 and FR−.526779. Those are policy-expressed rolling commands, not a remaining negative nominal being impossible to cancel.

These arithmetic facts distinguish sampling and filtering from range availability. They do not prove that changing cap, std, action sign or a single wheel would produce RL qualification, or identify wheel action as the unique cause of the current ground/contact state.

## Execution integrity and interpretation limits

All128 existing decision-end receipts report verified=true, setter=dispatch, actual-mapping=dispatch, same-tick counterfactual=true and previous-ACK reference independently verified=true. Every physical snapshot is valid and no hard physical termination is recorded; no observed mismatch in this scope. This is not a replacement full native/optimizer audit. Exact raw contact-point/force vectors, complete joint-velocity/body/CoM trajectories and all substep height extrema were not captured by this fixed compact extraction; geometry or force causes are not inferred from absent fields.

The observable shortfall is that **no genuine above-top active RL process is earned**: RL remains mostly loaded/on ground, briefly makes low obstacle contact and small below-top AIR attempts, with no Q/C/P by the fixed boundary. Both RL P12 action channels sampled only negative raw/request values, while actual joint positions did move. Teacher RR placement is not new RL competence or guaranteed current RR support. This report records incomplete exploration coverage, not a final episode failure, proposed gate, causal proof, parameter recommendation, completed block or future success. Fixed boundary review completed; stopped without reading108929 or later.
