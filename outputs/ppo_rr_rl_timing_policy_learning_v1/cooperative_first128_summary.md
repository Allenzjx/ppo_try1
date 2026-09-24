# New cooperative policy: sealed first128 read-only audit

Scope: only run `20260924T0038084596548Z_g49eb23163a6e_baf0b6006fea46a3a731db4c4d63d633`, decisions **223233–223360**, first completed optimizer block. No Torch, model loading, physical execution, or production edits in this audit. Subsequent active samples are not included.

## Count and checkpoint

- Saved checkpoint: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000223360.pt`
- SHA256: `e9c99a457d64f44ee3d3870094f7b4b1000d868b64e3c7bdf8c2d375a55bc409`
- Actual learning: **128 policy decisions / 1 PPO update / 20 optimizer steps**, cumulative CP223360 / update1710.
- Successful-N continuous prefix: P01→P07, 43.000s / tick5160 / 645 prefix decisions, **zero PPO credit**. This is not a natural-P01 policy success.
- Sample phases: P07=1, P08=1, P09=64, P10=2, P11=8, P12=52.
- Rear capture assists stayed disabled. The 14 RR-assist observation slots were zero.

## Actual gate and physics evidence

- Cooperative exploration gate: **29/128** requests; extra FL-wheel factor: **11/128**. RR hip total rear factor remains **×4** in current RR carry+reachable, otherwise ×1; no additional RR broadening.
- First active request: decision223288, request tick5600; measured endpoint tick5608 / 46.733s. Current RR qualified AIR, no TOP/no bearing, gap48.801mm. Observed gate was carry=true, reachable=true, RL-prep=false, parent-receiving=false. Multipliers [FLhip,FLknee,FRhip,FRknee,RLhip,RLknee,RRhip,RRknee,FLwheel,FRwheel,RLwheel,RRwheel] = [1,2,1,16,2,8,4,1,2,1,1,1].
- RR current-qualified: **90** decision endpoints; TOP and current bearing: **8** endpoints each. RL current-qualified: **4** endpoints. These are endpoint counts, not 120Hz contact durations, sustained capture, RL placement, or task completion.
- Final audited endpoint: tick6184 / **51.533s, P12**, RR AIR, gap**6.297mm**, front-distance−11.932mm, current bearing0. The first128 therefore does **not** prove retained RR support.
- Cooperative preparation diagnostic relevant:32 endpoints. It does not count AIR as support. At first active endpoint FL negative-limit margin15.263°; RL wheel AABB clearance proxy102.406mm, explicitly not whole-linkage collision verification.
- Counterroll diagnostic: 1,024 physical-step samples,107 eligible,44 positive-cost samples. Positive cost is diagnostic/reward activity, not proof of policy correction.

## Actual capability on the 29 prep-active requests

Raw / conditional mean / sigma are dimensionless Gaussian quantities. Servo request, post-headroom residual, and final targets are degrees; FL-wheel quantities are canonical rad/s. Every residual/final value below is from the **same decision's last physical-step receipt**, not an independently recomputed nominal subtraction. Ranges across requests are not isolated causal effects.

| Channel | Conditional mean | Effective sigma | Original raw | Requested residual | After headroom residual | Actual final target | Distinct final targets | Headroom clipped endpoints |
|---|---|---|---|---|---|---|---|---|
| FR_knee | -1.4621 … -0.0052 | 0.2359 … 0.4978 | -1.5841 … 0.4870 | -85.8658 … -48.3062 | -85.4794 … -48.3062 | -58.0000 … -20.8268 | 24/29 | 1 |
| FL_knee | -2.1773 … -1.1815 | 0.1733 … 0.3759 | -2.2490 … -1.1414 | -35.2073 … -29.3355 | -35.2073 … -29.3355 | -47.3573 … -41.4855 | 29/29 | 0 |
| RL_hip | -1.1240 … -0.2987 | 0.1578 … 0.1861 | -1.1855 … -0.2623 | -19.6262 … -6.1556 | -19.6262 … -6.1556 | -27.1456 … 22.0444 | 29/29 | 0 |
| RL_knee | -0.1329 … 0.0949 | 0.0354 … 0.1076 | -0.1414 … 0.0975 | -5.0573 … 3.4995 | -5.0573 … 3.4995 | -5.0573 … 37.6301 | 29/29 | 0 |
| FL_wheel | -2.3428 … -0.8446 | 0.0664 … 0.7769 | -2.4531 … -0.8145 | -1.1824 … -0.8065 | -1.1824 … -0.8065 | -1.4325 … -0.5065 | 29/29 | 0 |

All five channels had29 distinct raw values. FR-knee had24 distinct final targets; the only matched logged servo safety/hard boundary was−58° at one endpoint (one distinct raw). Other listed servo channels had no matched logged boundary. **The data do not show many varied raw samples collapsing to the same hard-bound target.** This does not exclude within-decision slew limitation or other clipping outside these endpoint receipts. FL-wheel's headroom receipt has no wheel hard-limit field; no wheel boundary is inferred.

**FL wheel remained negative in all29 prep-active final endpoint targets (−1.43245 to−0.50651 canonical rad/s).** The observed exploration gate is working, but this block does not show corrected forward FL execution. The residual requests were also negative; this is not evidence of an omitted wheel/mask.

## Sampling and PPO likelihood

-128/128 original requests, raw samples, conditional means, effective sigmas and old log probabilities exactly match saved rollout receipts.
-20 minibatches; each saved sample appears5 times. Shared cooperative gate/multipliers match request-time evidence. Single policy draw/forward per request; no teacher, diagnostic or prefix data credited.
- Scalar independent Normal log-probability check: sample max absolute error 0.000002662950414844545; minibatch current-logp error 0.0000023911222726269443. Sigma errors <=2.384185791015625e-7.
- Actor changed, finite nonzero gradient observed, LR1e−5, mean KL0.0325, clip fraction0.3375.
- Native target audit verified1,024 physical ticks. This check validates its audited stream; it does not convert commanded targets into a guarantee of physical tracking.
- Likelihood belongs to the original Gaussian raw sample, **not** the projected actuator target. No new action transformation or sigma change is inferred from target ranges.

Result: **new exploration/likelihood wiring is evidenced on29 active samples**, with actual target variation. Real RR touch/bearing occurred but was absent at the last audited endpoint. This is only the first sealed training block, not deterministic evaluation or whole-task success.

## Physical event attribution and support continuity follow-up

**This entire first128 was collected before PPO update1710.** The rear events below are the stochastic execution of saved CP223232 parameters under the new cooperative sigma and frozen49eb control/reward. They are useful training experience, **not evidence that the subsequent CP223360 optimizer update learned capture**. Unlike the old P10 teacher-placed block, these events occurred after this run's P07 learner start at tick5160.

| Event | Tick / time | Actual evidence |
|---|---|---|
| RR qualified lift |5428 /45.233s | Current unsupported measured AIR lift; before crossing |
| RR crossed |5602 /46.683s | Recorded actual event |
| RR placed history |5682 /47.350s | Recorded actual learner event, not prefix |
| First sampled RR TOP/bearing |5688 /47.400s |0.3643N, load fraction0.01269; consecutive TOP8 physical samples |
| First sampled loss |5696 /47.467s | AIR /0N; first loss bracket **(5688,5696]**, not an asserted exact120Hz loss tick |
| Stronger RR reacquisition |5768–5816 /48.067–48.467s | Seven successive decision endpoints TOP; force15.158,15.097,15.742,15.267,15.021,13.501,13.886N; consecutive TOP count reaches54 physical samples |
| Second sampled RR loss |5824 /48.533s | AIR /0N; loss bracket **(5816,5824]** |
| RL qualified current lift |5772 /48.100s | Real measured AIR rise9.036mm; four qualified endpoints5776,5784,5792,5800 |
| RL ground revocation |5802 /48.350s | Explicit qualification_revoked_ground_before_cross; not crossed, no TOP/placement |
| Audited ending |6184 /51.533s | RR AIR gap6.297mm; RL GROUND; neither retained RR support nor completed RL |

The stronger RR segment is more than a solitary low-force touch, but **support was subsequently lost**, so it is not sustained capture success. At the four RL-qualified endpoints, its front-distance moves−109.66→−167.55mm and gap remains below platform top (−36.81,−33.93,−38.67,−47.00mm): this is a brief genuine unload/lift followed by ground return, not RL crossing.

### Why prep is false at the end

The last request223360 observes tick6176; RR is current-qualified AIR, but `within_top_xy=false`, outside the accepted XY region by8.427mm. At endpoint6184 it is still outside by6.932mm (front-distance−11.932mm). Both have verified other supports FR+FL+RL, so **this particular reachable=false is not a shortage of other supports**. Current bearing=false and RL current swing=false also remove the alternative preparation branches. The unchanged reachable conjunction at `semantic_rr_capture_context.py:98–102` requires accepted XY in addition to the current AIR/actual-support evidence. Historical RR placed does not supply present support.

The full compact event/force endpoints are retained in `cooperative_first128_summary.json:physical_event_followup`. No later rollout samples were read.

