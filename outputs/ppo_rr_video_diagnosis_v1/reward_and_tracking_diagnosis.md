# RR early divergence: measured diagnosis

## Delivered first pair

B0 and C0 are fresh sequential natural-P01 runs, seed4001, currentHEAD4b2c038, identical372-observation/HISTORY12-action contract,120Hz physics/15Hz policy, no teacher and no optimizer updates. The initial recorded joint q/qd, standing offsets, wheel speeds, root pose/velocity, raw mass-weighted world CoM and contact pairs are exactly equal. The first-pair runtime content is ecee8d3abba1981b95d462a70c9ea39deb3e949878caea9d6589c95cdc043458. See `reference_and_runtime_identity.json` and `video_manifest.json`.

| Run | Measured outcome | First unfinished task |
| --- | --- | --- |
| B0, N+zero | P09,73.833333s,8860ticks,TASK_INCOMPLETE | RR lift/cross/place. Local age30.7s exceeded current finite30.6920087356s limit; source LOCAL_BOUNDED_RECOVERY_EXHAUSTED. RR grounded,front distance−49.6797mm,top gap−50.3546mm,no crossing/placement |
| C0, N+checkpoint166784 | P02,8.191667s,983ticks,SAFETY_ABORT/HARD_JOINT_LIMIT | FR sustained approach/cross/place. RR knee actual−60.007387692deg, final target−5.654126832deg. No measured base-obstacle contact |

The B0 local threshold is the frozen30s base deadline plus0.6920087356s current-progress allowance. It is not a file/encoder failure. Its numeric placed_RR progress0.26306 is a shaping quantity, not a true placement event. C0's first small hard-limit overrun does not explain its already54.353degree tracking error. Original unsuccessful manifests remain unchanged.

## What changed first

All12 N logical request channels are identical in the common983ticks. B0 all8860 native audits verify raw learned action, effective policy residual and same-tick direct policy effect are exactly zero. Its original mapper tracking feedback remains active: nonzero8823ticks, maximum10degrees; post-mapper controller bias happens to be zero. This is one shared N/control path, not a residual-zero shortcut to a different FSM.

At the first executed tick, PPO already changes all12 final channels. For example RL hip target changes fromB0+1.25deg toC0+0.75deg and RR knee from0 to−0.5deg. The first mapped-servo cross-run difference occurs later at FR knee,tick105/0.875s, after trajectories diverge. RR mapped N stays0 in both common prefixes.

| C0 observation | Start of first12-consecutive-tick interval | Actual confirmation tick | Meaning |
| --- | ---: | ---: | --- |
| RR knee abs tracking error≥1deg | 35 /0.291667s | 46 | target−2.255808deg,actual−3.281206deg at start |
| ≥2deg | 50 /0.416667s | 61 | diagnostic onset only |
| ≥5deg | 178 /1.483333s | 189 | target−5.200861deg,actual−10.205189deg at start |
| Real terminal | 983 /8.191667s | 983 | target−5.654127deg,actual−60.007388deg |

These thresholds select diagnostic windows; they were not introduced as task gates or safety limits. B0 RR knee peak absolute tracking error is only0.865149deg through its firstP03 prefix. The paired data alone demonstrate policy-altered targets followed by a growing actual tracking failure, but cannot isolate direct RR channels from remaining whole-body policy changes. The subsequent RR-off intervention below reproduces the failure with zero direct RR corrections. Small direct RR residual does not exclude indirect load/contact effects.

A bounded recheck of the already extracted records at ticks0/35/50/178/976 (five perrun) does **not** support an early loss of RR ground contact. RR remains GROUND with verified pair and positive normal force at all ten points. B0/C0 have the same contact set: all four at0, FL/RR at35/50, FL/RL/RR at178/976. At35 the RR ground normal force is15.007/14.845N (B0/C0); at178 it is12.509/13.098N; at976 it is12.226/14.845N. Thus tracking error grows while RR still has ground contact force, followed by lower body/CoM; it is not demonstrated RR contact loss. The exported normalized-bearing fraction and validity fields at these points are null, not an invented valid load fraction. Normal forces alone do not identify instantaneous joint torque or its causal pathway.

`rr_first_divergence.csv` contains306 exact120Hz rows:153 perrun, windows0–120,166–189,976–983. It does not imply continuous coverage between those windows. Missing substage/projection flags/torque fields stay blank, not false or zero. Native and physical are joined by the same post-step episode tick. Actual−final is e_tracking; final−same-run mappedN is e_residual, which may include non-policy bias and slew. A same-prestate native counterfactual isolates only the current policy correction, not accumulated policy effects across runs. Raw base Euler and calibrated body Euler are separate; world CoM is not a normalized observation feature.

## A versus current N

Frozen A reference is Trial043/v010_manual/RR_FIRST in `C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1`. This turn reuses the existing source/provenance report; it does not reclassify an old failed A run or claim a new A success. The scene factory, RobotAdapter and command_batch sources have no differences from A.

N is not temporally identical to A: A P01 ends at13.3333s, whereas both current semantic runs enterP02 after16ticks. The unfinished RL hip/wheel preparation owner continues across that task-stage boundary. Source P01 RL hip progression9→17.5→33.4→37.6degrees and wheel+.3 suggestion are retained; P02 FR knee progression is45.9degree final. N also has an explicitly conditional wheel approach after actual FR lift evidence. These are concrete scheduling/feedback differences, not a justification to silently omit the source motion. Old P03 correction and handoff fixes are already present; no new missing-source repair is asserted here. B0 retains front-leg task capability in this pair; its later RR failure remains real.

No cross-run same-tick difference against A is presented as residual causality. A event-window numerical e_reference remains separate from the exact B0/C0 first-divergence measurements; this turn did not replay all historical recordings.

## Reward: a tension in one component, not demonstrated reward preference for failure

Compare exactly122 returned decisions/976ticks/8.133333s. C0's final seven physics ticks did not return a reward record in the video callback path and are excluded; no missing−40 terminal penalty is invented. The training terminal path is separate.

| Signed contribution | B0 | C0 |
| --- | ---: | ---: |
| Potential shaping | +0.532875632 | +0.293800748 |
| Time | −0.162666667 | −0.162666667 |
| Body attitude | −0.038108920 | −0.010821611 |
| Euler rates | −0.000542194 | −0.000475790 |
| Angular acceleration | −0.006501020 | −0.005938633 |
| Contact | −0.000011567 | −0.000020098 |
| Applied first difference | −0.010348619 | −0.014740015 |
| Applied second difference | −0.016778141 | −0.022411975 |
| Residual regularization / recorded terminal | 0 /0 | 0 /0 |
| Total | +0.297918504 | +0.076725958 |

C0's weighted attitude cost is71.6% smaller, but its total is0.221192545 lower. At tick976 its base origin is21.975mm lower and CoM47.833mm lower than B0; angular speed0.20233 versus0.01479rad/s,FR top gap−13.607 versus+99.137mm. More level therefore does not mean better stability here. Existing potential/smoothness terms penalize C0 enough that the combined prefix reward does not prefer it.

Current body/contact weight is1−0.8*physical_transfer_fraction. It does not switch to full penalty merely onQ. The fraction can fade with measured motion/support evidence: B0 tick104 has FR I/Q/AIR still true but fraction0.22222,weight0.82222. In the common prefix C0 endpoint fractions stay0.875–1. Endpoint fractions are not120Hz integrals. Applied first/second-difference smoothness is not transfer-discounted. These facts justify checking incentives but not claiming the network intended to sink for points. No speculative reward, residual-bound, gain or hard-limit change was made.

See `combined_reward_interpretation.md` for exact component reconstruction and event-window limits. Actual torque saturation and per-channel reward saturation counts are unrecorded, not zero. A separate CPU regression verifies changed reward weights can change reward while the same frozen actor/observation produces bitwise identical actions; it is a synthetic test, not learning improvement.

## Actual drive readback and current follow-up

The single RR-raw-off diagnostic records actual PhysX getters after the ordinary reset/settle, before its first action. Actual readback: all eight servos K600/D60,maxforce2.7000000477Nm,maxspeed4.9999995232rad/s; four wheels K0/D20,maxforce1Nm,maxspeed2.0943951607rad/s. Actual backend/controller/mapper/motion executor/sensor module paths point into this worktree. These observations match inspected configured values; no evidence warrants increasing physical drive force or stiffness. This is a new diagnostic boundary readback, not a retroactively fabricated B0/C0 runtime receipt. Getters do not measure instantaneous drive torque or solver reaction.

Diagnostic run `runs/ppo_fsm_reference_p09_stable_v2/rr_channels_off_diagnostic/20260914T0503017583189Z_g4b2c038887c4_0d7983dafbee4b78ad64ba31363625f4` completed at998ticks/8.316667s, earlier than its15s bound, with P02 SAFETY_ABORT/HARD_JOINT_LIMIT. RR knee final target0deg,actual−60.018800938deg. RR hip target0deg,actual−1.301465466deg. It disabled only raw6/7 from the start, retained actual masked HISTORY and remaining actor control. Its125history checks and final actor/critic/Adam/normalizer unchanged assertion passed. This reproduces failure without direct RR learned corrections, so those corrections are not necessary for this failure. Other policy-channel effects on the shared whole-body dynamics remain sufficient to reproduce it; this does not prove all PPO actions are irrelevant or identify a reward motive. Instantaneous torque/contact causal details remain unknown. No on-policy samples, full-policy success or optimizer update are credited to the intervention. The separate diagnostic video is `videos/rr_channels_off_diagnostic.mp4`, not an after-repair or formal C0 video.

Production controller/reward/physics changes this turn: none. Material code work is confined to reusable diagnostic export/extraction and the explicit bounded intervention. The comparison exporter PTS bug was repaired and its invalid first attempt preserved. The three before videos were delivered before this follow-up diagnostic or training.

## Actual learning and independent reload evaluation

The bounded continuation completed640 new policy decisions,5 PPO updates,100 optimizer steps:512 naturalP01 decisions and128 physically initialized P06-precursor decisions. The latter contains P06=115/P07=1/P08=1/P09=11; all896 teacher-prefix decisions are excluded. The first32 suffix decisions ended inP09/FALL; the second96 stayed nonterminal inP06. The native stop request was consumed only after the full128-row update and verified checkpoint save, leaving384 of the originally requested second-block decisions unconsumed. No ordinary phase transition became done. Latest checkpoint167424 has1273 updates/25460 optimizer steps; the original166784 is preserved.

Its independently reloaded deterministic naturalP01 evaluation, same seed4001 and unchanged control/reward/physical version, again ended with P02 RR knee HARD_JOINT_LIMIT at879ticks/7.325s. Final RR knee target−6.358187707deg,actual−60.016799693deg,e_tracking−53.658611986deg. FR qualified-lift history exists, but current top clearance−16.112mm/front distance−147.151mm and no crossed/placed event leaveP02 FR clearance/approach unfinished. The terminal RR still has ground contact/bearing15.839683N; measured initial physical arrays and contact pairs equalB0. This single after-learning result is earlier failure, not improvement or proof of a uniquely identified reward cause. Video `videos/ppo_after_learning.mp4` is110 original frames/7.333333s, complete playable and shown; it is after-learning, not after-repair. Exact load/endpoint/scope evidence is in `after_learning_eval_receipt.json`.
