# C118016 final evaluation — FL crossed but did not capture

Run `validation/20260907T0707251670686Z_gf4bfe2560bfd_b8a1a4c5cb244de4951774a4483803e5`, runtime f4bfe2560bfd, checkpoint118016, N1/seed2001/natural P01/deterministic policy mean. Final lifecycle **SUCCEEDED execution**, with root-confirmed exit0, is **not task success**. Actual **669 decisions /5352 physics ticks /44.6s**, **P05 INCOMPLETE_CONTROLLER_BLOCKED**, stage age30s; taskfalse, physical evaluator validtrue/reason empty/physical terminationnull. Evaluation optimizer updates0 and `window_ended_before_task_terminal=false`.

Only final manifests, the completed669-row compact decision ledger, terminal raw5352 and existing quality metrics were read. No checkpoint/PT load, repeated hash, Python/GPU/Isaac, subsequent training, production/config/test change or causal experiment.

## First missing physical task

Phase counts P01–P13 are **[1,213,4,1,450,0,0,0,0,0,0,0,0]**. P02/P03/P04/P05 enter at ticks8/1712/1744/1752, each a nonterminal handoff with bootstrap allowed. No later phase is sampled.

| Leg | Hard Q / C / P ticks | Terminal current contact / load | Front / top gap mm |
|---|---|---|---|
| FR | 48 /1723 /1740 | TOP /.418400197 | +96.225444 /−.181654 |
| FL | **1843 /2869 /absent** | **AIR /0, not support** | **+6.726686 /+31.968420** |
| RL | None | GROUND /.507382481 | −559.271258 /−50.597036 |
| RR | None | GROUND /.074217323 | −556.258139 /−51.281403 |

The first missing task is **actual FL platform capture/placement**, not qualification or historical crossing. Terminal entry remains valid, `placed_FL=.7` is fractional predicate progress rather than placement. Current FL is within top XY, but top_geometry=false, obstacle pair inactive and consecutive_TOP_samples0. Its recorded terminal AIR streak3572 ticks corresponds to1781–5352; this is stored evaluator substep evidence, not a separately rescanned raw-contact history.

Of450 P05 decision endpoints,447 are AIR and3 are GROUND; obstacle-active0, trueTOP0 and top_geometry0. Historical FL crossing is present at311 endpoints. After crossing, maximum sampled frontdistance is+32.234672mm at2904, with gap+85.134020mm. The smallest sampled absolute top gap is **+25.384521mm at2944**, front+11.823360mm, AIR/load0. The unchanged runtime geometry bound is−15..+25mm, with5mm XY tolerance and two required consecutive loaded TOP samples. These are decision-end extrema, not120Hz extrema. No sampled proximity or historical Q/C establishes actual supporting contact, and the .384521mm excess is not itself evidence of a sensor or threshold bug.

## Terminal raw contact, geometry and safety

Raw5352 is finite; FL wheel geometry is verified. Center x .528038859367m minus obstacle front .521312173774m reproduces front+6.726686mm. Wheel bottom z **.081968419822m** minus obstacle top .05m reproduces gap+31.968420mm. Both exact FL ground/obstacle pairs are verified inactive with0N and no obstacle contact point. Thus neither the historical crossing nor the current XY location makes FL a support.

Current supports are **FR/RL/RR**, count3/valid. FR's verified obstacle force is approximately `[0,0,12.017849922]`N with contact point `[.611498833,−.331828624,.050888512]`m; RL/RR verified ground normal magnitudes14.573718071/2.131769180N. The measured CoM projection lies inside the support polygon by19.472737008mm. Body collision detected/real_pair_active/persistent are false, penetration0, reason `no exact base_link/obstacle contact`. The evaluator has no hard physical termination: this remains a true task deadline, not a collision, hard-joint abort or recording failure.

## Native continuity and terminal return

The669 eight-tick decisions sum exactly5352, with no episode tick/time discontinuity. Existing receipts report **5352 verified native ticks /5352 actual-effect ticks /5348 own-request-effect ticks**; the four ordinary handoff holds account for the difference. All endpoint setter/dispatch, actual mapping, same-tick counterfactual and independent previous-ACK checks pass. Four in-episode root pose/root velocity/force-or-impulse/gravity write totals are0; physical-invalid snapshots0 and finite fallbacks0.

First/last native dispatch187/5531 correspond to episode8/5352, a consistent179-tick offset. Their previous ACK/feedback samples186/5530 immediately precede dispatch, and mapper feedback ticks equal dispatch. This reconciles existing receipts and episode clocks; it does not independently replay all native buffers.

All668 nonterminals retain bootstrap, including four ordinary phase transitions; time_outs=false throughout. The sole true terminal has pre-potential .4099000024761695, absorbing next-potential0, PBRS−2.0495000123808476, event−40 and total reward−42.05153398637684, with bootstrap disabled. Whole evaluation return is−43.29768341637331. No PPO update or additional training credit comes from this evaluation.

## Existing quality statistics, incomplete window

| Diagnostic | Whole44.6s | P05-only30s |
|---|---:|---:|
| Roll RMS rad | .128848634 | .042961421 |
| Pitch RMS rad | .097607314 | .043467177 |
| Body x angular-speed RMS rad/s | .082592888 | .073210838 |
| Body y angular-speed RMS rad/s | .059360700 | .045013035 |
| Angular-acceleration RMS rad/s² | 4.092257956 | 3.074608160 |

Whole-window angular-acceleration peak78.455615434rad/s²; skipped nonfinite ticks0. **fixed_quality_score=null, all_phases_sampled=false** because P06–P13 are absent. The P05-only score .473270428749 is not a complete-task or paired-FSM score. No stability-superiority, gamma-causal, failure-cause or success claim follows from these diagnostics.

Final conclusion: FL earned genuine Q/C but remained airborne without loaded platform capture until P05's deadline. This report adds only the completed evaluation result; prior training ledgers remain intact. No gate, parameter/detector change, future result or successful-video claim is proposed. Report/master append finished; writing stopped.
