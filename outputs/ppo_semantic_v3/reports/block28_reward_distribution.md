# Block28 reward distribution — fixed1792-row read-only audit

Source: `runs/ppo_semantic_v3/train/20260907T0134080107746Z_g42b91e857a0a_a862752b4ec84772b24f061befcd1676`, runtime42b91e857a0a2da256dee85e8092642ca8e64aeb. Exactly1792 fixed policy rows were parsed once; no active block29, tensors, hashes, GPU/Isaac or optimizer was read/run. Only this report is written; master and all outcomes remain unchanged.

`applied_audit.reward_breakdown` exists for every row and includes all five weighted families, potential before/after, shaping, terminal event, elapsed physics time and detailed cost components. Family sums exactly equal the nested total. Stored outer PPO reward differs from the higher-precision nested total by at most9.049876e−7 (recorded numeric representation), not missing reward data.

Statistics below are **nonterminal per-issued-decision** values, not returns, value targets, gradients or successful-episode probabilities. P02 n394, P05 n308, P06 n1077 plus one separately reported terminal (1078 P06 total); the other12 rows are P01/P03/P04. Every interval in this block is1/15s. Quantiles use linear interpolation at rank(n−1)q.

## Five-family weighted means

| Weighted family | P02 | P05 | P06 nonterminal |
|---|---:|---:|---:|
| task_progress | -0.0018947 | -0.0040459 | -0.0123246 |
| body_stability | -0.0009669 | -0.0004086 | -0.0005247 |
| contact_motion_quality | -3.525e-6 | -1.015e-6 | -5.188e-6 |
| control_smoothness | -0.0023773 | -0.0024960 | -0.0031656 |
| control_regularization | 0 | 0 | 0 |
| Total reward | -0.0052424 | -0.0069515 | -0.0160201 |

All non-task families are nonpositive. `control_regularization` is exactly0 for every selected sample; the historical config disables it (weight0). These data do not show an explicit positive rest bonus or a global residual-magnitude penalty. Actual-drive difference costs are nevertheless real negative terms; absence of magnitude regularization does not mean motion is cost-free.

## Per-step distributions and potential

| Phase / quantity | Mean | p05 | Median | p95 | Minimum / maximum | Positive samples |
|---|---:|---:|---:|---:|---|---:|
| P02 / reward | -0.0052424 | -0.0171500 | -0.0068964 | 0.0048513 | -0.0264194 / 0.1688055 | 59/394 |
| P02 / shaping | -0.0005614 | -0.0119825 | -0.0022442 | 0.0089128 | -0.0216932 / 0.1735114 | 144/394 |
| P02 / phi_step | 0.0006887 | -0.0016324 | 0.0003208 | 0.0026777 | -0.0036128 / 0.0353117 | 241/394 |
| P05 / reward | -0.0069515 | -0.0545121 | -0.0114102 | 0.0550163 | -0.1352919 / 0.2965751 | 53/308 |
| P05 / shaping | -0.0027125 | -0.0503703 | -0.0072574 | 0.0593793 | -0.1306657 / 0.3007973 | 76/308 |
| P05 / phi_step | 0.0012158 | -0.0084406 | 0.0002463 | 0.0136497 | -0.0244807 / 0.0619266 | 174/308 |
| P06 / reward | -0.0160201 | -0.0299104 | -0.0158206 | -0.0034731 | -0.1212383 / 0.0689419 | 43/1077 |
| P06 / shaping | -0.0109913 | -0.0248367 | -0.0109109 | 0.0014960 | -0.1168750 / 0.0737095 | 67/1077 |
| P06 / phi_step | -8.218e-5 | -0.0028302 | 0 | 0.0023606 | -0.0212500 / 0.0169517 | 516/1077 |

P06 workspace `rear_approach` is exactly0 in all1077 nonterminal samples (and at its terminal), **but global Φ is not flat**: post-step Φ spans.391481937–.459198947, mean.423215938. ΔΦ is positive516 times, negative536 and exactly0 in25. The remaining potential reflects recorded global physical-task state; these aggregates do not attribute every change to a particular leg or command.

Historical42b91e8 `semantic_reward.py` and `reward_config.yaml` specify shaping `5*(.995*Φ_after−Φ_before)` and a time charge`.02*dt`; task family weight1, stability.4, contact.2, smoothness.1, regularization0. All three phase group means agree with this formula. Task-family remainder after shaping/event is consistently−.001333333 per decision.

For P06, mean shaping−.010991289 decomposes algebraically into **5*meanΔΦ=−.000410891** and **−.025*meanΦ_after=−.010580398**. A positive small ΔΦ need not yield positive one-step shaping. This is discounted potential shaping, not by itself a proof that moving forward is punished in cumulative PPO objectives or that policy-gradient training prefers stagnation. Comparing state-dependent one-step rewards without future/terminal terms would be misleading.

## Cost-component distributions

These are the stored integrated/normalized diagnostic components, **not raw angular units or additional reward families**. Cells show mean / p95; the weighted smoothness family uses only actual-drive first/second differences, so nominal/residual diagnostic differences must not be counted a second time.

| Cost component | P02 mean / p95 | P05 mean / p95 | P06 mean / p95 |
|---|---|---|---|
| confirmed_post_touchdown_rebound | 0 / 0 | 0 / 0 | 0 / 0 |
| contact_chatter_diagnostic | 0.0062817 / 0.0250000 | 0.0035985 / 0.0166667 | 0.0068284 / 0.0250000 |
| touchdown_events | 0.3781726 / 2.0000000 | 0.2110390 / 1.0000000 | 0.3909006 / 1.0000000 |
| nominal_first_difference | 0.0004218 / 0 | 0.0008485 / 0.0055556 | 6.319e-5 / 0 |
| residual_first_difference | 0.0338895 / 0.0450468 | 0.0341064 / 0.0440672 | 0.0465019 / 0.0563171 |
| actual_drive_first_difference | 0.0344105 / 0.0455275 | 0.0352364 / 0.0450579 | 0.0477546 / 0.0569805 |
| actual_drive_second_difference | 0.0131352 / 0.0150541 | 0.0146833 / 0.0178233 | 0.0155582 / 0.0175688 |
| gravity_attitude | 0.0019615 / 0.0022675 | 0.0002484 / 0.0004903 | 0.0004441 / 0.0013562 |
| euler_rate | 0.0002672 / 0.0006228 | 0.0004213 / 0.0013610 | 0.0004125 / 0.0012096 |
| angular_acceleration | 0.0050230 / 0.0134191 | 0.0023948 / 0.0049586 | 0.0030784 / 0.0081754 |
| contact_quality | 1.762e-5 / 0.0001131 | 5.077e-6 / 2.394e-5 | 2.594e-5 / 0.0001352 |

Historical configuration selects `smoothness_components=applied_only` and contact-loss semantics `recent_touchdown_without_active_command_change`. Confirmed rebound is0 in every selected sample; chatter/touchdown diagnostics do not themselves mean every contact departure was penalized as rebound. P06 contact-quality cost is nonzero462/1077 steps and its weighted mean is only−.000005188; smoothness is nonzero1077/1077. Nominal first difference is nonzero only4/1077 while actual/residual differences are nonzero throughout. Thus constant nominal is not evidence that actual policy actuation or measured body motion stopped.

## Bounded check of the “global stagnation preference” hypothesis

P06 samples were split descriptively at their recorded median body linear speed **.083985526m/s** (not a success/entry threshold). No samples, observations, actions or phase histories were matched, so this is an observational check only.

| Nonterminal P06 group | n | Mean body linear speed(m/s) | Mean total reward | Mean shaping | Mean stability | Mean smoothness |
|---|---:|---:|---:|---:|---:|---:|
| Lower-speed half | 539 | 0.0539128 | -0.0160061 | -0.0109952 | -0.0005122 | -0.0031628 |
| Higher-speed half | 538 | 0.1313056 | -0.0160342 | -0.0109874 | -0.0005371 | -0.0031685 |

The mean total-reward difference is only+0.000028066 in favor of the lower-speed half; shaping is nearly equal. These groups are not stationary: mean body speeds.053913/.131306m/s, maximum-wheel-speed means.633633/.667169rad/s, with changing controls and different states. **No controlled evidence of a global stop/stagnation preference is established.** The measured facts are weak P06 task progress, negative discounted shaping, persistent smoothness cost and negligible contact cost in this run. A cost for target changes can make smoother actions locally cheaper, but these data cannot prove that the learned policy deliberately traded completion for inactivity, or isolate reward causality from control/state/curriculum effects.

## Terminal preserved separately

The single terminal is global102333/tick7656/P06 INCOMPLETE: total **−41.990904857**, terminal event−40, Φ_before.397256509→0, shaping−1.986282547. Weighted task family−41.987615880, stability−.000142550, smoothness−.003146427, contact/regularization0. Do not mix this one terminal with the1077 ordinary P06 steps and then claim their normal reward is approximately−42. The835-decision second tail remains nonterminal, not a second failure or success.

Conclusion: the logs are sufficient for reward-distribution measurement, and show substantial non-workspace reward structure; they do **not** establish an execution defect, causal stagnation preference, need for a new hard gate or a justified parameter change. All failed/incomplete outcomes and planned-budget accounting remain preserved. No implementation or tuning proposal is issued.
