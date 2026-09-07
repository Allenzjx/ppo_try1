# C119040 natural-P01 evaluation — FL crossing without capture

Final, read-only. Run `validation/20260907T0752476310693Z_ge8462f066908_f4de72f296d54255bccbc7bd5b92c2ac`, runtime e8462f066908, saved `history/checkpoint_step_000119040.pt`, N1/seed2001/natural P01. The recorded policy is `history_conditioned_heteroscedastic_log_v1`, deterministic **conditional mean**, rho 0.9, observation324/action12; this is not a sample from its Gaussian innovation. Parent confirmed process exit0/PID123068 gone; finalized run lifecycle is `SUCCEEDED` execution, not task success.

This audit waited for the final evaluation manifest, then made one selected-field pass each over the completed decision, native, and physical-observation ledgers. No active/new training data, PT/Torch/Python, new simulation, production edit, or repeated file hashing was used.

## Final outcome and first missing task

- **658 decisions /5264 physics ticks /43.8666666667 s**. All decisions have8 ticks; there is no partial terminal interval.
- P05 age30 s: **INCOMPLETE_CONTROLLER_BLOCKED**, task/controller success false. Physical evaluator validtrue, physical terminationnull/reasonempty; not BODY_COLLISION, interface failure, or an unfinished evaluation window.
- Evaluation optimizer updates0; `window_ended_before_task_terminal=false`.
- First missing task: **real FL platform capture/placement after a valid active lift and front crossing**. Terminal `placed_FL=0.7` is partial progress, not completed placement.
- FL ends AIR/load0/supportfalse, front distance **−43.662306 mm**, clearance **+54.577028 mm**, outside top XY by38.662306 mm after the existing5mm measurement tolerance, TOP geometryfalse, TOP samples0. Both contact and current platform location remain missing.

## Event history is not current support

Q/C/P mean hard qualified lift/front crossing/placement. All events here belong to the natural-P01 evaluation, with no teacher prefix.

| Leg | Q tick | C tick | P tick | Actual terminal contact/load | Front / bottom clearance (mm) |
|---|---:|---:|---:|---|---:|
| FR | 60 | 1642 | 1654 | TOP, 0.423923333 | +35.796228 / −0.524380 |
| FL | 1750 | 2806 | none | AIR, 0 | −43.662306 / +54.577028 |
| RL | none | none | none | GROUND, 0.488419982 | −587.756429 / −50.293980 |
| RR | none | none | none | GROUND, 0.087656685 | −593.846176 / −50.313475 |

FL's hard crossing remains a valid historical event even though the wheel later retreats. It is not proof of placement or present support, and retreat does not retroactively falsify the observed crossing.

The complete120Hz raw P05 interval is tick1665–5264,3600 samples: **11 GROUND +3589 AIR**, with the continuous final AIR streak **1676–5264**. FL obstacle-active samples0; raw recomputed top-envelope samples0. Wheel collider geometry and both exact contact pairs are verified throughout this interval. The450 P05 decision ends independently contain449AIR/1GROUND,0 obstacle/TOP/TOP-geometry;308 ends are after the crossing, only3 within top XY.

Selected actual raw samples (all AIR, both FL pairs inactive, obstacle force0):

| Tick | Meaning | Front distance (mm) | Bottom clearance (mm) |
|---|---|---:|---:|
| 1750 | hard qualification | −142.429646 | +0.130517 |
| 2806 | first crossing; post-cross maximum height | +0.356380 | +94.134619 |
| 2813 | furthest post-cross forward position | +2.146367 | +88.706349 |
| 2824 | retreat behind front | −1.812574 | +66.512232 |
| 2855 | closest post-cross raw top gap | −18.278834 | +31.452109 |
| 5264 | terminal | −43.662306 | +54.577028 |

Decision-only extrema differ as expected: closest post-cross gap+31.481316mm at2856/front−18.416997mm; furthest front+1.818541mm at2816/gap+84.250008mm; highest gap+93.087634mm at2808. None is contact. The raw minimum is still above the existing+25mm top-envelope ceiling and behind the front; more fundamentally no exact loaded obstacle pair ever appears. No lower contact standard or event reclassification is proposed.

## Stage continuity and terminal accounting

Phase P01–P13 decision vector: **[1,203,3,1,450,0,0,0,0,0,0,0,0]**; the independent ledger matches the finalized manifest telemetry.

| Transition | Tick | Simulation time (s) | Terminal/bootstrap |
|---|---:|---:|---|
| P01→P02 | 8 | 0.066666667 | false/true |
| P02→P03 | 1632 | 13.600000000 | false/true |
| P03→P04 | 1656 | 13.800000000 | false/true |
| P04→P05 | 1664 | 13.866666667 | false/true |

These four forward handoffs preserve the physical clock and task history; there is no phase-label terminal or reset between them. P06–P13 were never entered. All657 nonterminal decisions allow bootstrap; the one true deadline terminal does not, has absorbing nextPhi0, and has `time_outs=false`. Adjacent recorded Phi matches exactly across all nonterminal decision boundaries, including handoffs. This is evaluation continuity evidence, not a claim that evaluation performed GAE or optimization.

Terminal reward: prePhi0.4052620115170668, PBRS−2.026310057585334, failure event−40, total−42.028362451227565. Complete evaluation return−43.24933885826473. Neither partial FL progress nor software execution is rewarded/reported as a task success.

## Native execution, physical validity, and endpoint command

All5264 native rows are sequential episode ticks1–5264, native dispatch ticks180–5443 (constant179 offset). Every row is verified, float32, finite12-target shaped, has changed-target effect, and passes setter/dispatched equality, actual mapping/dispatch equality, and same-tick counterfactual verification. Decision summaries reconcile **5264 verified/effect ticks,5260 own-phase effects +4 handoff holds**. The inner native receipt's source/request phase describes its own held request; the compact own-phase count is the cross-handoff ownership evidence.

All four in-episode root-pose/root-velocity/force-or-impulse/gravity write counters are zero. No inspected native-check, shape, finite-target, clock, or terminal-finite-fallback discrepancy occurred. All5265 raw samples including tick0 have `all_finite=true` and sequential clocks; body collision detected/real-pair/persistent counts are each0. This does not certify unobserved behavior or imply stability superiority.

The terminal raw independently gives FL centerx0.477649867535m against obstacle front0.521312173774m and bottomz0.104577027829m against top0.05m. Both exact FL ground/obstacle pairs are verifiedinactive, force `[0,0,0]`, pointnull. FR's true obstacle pair has force `[-1.543965,-0.004955,12.168188] N` and measured point approximately `[0.564999,-0.322934,0.051721] m`.

Actual supports are FR/RL/RR, not FL. Their measured polygon is valid, CoM projection inside, signed margin **+28.431977 mm**. Terminal base velocity is approximately `[-0.011477,-0.004964,+0.028361] m/s`; CoM velocity `[-0.014177,-0.011276,+0.010863] m/s`. These finite support/motion measurements explain valid-but-incomplete classification, not the cause of capture failure.

At the endpoint FL nominal hip/knee is `[22.8,-13.4]` degrees; policy residual `[+7.670141,-8.274923]`; actual canonical drive target `[29.892107,-20.424923]`. Four nominal wheel commands are0; actual FL/FR/RL/RR wheel commands are `[-0.08476180,-0.07783369,-0.06774518,+0.18284905] rad/s`. The terminal servo-clipped list is empty. These endpoint values do not establish an all-trajectory no-clipping claim or a Cartesian causal explanation. The independent prior C118016 control review is not a paired ablation of this changed conditional policy.

## Limited quality interpretation

Recorded incomplete-window roll/pitch RMS are **0.120451125 /0.095246567 rad**; body angular-acceleration RMS **4.438567835 rad/s²**, peak108.001050530. Quality covers5264 ticks, nonfinite terminal skips0, but `fixed_quality_score=null`, `all_phases_sampled=false`. These are finite diagnostics only—not a full-task score, controlled comparison, stable improvement, full/suffix success, or successful video.

Conclusion: this evaluation achieved real front-leg lift/crossing and FR placement, but FL never captured the platform and retreated while remaining airborne. The first missing task and unchanged hard-contact criterion agree with the actual raw sensors. No execution-chain anomaly was demonstrated. The subsequent same-MDP training run was not inspected or credited. This report is final; only this output and the authorized completed-evaluation master append were changed.
