# Natural P01 training: bounded P06 diagnostic

Read-only bound: global **227585–228683**, exactly 1,099 complete decision rows; final physics tick **8792**, simulation **73.2666667 s**. No later row, future episode, model, or simulator inspected. This is stochastic training evidence, not a frozen-checkpoint deterministic regression verdict.

Source: `runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0938059083357Z_g72e63592bdf4_889c2d62e7fe433ea0dc3e99787a26ca/residual_and_projection_audit.jsonl`. No runtime/config/source changes.

P05→P06 occurred at tick **6280 / 52.3333333 s** via the existing pending-capture continuation, explicitly without physical placement credit. The first actual FL `placed` event is **6289 / 52.4083333 s**. Adjacent measured endpoints show TOP contact/bearing: tick 6288, **5.227 N**; tick 6296, **5.155 N**. Endpoint forces are not falsely assigned to exact event tick 6289.

From P06 entry (6280) to this bound (8792):

- Body-forward coordinate relative to obstacle front: **−218.249 → −38.080 mm**, delta **+180.168 mm**.
- Mass-weighted CoM world xyz: **(285.015, −127.279, 154.814) → (464.279, −137.307, 165.638) mm**; delta **(+179.264, −10.028, +10.824) mm**.
- RR/RL front distances: **−521.928/−530.533 → −308.207/−321.119 mm**. Both rear wheels remain GROUND with current lift invalid. Forward progress occurred, but neither rear wheel is near its crossing window.
- The final 0.5 s window is different from total P06 progress: body xyz displacement **(−2.833, −0.052, −8.448) mm**, CoM **(+1.438, −1.125, −6.974) mm**. No claim that net forward motion proves side-directed transfer. Absolute body y/z are not inferred from collider bounds.

## Last same-tick four-wheel pathway

Canonical order FL, FR, RL, RR; positive means forward. Wheel N, effective correction, FINAL and measured speeds are rad/s. Mean/raw are dimensionless pre-tanh policy requests; they are not interchangeable with the effective correction. `mapped N` is the actual recorded same-tick mapper baseline, not an independently replayed nominal.

| Wheel | Source N / mapped N | Conditional mean / selected raw | Recorded effective correction | FINAL target | Measured velocity | Current contact / bearing N |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FL | +0.300 / +0.300 | -1.149298 / -1.221802 | -1.008222 | -0.708222 | -0.708432 | AIR / 0.000 |
| FR | +0.300 / +0.300 | +0.000245 / +0.008618 | +0.010341 | +0.310341 | +0.209796 | TOP / 12.206 |
| RL | +0.300 / +0.300 | -0.562899 / -0.617359 | -0.549287 | -0.249287 | -0.067323 | GROUND / 13.246 |
| RR | +0.300 / +0.300 | -0.056447 / -0.056212 | -0.033692 | +0.266308 | +0.257863 | GROUND / 3.567 |

All four residual permissions are **1**. Their tanh requests are **[−0.840185, +0.008618, −0.549286, −0.056153]**, caps **[1.2, 1.2, 1.0, 0.6]**; prior filtered requests are **[−0.983865, −0.003364, −0.529586, −0.037704]**. The table uses the logged final effective residual, retaining HISTORY/filter provenance rather than labeling an independently computed N difference as a fresh policy effect.

The actual native actuator wheel targets are **[+0.708222, +0.310341, +0.249287, +0.266308]**; native FL/RL signs differ from the canonical forward-positive axis. Same-pre-tick-state counterfactual without current PPO residual is native **[−0.3, +0.3, −0.3, +0.3]**. `same_tick_counterfactual`, `actual_mapping_matches_dispatch`, and `setter_dispatch_targets_equal` are all true; final audit is the robot target readback after the existing write-to-sim call. Thus FL/RL canonical reversal is not explained away by native axis signs or only by a mask=1 assertion.

At this bound P06's source wheel contribution is enabled, gain=1, source clock=2513; no source endpoint stop has issued. The existing FL capture assist owns only servo indices **0/1**, not any wheel; its before/after wheel values are identical. `all12_policy_channels_unmodified_at_actuator=false` is therefore not evidence of a wheel mask: the observable FL servo transform is active. Its current status is **BLOCKED / finite_hip_travel_or_margin**, holding FL hip **4.619735°** and knee **−31.985676°**. This separate control fact is not attributed to wheel residual.

Conclusion: N retains four forward wheel commands. The recorded policy/HISTORY requests and effective composition reverse **FL and RL**, not only RR rotating. FL currently has **0 N** contact and **27.622 mm** gap, so its motor rotation cannot be credited as ground traction. RL has real ground bearing and negative actual rotation, while FR/RR rotate forward under different tracking/load conditions. This is a concrete opposing-drive condition worth local follow-up; these simultaneous observations do **not** establish it as the sole cause of P06 delay or FL loss of contact. No new reward/control fix, optimizer credit, deterministic result, or future-run outcome is claimed here.

