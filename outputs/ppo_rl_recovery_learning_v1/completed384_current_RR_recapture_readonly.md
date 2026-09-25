# Completed384: current RR recapture read-only evidence

Only 225281–225664 / updates1726–1728 /60 Adam steps. Prefix and unfinished tail receive no credit.

| Physical endpoint window | Count |
| --- | ---: |
| RR_reachable_AIR_preparation | 48 |
| RR_actual_bearing_front_preparation | 28 |
| positive_FR_axis_CoM_body_projection_with_RL_fraction_decline | 22 |
| qualified_RL_edge_recovery | 0 |
| RL_qualified_AIR_capture_region | 0 |
| RL_actual_TOP_bearing | 0 |
| RL_placed_history | 0 |
| P13_actual_stage | 0 |
| all_task_complete | 0 |

Positive local fixed-FR-axis projections of both CoM and body displacement, plus declining RL load fraction while RR and a front leg currently bear. NOT verified lateral/rightward transfer, NOT absolute RL force decline, NOT qualified RL unloading. Forward movement alone can make the projection positive; a load fraction can decline because other legs load more. World xyz components and current absolute forces are reported separately; yaw/body-frame decomposition is not inferred from the projection.

Third episode RR-ground / unqualified-RL samples: 75. Cooperative relevant: 0.
RR recapture remains an explicit actor task and all12 are open. Strong dependent source lanes pause; changing policy requests can still move the suspended FL/RL targets. This is not a source-mask block on RR/FR/wheels.
However, after prior RR placement, illegal/ground RR has zero current-retention progress and skips workspace/lift/carry shaping. FL-range/RL-space shaping also disappears when cooperative relevance becomes false. Physical potential is not phase-only, but this recovery region is locally sparse in RR recovery guidance.
Observed owner H can remain different from actual q: freezing an already committed target does not stop residual tracking/inertia immediately. Numeric H/q/final/request separation and exact clocks are in JSON.

At decision225664 / episode tick7000 (58.333 s), current RR is ground+obstacle, RL is unqualified ground; all four suspension anchors remain active. Servo degrees below use current canonical coordinates. The q column is the measured state before the final dispatch (one physical tick before the reported endpoint); H is held from the earlier suspension, not redefined from q.

| Channel | Input H | Input r0 | Final target | Measured q | Final−q |
| --- | ---: | ---: | ---: | ---: | ---: |
| FL hip | 13.752 | −4.743 | −12.574 | −10.816 | −1.757 |
| FL knee | −55.303 | −20.816 | −58.000 | −57.974 | −0.026 |
| RL hip | −21.076 | −22.443 | −12.729 | −11.653 | −1.076 |
| RL knee | 36.417 | 1.117 | 38.200 | 38.313 | −0.113 |
| FR hip (not suspended) | N/A | N/A | 12.407 | 60.031 | −47.624 |
| FR knee (not suspended) | N/A | N/A | −20.820 | −15.323 | −5.497 |
| RR hip (not suspended) | N/A | N/A | 4.759 | 8.522 | −3.763 |
| RR knee (not suspended) | N/A | N/A | −41.388 | −42.393 | 1.004 |

The 75 RR-ground/unqualified-RL samples have FL knee final exactly −58° in all75, although its raw conditional mean varies from −2.085 to −0.946. This is policy/request-plus-projection saturation, not a missing residual permission. The anchor is not a neutral q hold: changing requests remain effective (e.g. FL/RL hips move substantially), and the single endpoint's large FR tracking error is a measured discrepancy, not proof of a mask or a cause of the earlier RR drop. These data do not justify turning off the policy or reopening strong source unloading while RR lacks support.

No production changes or intervention. No attribution of success to projection or learning; episode3 is still unfinished in this bounded prefix.

## Source facts

- semantic_rear_policy_timing.public_timing: current loss after RR placed sets rr_carry_capture even at P12, not hidden phase-only work.
- semantic_cooperative_preparation.measure_preparation: RR ground or outside topXY sets relevant=False, removing FL-range/RL-space proxy contributions.
- semantic_supervisor.physical_potential: history-placed RR bypasses workspace/lift/carry and gets .8+.2*current retention.
- semantic_supervisor._current_capture_retention: RR illegal/ground while RL not actual swing returns0; recapture direction outside legal region is not shaped by that retention share.
- RL without current RR support and no qualified AIR receives only .1*workspace; no false unload/lift progress from history placed RR.
- All12 actual mask stays1. Owner17 suspension changes dependent FL/RL composition, not raw Gaussian samples. Wheels, FR, RR remain ordinary compose/limit lanes.
- Observation includes actual joints, wheel geometry/velocities/forces and current explicit rr_carry_capture, dependency waits and owner17; no observation-only explanation that P12 must be RL transfer.
