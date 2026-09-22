# Physical task quality audit — read-only, 2026-09-20

Scope: latest attachment `07768750-afbe-4786-91ec-cda8b624bdb9` fully read;
production reward/supervisor/sensing/observation/height diagnostics inspected;
sealed block13 and block14 plus existing successful B only. No physics launch,
optimizer, production edit, checkpoint mutation, or synthetic training credit.
Current root-verified HEAD is 73c128d0f43c628a0d0c3bdb8e60c5f3414a6233;
runtime files are unchanged from e735. CP185856 is a saved training point, not
an already demonstrated full-task policy.

## Facts affecting the next implementation

1. **Height is already available in training task records.**
   `semantic_task.physical_evaluator.body_traversal_geometry.minimum_w_m` and
   `maximum_w_m` contain all three coordinates of the current-pose enabled
   base collider mesh AABB. `_all_stage_body_bounds` validates the live geometry
   and exact base/obstacle pair. `_all_stage_finish` copies this into the task
   snapshot at every valid tick, even before P13. Earlier reports correctly
   lacked the base-link origin z, but their stronger claim that training had no
   vertical body-collider bound overlooked this existing field.
2. **Placed FL can be AIR without losing any current capture-retention credit.**
   `physical_potential` assigns a placed leg `.8 + .2*retention`.
   `_current_capture_retention` tests legal XY and only the lower vertical
   bound; any positive gap inside XY gets full retention. This deliberately
   permits later transfer AIR, but also leaves early P06 dropout invisible to
   this share. `top_contact`, current bearing, and history remain distinct.
3. **P05 descent signal exists; do not append another touchdown bonus.**
   Current legal-XY FL proximity is `.5/(1+gap/.025)+.5/(1+gap/.003)`; it is
   .296822 at 26 mm, .472527 at 10 mm, .696429 at 3 mm, and 1 at zero/negative
   gap. With the existing leg/capture/geometric shares and weight 5, closing
   26 mm to zero changes the geometric component of `5*Phi` by .074713,
   before discount, contact, or other simultaneous state changes. The real TOP
   half still needs sensor samples. This is a weak but nonsaturated slope,
   not absent reward, and does not imply positive normalized advantage.
4. **P01/P02 quality really was optimized; later quality remains zero.**
   Block14 added 512 decisions / 4 PPO updates / 80 steps: request coverage
   P01/P02/P03/P04/P05/P06 = 2/204/4/1/148/153. Signed body-family contributions
   P01=-.0000206477530 and P02=-.0186889418982; all later body/contact/smoothness
   family contributions are zero. Existing body term is positive-floor
   beta=.015–.03 per second, half tilt and half actual Euler derivative;
   there is no current height term. Ordinary phase changes are not terminal.

## Sealed physical evidence

Block13 run: `20260918T0953571182841Z_ge73542cb57ad_fb762812ab4e4b52a77c8707d38e5d38`.
Block14 run: `20260918T1037290906944Z_ge73542cb57ad_befe799ac0b0461ca2c4f9189be96966`.
B: `20260918T0555572376368Z_gf2e552406ea7_1894338b350241fcbbac4868c5f7f3fb`.

The following are decision-endpoint samples, not 120 Hz contact occupancy or
causal matched-rollout measurements. B and C have different durations and
closed-loop states; block14 updates its network during collection.

| Window | Samples | Minimum collider world z | Actual contact evidence |
| --- | ---: | ---: | --- |
| B P02 | 182 height samples | 91.272 mm | Successful FR placement later at tick1502 |
| Block14 P02 | 204 learner endpoints | 86.671 mm | FR placed tick1680 |
| B P06 | 310 decision endpoints | 125.122 mm | FL/FR TOP at all 310; all four wheels non-AIR |
| Block14 P06 | 153 learner endpoints | 120.300 mm | FL AIR122, contact31; FR AIR4; FL placed2871 persists |
| Block13 P09 | 142 learner endpoints, multiple episodes | 49.998 mm | Three body-collision episodes; first RR Q later revoked on ground |

These z values are **not automatically body-to-obstacle clearance**. The collider
minimum is a real mesh extremum, but its XY can lie outside the platform. AABB
separation is a conservative geometric lower bound, not exact mesh distance.
The 50 mm obstacle top must not be subtracted and relabeled as exact clearance
without the projection/surface qualification. Do not use a historical base_z
target, reward taller forever, or weaken the existing collision evaluator.

Block14 P06 first/last endpoints2880–4096 span10.133333s: base forward distance
increased from -.174738 to +.040685 m (+215.423 mm); RL front distance improved
-.491133→-.288025 m and RR -.477508→-.264510 m. This is real forward motion,
not proven wheel stall. N stayed `[.3,.3,.3,.3]` rad/s on all four wheels.
Four-wheel final/actual values (FL,FR,RL,RR) include:

| Tick | Final targets rad/s | Measured rad/s |
| --- | --- | --- |
| 2880 | [.217616,.329856,.452931,.177651] | [.189032,-.027809,.369296,.174959] |
| 3120 | [.374792,.314640,.448701,.196196] | [.375047,.142712,.624566,.261446] |
| 3480 | [.402577,.541289,.435257,.207127] | [.402635,.591178,.335969,.610812] |
| 4096 | [.229963,.376315,.595342,.206018] | [.376136,.376732,.595368,.174019] |

Wheel actuation, contact load, and net propulsion are not interchangeable.
N's eight joint targets were held throughout these153 requests, while all
eight residual joints varied. This supports inspecting unnecessary linkage
motion, not yet the claim that linkage supplies most propulsion. In B P06,
held FL hip N was24.9deg; block14 held22.8deg after a different capture state.
Same nominal implementation is not identical N values at unmatched clock times.

## Minimal implementation options: at most two reward changes

**1. One explicit task-conditioned quality profile, retaining the existing front
tilt/rate budget and adding a bounded geometry-deficit component.** Reuse live
body collider bounds plus obstacle planes for a conservative minimum-separation
lower bound, and world minimum z for ground clearance. Give the deficit a finite
physical-mm margin and saturation, justified by the current probes/required
space rather than B's height trace; no benefit above sufficient space. Apply
while preparing as well as after lift, so no circular "must already succeed to
earn quality" gate. RR's component should measure insufficient space without
punishing necessary lateral transfer or imposing FL contact. Existing wheel-gap,
contact, and RR qualification remain task terms. A small P06-only linkage
rate/reversal cost may be a subcomponent of this same bounded profile only if
the new probe confirms useless cycling; use measured joint velocity, not the
existing all12 target-difference cost, and retire it at rolling-window exit.
Do not add it now merely because targets move. Keep actual forward displacement
and rear-edge progress visible; wheel absolute speed is never its reward.

**2. Replace—not append—the existing placed-front retention share during the
physical rolling window with current landing/support usability.** Keep .8 earned
history and reuse the .2 retention budget. Blend current verified TOP bearing
and legal near-top geometry so transient AIR is a recoverable soft regression,
never fake placed or new hard failure. Activate after both fronts are placed
while rear wheels are still behind the preparation region; smoothly retire by
measured rear-edge approach before RR transfer. Do not require both fronts to
remain contacting during P07–P09. Define this from current geometry/history,
not a fixed hold time or resets at phase labels. Continue ordinary PBRS
`5*(gamma*Phi_next-Phi_prev)`, terminal Phi=0, and no phase bonus. This directly
distinguishes a single touchdown followed by immediate lift from a useful P06
entry without a second gap/force bonus or unbounded downward pressure.

The hypotheses above require the root's physical probes and a declared version;
neither is a proven fix. No reward or sample was changed in this audit.

## Existing data/diagnostic interfaces, no actor schema expansion needed first

| Quantity | Reliable existing source | Interpretation/limit |
| --- | --- | --- |
| Body collider bounds | task.physical_evaluator.body_traversal_geometry | Current live-pose mesh extrema; all stages |
| Exact lowest collider point | HeightDiagnostics fresh_collider_bounds.base_link.lowest_collider_point | True mesh vertex xyz; not fake `(link.x,link.y,minZ)` |
| RR installation height | HeightDiagnostics rr_hip_mount_w_m | USD hip body0/localPos0 transformed by live parent body_link pose |
| Other hip mounts | Same USD joint resolution generalized by name in diagnostic only | Not currently emitted; never substitute upper-leg origin/CoM |
| CoM | raw.center_of_mass and task.transfer_roles.*.transfer_direction_context | Mass-weighted13 bodies, current xyz/velocity, direction projection; not proof receiver bearing |
| Current support | physical_evaluator.current_legs | ground/top/current bearing with validity; history alone insufficient |
| Limb usable space | wheel gap/XY; role receiver_workspace_state.joint_range_margin_deg | Joint-margin and radial-contraction proxy, not exact Cartesian reachability |
| Joint actual position/velocity | raw.joints or diagnostic native reads | Canonical8deg/deg/s vs native rad/rad/s must remain explicit |
| Wheels | N/projected/actual_drive_target plus evaluator measured_wheel_velocity | Canonical FL,FR,RL,RR; contact/traction separate |
| Policy decomposition | policy_request base_mean/history_center/conditional_mean/effective_sigma/selected_raw | Same draw used by real likelihood; conditional not all network mean |

`src/wlr50_clean/ppo/semantic_height_diagnostics.py`: HeightDiagnostics.start(frame),
sample(after, terminal=...), close(last_frame). Existing video integration is in
semantic_video.py around lines646–656 and714. Samples every8ticks plus start/end.
It owns only output files and issues no setter/step/reset. Missing getters remain
null with reasons. Its drive efforts are implicit-PD buffer estimates, not
measured PhysX drive torque. Diagnostic and task ticks must be preserved; native
target audit actual q is pre-dispatch while decision contact is post-step.

The372 actor already observes body orientation/obstacle planes, joint states,
wheel gap/contact, CoM, role history, and action/mapper history. Add needed
geometry to reward metrics/audit from existing raw/task fields, not new policy
dimensions by default. Exact hip mount heights can stay diagnostic initially;
only extend policy inputs if live evidence identifies an unrepresented ambiguity.
