# Block31 episode0: RR qualified and crossed, but never captured

Run `runs/ppo_semantic_v3/train/20260907T0313083674401Z_gf1a9bbf650b1_98167177c0ff4cd6b15dd57e491fa8c2`, runtime f1a9bbf650b1. This report is fixed to completed **episode0 only**, global108289–108740: **452 credited decisions /3616 physics ticks**, P07/P08/P09=**1/1/450**. No subsequent episode, optimizer log, checkpoint/hash, full-run ledger, raw/native full audit or active-course outcome was inspected. Only this report is written; no production/config/tests/master, Python/PT/GPU/Isaac or new gate.

The actual outcome is **P09 INCOMPLETE_CONTROLLER_BLOCKED at tick9568 /79.733333333s**, P09 age30s, taskfalse. The physical evaluator is valid with termination_reason=null and empty failure detail; entry remains valid. This is a task noncompletion, not a body collision, unqualified wheel climb, codec/I/O error or optimizer failure.

## Credit boundary and real rear events

The frozen-FSM teacher hands over P07 at tick5952/49.6s; `requested_phase_still_active_at_credit=true`, `from_P01_current_policy=false`. Front history was already earned by the teacher: FR Q/C/P71/1665/1695, FL2461/3115/3583. It is not new PPO credit. RR and RL had no hard Q/C/P at handoff.

Actual policy transitions are P07→P08 at5960 and P08→P09 at5968, both nonterminal. The first requires measured workspace_RR/workspace_RL/support_RR=1; the second workspace_RR/load_ready_RR=1. At5968 RR is still GROUND/load.154351, front−202.824mm and clearance−50.855mm, with no Q/C/P. Thus RR was not handed over already qualified or airborne.

Within the policy segment, RR has seven recorded whole-body initial-clearance events:6225/6272/6380/6458/6484/6518/6557. These are attempts, not seven qualifications. The first records3.066298mm upward excursion, own-joint motion4.396925°, whole-body joint motion15.642191°, commanded motion56.707946° and gravity-direction change.021957741. The detector uses the configured whole-body evidence, not a new requirement for a specific RR-only joint excursion.

**Actual hard RR Q is6588 (54.9s)**: recorded upward excursion50.717715mm and current clearance+1.192608mm. The qualification event's joint-motion diagnostic is11.844187°; it is not an added RR-only initial-lift gate. **Actual front crossing is6655 (55.458333s)**. RR has **no placed event**, RL no hard Q/C/P. Q/C remain earned history at the terminal; they do not mean current platform support.

## Geometry and support: a high airborne crossing followed by retreat

All values in this table are selected decision endpoints, not unrecorded substep extrema. `Body forward` is the recorded body-position scalar relative to the obstacle front, **not the robot CoM**; speed is a nonnegative magnitude, not signed longitudinal velocity. Support count is the measured-contact count.

| Tick | RR front mm | RR gap mm | RR current | FL current / load | Body forward mm | Body speed m/s | Support count |
|---:|---:|---:|---|---|---:|---:|---:|
| 5968 | −202.824 | −50.855 | GROUND/.154351 | AIR/0 | +93.835 | .116288 | 3 |
| 6224 | −67.591 | −46.835 | AIR/0 | AIR/0 | +109.445 | .108490 | 2 |
| 6584 | −45.161 | −4.295 | AIR/0, before hard Q | AIR/0 | +128.611 | .128396 | 2 |
| 6656 | +1.092 | +89.088 | AIR/0, Q+C | TOP/.190991 | +186.821 | .212161 | 3 |
| 6664 | −2.197 | +85.947 | AIR/0 | TOP/.163685 | +196.742 | .074742 | 3 |
| 6672 | −9.966 | +79.556 | AIR/0, outside top XY | TOP/.247554 | +198.908 | .029077 | 3 |
| 6776 | **+4.902** | +71.686 | AIR/0, inside top XY | TOP/.258248 | +206.232 | .060050 | 3 |
| 8000 | −68.518 | +109.300 | AIR/0 | TOP/.170448 | +155.533 | .003766 | 3 |
| 9216 | −145.745 | **+124.948** | AIR/0 | TOP/.154939 | +92.459 | .090516 | 3 |
| 9568 | **−150.051** | **+115.400** | **AIR/0** | **TOP/.274080** | +95.115 | .039449 | 3 |

Among the completed episode's post-Q decision endpoints, maximum forward distance occurs6776 (+4.902009mm), and maximum gap occurs9216 (+124.947892mm). The first sampled return behind the front after C is6664; the first sampled exit from tolerance-expanded top XY is6672. The wheel briefly re-enters the front region later (e.g.6776), so the earlier retreat is not described as irreversible at that instant. From6776 to9568, recorded body-forward decreases111.118mm and RR frontdistance decreases154.953mm. Both body motion and leg-relative motion participate in the observed geometry; this is association, not a force/actuator attribution.

RR has412 AIR,32 GROUND and8 obstacle-pair-active decision endpoints; **TOP0 and maximum consecutive_TOP_samples0**. Terminal recorded consecutive_AIR_samples=3013, covering6556–9568 inclusive and therefore the entire period from hard Q and crossing onward. This is the evaluator's actual recorded substep history, not an independent full raw-contact reparse. No sampled post-cross GROUND is found; do not describe the missing placement here as qualification revoked on a return to ground.

**The first unmet task is real RR capture/placement.** At the terminal, `placed_RR=.7` consists of earned lift/crossing but no current top geometry/contact credit. The RR wheel is150.051mm behind the front,115.400mm above top, outside top XY, load0, obstacle_pair_active=false, top_contact=false, top-count0. In-region geometry and actual loaded TOP capture never become placement. The high clearance is not evidence of safe arrival, and historical crossing is insufficient.

FL has101 AIR and351 trueTOP/obstacle-active decision endpoints, no sampled GROUND. It is indeed unloaded around the early RR attempts, but is truly loaded again at the observed crossing and late samples. At the terminal FR/FL are trueTOP with loads.403591/.274080, RL GROUND/.322329, RR AIR/0. Thus neither “FL always unsupported” nor “all historically placed legs always supported” describes this episode.

## Nominal, filtered request and actual drive are different records

Servo values below are canonical degrees; wheel values are rad/s in **FL/FR/RL/RR** order. `request` means the recorded projected/rate-filtered residual, not raw policy output. `native` is the mapped advisory target before post-mapper residual; geometry can modify that advisory and the final actuator stage can apply its own slew/hard bounds. `actual drive` is the recorded final canonical target, not measured joint position. It must not be equated indiscriminately with `nominal+request`.

| Tick | Nominal RR hip/knee | Request RR hip/knee | Native RR hip/knee | Actual drive RR hip/knee |
|---:|---|---|---|---|
| 6224 | −6.9 /−37.8 | −2.012593 /−8.626517 | −8.15 /−39.05 | −10.425185 /−48.426517 |
| 6584 | −6.9 /−37.8 | −5.555900 /−10.567572 | −8.15 /−39.05 | −13.705900 /−49.617572 |
| 6656 | −6.9 /−37.8 | +1.047652 /−8.581858 | −3.15 /−39.05 | −2.102348 /−47.631858 |
| 6776 | −6.9 /−37.8 | −1.763561 /−8.371101 | −7.624573 /−41.55 | −9.388134 /−49.921101 |
| 8000 | −6.9 /−37.8 | −4.063681 /−11.634294 | −7.624573 /−34.223111 | −11.688254 /−45.857405 |
| 9568 | −6.9 /−37.8 | −.725457 /−14.986794 | −7.624573 /−34.223111 | −8.350030 /−49.209905 |

Recorded controller drive bias is zero in these selected rows; the post-mapper combined bias retains the current policy residual. Native changes despite a fixed logical RR advisory are not automatically corruption: existing mapper compensation and its history remain part of dispatch. For example at9568, native RR knee−34.223111 plus residual−14.986794 gives actual−49.209905. Other joints can encounter a final limit: FL knee actual−58° here, rather than assuming every joint's final target equals an unconstrained sum.

There are real whole-body advisory and target changes around Q/C, not an isolated frozen RR command. The nominal eight servos are `[38.6,-13.4,0,31.1,31.2,0,-6.9,-37.8]` at6584, then `[-11.4,-31.4,0,31.1,15.4,19.4,-6.9,-37.8]` at6656, later `[-18.5,-31.4,0,31.1,15.4,19.4,-6.9,-37.8]`. Terminal actual eight-servo target is `[-17.892212,-58,-13.751088,-.820462,-4.930728,4.002147,-8.350030,-49.209905]`. These are commands, not proof that the corresponding joint movement caused upward/forward displacement.

| Tick | Nominal wheels | Residual wheels | Actual drive wheels |
|---:|---|---|---|
| 6584 | [.3,.3,.3,.3] | [−.330173,−.574011,−.421443,+.212203] | [−.030173,−.274011,−.121443,+.512203] |
| 6656 | [.125,.125,.125,.125] | [−.427955,−.717979,−.369494,+.359737] | [−.302955,−.592979,−.244494,+.484737] |
| 6776 | [.125,.125,.125,.125] | [−.296750,−.664222,−.118492,+.268289] | [−.171750,−.539222,+.006508,+.393289] |
| 8000 | [.3,.3,.3,.3] | [−.479575,−.697063,−.301947,+.449151] | [−.179575,−.397063,−.001947,+.749151] |
| 9568 | [.3,.3,.3,.3] | [−.440794,−.814754,−.421643,+.505471] | [−.140794,−.514754,−.121643,+.805471] |

Terminal raw wheel outputs are `[−.646778,−.827184,−.872690,+1.229561]`; they are not those filtered requests. Measured wheel velocities are `[−.217010,−.502947,−.171607,+.806045]`. Opposed wheel commands and observed retreat are concurrent facts, not a demonstrated unique cause; rotating an unloaded RR wheel is not evidence of useful platform traction. The terminal nominal diagnostics explicitly say **no live P06 layer**, so its .3 suggestion cannot be attributed to activation of the P06 endpoint-tail mechanism.

## Geometry branch and evidence limits

The recorded nominal geometry intervention is active in some preplace samples, but it does not guarantee physical forward motion or capture:

- At6224, status `projected_relaxed_forward`; the RR nominal adjustment is `[−.262593,−7.735080]` degrees, before residual/final dispatch. The final target is separately audited; this is not a request to impose the same geometry projection on the residual.
- At6384, `degraded_bypass_infeasible_box_downward`, conflict `box_vs_downward_halfspace`; nominal adjustment is zero and `physical_motion_guaranteed=false`. This is an explicit degraded branch, not a hidden claim that its linear constraint succeeded. It precedes the genuine Q/C later in the same episode.
- At9568, `identity_within_descent_allowance`, no adjustment. The source geometry at tick9567 has gap114.919mm and allowed descent99.919mm; its existing two-DOF, fixed-base **first-order nominal-only** estimate is Δx−4.127mm/Δz+2.324mm. These are modelled target-displacement diagnostics, not the measured next-step change, a full-body Jacobian inversion, or a causal prediction including residual/contact.

This compact log provides body-forward position, body linear/angular speed magnitudes and contact-count/load features. It does **not** provide robot CoM position, support-polygon margin, signed full body velocity, complete attitude trajectory, or exact obstacle contact-point/force vectors for this episode. Geometry's `link_minus_com_world_m` and COM-velocity identity checks concern the wheel rigid body's Jacobian origin; they cannot be relabelled robot-CoM motion/stability evidence. Partial RR measured joint coordinates exist in geometry contexts, not a complete measured whole-body trajectory.

The fixed452-row ledger is contiguous, every physical snapshot valid, every decision-end native audit verified, and the four in-episode state-write counts sum to0; teacher storage-credit flags are false. Selected dispatch checks (setter equality, mapping equality, same-tick counterfactual) all pass. No execution-chain mismatch is observed in this scope. This is deliberately not an optimizer or independent all-substep native audit, and does not prove the absence of all implementation defects.

**Conclusion:** the observed missing quantity is sustained actual RR platform capture after a real whole-body lift and legal crossing. RR stays unloaded and eventually high/behind the platform while front-leg support is partly recovered. That is valid incomplete exploration, not RR never qualifying, a return-to-ground revocation, or a proven actuator-chain break. No unique cause, new entry gate, RR-only actuation requirement, threshold change, suffix/full success or stability improvement is asserted. Fixed episode0 review completed; stopped without reading later episodes.
