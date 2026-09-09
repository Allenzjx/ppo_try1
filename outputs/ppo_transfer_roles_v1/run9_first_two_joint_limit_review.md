# Run9: first two HARD_JOINT_LIMIT episodes — bounded safety review

Run `20260908T0317319786616Z_gdb991aa68103_0ae9bc64aad14ce7b2943de60d407031`, frozen runtime `db991aa68103`. Scope: first two completed episode records, their terminal compact receipts, and existing source. No later episode analysis, production/test edits, Python, simulation, tensor load or hash work.

**Judgment:** the selected receipts do not demonstrate a target conversion, projection or native-dispatch defect requiring a production repair. They record actual runtime **SAFETY_ABORT / HARD_JOINT_LIMIT**, not ordinary incomplete task outcomes. Preserve both as failed physical exploration data. A 2° target reserve is not a guarantee on measured joint position; increasing it is not an evidence-supported general remedy, particularly for episode1.

| Actual terminal evidence | Episode0 | Episode1 |
|---|---:|---:|
| Credit decisions / terminal phase | 241 / P12 | 10 / P12 |
| Episode tick / time (includes prefix) | 9505 / 79.208333 s | 7664 / 63.866667 s |
| Named joint / canonical index | FR knee / 3 | RR knee / 7 |
| Same-dispatch pre-step measured canonical angle | −59.9874319338° | −59.9995671255° |
| Pre-step margin to −60° | +.0125680662° | +.0004328745° |
| Mapped nominal before residual | +21.1° | −17.2° |
| Requested → effective residual | −109.2475412479 → −79.1° | +14.1377343625 → same |
| Final canonical target / previous final target | −58° / −58° | −3.0622656375° / same |
| Reconstructed dispatched float32 target | −57.9999971495° | −3.0622656087° |
| Target margin to lower hard bound | +2.0000028505° | +56.9377343913° |
| Headroom clipped indices | [3] | [] |
| Final decision native verified/effect ticks | 1/1 | 8/8 |

Both final native audits report setter/dispatch equality, actual mapping match and verified=true; all four state-write counters are0, terminal bootstrap=false, finite fallback=false. These are terminal-receipt checks, not an independent whole-run audit.

## Why this is not an observed out-of-range target

`semantic_residual_adapter.py` composes the unique mapper/geometry result with same-tick residual headroom, then applies `bounded_drive_feedback_step`, hard clamping and one physical sign/standing conversion before dispatch. `semantic_headroom.py` explicitly describes its original2° reserve as a **target** interval, not a physical motion guarantee. Episode0's clipping arithmetic is exactly `21.1−79.1=−58`; the output was already held at−58 before the terminal tick, so this was not a new outward terminal target jump. Episode1's commanded RR target is far inward and unchanged; its endpoint clipping would remain inactive for ordinary small increases in the reserve.

`command_batch.py` uses front sign+1/rear sign−1 and `physical_rad = radians(standing_deg + sign*canonical_deg)`. RR standing+.2152439935° with measured physical1.0509467125rad gives canonical−59.9995671255°; dispatched physical.0572033338rad gives−3.0622656087°. There is no sign or degree/radian discrepancy in this example. Native verification proves target delivery, not achievable tracking torque or absence of contact/inertial forcing.

## Measurement and numerical boundary

`guard_state._joint_limit_violation` and `TaskEvaluator.observe` both test measured canonical knees strictly within[−60,210] without epsilon. The named physical evaluator reason independently agrees with the backend HARD_JOINT_LIMIT classification; physical.valid=true means the observation contract is usable, **not safety success**.

The compact terminal does not retain the **post-step exact offending q/qd, drive torque, resolved external joint load, or solver residual**. Its tracking receipt is pre-step evidence. Thus it does not quantify how far beyond−60° the next measurement moved. `robot_adapter.py` deliberately gives native PhysX endpoints a `PHYSX_TARGET_QUANTIZATION_MARGIN_RAD=1e−5` envelope (about .000572958°), while logical guards remain strict. A tiny numerical/solver boundary crossing versus a larger dynamic excursion cannot be separated from these records. Do not invent a sensor error, relax the hard criterion, or claim that all overshoot is a large mechanical event.

The measured whole-body context is non-static: episode0 has FR/RL supports with loads .510675/.489325, FL/RR AIR, body speeds .044680m/s and .205672rad/s; episode1 has FL/RR supports .458478/.541522, FR/RL AIR, .463947m/s and .623695rad/s. Actual wheel commands FL/FR/RL/RR are respectively [−1.060397,−.297944,−.011071,+.583305] and [−1.590000,−.249437,−.092271,+.292878]rad/s. These are relevant coupled-motion observations, not proof that a particular wheel, support pair or reward caused the limit crossing.

**Boundary-after-run implication:** retain the actual safety failures and native receipts. No target/threshold/physics change is warranted solely by these two endpoints. If a later diagnosis needs to distinguish strict-boundary quantization from dynamic overshoot, the missing post-step offending q/qd and actuator/contact dynamics are the discriminating evidence; a larger static reserve or a rerun of A is not a substitute. No new execution gate is introduced here.
