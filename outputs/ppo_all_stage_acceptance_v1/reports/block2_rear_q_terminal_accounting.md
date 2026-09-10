# Block2 RR Q / P09 terminal accounting — saved windows only

Source: `block2_rear_windows/episode_000_windows.json` through `episode_002_windows.json`; no raw audit, active run, configuration, tensor, or simulation was read in this analysis. The 141 retained rows comprise P06=122, P07=2, P08=2, P09=15. They are event/extreme-selected, **not an unbiased sample or the entire 384-decision block**. The saved summary confirms no P01–P05 policy samples; two FALL terminals and one nonterminal P06 tail remain unchanged.

## Clock and physical anchors

Q is the original event tick; the controls below belong to its first-observed decision-end row, not reconstructed event-tick state.

| Anchor | Global / source phase | Event tick → decision episode tick / native command tick | RR front / clearance (mm) | Current FL / FR bearing-load fractions |
|---|---|---|---|---|
| A: ep0 Q | 139682 / P06→P07 | 4873→4880 / 5059 | −312.908 / −26.134, AIR | .065541 TOP / .463977 TOP |
| B: ep0 FALL | 139698 / P09 | 5008 / 5187 | −364.795 / +145.123, AIR | 0 AIR / .664191 TOP |
| C: ep1 Q | 139810 / P06→P07 | 4477→4480 / 4659 | −319.571 / −37.832, AIR | .045427 TOP / .482189 TOP |
| D: ep1 FALL | 139815 / P09 | 4519 / 4698 | −285.138 / +14.063, AIR | .204654 TOP / 0 AIR |

Neither rear leg has C/P in these windows. Current AIR/support and historical placement are separate. Q below platform-top height is recorded under the current common evaluator; it is not a missing-Q or wheel-only reclassification.

## Actual control accounting

The stored `canonical_order` is exactly `[FL hip, FL knee, FR hip, FR knee, RL hip, RL knee, RR hip, RR knee, FL wheel, FR wheel, RL wheel, RR wheel]`. Servo entries below are **canonical degrees relative to standing**; wheel entries are **canonical rad/s** unless explicitly marked native. Rounded here; original precision remains in the window JSONs.

`N` = logical nominal; `R` = filtered/projected REQUEST; `M` = `native_drive_target_full12` (mapped nominal, **not the final dispatched target**); `F` = final canonical actual-drive target. Controller-bias vectors are zero at A–D.

| Anchor | Front N: FLh,FLk,FRh,FRk | Front R | Front M | Front F |
|---|---|---|---|---|
| A | 22.8, −13.4, 0, 45.9 | 25.562, 13.722, .079, −74.523 | 21.550, −12.150, −.800, 45.347 | 47.112, 1.572, −.721, −29.176 |
| B | 49.2, −13.4, 0, 31.1 | 17.331, 18.283, .560, −44.023 | 52.277, −12.150, −.800, 27.716 | 69.608, 6.133, −.240, −16.307 |
| C | 22.8, −13.4, 0, 45.9 | 10.907, 18.279, −22.164, −46.229 | 21.550, −12.150, −.800, 45.347 | 32.457, 6.129, −22.964, −.882 |
| D | 40.8, −13.4, 0, 45.9 | 28.907, 20.387, −22.207, −64.229 | 40.800, −12.150, −.800, 45.347 | 69.707, 8.237, −23.006, −18.882 |

| Anchor | RR hip/knee N | RR R | RR M | RR F |
|---|---|---|---|---|
| A | 0, 0 | −20.242, 9.683 | −1.350, .954 | −21.592, 10.637 |
| B | 55.6, −19.8 | −22.950, 7.005 | 56.850, −21.050 | 33.900, −14.045 |
| C | 0, 0 | −23.688, −6.809 | −1.350, .954 | −25.038, −5.855 |
| D | 17.5, 0 | −12.222, −7.971 | 17.500, .954 | 1.128, −7.016 |

All four nominal wheels are +.3 at A–D and all 15 retained P09 rows. Actual wheels do not therefore all roll positively:

| Anchor | Wheel R: FL,FR,RL,RR | Wheel F, canonical | Actual float32 native wheel readback, signed rad/s |
|---|---|---|---|
| A | 1.176598, .637057, −.917863, .175085 | 1.476598, .937057, −.617863, .475085 | −1.476598, .937057, .617863, .475085 |
| B | .195917, −.763495, −.664673, .560667 | .495917, −.463495, −.364673, .860667 | −.495917, −.463495, .364673, .860667 |
| C | −1.199946, −1.195555, −.677487, .568933 | −.899946, −.895555, −.377487, .868933 | .899946, −.895555, .377487, .868933 |
| D | −1.199997, −1.036261, −.815457, .444713 | −.899997, −.736261, −.515457, .744713 | .899997, −.736261, .515457, .744713 |

The observed FL/RL native sign reversal is not a request reversal/mismatch. Native servo `servo_position_rad` likewise is physical radians with standing/sign transformation, not the canonical-degree vectors above.

**Mean/std, filtering and slew:** all 141 retained rows have identical requested/effective headroom residuals (maximum difference 0; no clipped servo indices). Five retained rows have a later candidate-versus-final servo difference: RR hip on four, RL hip on one; maximum 5.82532°. This does not establish that their sampled latent wanted a dynamically feasible movement.

- A FR knee: latent mean −.139814, std .459842, sampled raw −.048097 (dimensionless), yet R=−74.523104°. The preceding retained decision R=−78.523104°, so the actual change is +4° across 8 physics ticks; the last native target increment is +.5°. A near-zero current sample is not an instant zero residual or zero drive command.
- C FL hip: latent mean 6.441553, std 1.768689, sampled raw 3.524879; filtered R=10.907020°, final 32.457020°. These are distinct quantities, not a direct mean-to-target conversion.
- D RR hip: headroom candidate +5.278421°, preceding final −.121579°, dispatched final +1.128421°: **+1.25° last-tick advance**, 4.15° below the candidate. This is concrete final-slew limitation, not headroom clipping or proof of a mapping defect. A/B/C candidates equal their final canonical targets at the reported endpoints.

**Geometry:** all 15 retained P09 rows report `identity_within_descent_allowance`; the live model exists but made no correction there. At first P09 rows 139685/t4904 and 139813/t4504 its nominal-only linear `(x,z)` is respectively `(−25.995,+17.054)` and `(−27.910,+3.261)` mm. At B it is `(+28.478,−23.660)` mm, permitted by 133.723 mm available descent. These are fixed-base, two-DOF, zero-current-policy linear target predictions, not measured wheel travel or a forward-carry guarantee.

## Five weighted reward families

Values are the original `reward_breakdown.families`, already weighted. `Reg` is control_regularization. No endpoint re-evaluation was performed.

| Anchor | Task | Body | Contact | Smooth | Reg | Total |
|---|---:|---:|---:|---:|---:|---:|
| A, nonterminal Q | .00808127 | −.000308971 | −.000000796 | −.00235481 | 0 | .00541669 |
| B, FALL | −42.64377234 | −.000926181 | 0 | −.00329960 | 0 | −42.64799813 |
| C, nonterminal Q | −.02626598 | −.000290341 | −.000007366 | −.00245999 | 0 | −.02902368 |
| D, FALL | −42.64839705 | −.003140790 | −.000009555 | −.00241416 | 0 | −42.65396156 |

A Phi `.49194828→.49457306`, shaping `+.00941460`; C Phi `.49997956→.49573664`, shaping `−.02493265`. Thus a Q event can coincide with either sign of the **global** potential change; these rows alone do not isolate the contributing legs or prove a reward conflict.

B/D use recorded terminal event −40 and next Phi 0; shaping is −2.64243901/−2.64723038. Remaining task deductions are −.00133333/−.00116667 over 8/7 physics ticks. The task family accounts for about 99.99% of each terminal reward magnitude; body/contact/smooth penalties are much smaller in those intervals. Both terminal rows disallow bootstrap. Nonterminal event reward is 0.

For the retained **nonterminal P09** rows only:

| Episode / selected count | Task sum | Body sum | Contact sum | Smooth sum | Reg sum | Total sum / per-row range |
|---|---:|---:|---:|---:|---:|---|
| ep0 / 11 | −.03499278 | −.00976418 | −.00025199 | −.03220106 | 0 | −.07721001 / [−.01066682,+.01761673] |
| ep1 / 2 | .10121093 | −.00157572 | −.00003207 | −.00501646 | 0 | .09458669 / [−.00732858,+.10191527] |

This is 13 nonterminal rows plus two terminals, **not all 17 P09 decisions**. No complete-episode return or objective preference is inferred. `confirmed_post_touchdown_rebound=0` at A–D; nominal/residual difference entries remain diagnostics separate from the charged family totals.

## Which new range was actually exercised?

- **P03–P05 caps: untested by Block2 PPO.** There are zero credited rows in those phases; inherited teacher achievements cannot test their residual range. Large FR-knee residuals observed later in P06 do not change this conclusion.
- **P06 RL wheel negative expansion beyond the old ±.6 comparison range: exercised.** Of 122 retained P06 rows, 51 have R<−.6, none has R>+.6; range `[−.95543324,+.12000000]` rad/s. At **139679/t4856**, latent mean/std/raw=`−1.290951/.273653/−1.890689`; N=+.3, R=−.95543324, canonical final=−.65543324 and actual signed native=+.65543324 rad/s. This is an executed expanded negative residual, not merely a high cap or a large final wheel target.
- Retained P09 RL R is `[−.82861387,−.59826773]`: 13/15 below −.6, zero above +.6. P07/P08 each have two retained negative-expanded rows. These are bounded-window occurrence counts, not run-wide frequencies; positive expanded authority has **no observed exercise in these windows**. Exact phase-cap configuration values were not serialized into these window artifacts and are not reconstructed from one latent sample.

`geometry_sample_wall_s` and `geometry_point_counts`: **UNAVAILABLE in the generated window JSONs**. No conclusion about their presence in the original raw stream or actual geometry timing is drawn. RPY/gravity/raw base pose also remain unavailable; no unique FALL cause is reconstructed. This report recommends no reward/range changes and introduces no success or optimizer gate.
