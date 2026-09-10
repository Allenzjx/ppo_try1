# Block3 geometry inverse: bounded target versus large reported offset

Read-only scope: current `a8b148463115` source and existing retained ep2 g139970/g139971, ep3 g140002/g140021. No new audit pass, simulation, Torch, test execution or production edit.

**Finding:** the +54.514° RR-hip entry is an active-branch inverse offset relative to the raw mapper target, not a 54° commanded jump or an accumulated source pose. The recorded behavior matches the explicit current contract and existing identity/active-inverse tests. There is a real piecewise change in residual effectiveness when the geometry branch switches; the current contract does not promise continuity of that effectiveness. No new contract-violating defect is established here.

## Exact order and what is bounded

`semantic_residual_adapter.apply_semantic_residual` advances the nominal mapper once, then calls geometry **without the current policy residual**, then computes residual headroom around that corrected nominal plus bounded controller bias, then applies the single shared final hard clamp/slew, float32 conversion and dispatch. It updates only the actual final-drive history after dispatch.

`semantic_nominal_geometry.correct_nominal_geometry` (lines198–266) first computes the hypothetical **zero-current-policy final target** using actual previous final drive, raw mapper nominal and controller bias. Its projection box is original physical hard bounds intersected with **previous final ±1.25° this tick**. The two-DOF fixed-base Jacobian projector protects the linear nominal descent allowance and preserves the requested world-x direction/closest feasible x. It is not a full-body/contact motion guarantee and does not constrain current residual.

On a nonzero feasible correction, the inverse is `adjusted_native = desired_zero_policy_final − controller_bias`; on identity or explicit infeasibility it leaves **raw native** unchanged. The logged adjustment is `adjusted_native − raw_native`, not `desired_final − old_bounded_final`. Hence a large adjustment can merely bridge an existing raw-target/final-history gap.

## Ep2: arithmetic of the two retained endpoints

Canonical degrees, RR hip/knee; controller bias is zero. Rows are six physics ticks apart, not consecutive states; `previous` is each row's same-tick pre-dispatch history. Both happen to record the same previous pair.

| Quantity | g139970 / t6216, identity | g139971 / t6222, projected exact x |
|---|---|---|
| Raw mapper nominal | [−8.15, −39.05] | [−8.15, −39.05] |
| Previous actual final | [+47.58973952, −58] | [+47.58973952, −58] |
| Old zero-current-policy bounded final | [+46.33973952, −56.75] | [+46.33973952, −56.75] |
| Projected desired zero-current-policy final | Identity, no replacement | [+46.36400704, −56.75892077] |
| Actual change of zero-policy final | [0, 0] | **[+.02426752, −.00892077]** |
| Logged offset from raw mapper | [0, 0] | **[+54.51400704, −17.70892077]** |
| Requested residual | [+5.05653839, −29.66521589] | [+8.05653839, −30.65335087] |
| Headroom-effective residual | [+5.05653839, −18.95] | [+8.05653839, −1.24107923] |
| Pre-final-slew candidate | [−3.09346161, −58] | [+54.42054543, −58] |
| Actual canonical final | [+46.33973952, −58] | [+48.83973952, −58] |

At g139970 the hip candidate is 49.433201° below the actual final, but final is exactly **previous−1.25°**. At g139971 it is exactly **previous+1.25°**. Neither dispatch jumps 54°. The knee's −58° value is the original −60° lower hard bound inset by the existing 2° residual reserve.

At terminal the projection changes predicted nominal vertical displacement from **−.03163290 mm to 0**, while keeping predicted world-x **−3.02729549 mm**. `projected_exact_forward` means exact requested x, including its negative direction; it does not promise positive world-x motion. The actual small joint correction has norm .000451259 rad. Both recorded mathematical box and half-space checks are true; subsequent residual/headroom/final slew need not preserve that nominal-only prediction.

## Authority: ±24° is not to be compared directly with +54°

P09 RR hip cap is ±24°, knee ±36°; residual slew is 60°/s (.5° per 120-Hz own tick). In the **active** terminal branch, a static hip residual of **−.02426752°** would restore the old zero-policy bounded hip target; **+1.22573248°** would hold the previous hip target. Both are inside the cap/headroom. Reproducing raw −8.15° immediately is impossible under the existing final slew regardless of cap and is not the correct cancellation baseline.

These are static arithmetic possibilities, **not actions already requested or instantly reachable**. The actual filtered hip residual is +8.05653839°; changing it to the first value would require at least 17 own ticks under the existing .5°/tick residual slew if that target/state were otherwise frozen. The live state and policy would not remain frozen.

There is a genuine branch-sensitive map: in identity, raw `m+c+r` can remain far outside the final-slew interval, swallowing a modest opposing residual. In active projection, the inverse places the zero-policy baseline inside that interval, making the same residual effective near the current final target. A very small change of the nominal projection can therefore switch which end of the permitted final-slew interval is reached. This is not an assertion of globally smooth authority, but it is also **not a new violation**: identity's old mapping and active inverse are deliberately different contracts.

## Ep3 checks and accumulation boundary

At **g140002/t6160**, the high-clearance extreme, geometry is identity with zero offset; original predicted nominal z is +21.033 mm versus an available descent allowance35.918 mm. Its candidate equals final `[+52.87789016,+6.16843605]°`; no retained earlier projected peak is substituted there.

At **g140021/t6307**, geometry uses `projected_relaxed_forward`. Old zero-policy final is `[−12.98682049,−47.70369332]°`; desired is `[−12.98682049,−49.57114617]°`: only the knee changes, by **−1.86745285°**, within the 2.5° total width of the same-tick box. The logged raw-mapper offsets are `[−4.83682049,−10.52114617]°`. Adding the actual residual produces candidate `[−17.32364099,−57.72483949]°`, but final is `[−15.48682049,−50.20369332]°`, each previous−1.25°. This is still bounded dispatch, not a held historical lift or unconditional phase reentry.

No geometry offset accumulator exists: each call starts from `list(native)`, reads current measured q/J and actual final history, and does not mutate the nominal source/mapper. Tracking feedback consumes previous **independent REQUEST residual**, not geometry offset or realized final residual (`semantic_tracking_reference.py:130,254`). Repeated helper calls with identical complete inputs do not add the prior offset. Actual dispatch does advance final history, so closed-loop target lag, oscillation or state-dependent drift remains possible; “no additive accumulator” is not a proof of dynamic stability or absence of long-term motion.

## Existing contract tests, not newly executed

* `test_semantic_nominal_geometry_context.py::test_active_inverse_subtracts_controller_bias_from_desired_not_bounded_delta` and `::test_identity_keeps_raw_native_not_slew_bounded_reconstruction` explicitly require these two mappings.
* `test_semantic_nominal_geometry_dispatch.py::test_bounded_delta_is_not_a_correct_inverse_and_identity_must_not_recenter` demonstrates why adding only the bounded target correction to a far-away raw mapper target can be swallowed by slew, and why recentering identity would alter old residual behavior.
* `::test_active_inverse_composition_one_advance_write_and_three_targets` and `::test_identity_and_degraded_keep_raw_native_and_old_residual_mapping` specify composition, single dispatch and unchanged identity behavior.

No failing-contract counterexample is established from this source/window review, so no production correction or new gate is proposed. The observed branch-sensitive control map and limited first-order/zero-current-policy scope should remain explicit; neither proves a cause of either safety outcome.
