# Same-tick post-mapper servo headroom

The isolated candidate has been published, by explicit authorization, as
`src/wlr50_clean/ppo/semantic_headroom.py`. Its focused production tests are
`tests/unit/test_semantic_headroom.py`. This receipt is not a physical evaluation
and does not establish that the integrated environment is fixed.

## API and boundary

`HEADROOM_MODE = 'same_tick_post_mapper_servo_margin_v1'`

`project_semantic_servo_headroom(native_full12, controller_bias_full12, projected_residual_full12)`
accepts three finite canonical full12 sequences. The native input must be the
unique same-tick geometry-corrected mapper output. The controller bias must
already satisfy its original envelope; the policy residual is independent of
that controller-only 10-degree limit. The helper imports no Torch or Isaac,
reads no files, and neither reads nor modifies mapper state.

For the first eight servo channels, with `base = geometry_native + controller`,
the interval is `[min(0, L-base), max(0, U-base)]`; only the requested policy
residual is clipped. `L,U` are frozen `servo_limits_deg` inset by the original
2-degree reserve (hips [-133,133], knees [-58,208]). The four wheel residuals
pass through unchanged. Logical nominal, previous slewed target, current
physical joint position, and policy phase are not inputs.

The startup loader must call
`validate_semantic_servo_headroom_config(mode, actual_base_config['joint_safety_margin_deg'])`
and verify the real base configuration still contains exactly 2 degrees for
both hip and knee. Unknown modes, malformed/finite-invalid inputs, incompatible
margins, unbounded controller bias, and arithmetic overflow are rejected.
The function cannot prove the caller supplied same-tick geometry provenance.

Zero (including signed zero) does not relocate an existing baseline outside the
reserve. At a knee baseline of -59 degrees, zero stays -59, an inward +0.5-degree
residual is retained (target -58.5), and an outward -0.5 is clipped to zero.
At native -27.8, residual -24.98 is feasible and gives -52.78; a logical nominal
of -37.8 is irrelevant at this boundary. Tiny residual values are retained
directly where feasible, but this does not promise representable actuator
effects after float32 conversion.

The real adapter must still own final hard clamping, the existing single slew
history, dtype conversion, and actual dispatch. The helper's diagnostic target
is before all of those operations. It may remain outside the reserve when the
legacy baseline is already outside, and unconstrained wheel arithmetic is not
a wheel target safety guarantee.

## Evidence fields

The returned dictionary contains the locked keys `mode`,
`servo_safety_limits_deg`, `requested_policy_residual_full12`,
`effective_policy_residual_full12`,
`effective_combined_post_mapper_bias_full12`,
`policy_residual_intervals_servo_deg`, and `clipped_servo_indices`.
Additional fields identify input geometry-native/controller values, baseline,
hard limits/reserve, channels already outside reserve, and the no-slew
arithmetic target. It explicitly labels same-tick provenance as the caller's
responsibility. These values are not an acknowledgement of actual actuation.

## Executed CPU validation

- Initial candidate collection failed because `request` is a pytest-reserved
  parametrization name. Zero tests ran; `junit.xml` preserves that failure.
- After fixing the test name and publishing the production module, the focused
  production suite completed: **96 passed in 0.26 seconds**, process exit 0.
  Receipt: `junit_production.xml`. The candidate suite was not separately rerun;
  its cases were transferred to the production suite.
- Independent integration review then identified a wheel signed-zero edge:
  unlike servo zero-preserving arithmetic, wheels must retain the old direct
  `c+r` and `native+(c+r)` sums. The production module now does so, with finite
  checks. After the authorized narrow fix and an exhaustive eight-combination
  signed-zero test across all wheels, the final focused production suite is
  **97 passed in 0.27 seconds**, process exit 0; receipt:
  `junit_production_final.xml`. These are successive versions, not 193 distinct
  passing tests. The original isolated candidate remains an earlier draft;
  the production module is the reviewed implementation.
- Coverage includes the real base YAML/frozen limits, all eight servo upper and
  lower boundaries, reserve violations, geometry-corrected versus other
  baselines, bounded-controller composition, signed zero/tiny values, wheel
  pass-through, malformed inputs, overflow, and 1,000 finite random cases over
  four deterministic seeds.
- The interpreter ran with CUDA hidden, BLAS/OpenMP thread limits of 1, bytecode
  writes disabled, and third-party pytest auto-loading disabled. No Torch,
  optimizer, model weights, Isaac, or production simulator was invoked.

Only the new pure production module and its new dedicated test file were added
outside this report directory. No adapter, audit, runtime/configuration,
historical test, or checkpoint was changed by this subtask; no commit was made.
