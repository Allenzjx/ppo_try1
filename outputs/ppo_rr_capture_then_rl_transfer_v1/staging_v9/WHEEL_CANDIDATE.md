# P12 post-authored-stop retention candidate — NOT applied or executed

Prepared while the v8 video is active. Only this staging directory was edited;
no Python, pytest, actor, simulator, or encoder was started. Production module
and its existing test file remain unchanged. Static text review is not a test
pass and cannot demonstrate physical retention.

Files:

- `semantic_rr_carry_wheel.py`: same public builder/projector signatures; 86 additions / 20 deletions versus v8.
- `test_semantic_rr_carry_wheel_v9.py`: normal-package imports, real source contract and MotionExecutor fixtures; synthetic sensors/ACKs, not physical evidence.
- `wheel_production.patch`: unapplied apply_patch-compatible module + new unit test patch.

MODE remains `rr_capture_support_forward_projection_v1`.
SEMANTICS is exactly:

```
P09_v8_unchanged_P12_RR_placed_RL_unplaced_post_authored_wheel_stop_current_TOP_or_qualified_signed_AIR_bearing_only_depth_floor_previous_FINAL_1p8_slew
```

P09 retains its previous support set, gap taper, TOP release, source proof and
projection behavior. New P12 uses its own source layer and adjacent one-write
ACK, never the old P09 receipt. The new source fields are:

- `source_full_endpoint_issued`: whole P12/RL source completion, merely recorded.
- `wheel_stop_endpoint_issued`: all-four authored final wheel stop has occurred.
- `authored_reverse_pulse_verified`: exact existing -0.3 four-wheel pulse exists before that stop.
- `wheel_stop_completion_semantics`: `P12_all4_authored_final_wheel_stop_not_full_RL_source_endpoint`.

The real source wheel stop is tick320 (2.6667 s), while full RL source ends at
tick560 (4.6667 s). P12 can arm from tick321 after the stop's actual adjacent ACK.
Fresh wheel events from any source win. On the stop tick prior requested N must
be zero; afterwards zero or the existing all-four +0.3 live approach advice is
allowed. The original P12 -0.3 pulse is neither deleted nor shortened.

Only current verified-bearing wheels receive the floor: RR is included only
while bearing; AIR RR is untouched. FL plus one other current support is needed;
FL+RR is allowed without declaring two-point stability. Historical RR placed,
current cross and legal TOP/qualified signed AIR, and RL not placed bound scope.
TOP retention uses only depth taper; its near-zero contact gap does not erase
the floor. AIR keeps the existing gap taper. Deep-inside floor is zero, not a
return to negative policy. Geometry loss releases through the existing slew;
support/safety/phase/fresh-source loss bypasses. No new cursor, latch or timer.

Eight servo channels, raw12/logp/masks, all source clocks, reward, sigma, assist
budgets, physical limits and 1.8 rad/s² previous-FINAL wheel slew are unchanged.
X409 remains armed-envelope evidence, not actual projection or traction.

After the live run ends, first run the unchanged existing wheel tests together
with the new file against an explicit candidate overlay or authorized adoption.
Tests cover old-P09-ACK rejection, P12 full-endpoint distinction, pulse/fresh
stop, +0.3 nominal feedback, bearing-only RR selection, support/geometry/phase
negative cases, and same-prestate actual/zero-policy mapping. Gaussian sampling
itself is outside this controller module and is not rerun here.

Physical hypothesis is narrower than the first failure: preventing later
post-stop retreat after a real recapture. It does not claim to prevent first
loss at8401 during the legitimate negative source pulse, or establish whole
episode success. Adoption requires the separately reviewed explicit control
semantics migration; this text/patch does not authorize it.

Static SHA256:

- module: `00be1432b91431c3a10486700364e3b60a85c998245e3d4c8d312944645b5892`
- test: `96a40adbc4401223f8ce8981661ad8c7b28bdefc02389bf0d1a746f938796666`
- patch: `a51d9311b4333f3c2ea8f535b198aeb35d5e2e88143134d315872ff9593a197f`
