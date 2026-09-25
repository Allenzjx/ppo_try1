# Frozen-student one-shot direction probe v2

This is a deterministic natural-P01 **diagnostic**, not a formal policy result.
It used the frozen student prefix, no successful-N prefix or state injection,
and wrote no PPO/optimizer/training storage.  The candidate replaced only raw
FL-knee `+0.25`, FR-knee `+0.10`, and RR-hip `-0.20`; RR knee and the other
channels remained the student's requests.  The ordinary projector, mapper,
caps, slew, ownership, and single native write remained authoritative.

## Exact intervention window

- ACTIVE from tick 7984 / 66.533333 s to release tick 8008 / 66.733333 s:
  **3 decisions, 24 physics rows, 0.200000 s**.
- Conservative release reason:
  `entry_eligibility_lost:RR_front_distance_not_nonnegative`.
- Across the 24 measured response rows, actual FL knee moved `+6.9768 deg`,
  FR knee `+5.7442 deg`, RR hip `-6.9695 deg`, and the unmodified RR knee only
  `+0.01095 deg`.
- The **mapper nominal N** stayed FL knee `-12.15 deg`, FR knee `+31.35570
  deg`, RR hip `-8.15 deg`, RR knee `-39.05 deg`. It is not FINAL. The actual
  executed FINAL from `atomic_ACK.drive_target_full12` moved over ACTIVE as
  follows: FL knee `-44.3710 -> -32.8710 deg`, FR knee `-18.8253 -> -7.3253
  deg`, RR hip `+6.9851 -> -4.5149 deg`, while RR knee remained `-58 deg`.
  Thus the finite raw directions did reach the executed command path, while the
  measured joint motion still cannot be treated as an isolated causal effect.
- RR gap moved **59.003 -> 63.455 mm** during ACTIVE while RR front distance
  moved **+0.709 -> -0.139 mm**.  There was no RR TOP contact or load in the
  ACTIVE rows.  FL remained verified TOP support, with measured force
  `2.054..6.507 N`.  COM displacement over the actual ACTIVE rows was
  `[-5.919,+0.946,+2.559] mm` in world x/y/z.

## Delayed post-release minimum (separate from ACTIVE)

The minimum legally qualified post-release gap on the complete 15 Hz
decision-start clock was tick **8264 / 68.866667 s**: RR gap **15.168434 mm**,
front distance **+51.708642 mm**, qualified AIR, no TOP/ground contact, and
FR/FL/RL all recorded as supporting.  This is 2.133333 s after release and is
therefore a delayed response, **not maintained capture by the three ACTIVE
decisions**.

At that exact 120 Hz tick, FL was verified TOP support at `1.522373 N`, COM was
`[0.745139,-0.124635,0.150135] m`, and the last committed FINAL/actual values
were:

| channel | current student raw | mapper nominal N | executed FINAL | actual |
|---|---:|---:|---:|---:|
| FL knee | -1.475655 | -12.150000 deg | -44.527820 deg | -44.769682 deg |
| FR knee | -0.472723 | +31.355701 deg | -17.760170 deg | -16.523016 deg |
| RR hip | +0.753322 | -8.150000 deg | +7.087761 deg | +6.786870 deg |
| RR knee | -0.603469 | -39.050000 deg | -58.000000 deg | -57.754129 deg |

The raw request belongs to decision 1033 starting at tick 8264. The executed
FINAL and actual values are the exactly matched 120 Hz state at that start
tick; mapper nominal N is separately labelled and never called FINAL. The JSON
separately records first-effect and decision-end states rather than pretending
they share the start clock.  By the run endpoint (91.166667 s), RR gap was back
to `57.200967 mm`; the episode remained P09 and ended
`INCOMPLETE_CONTROLLER_BLOCKED`.

## Evidence

- Authoritative compact result:
  `student_entry_direction_probe_v2_analysis_ACK_FINAL.json`
- `student_entry_direction_probe_v2_analysis.json`,
  `student_entry_direction_probe_v2_analysis_clock_aligned.json`, and
  `student_entry_direction_probe_v2_analysis_final.json` are superseded: they
  incorrectly labelled mapper nominal N as executed FINAL.
- Physics rows: 10,940; SHA-256
  `915187c734e5a2a4f6baeede88dd2345cdb712f08d40a48a41d3298eaa36dcd3`
- Decision rows: 1,368; SHA-256
  `2a7e06b80fc813ccdd702450b93926c12f2437ce152aea6116de680f2f68ad79`
- No missing tick was interpolated. Missing augmented 120 Hz eligibility fields
  remain N/A; legal-min selection uses the complete 15 Hz decision-start
  eligibility only.
- Every selected physics row requires one atomic write plus verified matching
  actuator readback. Executed FINAL comes only from
  `atomic_ACK.drive_target_full12`; mapper N comes from
  `native_readback_audit.native_drive_target_full12`.
