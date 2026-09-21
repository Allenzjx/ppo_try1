# Targeted verification

- Root selected integration regression: **220 passed**, 15.95 s. Receipt:
  `root_targeted_tests.xml`. Covers new capture/front quality, original residual
  composition and headroom wiring, and current RR free-AIR/source carry rules.
- Reward-specific new tests: **34 passed** (`fl_capture_front_quality_tests.xml`).
  These overlap the 220; do not add them as independent test counts.
- Old reward/capture/retention/carry/transfer/observation subset: **268 passed**
  (`old_reward_capture_focused.xml`), also overlapping the root suite.
- Probe finite ramp/hold/release and non-accumulating target tests: 6 passed.

Pre-existing failures are not hidden: three parameterizations of
`test_semantic_headroom_dispatch.py::test_real_geometry_precedes_headroom_three_branches_and_wheels_unchanged`
expect geometry native -2 but receive 20. They reproduce with both modified
and HEAD-original reward/supervisor loaded in memory. Adapter, geometry,
mapper and that test are unchanged by this implementation. The fixture has
nominal command history 20 but residual-modified actual target history 0;
its old expected projection uses the latter. Do not restore residual-to-N
feedback to satisfy these assertions. No claim that the entire repository
test suite passes is made.

The reward agent separately reproduced an old workspace-integration numeric
expectation mismatch under HEAD-original supervisor. This is not a physical
task or acceptance failure. Original tests/results are preserved.

Unit/CPU tests do not demonstrate FL contact, stability improvement or full
obstacle traversal. Those require the new actual physical run receipts.
