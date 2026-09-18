# Terminal v2 independent CPU review

Scope: read-only review of `NominalMotionProvider._final_stop_request`, its owner handoff, existing mapper behavior, and focused tests. No simulator, policy forward pass, optimizer update, or production edit was performed. Receipt: `terminal_v2_independent_review.json`; reviewed supervisor SHA-256: `7c244749881e9055012881dbd3731cfd3dcc9ccb0013523ce7ad50b9cb6c3b1c`.

## Result

No confirmed blocking implementation defect was found in this bounded review. The new v2 home path starts from the retained previous nominal, uses elapsed observation ticks for its 0.5-second quintic ramp, does not restore the old wheel pulse, and does not mask residual channels. This is not evidence of physical success.

Using the sealed v1 run's actual home-entry nominal and 120 terminal observations (8737–8856):

| Check | Result |
|---|---|
| Replayed v1 nominal vs recorded native nominal | Maximum error 0° |
| First v2 request vs retained owner nominal | Exactly equal |
| Largest requested servo delta | v1 30.700°; v2 0.958664589°/tick |
| Continuous quintic velocity upper bound | 115.125°/s, below mature mapper 150°/s |
| Exact home target reached | Observation 8797, first dispatch 8798 |
| Repeated request at one observation tick | Idempotent |
| Nonadjacent ticks | Advance by physical elapsed time, not call count |
| Recorded transient control loss | Does not restart home ramp |
| Four nominal wheel targets during home | All zero; no legacy pulse |

The start is read before `evaluate` assigns the next nominal, so it is the held nominal rather than a future source sample or measured joint angle. Production adapter monotonic-time checks still reject duplicate complete controller steps; the request-seam idempotence test does not bypass that check.

A representative, explicitly synthetic seeded official mapper retained its object and advanced feedback tick 500→620. Its largest applied step was 0.958664589°, below the 1.25°/tick bound. This does not reconstruct the old run's exact mapper internals. Important wording: `_applied` and feedback history are not reset, but the mapper's existing changed-target semantics reset changed-channel compensation and nominal-reached flags as nominal moves. Tracking and normal bias are explicitly retired at home. Do not describe every internal mapper field as unchanged.

## Tests

- Existing production-facing terminal home/stop-owner tests: **41 passed** (`terminal_v2_independent_unit_tests.xml`).
- Additional output-only independent tests: **3 passed** (`terminal_v2_additional_checks.xml`): exact v1/v2 pre-home equivalence over 130 synthetic calls; delayed home starts from retained stop-owner nominal; nonzero Full12 residual remains active on all 12 channels through the actual bridge/projector.
- Reusable replay: `review_terminal_v2.py`; additional tests: `test_terminal_v2_independent.py`.

## Limits and next physical check

Old v1 measured states are not counterfactual v2 closed-loop states. The replay verifies command construction, not improved wheel tracking, elimination of RR oscillation, or sufficient settling in the remaining fixed post-completion window. The unchanged terminal physical criterion must be evaluated on the next sealed real run; the previous endpoint failure remains a failure.
