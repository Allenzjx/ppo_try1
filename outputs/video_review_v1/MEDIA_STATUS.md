# Video review v1

Approved historical diagnostic export:

- `historical_cp177152_RR_contact_review_native_tick.mp4`
- Exact source endpoint ticks 4352–5440; 137 frames at 15 fps; 9.1333 s.
- Full decode and PTS checked; decoded ticks 4680 (first real AIR), 4944 (second contact ascent), 5176, and 5440 visually checked. Header values are bound to each actual source frame's ledger tick.
- Historical CP177152 evaluation only, **RR acceptance under review**. Neither latest CP177792 evaluation nor a new success. First real AIR lift is retained; powered obstacle-contact ascent does not isolate motive-force causality.
- Original full video and its historical evaluator result remain unmodified.

Unpublished output-only QA drafts (do not link as evidence):

- `historical_cp177152_RR_under_review.mp4`: full decode passed but container duration/FPS QA failed.
- `historical_cp177152_RR_contact_review.mp4`: container passed, but ASS centisecond rounding could show the preceding tick on some frame labels. Superseded by the approved `native_tick` version with floored subtitle boundaries. Associated drafts/receipts remain for traceability.

Camera preflight:

- Historical camera remains unchanged. The new non_residual_refine_v1 and residual_rr_fix_v1 camera is eye [1.85, -1.65, 1.15], target [0.70, -0.15, 0.15].
- `camera_unit_tests.xml`: 217 CPU tests passed. No physical success is inferred from tests.
- `camera_trajectory_preflight.json`: 8858 retained-zero observations, measured base/wheel collider AABBs plus an approximate 75 mm margin about other body origins. New view requires about 53.07 degrees horizontal FOV to enclose this conservative trajectory envelope, versus 87.63 degrees for the historical view.
- This calculation does not establish visibility, lens settings, or absence of occlusion. New full-run first, FR, FL, RR, RL, and ending frames must be checked. No new physical simulation was performed by these helpers.
