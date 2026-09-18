# CP178432 sealed delivery and QA

The formal deterministic FULL12 evaluation naturally ended at **P05 / FL placement incomplete**, tick5907 /49.225s. FL had crossed the front edge but had not placed. Termination: `INCOMPLETE_CONTROLLER_BLOCKED`, source `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`; stage age37.225s. RR preparation/traversal was not reached. These are **not RR failure or strict-RR-success videos**.

## Delivery

- `PPO_full_native.mp4` — entire original single episode, native remux,739 actual frames,15fps, no retiming; physical49.225s /encoded49.266667s.
- `PPO_full_actual_wheels_RR.mp4` — entire episode with actual native-axis FL/FR/RL/RR angular velocities, RR diagnostic fields and explicit source-observation tick when recorded. RR in this filename names the diagnostic panel, not a reached RR task. Original1280×720 view is uncropped;160px panel is outside it.
- **`PPO_P05_FL_capture_failure_actual_30s.mp4`** — preferred failure explanation:450frames/30.000s, actual ticks2320–5907; FL gap/front distance/contact/bearing/support, hip/knee target versus actual, and four wheel qd. No zoom, synthesized frame or guessed leg-image coordinates.
- `zero_vs_PPO_same_camera.mp4` — new sealed zero v2 on left; this PPO on right.1108frames/73.866667s; PPO's369 static terminal frames are explicitly labelled `END - frozen last frame, not new simulation`. Same P01 elapsed-time alignment, not phase synchronization.
- `PPO_latest_failure_context.mp4` — earlier generic context retained unchanged;451frames/30.066667s due endpoint-frame quantization. Prefer the exact30s FL-specific variant above for this failure.
- Corresponding `.media.json` receipts preserve full decode, dimensions, PTS, source hashes and provenance. Per-frame data is retained in `.frame_data.json`; no unknown physical values are replaced with zero.

The checkpoint official-load immutable manifest verifies new RR-branch640 decisions /5 PPO updates /100 optimizer steps since177792. Those counts describe training, not successful traversal. This export performed no physics, policy forward pass or optimizer update.

## What the failure panel establishes

At5907: FL bottom-to-platform gap **+3.470128mm**, front distance **+197.938267mm**, `AIR`, no ground/obstacle contact, bearing **0N** with bearing source verified, no top contact/support. FL hip target/actual **24.154553/23.000725°**, knee **−12.709240/−12.715437°**. Four native measured qd FL/FR/RL/RR = **[−.158582,−.074058,−.129269,+.052539]rad/s**. These are same-tick records, not wheel-center displacement. They show missing placement despite close knee tracking, not a mask diagnosis or isolated motor/contact causality proof.

## P02 four-wheel result

Separate bounded report: `../cp178432_run051034_p02/p02_wheel_chain.md` and JSON. At tick400, all four nominal targets were+.3rad/s; final canonical targets were[+.35267,+.30588,+.34967,+.30364], actual canonical qd[+.45378,+.30808,+.42431,+.30958]. Each current policy wheel effect was positive; FR was airborne and therefore rotation is not proof of traction.

Across P02's1352 ticks with positive N, no wheel had negative same-tick policy target effect, final target below half N, or final target near zero. FL alone had4 measured near-zero samples (first230) despite a nonzero target; the other three had none. The reviewed bounded dispatch/mapping checks all pass, with float32 arithmetic residual2.64e−8rad/s. This rules out the proposed persistent “only RR commanded” pattern in this particular run/window; it does not certify all-phase coordination or traction.

## Comparison disclosure

Same committed code/inventory, review camera, seed, scene/actuator/action settings and five non-stage configuration hashes. Stage-spec RR free-air qualification and current-lift carry readiness intentionally differ between isolated zero and residual profiles; terminal home v2 is the same. Therefore this is **not identical nominal scheduling or identical task acceptance**, and identical measured initial physical states are not claimed. No sealed fields were altered to authorize the comparison.

## Visual and caption QA

Inspected all7 distinct C start/FR/FL/terminal keyframes, generic failure first/last, FL-specific failure first/middle/last, and comparison first/last. Whole robot outline is within the inspected frames; no panel overlays the robot. Far-side contact occlusion remains possible, so no all-four-contact/all-frame visibility claim is made.

Full-frame tool previews initially appeared to omit letters. This was **not established as video encoding damage**: `caption_pixel_QA.json` checks10 real decoded caption samples against their exact same-frame source panels. Every expected bright glyph pixel remains nonblack (0 missing-bright-region pixels in each); average RGB differences0.692–0.887 levels are normal lossy encoding. Independent single-frame decoded `caption_QA_terminal_crop.png` visibly contains the complete P05/native/qd text. `FL_failure_terminal_caption_exact.png` likewise contains all FL failure values. Root also inspected these crops. Existing full/comparison videos were not needlessly re-encoded.

Automated receipts retain their original manual-QA-pending flag; this note records the completed manual scope and limits without rewriting the sealed receipts. Original source, historical media and production files are untouched.
