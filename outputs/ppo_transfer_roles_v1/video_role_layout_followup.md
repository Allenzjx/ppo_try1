# Video follow-up after append48 — read-only, not implemented

Reviewed the existing source while Run7 is live at `db991aa68103642c179e130fe4cce6ee892c89ab`. No production/test/config edits, Python, decoding, hashing, simulation, or video artifacts. The earlier namespace patch remains **unapplied**; its “324/12 unchanged” note does not cover the now-explicit 372/12 layout.

## Concrete minimum integration gaps

| Existing seam | Required bounded follow-up |
|---|---|
| `semantic_video_cli.main`, `semantic_video.validate_semantic_video_source`, `scripts/run_semantic_video.ps1` | Implement the existing three-file namespace proposal: propagate optional `experiment_id` consistently into preflight and capture/end validation, reconstruct the recorded experiment contract during source validation, and route both run/launcher paths under `runs/ppo_transfer_roles_v1/video_eval`. Keep v3 configuration paths and legacy defaults. |
| `semantic_video_cli.checkpoint_loader` (runner construction around line30) | It currently passes only `policy_version`. Shared preflight already derives and validates the complete source/target layout, but this call omits it. Pass `**_observation_layout_options(args)` alongside the existing version. Record `_resolved_policy_contract(args)`/layout/dimension in load provenance, and reject an observation width inconsistent with that verified contract. Do not infer372 merely from array length, truncate to324, zero-pad at evaluation, or apply another append migration here. |
| `semantic_training.load_semantic_checkpoint` exact-resume migration branch (around455–463) | **Additional actual blocker:** any supplied ResumeMigration still requires width324 and rebuilds runner configuration without `observation_layout`. A correctly constructed372 actor would therefore fail the future video-only migration. For the existing reviewed video factor, validate actual fresh storage against the complete unchanged source/target policy contract and layout; rebuild the exact runner configuration with the same validated layout/return profile. Preserve source state, optimizer, normalizer, RNG and counters; never replace this with NewMdpWarmStart. Retain old324 behavior and reject mixed architecture/config/topology changes. |
| `semantic_migration.build_migration_plan` (around725) | Its plan still reports `observation_dimension:324`. Bind the video-only factor's dimension/layout to strict checkpoint metadata and unchanged target schema instead; do not emit a false324 receipt for372. `validate_migration_plan` must reconstruct the same result. Existing `VIDEO_FILES` plus `INSTRUMENTATION_FILES` already cover these five runtime files; enumerate the exact committed delta, not an expanded generic exemption. |
| `build_video_core` / `video_configuration` / capture recorder | These already select the v3 observation schema through the current semantic core, rather than a hardcoded324 vector. No second encoder or recorder is needed. Keep original324 positions, appended48 fields, existing HISTORY kernel/rho0.9, history slice195:207, identity normalization and12 outputs. Actual capture continues to use the current shared evaluator and0-extra-tick existing180-settle capture. |

The current strict loader failure is an artifact-path compatibility issue, **not** evidence that live Run7 training or its checkpoint is invalid.

## Historical A: explicit reference path, not success-validator relaxation

The current `publish_success_comparison` requires **both** sources to pass `validate_semantic_video_source`, plus matching seed/runtime/version/reset conditions. That correctly remains the matched-success API but cannot represent this preserved A. The existing output-only `render_outcome_comparison.py` draft permits failed A, yet still requires managed current-v3 sources and equal conditions; it is not presently usable for the legacy A either.

A narrow future change should extend that existing neutral artifact helper with an explicit **historical-A / unpaired-reference** mode:

- Bind the known historical A record and its original media/evaluation/recorded technical provenance without making A pass today's TaskEvaluator, replacing its failed outcome, or requiring an A rerun. Preserve P10 WAIT_ENTRY incomplete, seed4001,180+64 pre-action ticks and legacy evaluator provenance. Do not interpret legacy commanded-wheel-speed0 as measured zero actuation.
- Keep C on the complete current success-only source validator: verified saved372 HISTORY policy, deterministic natural P01 with no prefix,0 optimizer updates/unchanged learned state, real physical success,184 real stable post-success ticks, native/no-state-write evidence and full technical validation. A successful prior evaluation cannot stand in for success of this actual video capture.
- Emit neutral titles such as **UNPAIRED REFERENCE — A INCOMPLETE / C TASK SUCCESS**, `paired=false`, explicit condition differences and null paired-improvement/stability-superiority claims. Use a fresh neutral comparison filename, not `fsm_vs_ppo_success.mp4` or an “improved” name.
- Use each source's **actual** frame count and performed context. Preserved A is989 frames at15fps with recorded decoded duration65.933334s, no184-tick post-success hold. Do not use `frames_for_episode`'s success-post assumption for A. Preserve both complete continuous sources at real speed; hold the shorter side's final frame visibly labelled “SOURCE ENDED”, never present it as ongoing physics, crop, interpolate or stitch episodes.
- A's recorded container duration is invalid while its stored decoded PTS passed. Preserve that limitation and original file; no standalone repair/remux is required by this proposal. Any later actual composite must be validated from its real decoded frames/PTS and sane new container, <=200s, without overwriting A. Stored evidence alone cannot certify a future rendered output.

The video CLI currently locks seed4001. Keeping that seed is compatible with an honestly labelled unpaired reference; it does **not** reproduce the formal seed2001 C evaluation. Seed changes are unnecessary to solve372 loading and must not be silently implied.

## Small future verification set and boundary

After a completed simulation boundary, if this follow-up is selected:

1. Extend the existing saved-policy video-loader test with real HISTORY324 and explicit HISTORY372 cases: exact layout passed, initial/current width preserved,12 deterministic actions, no weights/cache/optimizer/normalizer changes; reject wrong/missing marker, incomplete contract and source372/target324 mismatch before native launch.
2. Add namespace and source-contract rebuild tests in existing video suites; retain all success, partial-terminal-tick, original180 settle/no-extra-step, ACK post-hold, codec/PTS, failed-C and no-improvement negative assertions.
3. Extend existing video-migration tests with a strict372 checkpoint/runner round trip through the **same** reviewed video-only plan; assert correct372 plan provenance, exact actor/std/critic/Adam/RNG/budget preservation and empty rollout. Reject schema/reward/kernel/topology changes and fake mixed-layout metadata. No new migration lane.
4. For the existing artifact helper's explicit unpaired mode, test historical incomplete A + valid C acceptance, failed C rejection, undeclared mismatch rejection on the matched path,989-frame A accounting, labelled final-frame hold and immutable source/new-output handling. This does not certify the unexecuted helper or create a video now.

Targeted existing files: `test_semantic_policy_cli.py`, `test_semantic_video.py`, `test_semantic_video_v3.py`, `test_semantic_video_migration.py`; reuse `test_semantic_role_policy_layout.py` fixtures where appropriate. No commands or tests were executed in this review.

Commit only a reviewed video/instrumentation delta and use the existing `video_review` + `ResumeMigration` at an immutable checkpoint boundary. Keep the already-learned372 layout and physical MDP fixed. This preparation introduces no A5/5, probe, repeated-evaluation or optimizer gate; no present C success or paired video is claimed.

