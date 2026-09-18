# Non-residual terminal refinement — measured v1 failure, verified v2 success

The protected baseline remains the unmodified N+0 run `20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6`, 8857 physics ticks / 73.808333 s. It has no learned checkpoint: its control is the successful nominal with zero residual. Original source/config hashes, video and result are referenced in `outputs/diagnostics_v1/checkpoint_and_protected_baseline_references.json`.

The current repository includes the user's artifact snapshot commit `5268b8f97bcbd88dc8c36e75aeb1fccd3b91f42c`; it changes no production source/config relative to `fc14a68`. This work continues on top of it.

## Confirmed terminal issue

The old stop owner acquired at zero observation tick8737, retained servo requests `[-18.5,-31.4,3.7,31.1,-6.9,-18.7,-6.9,-27.2] deg`, and deliberately retired the later full12 home+wheel-pulse group. This protected physical controlled completion, but also removed the requested visible home recovery. Actual final maximum home error was about30.23deg. The historical source's P13 endpoint servo request is `[.5,-.7,3.7,.4,-2.6,-3.9,-.5,-6] deg`; its intervening wheel pulse `[1.09,-.72,-.43,-.39] rad/s` is not required as an acceptance rule.

## Isolated candidate

`source_home_after_physical_stop_v1` is an opt-in nominal owner, not a zero-residual-specific controller. Before the existing real post-completion stop acquisition, control remains unchanged. After the owner has acquired and current verified control/support/region conditions hold, it issues the source's eight home-like servo requests once and keeps nominal wheels stopped. This is an ordinary new authored command through the existing mapper, with no actual-joint write or history reset. The home group retires preceding carry tracking/bias ownership; it does not reset the physical mapper, residual HISTORY, or learned residual.

The original final-stop mode still holds its original pose. The reference recording, successful baseline configs, physics, actuator strength, hard limits, whole-body traversal criteria and fixed1s observation are unchanged. Exact home angles are suggestions and diagnostics, not a newly imposed task-success gate. New physical motion can still fail the unchanged safety/completion checks; synthetic tests do not establish its success.

The initial candidate has passed39 focused terminal ownership/refinement checks. Additional source-order, P05 handoff, height, video and routing regressions passed before physical launch. New camera framing and a complete natural-P01 N+0 run are required before calling this improvement verified.

## Residual boundary

The latest saved PPO is CP177792 /1354 updates /27080 optimizer steps; its512-decision block is sealed. Do not downgrade or erase it. Old CP177152 remains immutable with its historical evaluator label, but its RR edge-contact ascent is under a separate stricter physical audit. It is not currently evidence that wheel-supported climbing has been excluded. No new quality reward has been enabled.

## First physical candidate, sealed September17

Run `20260917T0343245466199Z_g3d231897c91c_0e58ddffca37441089dd17f34f4ed5d8` naturally completed8857 ticks/73.808333s with `INCOMPLETE_CONTROLLER_BLOCKED`, reason `not currently controlled at fixed post-completion observation end`. Four-leg placement, platform region and support remained valid; no region loss was observed. This is a real terminal failure, not a media/export failure, and does not replace the protected successful zero.

Home/stop owner acquired8737; its first drive tick was8738. At8857 body linear speed=.012815m/s and angular speed=.061457rad/s were below unchanged limits; all wheel targets were0. RR measured velocity=-.370923rad/s exceeded the unchanged .25rad/s limit. The final .5s contained17 RR violations among60 physical samples, so it is not merely one rare endpoint measurement. Peak body speed during the home movement was .31342m/s, peak angular speed .46232rad/s. Actual final maximum error versus the source-home request was5.438deg; versus all-zero home it was8.115deg. The source-home intent therefore executed visibly, but it excited loaded-wheel motion and did not satisfy controlled completion.

## Second candidate: implementation and CPU checks

Opt-in `source_home_after_physical_stop_v2` keeps the same acquisition, source target, zero nominal wheels and all12 residual permissions. It replaces the abrupt home suggestion with a finite quintic nominal ramp, using the existing .5s stable-duration scale, then holds the exact source target. The ramp starts at the last issued nominal request, not an actual-joint pose reset. Its purpose is to reduce command excitation while reserving the remaining part of the unchanged1s observation for settling. The mature mapper, actuator limits, actual stop thresholds, evaluation window and failure classification are unchanged. Existing v1 and protected historical modes remain available.

The41 focused terminal tests pass, including unchanged legacy behavior, monotonic bounded ramp, exact endpoint hold, no re-start on transient speed loss and no residual restriction. These tests prove control semantics only. The subsequent physical result is recorded separately below.

## Second physical run: sealed v2 success, September 17

Run `20260917T0424208857504Z_g6c2121b68654_13804a86a302480ab61ad2af0dcb8dcd` at commit `6c2121b68654eaaa2afa7c19e9d02ae770493d2f` naturally completed P01–P13 with **SUCCEEDED**, 8857 physics ticks / 73.808333 s. Recorded region, support, final controlled state, task-completed-controlled and fixed post-observation completion were all true; region-loss latch was false; `source_acceptance_error` was null. Every issued raw action and dispatched residual was zero: this is **N+0, not a learned PPO checkpoint**. The v1 failure and earlier protected baseline remain unchanged.

Home starts at observation8737 in both runs. V2 reaches the exact source nominal target at dispatch8798 instead of v1 dispatch8738. Its measured maximum nominal request step is0.958665deg/tick rather than30.7deg. The available pre-home tail8617–8737 (121 samples) has exactly equal actualFull12 and body-position values between v1 and v2; this is a bounded tail comparison, not a claimed full-run byte-equivalence proof.

| Measured quantity | v1 failure | v2 success |
|---|---:|---:|
| Terminal canonical wheel qd FL/FR/RL/RR (rad/s) | +.060021 / -.034760 / -.001982 / -.370923 | +.149182 / +.002094 / +.000390 / -.239496 |
| Terminal body linear speed (m/s; limit .05) | .012815 | .021498 |
| Terminal body angular speed (rad/s; limit .30) | .061457 | .059457 |
| Final .5s RR samples above abs(.25rad/s) | 17/60 | 5/60 |
| Final .5s all measured control thresholds passing | 43/60 | 48/60 |
| Longest post-home consecutive controlled interval | 20 samples / .166667s exposure | 27 samples / .225s exposure, ending8857 |
| Terminal maximum error vs source-home target | 5.437734deg | 7.869146deg (FL knee) |

The terminal RR wheel is within the unchanged limit by only0.010504rad/s; no wider robustness margin is claimed. Four terminal nominal wheel commands are zero. The result improves actual controlled completion and reduces late RR speed excursions, but **not every stability or home-accuracy metric improves**: across the full post-home second, body linear-speed exceedances increase41→57/120 and angular-speed exceedances17→22/120; total controlled samples change62→59/120. The source-home request is fully issued, but actual joint convergence is not exact. This tradeoff remains visible in the data rather than being relabelled as general stability superiority.

Machine-readable same-tick comparison: `outputs/video_review_v1/zero_refine_6c2121b_run042420/zero_home_v1_v2_comparison.json`. It preserves both outcomes, original speed thresholds, every0.1s error/velocity trace, exceedance intervals and exact endpoint values, without smoothing.

## V2 delivered media and visual scope

- Native whole episode: `outputs/video_review_v1/zero_refine_6c2121b_run042420/zero_Nplus0_full_native.mp4`.
- Preferred labelled delivery: `outputs/video_review_v1/zero_refine_6c2121b_run042420/zero_Nplus0_full_review_clean.mp4`, with corresponding `.media.json`.
- Both contain1108 actual frames at15fps; physical73.808333s, encoded73.866667s due the terminal frame interval. No slow motion, interpolation, cross-run stitching or added simulation. Native remux retains exact decoded frames/PTS/keyframe sequence; the labelled derivative adds only a60px external header and re-encodes. Full decode and N/15 timestamp checks pass.
- Fifteen distinct event/start/home/end keyframes were manually inspected. The whole robot outline remains inside the view in those frames, including the formerly clipped near-side FR and the visible terminal pose change. The far-side RL contact is partly self-occluded in some frames; **four contacts visible in every frame is not certified**. The whole file was decoded, not manually inspected frame-by-frame.
- The first drawtext-labelled derivative is retained. An early full-frame preview raised a missing-glyph concern, but later decoded-image pixel checks found0 expected bright-caption pixels lost to black; encoding damage was not established. The preferred clean version had already been produced using a static Pillow-rendered title PNG, with checked clean early/RR/RL/final keyframes; it remains a valid additional derivative, not proof that original footage was corrupt. Original native footage and both original run directories are untouched.
