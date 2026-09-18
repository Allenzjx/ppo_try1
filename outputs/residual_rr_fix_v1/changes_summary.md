# Residual RR physical correction — implementation and actual training

## Protected state and current result

The successful73.808333s N+0 baseline, its source/config/video and all old checkpoints remain intact. N+0 has no learned policy checkpoint. Source CP177792 /1354 PPO updates /27080 optimizer steps is preserved. This isolated branch has actually earned640 decisions/5 PPO updates/100 optimizer steps; latest saved checkpoint is `outputs/ppo_residual_rr_fix_v1/checkpoints/history/checkpoint_step_000178432.pt`, cumulative178432/1359/27180. Training execution completion is not physical task success.

The new-camera zero run `20260917T0343245466199Z_g3d231897c91c_0e58ddffca37441089dd17f34f4ed5d8` completed all four leg placements but failed final controlled-stop acceptance. Its abrupt home request improved pose recovery while exciting RR wheel contact; final RR velocity=-.370923rad/s, limit=.25. The last .5s had17/60 RR speed violations. Body speed, platform region and support passed at the endpoint. The result remains failure; original successful zero is not downgraded. See `../non_residual_refine_v1/changes_summary.md` and the fully decoded normal-speed video in `../video_review_v1/zero_refine_3d23189_run034324/`.

## Evidence-backed changes committed in6c2121b

1. New opt-in RR free-AIR semantics: initial/qualified lift is earned from measured unsupported rise within the current continuous AIR segment. Contact ascent is not credited as new free rise. Ground recontact revokes the attempt; genuine whole-body lift does not require RR-only joint movement or imagined FL bearing.
2. Pending P09 source events consult current physical lift/clearance evidence before consuming the next knee waypoint or first positive four-wheel group. No phase reset, forced old pose, masked residual, fixed dwell, or global wheel stop is introduced. Negative top-relative clearance during genuine AIR does not by itself block the first rolling group.
3. The existing RR mode and successful zero retain their prior qualification/source semantics. All12 residual channels, additive composition, actuator limits, nominal geometry and quarter-temperature policy remain unchanged.
4. Terminal home v2 uses a finite .5s quintic nominal ramp from the previously issued nominal toward the source-home request, then holds it. The new zero natural-P01 run042420 actually succeeded in8857 ticks/73.808333s. The original1s physical observation and measured stop limits remain unchanged. Final RR wheel speed−.239496rad/s has only.010504rad/s margin to the unchanged.25 limit; source-home error7.869deg. This repairs visible recovery and terminal completion, not every stability metric or exact home convergence.
5. The new review camera is viewport-only and shared by new zero/PPO videos. Checked keyframes no longer crop the front leg at the ending; far-side wheel contact can still be self-occluded by the body. Video annotation is not physical evidence.
6. An explicit same372 semantic migration preserves complete actor/critic/Adam/Identity/RNG/counters from CP177792, discards the old unfinished rollout and collects fresh on-policy data. Task-derived observation and reward-event meanings change, so full MDP equivalence is explicitly not claimed.

## Actual training and remaining evaluation

After targeted tests and a committed clean runtime, the natural-P01 zero v2 succeeded. Actual training then completed128 natural-P01 decisions plus512 from a physically rebuilt successful-N P01→P06 preparation prefix, saving every update. A failed zero verification was never a training-start gate. The335 teacher-prefix decisions/2680 physics ticks are excluded. Actual learned samples: P01=2,P02=126,P03–P05=0,P06=166,P07=1,P08=1,P09=214,P10=1,P11=19,P12=110,P13=0. All640 decisions retained full12 additive residual permissions and epsilon0; all5120 physical learning ticks passed native dispatch/effect checks. Six ordinary suffix stage transitions did not produce false done. Actor/critic/Adam/Identity/RNG/counters were resumed; no old partial rollout was reused.

The suffix was nonterminal at tick6776/P12, not a full episode success. RR genuinely requalified after actual ground revocations, crossed5726/placed5731, then grounded again6053; final RR ground/front−.130729m, RL not placed. The seven pending-source readiness snapshots were all legitimately ready; the new pending-event guard did not hold this particular trajectory. It must not be credited as the cause of this suffix's transient crossing. Source P09/P12 commands executed, with subsequent learned corrections and support loss; no mask/omitted stop/illegal overwrite was established. See `../diagnostics_v1/training_RR_carry_178432.md`.

CP178432 was reloaded for formal full natural-P01 run051034, same review camera, no teacher or mask intervention. It naturally ended at5907 ticks/49.225s, P05 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`. FR qualified23/crossed1406/placed1425; FL qualified1484/crossed2106 but never placed. Final FL AIR gap+.003470128m, zero bearing; stage age37.225s equals unchanged30s+7.225s progress allowance. Physical evaluator remained valid/VERIFIED with no collision/nonfinite/hard-limit event. This is a real incomplete task, not an Isaac crash or video failure. Wrapper exit1 reports the diagnostic-failure lifecycle. P06–P13 were not reached by this formal evaluation; full-P01 RR success is therefore unproven.

P02 four-wheel chain was measured through first P03: tick400 N all+.3rad/s, final canonical FL/FR/RL/RR `[.35267,.30588,.34967,.30364]`, actual `[.45378,.30808,.42431,.30958]`. Actual masks all1 apply to residual; no three-wheel cancellation or lost command was found. FR was AIR, so its rotation is not ground traction. See `../video_review_v1/cp178432_run051034_p02/p02_wheel_chain.md` for same-tick raw/filtered/effective/native/contact details and limitations.

Quality epsilon remains0; front-leg secondary shaping is deferred while task completion is unproven. This branch collected no new P05 samples; that is a current curriculum coverage gap, not a claim that the ancestor policy never trained there. Current task capability must first close FL capture and then preserve RR placement through later whole-body cooperation. Synthetic CPU optimizer tests, sealed-input replays and video exports never add Isaac training credit.

## Production file scope

All runtime changes are committed in3d23189 and6c2121b, on the current repository history. Runtime paths are clean after the actual runs; output reports/media remain separate. Original FSM data/configs and all historical runs are preserved.

| Files relative to repository | Change |
|---|---|
| `src/wlr50_clean/ppo/semantic_supervisor.py` | Versioned RR unsupported-rise qualification and pending-source readiness; versioned post-stop source-home ramp. No phase-wide residual/history reset. |
| `src/wlr50_clean/ppo/semantic_backend.py`, `semantic_transfer_roles.py` | Allow the new explicit RR semantics through existing backend/transfer logic. |
| `src/wlr50_clean/ppo/semantic_migration.py`, `semantic_training.py`, `semantic_cli.py` | Isolated same372 migration, exact state continuation/ancestry/accounting, namespace routing and likelihood audit. No PPO algorithm or network reset. |
| `src/wlr50_clean/ppo/isaac_fsm_backend.py`, `semantic_video.py`, `semantic_video_cli.py` | Pre-reset viewport-only camera selection, isolated video routing/task window and common evaluation dispatch. No asset/actuator/physics modification. |
| `scripts/run_semantic_ppo.ps1`, `scripts/run_semantic_video.ps1` | New isolated experiment names; existing sequential-process and pinned-runtime checks retained. |
| `configs/ppo_non_residual_refine_v1/`, `configs/ppo_residual_rr_fix_v1/` | Six-file isolated profiles each; changes to stage/nominal semantics only, action ranges/reward/normalizer/physical capability unchanged. |

Targeted tests include41 terminal tests,155 RR/source tests,176 related contracts,33 migration tests and181 migration regressions; groups overlap and must not be summed as unique tests. Old-input RR replay2202frames and CPU official optimizer/save-load proof are component evidence, not physical success credit.
