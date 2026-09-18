# Live continuation state — 2026-09-17

Production commit: `6c2121b68654eaaa2afa7c19e9d02ae770493d2f`. Runtime/source/config must not be hot-edited while the active Isaac run continues. One execution resource only.

SEALED SUCCESS: zero home-v2 full natural-P01 video, seed4001, N+0. 8857 physics ticks / 73.808333 seconds, all four crossed/placed and final controlled stop true. Session35655 exited0; original success remains untouched. Media export is in progress with learning_signal_recovery.

Run: `runs/ppo_non_residual_refine_v1/video_eval/prior_B/20260917T0424208857504Z_g6c2121b68654_13804a86a302480ab61ad2af0dcb8dcd`.

Bounded read-only live status:
`python outputs/ppo_task_first_recovery_v1/peek_video.py <run>`.

Prior home-v1 new-camera run is sealed terminal failure, not discarded:
`runs/ppo_non_residual_refine_v1/video_eval/prior_B/20260917T0343245466199Z_g3d231897c91c_0e58ddffca37441089dd17f34f4ed5d8`.
Its full video has already been delivered: `outputs/video_review_v1/zero_refine_3d23189_run034324/zero_Nplus0_full_review.mp4`.
Final RR speed=-.370923rad/s;17/60 final-half-second samples exceed .25. Body/region/support passed; home recovery happened but stopping did not pass. Original73.808333s successful zero remains preserved separately.

Source remains preserved at `outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000177792.pt`:177792 decisions/1354 updates/27080 optimizer steps. Source SHA `e4552bb18ef223fb51da31a616cf57cf3291ba1a108b08e7da40c07bee0db0ba`.

SEALED new RR P01 training: `runs/ppo_residual_rr_fix_v1/train/20260917T0450450693297Z_g6c2121b68654_8996bc65765a494c877b12dae8792f81`, SUCCEEDED execution, +128 decisions/+1 update/+20 optimizer steps. Latest saved `outputs/ppo_residual_rr_fix_v1/checkpoints/history/checkpoint_step_000177920.pt`. Actual1355th update LR1e-5, KL.0231517481. Ended at nonterminal P02 rollout boundary, not full task success.

SEALED P06 suffix512: session44166 exited0, run `runs/ppo_residual_rr_fix_v1/train/20260917T0454003203604Z_g6c2121b68654_7079dc3b6d4f471d8d3dfe5404d49924`. Fresh physical successful_nominal P01 prefix, no second migration. Saved latest `outputs/ppo_residual_rr_fix_v1/checkpoints/history/checkpoint_step_000178432.pt`, cumulative178432/1359/27180. Total newRR640 decisions/5 updates/100 optimizer steps. End at nonterminal tick6776/P12: RR historical placement true but current ground/front−.130729m, RL not placed; no full task success. Agents are analyzing sealed accounting and actual RR history.

SEALED formal C natural-P01 video: session75309 exited1 because the wrapper rejects DIAGNOSTIC_FAILURE, not due to simulator crash; no active Isaac process remains. Run `runs/ppo_residual_rr_fix_v1/video_eval/validation/20260917T0510347247858Z_g6c2121b68654_35383f247031424e9b890de765bc2b13`. Real reload CP178432, seed4001, full12 deterministic policy, no teacher/intervention, same review camera. Terminal5907/49.225s atP05, INCOMPLETE_CONTROLLER_BLOCKED/LOCAL_BOUNDED_RECOVERY_EXHAUSTED. FR placed; FL crossed but AIR gap3.470128mm and not placed. Physical evaluator valid/VERIFIED, no collision/nonfinite/hardlimit event. Formal run never reached RR. Model remains candidate, not success.

All media/reports are finished. `outputs/video_review_v1/cp178432_run051034/PPO_full_actual_wheels_RR.mp4`, `zero_vs_PPO_same_camera.mp4`, and the preferred `PPO_P05_FL_capture_failure_actual_30s.mp4` are exported/decoded/PTS checked. Root personally inspected zero and C event/end keyframes, comparison endpoint and FL failure end; independent decoded-caption crop confirmed no text corruption from the earlier scaled-preview concern. Media QA receipt is `DELIVERY_AND_QA.md` in that directory. Full PPO and zero videos were already delivered in commentary.

`outputs/diagnostics_v1/formal_P01_CP178432_failure.md/json` is complete (JSON41MB; read MD or selected summary keys, never dump whole JSON). Actual late FL had no contact, gap1.387–6.641mm, finite N endpoint hold after2609; knee tracking error small, hip compensation oscillates. Small own FL residual is not proof of a sole-channel cause; full-body mean policy/configuration keeps FL AIR. Formal ledger/resume and `outputs/residual_rr_fix_v1/DELIVERY.md` are final. No further training is active or scheduled. Latest user spec permits exact remaining blocker plus best failure evidence; do not claim completed PPO task or stability superiority.

Ready migration: `outputs/residual_rr_fix_v1/checkpoint177792_rr_physical_acceptance_migration.json`; already built and revalidated against actual committed bytes. RR target runtime `cfad9c3664f4e479fd16a9902178cc4e84327ab06aa03dd3db83e3d96a89aed1`. Preserves complete quarter372/full12 actor/critic/Adam/effectiveLR1e-5/Identity/RNG/counters; old rollout discarded. Semantics/nominal transition change is explicit; no MDP equivalence claimed.

After active zero naturally seals:

1. Export/inspect its real result and new-camera/home video; keep failure if it fails. A second zero failure is not a gate against valid PPO training.
2. Run128 actual decisions from naturalP01/full_episode with `residual_rr_fix_v1`, source CP177792 and the above migration, seed1001/N1/cuda:0/checkpoint-every-update.
3. Reuse its actual newest checkpoint for512 P06 phase_suffix decisions with `PrefixSource successful_nominal`; preparation comes from fresh physical P01 prefix, teacher steps excluded. Same runtime, no second migration.
4. Reload actual newest checkpoint for full naturalP01 semantic_residual_eval video, seed4001, new RR profile/review camera, no mask/teacher/intervention.
5. Publish latest full attempt/failure and same-camera N+0 comparison; inspect first incomplete task and RR physical evidence. Do not present old CP177152 historical evaluator success as newly validated RR success.

Planned128+512 is not completed credit. Training summary command: Node `outputs/ppo_task_first_recovery_v1/summarize_recovery_training.mjs --run <finalized run> --branch residual_rr_fix_v1`; save receipts only under new RR output folder. Reward supplementary helper is `summarize_task_reward_signal.mjs`. Live training helper is `peek_run.py`.

Agents: recovery_branch_migration owns outputs-only RR diagnosis/new sealed analysis; learning_signal_recovery owns outputs-only sealed media/QA; task_reward_profile owns outputs-only continuation/accounting and bounded front-quality baseline review. All production implementations are finished/frozen. Tests include33 RR migration +181 migration regression;155 RR/source +176 related contracts;41 terminal;154 zero/camera/video. Groups overlap and are not additive totals. CPU/replay tests are not physical training/success.
