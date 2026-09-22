# New task-conditioned review adapter

This is an output-only copy/adaptation of the existing 238-line FL review. Old review and old paired/cross-HEAD helpers are unchanged. It does not launch Isaac, run a policy or create missing recordings.

Formal C requires a sealed natural-P01 single-episode source, no intervention, the `task_conditioned_hip_wheel_v1` runtime, `task_conditioned_hip_wheel_sigma_v1`, verified official checkpoint load/save roundtrip, immutable checkpoint/manifest hashes and **positive** `task_conditioned_hip_wheel_branch_counts` in all three actual-learning counters. Initial migration/old FL branch credit cannot satisfy this. Deterministic/stochastic labeling must agree with actual sampling provenance.

Names follow the requested neutral whole-attempt scheme. A failed attempt still uses `CPxxxx_deterministic_P01_full.mp4`; its physical failure is explicit on every frame and in the receipt, not hidden by the filename. No failure tail or internal waiting is removed. Native 15fps/1x view and synchronized four-wheel measured native qd remain. qd is joint rotation, not endpoint displacement or proven traction. Unavailable synchronized body/hip overlay is null, not filled from a nearby tick.

## Commands after actual new runs exist

The paths below are placeholders, **not existing/newly evaluated runs**. Substitute actual sealed source directories and actual saved CP counts. Use the existing Isaac environment Python.

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_task_conditioned_hip_wheel_v1/review_video.py export --source 'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/ACTUAL_DET_RUN/source' --destination 'outputs/ppo_task_conditioned_hip_wheel_v1/CPxxxx_deterministic_review'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_task_conditioned_hip_wheel_v1/review_video.py export --source 'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/ACTUAL_STOCH_RUN/source' --destination 'outputs/ppo_task_conditioned_hip_wheel_v1/CPxxxx_stochastic_seed4101_review'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_task_conditioned_hip_wheel_v1/review_video.py export --source 'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/ACTUAL_B_RUN/source' --destination 'outputs/ppo_task_conditioned_hip_wheel_v1/B_current_review'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_task_conditioned_hip_wheel_v1/review_video.py verify-modes --deterministic-receipt 'outputs/ppo_task_conditioned_hip_wheel_v1/CPxxxx_deterministic_review/CPxxxx_deterministic_P01_full.media.json' --stochastic-receipt 'outputs/ppo_task_conditioned_hip_wheel_v1/CPxxxx_stochastic_seed4101_review/CPxxxx_stochastic_P01_full_seed4101.media.json' --output 'outputs/ppo_task_conditioned_hip_wheel_v1/CPxxxx_same_model_modes.json'
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' outputs/ppo_task_conditioned_hip_wheel_v1/review_video.py pair --b-receipt 'outputs/ppo_task_conditioned_hip_wheel_v1/B_current_review/B_current_Nplus0_P01_full.media.json' --deterministic-receipt 'outputs/ppo_task_conditioned_hip_wheel_v1/CPxxxx_deterministic_review/CPxxxx_deterministic_P01_full.media.json' --destination 'outputs/ppo_task_conditioned_hip_wheel_v1/CPxxxx_N_pair'
```

`verify-modes` requires the same exact checkpoint and runtime/scene/camera/evaluation bindings. `pair` requires same-version current B=N+0 and deterministic C; it does not silently relabel accepted historical N_ref/B as current B. If historical B is reused, that needs a separately disclosed, specifically reviewed cross-version comparison; do not weaken this adapter or edit old receipts.

The paired encoder is reused from the old helper only after the new adapter validates its own inputs. It keeps full attempts at the same elapsed P01 origin; the shorter side is explicitly labeled `RUN WINDOW ENDED - FROZEN FRAME`. Freeze has zero additional physical evidence. No phase time-warping, derived quality improvement claim or success inference is added. Event-window quantitative quality remains a separate analysis, not a condition for exporting failure footage.

Synthetic metadata tests are under `candidate/test_task_review_video.py`. They test branch/mode/checkpoint bindings and positive/negative cases only; no fake media is generated and no real run is claimed. Full decoding/render preview remains mandatory on the first actual export.
