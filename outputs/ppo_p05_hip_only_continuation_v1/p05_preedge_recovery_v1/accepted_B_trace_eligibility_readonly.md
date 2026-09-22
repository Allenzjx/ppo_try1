# Accepted zero-residual B: old-trace eligibility protection

Only the accepted source's 28-row `stage_transition_evidence.jsonl` and existing `semantic_video_source_manifest.json` were read. No Recording scan, video hashing, physics, policy evaluation or production edits.

Source: `runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source`.
Manifest identifies role B, `semantic_prior_eval`, P01 reset, seed 4001, zero optimizer updates, physical success true, duration 73.8083333333 s.

| Recorded event | Episode tick | Simulation time | P05 age |
|---|---:|---:|---:|
| P04 → P05 | 1512 | 12.600000 s | 0 |
| FL qualified lift | 1559 | 12.991667 s | 0.391667 s |
| FL front-edge crossed | 2468 | 20.566667 s | 7.966667 s |
| FL placed | 2677 | 22.308333 s | 9.708333 s |
| P05 → P06 | 2680 | 22.333333 s | 9.733333 s |

P05 lasts 1,168 physics ticks / 9.733333 s until P06 entry. Its last P05 frame is at age 9.725 s, well before the new age-30 gate. Age 30 would correspond to episode tick 5112 / 42.600 s, but this trace already left P05. FL crossing and placement additionally disable the pre-edge predicate.

Therefore the new P05-only predicate has **zero activation opportunity on this accepted old trace**. This is trace-eligibility protection, not a full new-version B physical run and not proof that unrelated new control versions preserve the trajectory.
