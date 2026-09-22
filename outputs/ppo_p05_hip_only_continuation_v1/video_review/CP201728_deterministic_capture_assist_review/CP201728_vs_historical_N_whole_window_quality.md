# CP201728 versus historical N: whole-window descriptive quality

These values come directly from each sealed source manifest at `physical_episode.quality_metrics.global`. They cover each source's complete physical window; they are not phase- or event-aligned.

| Source | Real task result | Physical window | Roll RMS / peak (rad) | Pitch RMS / peak (rad) | Roll-rate RMS (rad/s) | Pitch-rate RMS (rad/s) |
|:--|:--|:--|--:|--:|--:|--:|
| CP201728, deterministic capture-assist checkpoint, +16 P05 PPO updates | **DIAGNOSTIC_FAILURE** — `INCOMPLETE_CONTROLLER_BLOCKED` in P09 | 10084 ticks, 84.033333 s | 0.131952 / 0.199699 | 0.078493 / 0.176746 | 0.052028 | 0.054879 |
| Historical N_ref, commit `ee5a9651591d`, nominal zero residual | **SUCCESS** | 8857 ticks, 73.808333 s | 0.119031 / 0.243379 | 0.081308 / 0.191626 | 0.067664 | 0.065342 |

The CP201728 terminal reason is taken from the last sealed `video_policy_decisions.jsonl` row (`decision=1261`, request phase P09, tick 10080→10084, `environment_step_returned=true`). The generic video-source error is only `SemanticVideoError: episode did not meet common physical task`; it does not replace the exact controller reason.

## Interpretation limit

This is a descriptive side-by-side only. CP201728 and N_ref use different runtime/experiment versions, different control methods, different durations, and different task outcomes (failure versus success). Numerically lower raw values do **not** establish a stability or policy improvement here, and this is not a causal counterfactual.
