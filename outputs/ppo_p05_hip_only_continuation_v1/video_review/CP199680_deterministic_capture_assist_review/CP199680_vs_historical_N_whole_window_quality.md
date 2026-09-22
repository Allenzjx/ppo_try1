# CP199680 versus historical N: whole-window descriptive quality

These values come directly from each sealed source manifest at `physical_episode.quality_metrics.global`. They cover each source's complete physical window; they are not phase- or event-aligned.

| Source | Real task result | Physical window | Roll RMS / peak (rad) | Pitch RMS / peak (rad) | Roll-rate RMS (rad/s) | Pitch-rate RMS (rad/s) |
|:--|:--|:--|--:|--:|--:|--:|
| CP199680, capture-assist migrated warm start, +0 P05 PPO updates | **DIAGNOSTIC_FAILURE** — `INCOMPLETE_CONTROLLER_BLOCKED` in P09 | 8697 ticks, 72.475000 s | 0.229407 / 0.365985 | 0.226745 / 0.357775 | 0.072069 | 0.060016 |
| Historical N_ref, commit `ee5a9651591d`, nominal zero residual | **SUCCESS** | 8857 ticks, 73.808333 s | 0.119031 / 0.243379 | 0.081308 / 0.191626 | 0.067664 | 0.065342 |

The CP199680 terminal reason is taken from the last sealed `video_policy_decisions.jsonl` row (`decision=1088`, request phase P09, tick 8696→8697, `environment_step_returned=true`). The generic video-source error is only `SemanticVideoError: episode did not meet common physical task`; it does not replace the exact controller reason.

## Interpretation limit

This is a descriptive side-by-side only. CP199680 and N_ref use different runtime/experiment versions, different control methods, different durations, and different task outcomes (failure versus success). The raw differences therefore do **not** establish a stability or policy improvement, and they are not a causal counterfactual.
