# CP223616 vs historical N: descriptive body-motion table

These are separate sealed physical episodes with different task outcomes and durations. The table is descriptive only: it does **not** establish stability superiority, causality, or a time-normalized ranking. Historical N is not a fresh same-controller B. Video freeze is excluded.

| Run / physical window | Outcome | Duration s | Samples | Linear speed RMS / max (m/s) | Angular speed RMS / max (rad/s) | Roll min..max / RMS0 (deg) | Pitch min..max / RMS0 (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| CP223616_candidate / full_physical_episode | INCOMPLETE | 85.108333 | 1278 | 0.024564 / 0.172012 | 0.096186 / 1.219997 | -11.673605..6.778370 / 8.041836 | -10.287398..0.342895 / 5.516429 |
| CP223616_candidate / RR_to_RL_window | INCOMPLETE | 35.041667 | 527 | 0.023945 / 0.168899 | 0.088121 / 0.900706 | -8.940930..6.778370 / 8.548608 | -8.076416..-0.831788 / 2.225383 |
| historical_N_reference / full_physical_episode | SUCCESS | 73.808333 | 1109 | 0.034242 / 0.245269 | 0.106492 / 1.198817 | -13.943172..1.139552 / 6.819658 | -10.977186..1.660002 / 4.658435 |
| historical_N_reference / RR_to_RL_window | SUCCESS | 30.808333 | 464 | 0.044794 / 0.245269 | 0.109774 / 0.781008 | -11.678968..1.139552 / 5.364660 | -8.821180..0.860351 / 2.092562 |

## Sampling and source boundary

Both runs use their sealed 15 Hz `height_diagnostics.jsonl` sensor stream (dominant stride: 8 physics ticks at 120 Hz). RMS uses a documented left-sample piecewise-constant weighting over the exact physical duration; it is an endpoint-weighted approximation, not a 120 Hz reconstruction or interpolation.

Only each source manifest and height-diagnostic JSONL are used and SHA-verified. Full source paths and hashes are retained in the JSON report.

Machine-readable report: [CP223616_descriptive_stability.json](CP223616_descriptive_stability.json)
