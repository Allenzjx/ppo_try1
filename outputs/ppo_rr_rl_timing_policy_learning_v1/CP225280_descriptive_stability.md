# CP225280 vs historical N: descriptive body-motion table

These are separate sealed physical episodes with different task outcomes and durations. The table is descriptive only: it does **not** establish stability superiority, causality, or a time-normalized ranking. Historical N is not a fresh same-controller B. Video freeze is excluded.

| Run / physical window | Outcome | Duration s | Samples | Linear speed RMS / max (m/s) | Angular speed RMS / max (rad/s) | Roll min..max / RMS0 (deg) | Pitch min..max / RMS0 (deg) |
|---|---|---:|---:|---:|---:|---:|---:|
| CP225280_candidate / full_physical_episode | INCOMPLETE | 91.166667 | 1369 | 0.023466 / 0.158111 | 0.097475 / 1.208354 | -11.618396..7.069788 / 8.063903 | -9.962396..0.649630 / 5.482583 |
| CP225280_candidate / RR_to_RL_window | INCOMPLETE | 35.033333 | 527 | 0.023243 / 0.153300 | 0.095569 / 0.925086 | -9.362101..7.069788 / 8.872302 | -7.892506..-0.190660 / 2.080164 |
| historical_N_reference / full_physical_episode | SUCCESS | 73.808333 | 1109 | 0.034242 / 0.245269 | 0.106492 / 1.198817 | -13.943172..1.139552 / 6.819658 | -10.977186..1.660002 / 4.658435 |
| historical_N_reference / RR_to_RL_window | SUCCESS | 30.808333 | 464 | 0.044794 / 0.245269 | 0.109774 / 0.781008 | -11.678968..1.139552 / 5.364660 | -8.821180..0.860351 / 2.092562 |

## Sampling and source boundary

Both runs use their sealed 15 Hz `height_diagnostics.jsonl` sensor stream (dominant stride: 8 physics ticks at 120 Hz). RMS uses a documented left-sample piecewise-constant weighting over the exact physical duration; it is an endpoint-weighted approximation, not a 120 Hz reconstruction or interpolation.

Candidate label `CP225280` comes from the lifetime counter in the exact sidecar referenced and byte-verified by the sealed source's checkpoint load provenance; it is not caller-provided.

Only the candidate run/source manifests, its checkpoint/sidecar, and each source's height-diagnostic JSONL are used and SHA-verified. Full paths and hashes are retained in the JSON report.

Machine-readable report: [CP225280_descriptive_stability.json](CP225280_descriptive_stability.json)
