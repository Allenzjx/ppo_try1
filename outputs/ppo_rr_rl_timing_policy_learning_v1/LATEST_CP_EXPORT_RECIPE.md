# Latest cooperative-policy DET delivery recipe

Use `prepare_latest_cooperative_delivery.ps1` only after the natural-P01 video run is sealed. The helper is dry-run by default. It reads the sealed source's `checkpoint_load_provenance`, verifies the bound checkpoint and sidecar bytes, takes the lifetime counters from that exact sidecar, and renders them back into the existing exporter's explicit `--expected-*` pins. The exporter then independently verifies those counters against both the sidecar and video load proof.

The source manifest does not directly replace the counter pins: it identifies the exact bound sidecar from which they are derived. This retains the existing fail-closed hash/counter boundary.

## 1. Review the immutable plan

```powershell
& outputs/ppo_rr_rl_timing_policy_learning_v1/prepare_latest_cooperative_delivery.ps1 `
  -Source '<sealed-run>\source' `
  -Destination 'outputs/ppo_rr_rl_timing_policy_learning_v1/video_review/CP<actual>_deterministic_cooperative_prep_v4_review' `
  -ExpectedHead '49eb23163a6e20bc56301dbafb59b137ecebce66'
```

The command must report `DRY_RUN_ONLY`. Review its source/run-manifest hashes, checkpoint/sidecar hashes, and actual lifetime counters. An active source, an existing destination, a non-deterministic evaluation, an unverified load, or an existing same-step audit fails before video processing.

## 2. Export and audit

Repeat the same command with `-Execute`. It runs, in order:

1. `export_policy_rear_no_assist_video.py` using the existing full/decode/PTS/black-frame QA, immutable destination, historical-N freeze label, and strict `FL ON / rear task assist OFF` validation.
2. `analyze_cooperative_rr_rl_window.py` against the same sealed source/checkpoint pins.

The exporter prints `FULL_PLAYABLE_VALIDATED <absolute path>` as soon as the full video is usable, then finishes the real RR→RL detail (or an honest not-reached detail), historical-N comparison, previews, hashes, and `export_receipt.json`.

## 3. Final checks

- Receipt: all three `validation.valid=true`, `full_decode=true`, `timestamps_monotonic=true`, `timestamps_continuous=true`, and `black_like_frame_count=0`.
- Full: exact source frame count and complete physical failure/success tail at normal 15 fps.
- Detail: one contiguous same-episode interval; no retiming or removed stalls.
- Comparison: historical N is explicitly not a fresh same-controller B and freezes only after its own endpoint.
- Visual: inspect full first/middle/terminal and comparison terminal PNGs; confirm the HUD says `FL ASSIST ON`, `REAR TASK ASSIST OFF`, the actual checkpoint step, and the observed physical outcome.
- Stop condition: Python/FFmpeg processes from this helper have exited before another viewport capture starts.
