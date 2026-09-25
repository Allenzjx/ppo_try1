# Direct-recorder media interface smoke

Status: **PASS** (output-only; no source MP4 was opened).

The existing shared helper does not require the old semantic-video manifest
inside the functions reused by the new adapter:

- `checked_media(context, ffmpeg=...)` reads only `context.video`,
  `context.ledger_path`, `context.endpoint`, and
  `context.capture.{frame_count,video_sha256}`. It reconstructs the exact
  8-physics-tick/15-fps grid and performs the full decode at final export.
- `encode_full(...)` reads `context.video`, the already joined rows, the
  adapter-provided outcome/identity and the adapter's temporarily installed
  `panel_lines`. It does not inspect role, legacy checkpoint-load provenance,
  or `episode_physics_ticks` in a source manifest.
- `encode_detail(...)` consumes only the completed full video, joined rows and
  the adapter's explicit detail plan.
- `validate_output(...)` and `preview(...)` consume only the derived video and
  its expected frame geometry/count.

The adapter itself reconstructs the direct context from the new
`source_manifest.json`, including `actual_ticks` as `context.endpoint`; it does
not pass that manifest to an old candidate validator.

One real, fully written endpoint from the live diagnostic source was joined at
episode tick 8 / native raw tick 187. Calling the exact `panel_lines` and
`rgba_panels` path used by `encode_full` produced 15 lines, 1,187,840 RGBA
bytes (`1280 x 232 x 4`), and a maximum measured line width of 1080 px within
the 1280 px panel. The isolated visual artifact is:

`test_preview/actual_endpoint_000008_overlay_smoke.png`

This image is a field/layout smoke only. It is not a video, not a sealed-run
result, and not evidence of diagnostic or PPO success. The growing source MP4
was not read.
