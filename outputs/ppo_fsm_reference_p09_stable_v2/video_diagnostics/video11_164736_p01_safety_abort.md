# Video11: playable safety-abort diagnostic, not success

Source run: `20260911T0130039409655Z_gd4e46006b382_4c377fe16f9a405ea1289f599bc2a6f6`; loaded checkpoint counter 164736. The unchanged source verdict is `physical_task_success=false`, `diagnostic_only=true`, physical termination `SAFETY_ABORT`. This natural-P01 run stopped at P02; it is not a successful crossing or evidence of improved stability.

- Actual source capture: 165 frames, 15 fps, 1280×720 H.264/yuv420p; full decode passed, 165 unique frame checksums, zero black-like frames. The original container duration was invalid at 286331153 seconds.
- The diagnostic copy has a valid 11.0-second container and decoded duration. Physical execution was 1317 ticks / 10.975 seconds: 165 issued decisions, 164 returned complete eight-tick steps, then a five-tick safety-interrupted final decision. The existing 25 ms last-frame quantization is retained, not treated as extra simulation.
- The helper exited 0. Stream copy only: no reencoding, trimming, stitching, speed change, interpolation, or added frames. Encoded packet payloads, absolute rational PTS/DTS, packet durations and key flags match exactly; decoded frame identities, absolute PTS/key flags and PTS deltas also match exactly. Output strict duration/codec/full-decode checks passed.
- Original MP4 byte count/SHA and both original manifest byte sequences were verified unchanged. No success publisher, source-verdict edits, production changes, checkpoint loads, or extra Isaac process were used.

Output: `video11_164736_p01_safety_abort.mp4` (15069197 bytes; SHA256 `cfded0595d6af139d3b8355a1bd0d738690c0a5fbc817cddd015026a622a0795`). Detailed receipt: `video11_164736_p01_safety_abort.remux_receipt.json`. Original SHA256 remains `9c75513b6d17543b94118d3efdab5c5ed73397db0b79170a1edde0961359f9d5` (15088081 bytes).

Endpoint QA images: `video11_164736_decoded_01.png` (frame 0, PTS 0) and `video11_164736_decoded_02.png` (frame 164, PTS 10.933333). PNG format/dimensions were checked; visual review is left to the parent agent and is not claimed here.
