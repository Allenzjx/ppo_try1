# SYNTHETIC_QA — limited comparison-renderer check

Result: the existing comparison script's exact rendering expressions passed a
small offline synthetic check. **This is not a robot video, task success,
stability comparison, or end-to-end publication acceptance.** No actual video,
checkpoint, runtime/configuration file, or historical artifact was opened or
changed. Do not present these clips as user deliverables.

The script and its README were read without alteration. To avoid its top-level
training/runtime imports while Isaac was running, one standard-library Python
helper extracted the exact renderer AST and selected rejection functions.
Synthetic dictionaries replaced managed source validation only for this isolated
renderer test; the real CLI and its provenance/success validators were not run.
The original script still requires actual validated C success and compatible
fresh A/C source evidence before real publication.

## Actual result

- Existing local FFmpeg: imageio_ffmpeg's bundled `ffmpeg-win-x86_64-v7.1.exe`;
  no installation/network. Its drawtext/tpad/hstack filters were present.
- Input A: 6 frames / 0.4 seconds; input C: 12 frames / 0.8 seconds. Both are
  1280×720 synthetic moving test cards, explicitly watermarked `SYNTHETIC_QA`
  and `NOT A ROBOT VIDEO`, with synthetic sine audio.
- Output: exactly 12 frames, 2560×720, 15 fps, 0.8-second container, H.264,
  yuv420p, no audio. Fully decoded PTS match `frame_index/15` within 1e-5 s.
- The unchanged filter graph has no crop, scale, interpolation, rate conversion,
  or playback-speed expression. A's source indices 0–5 are retained, followed
  by its final frame at output indices 6–11; C retains source indices 0–11.
  A content-region sampled RGB error against that correspondence was at most
  0.328/255; C at most 0.337/255 after lossy re-encoding. Analysis crops were
  read-only measurement regions, not edits to the composed video.
- The actual executed filter contains neutral `FSM A - INCOMPLETE` and
  `PPO C - SYNTHETIC QA ONLY` labels, plus
  `SOURCE ENDED - last recorded frame held` enabled exactly at `n>=6`.
  Text rendering completed, but independent OCR/manual visual inspection was
  not performed; real-video visual review remains necessary.
- Extracted real rejection functions refused this unmanaged synthetic source,
  a boolean endpoint, and a negative endpoint before loading any physical data.
  Existing output refusal was also checked with an unchanged byte hash.

## Small harness correction, preserved honestly

The first run rendered and decoded successfully but the final QA assertion
incorrectly assumed FFmpeg `-n` must return nonzero for an existing destination.
This binary prints `already exists. Exiting.` and returns **0** while leaving
the file unchanged. `SYNTHETIC_QA_receipt.json` preserves that first harness
failure; the corrected bounded check uses explicit refusal text plus unchanged
size/SHA. The second invocation reused the already-created synthetic clips
without overwriting them and passed. This is not evidence of a production
overwrite bug: the original script separately rejects an existing output
directory before encoding.

Final receipt: `SYNTHETIC_QA_receipt_final.json`, status
`LIMITED_SYNTHETIC_RENDER_QA_PASSED`. The script SHA bound there is
`b9f4d7cf264d85763a8611fffce2cecfecf1a01272d3e053e22729f42e385c36`.
The two helper invocations took 3.391 and 2.844 seconds internally. FFmpeg
software encoder/decoder/filter threads were limited to one; no Torch, CUDA,
Isaac, or optimizer was imported or invoked. Disk artifacts before this report
were about 1.152 MB, well below 100 MB. All helper processes have exited.

No conclusion is made about real source compatibility, checkpoint provenance,
independent physical success, full CLI failure-file handling, or long-video
resource use. Those paths were deliberately outside this minimal offline QA.
