# Video7: playable safety-abort diagnostic, not success

Verified 2026-09-10 17:29:34Z (receipt records the equivalent offset-aware timestamp). Source run: `20260910T1711073721697Z_gd4e46006b382_c8a157341180454faab7e89333997585`.

The physical result remains **incomplete: P06 RR-knee hard-limit safety abort**, not a successful crossing. The completed source records 732 issued decisions: 731 full eight-tick decisions plus five final ticks = 5853 physics ticks / 48.775 s. No success publisher was called and no task, source-manifest, checkpoint, or production file was modified.

Only the MP4 container was repaired with the existing diagnostic stream-copy workflow (`-c:v copy`, no overwrite). The original's 286331153 s container duration was invalid despite valid 15 fps packets and full decoding. The separate output has a valid **48.8 s** container, below 200 s. Its existing frame quantization is 0.025 s longer than the physical episode; no frame was added to cover that difference.

- Source and output both fully decode to 732 unique 1280×720 H.264/yuv420p frames; black-like frames = 0. First/last decoded PTS: 0 / 48.733333 s; timestamps are monotonic and native-rate continuous.
- All 732 encoded packet payload SHA256 values, normalized exact rational PTS/DTS/durations, packet key flags (25 key frames), decoded frame checksums, absolute frame PTS, and PTS deltas are identical. Container time base changes from 1/15 to 1/15360 without changing time in seconds. Ordered packet-record SHA256: `fd4fff40553eaabf01223edaca2c53e85502fb76ea4890819b862d6cf26b2d62`.
- Source bytes before/after: 69230500; SHA256 `e2e1cca9b40aae125229a52b7e90c102c443c783a9949581636565007445e843`. Source manifest bytes also unchanged (SHA256 `c697bbe048db59225f2670c196ee12af6734d2b223ce6cb02debe21bd379728f`).
- Output: 69146344 bytes; SHA256 `32c5efab7232cbb865d927a7fe1454fd1dfce772541e4b2a751a336ffa7f7a16`. No re-encoding, trimming, stitching, speed change, interpolation, or success relabeling occurred.

Artifacts in this directory:

- `video7_154240_p01_safety_abort.mp4`
- `video7_154240_p01_safety_abort.remux_receipt.json` (commands and identity evidence)
- `video7_154240_decoded_01.png` (source-identical frame 0 / PTS 0)
- `video7_154240_decoded_02.png` (source-identical frame 731 / PTS 48.733333)
- `remux_video7_diagnostic.py` (video7-only output helper; old helpers unchanged)

The PNGs are decoded source-identical frames, not overlays or synthesized views. PNG structural checks passed; human visual review is left to the parent agent. This media check is independent of training and establishes no task success or comparative stability claim.
