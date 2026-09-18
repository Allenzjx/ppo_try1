# Zero v2 sealed review

Recorded outcome: **SUCCESS**, natural P01–P13, 8857 physics ticks /73.808333s. This is N+0 nominal control, not learned residual PPO. The previous v1 run remains a terminal failure; its outcome is not overwritten.

Deliver `zero_Nplus0_full_review_clean.mp4` and its source-bound receipt. `zero_Nplus0_full_native.mp4` is the unmodified-frame native remux. Both fully decode:1108frames,15fps, monotonic N/15 PTS, ≤200s, no interpolation or cross-run edit. Original camera pixels are uncropped. The old `zero_Nplus0_full_review.mp4` drawtext derivative is retained. An early full-frame preview suggested missing title glyphs, leading to the additional static-PNG-title derivative; later pixel comparison of original keyframe01 versus02 found0 expected bright-caption pixels lost to black. Encoding damage was not established, and the clean derivative should not be cited as proof that source or original video was corrupt.

## Terminal facts

Home entry8737; quintic ramp0.5s; full source nominal first dispatched8798. Terminal four nominal wheels0. Canonical measured qd FL/FR/RL/RR: `[+0.149181992,+0.002094464,+0.000390270,-0.239495918] rad/s`. All meet abs(.25); RR margin is only .010504082rad/s. Body speed .021497636m/s and angular speed .059457006rad/s meet .05/.30. Region/support/controlled/fixed-post-observation completion all true, region loss false.

Vs v1: final .5s RR exceedances17/60→5/60; measured control passing43/60→48/60; final continuous controlled segment8831–8857,27samples (.225s sample exposure; .216667s endpoint span). V1's longest segment was20samples and did not include its failed endpoint. But full-second linear-speed exceedances41→57 and angular17→22; actual max source-home error5.438°→7.869°. Thus terminal completion is repaired; universal stability superiority or exact home convergence is not claimed. See `zero_home_v1_v2_comparison.json` for exact values and intervals.

## Visual QA

Manually inspected all15 distinct original review keyframes: start, FR lift/cross/place, FL lift/cross/place, RR lift/cross-place, RL lift/cross/place, P13 entry, home entry and terminal. Also inspected clean-rendered early frame2, RR frame8, RL frame12 and terminal frame15. The robot outline stays in frame in these views; near-side FR clipping is fixed. Obstacle leading edge and near-side RR motion are readable; distant RL contact remains partly self-occluded. This is not a certificate that all four contacts are unobstructed throughout all1108frames, nor an independent wheel-force causal isolation.

Relevant clean images: `clean_camera_key_01.png` start; `08` RR lift; `09` RR crossing/placement; `11` RL crossing; `12` RL placement; `14` home entry; `15` final. Actual event-to-image offsets0–7ticks are documented in the media receipt; the image is never assigned an earlier event tick than its actual ledger sample.

The exporter receipts retain `manual_camera_QA_pending=true` as their original automated-stage state. This separate review records the completed manual scope and its limitations without rewriting the sealed receipts.
