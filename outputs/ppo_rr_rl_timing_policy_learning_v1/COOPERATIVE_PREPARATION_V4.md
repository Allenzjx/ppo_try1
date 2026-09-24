# Cooperative preparation v4 — implementation boundary

Latest result: the full2048/16 course and CP225280 deterministic naturalP01 evaluation have now sealed. See `CP225280_RESULT.md`: front placements and RR lift/crossing retained;91.166667s/P09 incomplete with RR gap56.118904mm/0N, no RL placement. Rear OFF/FL ON. Historical design-time and in-progress entries below are retained as chronology, not a claim that training or evaluation is still running.

## Actual frozen course completed

HEAD49eb23163a6e stayed fixed throughout all three blocks, including the early CP223616 video insertion at a completed384-decision block boundary. Real student budget **2048/16 PPO/320 Adam** completed with no new AUX. Full actor/critic/Adam/LR/Identity/HISTORY state was retained, and the same state-dependent distribution defined sample/current likelihood. Actual latest checkpoint CP225280 has225280/1725/34500 lifetime counters and verified full-state reload; its natural-P01 deterministic video is in progress, not yet a proved result at this heading's creation.

| Real optimized block | Decisions / PPO / Adam | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P07 real-prefix suffix |384 /3 /60|0|0|0|0|0|0|1|1|64|2|8|308|0|
| P10 real-prefix suffix |384 /3 /60|0|0|0|0|0|0|0|0|0|1|1|382|0|
| Natural P01 |1280 /10 /200|4|348|4|1|279|191|1|1|451|0|0|0|0|
| Total |2048 /16 /320|4|348|4|1|279|191|2|2|515|3|9|690|0|

P07 student sampling acquired RR TOP contact and short RL lift, but did not retain support/cross RL; initial contact preceded its first new update. P10 RR initial placement belongs to the zero-credit N prefix, followed by intermittent student TOP and eventual real ground revocation. Natural training episode0 ended85.141667s/P09: RR qualification at75.233333s was revoked by ground at75.358333s; no RR crossing/TOP capture. Its prep gate stayed inactive; two new natural-reset samples completed the last rollout. These separate physical outcomes cannot be combined into one successful natural-P01 policy trajectory. Source clocks and rear-off rules were not changed mid-course.

## Preserved design-time source and implementation

The d7e97ee course ended normally on 2026-09-23 ~20:18 EDT; no episode was interrupted. Its final actual source is selected-branch CP223232, SHA256 `2739173651e516ab19a5213e6d3206f7c970e7d76ed0f3adbde1bc95b7dc86c7`, sidecar SHA256 `c6e0cd978202509cf73d41e5814fe9ad59051a6e3c9f1f14d5f689c8b7c4b550`. Lifetime counters: 223232 decisions / 1709 PPO updates / 34180 Adam steps. The completed course adds 1536 / 12 / 240; selected continuation from CP220544 totals 2688 / 21 / 420. New AUX remains zero.

| Actual course phase | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Optimized decisions |2|422|4|1|163|269|2|2|415|1|1|254|0|

All prefixes are actual continuous physics and receive zero PPO credit. Course budget completion is not task success. In the last P10-prefix block, RR was already placed by the zero-credit prefix; learner decision endpoints show 40 actual RR TOP samples but no RL TOP. End 68.2 s/P12: RR AIR, gap46.608 mm, zero bearing; RL GROUND, front distance−57.221 mm. Historical RR placed is explicitly not current support.

## Control and learning split

RR qualified carry toward FL → capturable RR above TOP → overlapping learned FR receiving-space / FL available-range / RL wheel-space / RR descent → real RR support → FR-side transfer → qualified RL swing/placement.

The existing source gate still prevents initiating the strong P09 late FL/RL group from AIR RR. That group's actual changed channels are FL/RL hip/knee and a finite FL reverse wheel pulse; FR is already held at source knee+31.1°, and the group has no new RR descent channel. There is no harmless FR/+FL preparation sub-group to simply release. All12 policy channels were already open; the new version does not force angles or wheel signs.

Same422 observations and same learned parameter topology. Conditional preparation sigma broadens FR knee, FL knee, RL hip/knee, and locally FL wheel using already observed carry/reachable/prep flags. RR hip's existing ×4 remains unchanged. One shared kernel defines sampling and current likelihood; raw samples/log probability precede the unchanged physical projection. This is a changed stochastic distribution, not a Gaussian-equivalent migration.

The existing RL workspace potential budget is redistributed to edge .25 / FR receiver .5 / FL available range .125 / RL wheel collider clearance .125. These are bounded progress proxies, not pose acceptance tests. Already adequate FL range saturates; no requirement to reach +30°. RL wheel AABB is not whole-linkage clearance or a collision classification. Legitimate actual RL swing retires preparation components. RR post-cross FL receiver retirement stays intact.

A 0.01/s maximum task-family soft prior only prices supported, actually measured FL counterrolling under positive source suggestion with no RR forward or legal TOP-gap descent. Legitimate negative source pulses/stops, front P01–P06, AIR FL and progress-producing counterroll are excluded. It changes no target. Rear tilt is not newly penalized; mount geometry never enters reward.

The four fixed hip installation points resolve from actual live USD joint body0/localPos0 on the same base_link and transform through its current world pose. Diagnostics report left−right and rear−front average world heights, FR height, source and timestamp. Missing/ambiguous live definitions are N/A, never fabricated from camera orientation or URDF. They are not CoM transfer or bearing evidence.

## Evidence at design time

Accepted CP221696 frame tick9656/80.4667 s: nominal four wheels +.3 rad/s, final FL−.7164/FR+.2532/RL+.2732/RR+.1473; measured−.7705/+.1141/+.0869/+.1463. All four motors have nonzero command and response. FL reversal is policy cancellation, not a mask. RR is AIR, so its rotation is not traction.

Production semantic replay of this saved frame gives FL knee−45.11984°, physical negative margin14.88016° and saturated available-range proxy; RL wheel conservative AABB clearance0. It does not invent current FL negative-bound saturation or whole-linkage collision. Replay adds no physics, PPO or AUX credit.

## Checks and limits

New pure preparation tests34 PASS; P02/functional-carry/transfer/task-first tests143 PASS after making irrelevant front preparation explicitly N/A. Shared-kernel/422/history/prefix CPU regressions156 PASS; four-mount/height diagnostics29 PASS. Tests and synthetic optimizer updates are not real training. One old blanket frozen-file test still expects the pre-existing video_capture.py bytes; the runtime already records that authorized earlier media-only revision. No physics/FSM asset was changed here.

New code is not a claim of RR capture or RL task success. Migration, fresh training and reloaded natural-P01 video must be recorded after their actual completion. The accepted CP221696 video remains immutable and is not relabeled as latest weights.
