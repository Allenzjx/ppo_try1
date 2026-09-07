# C68224 fixed-policy P01 evaluation: first unfinished FL capture

## Scope and completed result

Source: `runs/ppo_semantic_v3/validation/20260906T1628525452104Z_g68cd9f5fca6c_fac838181e114f68954319480dae7b77`, completed evaluation only. Read-only PowerShell inspection; no Python, checkpoint tensor loading, simulation, or production changes.

- Recorded runtime HEAD: `68cd9f5fca6c9ba06c38666fb55024b9ba04f9b0`; checkpoint `checkpoint_step_000068224.pt`. Its manifest-recorded SHA-256 is `0bc86a834f7baeb6fb35a5c27d9d31dab6738f7656f528dc8cf101866af26897` (not independently rehashed here).
- Seed 2001, deterministic policy, natural P01 start, **0 optimizer updates** during evaluation.
- Execution lifecycle `SUCCEEDED`; **task success false**. 632 decisions, 5,056 physics ticks, 42.1333333333 s. Terminal P05 age 30.0 s, `INCOMPLETE_CONTROLLER_BLOCKED`.
- Physical evaluator valid true, physical `termination_reason=null`; no recorded body collision, wheel-only failure, or safety failure. Deadline is a true terminal: bootstrap false, terminal-event reward -40, absorbing next potential 0.
- Decision counts P01–P13: `[1,177,3,1,450,0,0,0,0,0,0,0,0]`. This trajectory never entered P06; the selected P06-tail change cannot explain this P05 outcome.

## Continuous front-leg chronology

| Event | Tick | Time (s) | Evidence |
| --- | ---: | ---: | --- |
| P01 → P02 | 8 | 0.066667 | Ordinary stage transition |
| FR qualified lift | 49 | 0.408333 | Real lift history |
| P02 → P03 | 1424 | 11.866667 | Ordinary stage transition |
| FR front crossing | 1434 | 11.950000 | Real crossing history |
| FR placed / P03 → P04 | 1448 | 12.066667 | Real placement history |
| P04 → P05 | 1456 | 12.133333 | FL still ground-contacting at entry |
| Last FL ground contact | 1468 | 12.233333 | Ground active, obstacle inactive |
| FL enters uninterrupted AIR | 1469 | 12.241667 | No ground or obstacle contact thereafter |
| FL initial clearance | 1483 | 12.358333 | Initial event; not qualified or placed |
| FL qualified lift | 1532 | 12.766667 | Gain 49.950253 mm; clearance +0.344654 mm above top |
| FL front crossing | 2561 | 21.341667 | AIR, front +1.040402 mm, clearance +63.292508 mm |
| Terminal P05 | 5056 | 42.133333 | FL qualified/crossed true, **placed false** |

The FL AIR suffix contains 3,588 raw samples, ticks 1469–5056 inclusive. Across the whole P05 interval including entry, FL has 13 ground-active samples (1456–1468), **zero obstacle-active samples**, and zero consecutive TOP samples at every one of its 450 decision-end snapshots. Historical FR placement is genuine; it is not a substitute for current FL capture.

## Closest approach to actual FL placement

| Raw sample | Tick / time | Front relative to obstacle (mm) | Bottom above top (mm) | Current contact |
| --- | --- | ---: | ---: | --- |
| Peak post-cross height | 2594 / 21.616667 s | +50.085250 | +85.486805 | AIR, obstacle force 0 N |
| Furthest post-cross advance | 2606 / 21.716667 s | +53.853575 | +75.230156 | AIR, obstacle force 0 N |
| Closest recorded bottom to top | 2657 / 22.141667 s | +27.759853 | **+0.402172** | AIR, obstacle force 0 N |
| Final | 5056 / 42.133333 s | +29.225651 | +2.212522 | AIR, obstacle force 0 N |

After crossing, FL front remained positive and its recorded clearance remained positive. At the closest sample the verified wheel/obstacle pair is inactive, its force is zero, and there is no TOP-contact sequence. A contact-point field exists even for this inactive zero-force pair; point existence alone is **not** evidence of touchdown or support. Sub-millimetre clearance from the cached live collider-extent geometry is also not proof of a contact-detector error.

The best `placed_FL` progress is 0.85 (first at decision 329 / tick 2632; 304 of 450 P05 decision-end samples). The existing predicate explains it exactly: qualified lift contributes 0.35, crossing 0.35, and valid top geometry without TOP samples contributes `0.30 × 0.5 = 0.15`. Total **0.85**, not a placement event or “85% task success.” The missing physical component is sustained actual FL capture/contact, not an absent stage label.

## Current support and control at the terminal

| Leg | Current state | Front (mm) | Clearance (mm) | Load fraction | Historical placed |
| --- | --- | ---: | ---: | ---: | --- |
| FL | AIR | +29.225651 | +2.212522 | 0 | false |
| FR | TOP | +128.634873 | -0.155934 | 0.424179 | true |
| RL | GROUND | -555.168204 | -50.451306 | 0.498911 | false |
| RR | GROUND | -545.335913 | -50.667035 | 0.076910 | false |

FL is inside the measured top XY region and its top geometry is valid, but it carries no load. FR remains TOP-supporting; both rear legs remain ground-supporting. RR/RL have no qualified/crossed/placed events. The **first unfinished subtask is FL placement in P05**, before rear approach or P13 stopping.

PPO was not disabled: final nominal full12 is `[22.8,-13.4,0,45.9,6.9,0,0,0,0,0,0,0]`; final logical residual wheels are `[-0.00598130,-0.01066752,+0.02495894,+0.06776555]` rad/s. The dispatched native float32 wheel targets are `[+0.00598130,-0.01066752,-0.02495894,+0.06776555]` after the existing physical sign mapping. FL measured wheel speed is -0.00648454 rad/s in its logical observation convention. These are command/measurement evidence, not a claim that a particular residual caused the missing capture.

## Audit completeness and conclusion boundary

The 632 completed decision audit records sum to **5,056 verified native ticks**. Own-phase request effect is 5,052 ticks; the four ordinary incoming handoff-first ticks are excluded. Every decision reports verified no-in-episode state writes. All 5,057 raw observation rows including reset are finite; none reports body collision. No scene/root rewrite or new optimizer activity was used in this evaluation.

This is a valid incomplete fixed-mean trajectory with real FL lift and crossing but no recorded FL touchdown. It does not establish a detector bug, a reward-sign bug, or causality from one control component. It also does not support assigning this failure to P06-tail behaviour, because P06 was never executed. Preserve the physical failure/incomplete distinction and retain the actual placement criterion; the next P01 training block keeps this unfinished front-leg task in its sampling distribution.

Sources: completed `evaluation_manifest.json`, `stage_transition_evidence.jsonl`, all decision records in `residual_and_projection_audit.jsonl`, and the 120 Hz `physical_observations.jsonl`. Source-level interpretation uses the existing `TaskEvaluator.predicate("placed_FL", ...)` formula; no thresholds or hard history were changed.
