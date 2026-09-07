# C72320 fixed-mean P01 evaluation: FL crosses but never captures

## Completed outcome and provenance

Run: `runs/ppo_semantic_v3/validation/20260906T1717048576808Z_g73e937039708_b1b7a8788eb545698c65882674df3153`.

- Runtime `73e937039708a7306b2b2c941e1f9108b8fa8b3d`, checkpoint `checkpoint_step_000072320.pt`. Manifest-recorded checkpoint SHA-256: `96d26c928dcd9d07f95bf4f33cd5fade64220de7bdffbb5b581f0dcadc64c10f`; no new hash or tensor load was performed for this report.
- Seed 2001, deterministic policy, natural P01 start, no teacher, **0 optimizer updates** during evaluation.
- Final run lifecycle **SUCCEEDED (execution)**; **task success false**. 647 decisions, 5,176 ticks, **43.1333333333 s**, P05 stage age 30.0 s, `INCOMPLETE_CONTROLLER_BLOCKED`.
- Physical evaluator valid true, physical termination reason null, success false. This is a true task deadline, not a body collision or wheel-only failure. Terminal bootstrap false, absorbing next potential 0, terminal event -40; total terminal reward -41.9145453 includes the potential/cost terms.
- Phase decisions P01–P13: `[1,191,4,1,450,0,0,0,0,0,0,0,0]`.

The first unfinished task is **FL actual placement/capture**, despite genuine lift and crossing. There is no P06 layer in any decision and zero new-tail activations. Consequently there is no P06–P09 or later-stage evaluation evidence in this trajectory; the new tail cannot be assigned as the cause of this P05 outcome.

## Physical history and stage chronology

| Event | Tick | Time (s) | Meaning |
| --- | ---: | ---: | --- |
| P01→P02 | 8 | .066667 | Normal task transition |
| FR qualified | 48 | .400000 | Hard measured lift |
| P02→P03 | 1536 | 12.800000 | Front-right approach task complete |
| FR crossed / placed | 1546 / 1561 | 12.883333 / 13.008333 | Both real hard events |
| P03→P04 | 1568 | 13.066667 | FR placed history confirmed |
| P04→P05 | 1576 | 13.133333 | FL still ground-contacting |
| Last FL ground sample | 1587 | 13.225000 | Ground active, obstacle inactive |
| First uninterrupted FL AIR sample | 1588 | 13.233333 | Neither contact pair active thereafter |
| FL initial clearance | 1598 | 13.316667 | Gain 3.107831 mm; **not** placement |
| FL qualified | 1652 | 13.766667 | Gain 49.419917 mm; current clearance +.118708 mm above top |
| FL crossed | 2690 | 22.416667 | AIR, front +1.115206 mm, clearance +68.132435 mm |
| P05 terminal | 5176 | 43.133333 | Qualified/crossed true, placed false |

These event timestamps are consistent with this run's full raw observations. For example, FL front changes from -.905391 mm at tick 2689 to +1.115206 mm at 2690 while its wheel remains clearly airborne. Neither rear leg has a qualified/crossed/placed event.

## No FL touchdown after the real crossing

The complete raw file contains 5,177 observations including reset. Within the P05 interval, FL has 12 ground-active samples (1576–1587), **zero obstacle-active samples and zero obstacle force**. Ticks **1588–5176** are an uninterrupted AIR suffix of 3,589 samples. All 450 P05 decision-end snapshots have zero consecutive TOP samples.

| Post-cross sample | Tick / time | FL front (mm) | FL bottom above obstacle top (mm) | Exact pair status |
| --- | --- | ---: | ---: | --- |
| Maximum height | 2714 / 22.616667 s | +33.183697 | +81.746568 | AIR, obstacle inactive, force 0 |
| Furthest advance | 2722 / 22.683333 s | +36.417667 | +76.655393 | AIR, obstacle inactive, force 0 |
| Closest recorded bottom to top | **2774 / 23.116667 s** | **+9.146396** | **+2.871861** | AIR, obstacle inactive, force vector `[0,0,0]` N |
| Terminal | 5176 / 43.133333 s | +11.367742 | +5.506163 | AIR, obstacle inactive, force vector `[0,0,0]` N |

Post-cross front remains positive; its minimum is +1.115206 mm at the crossing. The closest recorded top clearance is still positive. The inactive pair contains a contact-point field at tick 2774 (`z=.0533651784 m`), but zero force and inactive status mean that point's existence is **not** support or touchdown evidence. The cached verified collider-extent wheel geometry is a measurement representation, not a reason to infer a sensor bug or to relabel AIR as TOP.

`placed_FL` reaches **.85**, first at decision 344 / tick 2752, and equals .85 in 304 of the 450 P05 decision-end snapshots. The unchanged predicate accounts for this exactly: `.35 × qualified + .35 × crossed + .30 × (.5 × top_geometry + .5 × TOP_sequence_fraction)`. With genuine lift/cross and current top geometry but zero TOP samples, the result is `.35+.35+.15=.85`. It is neither a placement event nor “85% complete-task success.”

## Current contact/support is not historical completion

| Leg at terminal | Contact | Front (mm) | Clearance (mm) | Load fraction | Placed history |
| --- | --- | ---: | ---: | ---: | --- |
| FL | AIR | +11.367742 | +5.506163 | 0 | false |
| FR | TOP | +110.507015 | -.104913 | .430306 | true |
| RL | GROUND | -569.116392 | -50.299836 | .501108 | false |
| RR | GROUND | -549.106298 | -50.619068 | .068586 | false |

There are three current supports. FL is within the measured top XY region and its top geometry is valid, yet it has no force/load or TOP contact. Its AIR state is not itself an automatic failure; the unfinished requirement is actual FL capture before its deadline. Current rear positions are not evidence of a failed P06 task, because P06 never began.

## Native control and safety evidence

At terminal, the logical nominal remains `[22.8,-13.4,0,45.9,6.9,0,0,0,0,0,0,0]`. Residual wheel targets in FL/FR/RL/RR order are `[-.004202996,-.048674811,-.020779896,+.055455843]` rad/s; actual native float32 wheel targets are `[+.004202996,-.048674811,+.020779897,+.055455845]` after the existing physical sign mapping. FL measured logical wheel velocity is -.004485742 rad/s. PPO/native control is therefore not absent merely because nominal wheels have stopped; these values are not evidence that a particular residual caused the missing contact.

All **5,176** ticks have verified actual native effects; **5,172** have own-phase request effects, excluding four ordinary incoming handoff-first ticks. All decisions verify zero in-episode root-pose, root-velocity, force/impulse and gravity writes. Every raw observation is finite. Across all 5,177 raw rows, body collision detected, exact-body-pair active and persistent flags are each false. No collision-pair positive claim or hidden physical failure is needed to explain this valid deadline outcome.

## Conclusion boundary

The completed fixed-mean evaluation demonstrates real FR placement, real FL lift/cross, and persistent missing FL touchdown—not full-task success. It supplies **no sample of the new P06 tail**. The fact that stochastic training episodes sometimes captured FL does not replace this deterministic result or make a paired stability/cause comparison. Preserve the existing contact and success semantics; this report selects no parameter change, detector adjustment or new training gate.

Sources: finalized `evaluation_manifest.json` and `run_manifest.json`; all decision audits; complete 120 Hz `physical_observations.jsonl`; recorded stage/history events. Only bounded PowerShell reads and this new report write were performed. No Python, new simulation, production edits, repeated hashing or historical artifact changes.
