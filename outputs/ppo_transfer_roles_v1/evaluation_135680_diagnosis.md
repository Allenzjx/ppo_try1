# Natural-P01 C / checkpoint 135680

Run `20260908T0701082482598Z_gd7479d9fc41c_da613885bce34f5f8d1a9b7deba6459b`; runtime `d7479d9fc41c`; deterministic N1, seed 2001, no prefix, optimizer 0.

**592 decisions / 4736 ticks / 39.466667s; task success=False.** Reason `FALL`; window ended=False; physical valid=True, physical reason=SAFETY_ABORT. Class `independent_safety_abort`; independent safety abort=True.

First unfinished: **P06** / EXECUTION, age 15.9333s. Current completion values: `{"rear_approach": 0.0}`. Pending labels are not success.

| Leg | historical Q/C/P ticks | current contact / support / load | front / clearance (m) |
|---|---|---|---|
|FL|2012 / 2316 / 2823|GROUND / True / 0.43877|-0.0681163 / -0.0490167|
|FR|53 / 1800 / 1849|AIR / False / 0|-0.0496954 / -0.0326279|
|RL|unavailable / unavailable / unavailable|GROUND / True / 0.0642702|-0.721292 / -0.049744|
|RR|unavailable / unavailable / unavailable|GROUND / True / 0.496959|-0.746961 / -0.0503842|

P01–P13 decision counts: **2 / 222 / 8 / 7 / 114 / 239 / 0 / 0 / 0 / 0 / 0 / 0 / 0**.

Global recorded stability: roll_rms_rad=0.132889; pitch_rms_rad=0.138036; angular_acceleration_rms_rad_s2=6.29202. All-phase quality score=unavailable; unsampled phases/windows remain null.

Final-decision native recorded verified=True; state-write verification=True; bootstrap=False. Terminal raw finite=True; collision record=`{"detected": false, "real_pair_active": false, "persistent": false, "geometry_penetration_m": 0.0, "reason": "no exact base_link/obstacle contact"}`.

Scope: manifest accounting and trajectory head/tail, not a full stream re-audit. Historical placement is not current load; an unfinished external window is not task success. No causal or stability-superiority claim.
