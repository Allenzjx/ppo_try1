# Natural-P01 C / checkpoint 136192

Run `20260908T0747015601267Z_gd7479d9fc41c_2eaa941373dd4eeaa572d4805d297880`; runtime `d7479d9fc41c`; deterministic N1, seed 2001, no prefix, optimizer 0.

**895 decisions / 7160 ticks / 59.666667s; task success=False.** Reason `INCOMPLETE_CONTROLLER_BLOCKED`; window ended=False; physical valid=True, physical reason=None. Class `incompletion`; independent safety abort=False.

First unfinished: **P06** / EXECUTION, age 40s. Current completion values: `{"rear_approach": 0.3353188817911007}`. Pending labels are not success.

| Leg | historical Q/C/P ticks | current contact / support / load | front / clearance (m) |
|---|---|---|---|
|FL|1793 / 1969 / 2357|OBSTACLE / True / 0.380411|0.287853 / 0.000525018|
|FR|54 / 1536 / 1580|OBSTACLE / True / 0.0857819|0.300837 / -0.000569166|
|RL|unavailable / unavailable / unavailable|GROUND / True / 0.126969|-0.352965 / -0.0500149|
|RR|unavailable / unavailable / unavailable|GROUND / True / 0.406838|-0.38617 / -0.0500689|

P01–P13 decision counts: **2 / 189 / 7 / 14 / 83 / 600 / 0 / 0 / 0 / 0 / 0 / 0 / 0**.

Global recorded stability: roll_rms_rad=0.104372; pitch_rms_rad=0.108379; angular_acceleration_rms_rad_s2=5.69353. All-phase quality score=unavailable; unsampled phases/windows remain null.

Final-decision native recorded verified=True; state-write verification=True; bootstrap=False. Terminal raw finite=True; collision record=`{"detected": false, "real_pair_active": false, "persistent": false, "geometry_penetration_m": 0.0, "reason": "no exact base_link/obstacle contact"}`.

Scope: manifest accounting and trajectory head/tail, not a full stream re-audit. Historical placement is not current load; an unfinished external window is not task success. No causal or stability-superiority claim.
