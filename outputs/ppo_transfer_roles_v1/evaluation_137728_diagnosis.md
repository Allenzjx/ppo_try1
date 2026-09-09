# Natural-P01 C / checkpoint 137728

Run `20260908T0838185133094Z_gd7479d9fc41c_8765f17661b741d4b8dd4e0310636337`; runtime `d7479d9fc41c`; deterministic N1, seed 2001, no prefix, optimizer 0.

**683 decisions / 5464 ticks / 45.533333s; task success=False.** Reason `INCOMPLETE_CONTROLLER_BLOCKED`; window ended=False; physical valid=True, physical reason=None. Class `incompletion`; independent safety abort=False.

First unfinished: **P05** / EXECUTION, age 30s. Current completion values: `{"placed_FL": 0.09262439310854055}`. Pending labels are not success.

| Leg | historical Q/C/P ticks | current contact / support / load | front / clearance (m) |
|---|---|---|---|
|FL|unavailable / unavailable / unavailable|GROUND / True / 0.441729|-0.147072 / -0.049641|
|FR|55 / 1761 / 1819|GROUND_AND_OBSTACLE / True / 0.0482833|-0.0511056 / -0.0502369|
|RL|unavailable / unavailable / unavailable|GROUND / True / 0.0330758|-0.770773 / -0.0496061|
|RR|unavailable / unavailable / unavailable|GROUND / True / 0.476911|-0.727935 / -0.0503172|

P01–P13 decision counts: **2 / 217 / 9 / 5 / 450 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0**.

Global recorded stability: roll_rms_rad=0.101436; pitch_rms_rad=0.135619; angular_acceleration_rms_rad_s2=6.55945. All-phase quality score=unavailable; unsampled phases/windows remain null.

Final-decision native recorded verified=True; state-write verification=True; bootstrap=False. Terminal raw finite=True; collision record=`{"detected": false, "real_pair_active": false, "persistent": false, "geometry_penetration_m": 0.0, "reason": "no exact base_link/obstacle contact"}`.

Scope: manifest accounting and trajectory head/tail, not a full stream re-audit. Historical placement is not current load; an unfinished external window is not task success. No causal or stability-superiority claim.
