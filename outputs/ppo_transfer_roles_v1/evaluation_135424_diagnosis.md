# Natural-P01 C / checkpoint 135424

Run `20260908T0539048361640Z_gd7479d9fc41c_f892663561a243ae8f8b18e31e48f320`; runtime `d7479d9fc41c`; deterministic N1, seed 2001, no prefix, optimizer 0.

**227 decisions / 1816 ticks / 15.133333s; task success=False.** Reason `INCOMPLETE_CONTROLLER_BLOCKED`; window ended=False; physical valid=True, physical reason=None. Class `incompletion`; independent safety abort=False.

First unfinished: **P02** / CAPTURE, age 15s. Current completion values: `{"lifted_FR": 1.0, "clear_FR": 1.0, "approach_FR": 0.8197293608598568}`. Pending labels are not success.

| Leg | historical Q/C/P ticks | current contact / support / load | front / clearance (m) |
|---|---|---|---|
|FL|unavailable / unavailable / unavailable|GROUND / True / 0.389899|-0.146591 / -0.0489351|
|FR|53 / unavailable / unavailable|AIR / False / 0|-0.0500677 / 0.0643111|
|RL|unavailable / unavailable / unavailable|GROUND / True / 0.107614|-0.811008 / -0.0491897|
|RR|unavailable / unavailable / unavailable|GROUND / True / 0.502487|-0.62534 / -0.0502487|

P01–P13 decision counts: **2 / 225 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0**.

Global recorded stability: roll_rms_rad=0.174334; pitch_rms_rad=0.162984; angular_acceleration_rms_rad_s2=5.39279. All-phase quality score=unavailable; unsampled phases/windows remain null.

Final-decision native recorded verified=True; state-write verification=True; bootstrap=False. Terminal raw finite=True; collision record=`{"detected": false, "real_pair_active": false, "persistent": false, "geometry_penetration_m": 0.0, "reason": "no exact base_link/obstacle contact"}`.

Scope: manifest accounting and trajectory head/tail, not a full stream re-audit. Historical placement is not current load; an unfinished external window is not task success. No causal or stability-superiority claim.
