# Natural-P01 C / checkpoint 133248

Run `20260908T0348320962319Z_gdb991aa68103_ef640fd945134f319e9affad5e47f0fe`; runtime `db991aa68103`; deterministic N1, seed 2001, no prefix, optimizer 0.

**685 decisions / 5480 ticks / 45.666667s; task success=False.** Reason `INCOMPLETE_CONTROLLER_BLOCKED`; window ended=False; physical valid=True, physical reason=None. Class `incompletion`; independent safety abort=False.

First unfinished: **P05** / CAPTURE, age 30s. Current completion values: `{"placed_FL": 0.85}`. Pending labels are not success.

| Leg | historical Q/C/P ticks | current contact / support / load | front / clearance (m) |
|---|---|---|---|
|FL|1971 / 2102 / unavailable|AIR / False / 0|0.116066 / 0.0140551|
|FR|55 / 1753 / 1784|OBSTACLE / True / 0.430575|0.180855 / -2.29637e-05|
|RL|unavailable / unavailable / unavailable|GROUND / True / 0.497885|-0.519683 / -0.049987|
|RR|unavailable / unavailable / unavailable|GROUND / True / 0.0715399|-0.501934 / -0.0502008|

P01–P13 decision counts: **2 / 216 / 5 / 12 / 450 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0**.

Global recorded stability: roll_rms_rad=0.130458; pitch_rms_rad=0.0966915; angular_acceleration_rms_rad_s2=5.23894. All-phase quality score=unavailable; unsampled phases/windows remain null.

Final-decision native recorded verified=True; state-write verification=True; bootstrap=False. Terminal raw finite=True; collision record=`{"detected": false, "real_pair_active": false, "persistent": false, "geometry_penetration_m": 0.0, "reason": "no exact base_link/obstacle contact"}`.

Scope: manifest accounting and trajectory head/tail, not a full stream re-audit. Historical placement is not current load; an unfinished external window is not task success. No causal or stability-superiority claim.
