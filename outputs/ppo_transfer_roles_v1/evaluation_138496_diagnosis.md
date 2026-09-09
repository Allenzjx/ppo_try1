# Natural-P01 C / checkpoint 138496

Run `20260908T0953095796695Z_gd7479d9fc41c_708d4bd483f4419382addb1c487cc75e`; runtime `d7479d9fc41c`; deterministic N1, seed 2001, no prefix, optimizer 0.

**227 decisions / 1816 ticks / 15.133333s; task success=False.** Reason `INCOMPLETE_CONTROLLER_BLOCKED`; window ended=False; physical valid=True, physical reason=None. Class `incompletion`; independent safety abort=False.

First unfinished: **P02** / CAPTURE, age 15s. Current completion values: `{"lifted_FR": 1.0, "clear_FR": 1.0, "approach_FR": 0.9964065878801205}`. Pending labels are not success.

| Leg | historical Q/C/P ticks | current contact / support / load | front / clearance (m) |
|---|---|---|---|
|FL|unavailable / unavailable / unavailable|GROUND / True / 0.387753|-0.0945449 / -0.0504703|
|FR|56 / unavailable / unavailable|AIR / False / 0|-0.00589835 / 0.0243935|
|RL|unavailable / unavailable / unavailable|GROUND / True / 0.118627|-0.764697 / -0.050125|
|RR|unavailable / unavailable / unavailable|GROUND / True / 0.49362|-0.567092 / -0.0499496|

P01–P13 decision counts: **2 / 225 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0 / 0**.

Global recorded stability: roll_rms_rad=0.160576; pitch_rms_rad=0.148165; angular_acceleration_rms_rad_s2=5.49827. All-phase quality score=unavailable; unsampled phases/windows remain null.

Final-decision native recorded verified=True; state-write verification=True; bootstrap=False. Terminal raw finite=True; collision record=`{"detected": false, "real_pair_active": false, "persistent": false, "geometry_penetration_m": 0.0, "reason": "no exact base_link/obstacle contact"}`.

Scope: manifest accounting and trajectory head/tail, not a full stream re-audit. Historical placement is not current load; an unfinished external window is not task success. No causal or stability-superiority claim.
