# HEAD a546 sealed RR-window audit

Run: `runs/ppo_rr_capture_then_rl_transfer_v1/validation/20260923T0310001450430Z_ga54678ceef58_c5e1fa8c5a5f42089a96c3cf709bf324`

Status: bounded read-only audit in progress. This short note freezes the already checked core evidence while the next Isaac run owns the machine. No video exists for this headless run and none is synthesized.

## Sampling boundary

Values below marked **decision-grid** come from `residual_and_projection_audit.jsonl`: one completed policy decision per 8 physics ticks. Its assist context describes `source_observation_tick`, normally one episode tick before the decision endpoint. Stage-event ticks come independently from sealed `stage_transition_evidence.jsonl`. They must not be read as a complete per-tick trace. A later bounded pass will add exact native/physical rows.

## Core sealed facts

- Exact stage history: FL first cross tick `3544`, FL first placement tick `6150`, RR qualified tick `8553`, RR cross tick `9520`.
- P09 late nominal group starts at source-observation tick `9520`, the same exact tick recorded for RR crossing.
- Decision tick `9512` (`t=79.266667s`, context observation `9511`): RR is not yet within top XY; front distance `−5.27 mm`, gap `27.93 mm`, knee final/actual approximately `−45.37° / −45.20°`.
- Decision tick `9520` (`t=79.333333s`, context observation `9519`): current XY is true, front distance `+0.57 mm`, gap `28.383 mm`; the pre-context has not yet committed the cross event. RR knee final has already changed to `−51.611°` while measured actual is `−46.111°`. This change precedes assist anchoring.
- Assist state records an exact window-start gap of `28.662031 mm`. First active decision sample is tick `9528` (`t=79.400000s`, context observation `9527`): `DESCEND`, gap `34.454 mm`, hip entry `11.964087°`, hip target `11.947420°`, hip actual `14.379465°`; knee hold/final `−51.611176°`, knee actual `−48.662809°`.
- Post-anchor decision-grid gap peak is `69.092285 mm` at context observation `9591` / decision tick `9592` (`t=79.933333s`).
- The hip target reaches `7.964087°` (exactly `−4°` from entry) at the first sampled terminal block: decision tick `9768`, `t=81.400000s`, reason `gap_not_improving`, gap `66.853639 mm`; measured hip/knee are `7.878615° / −51.397982°`.
- Later decision-grid minimum is `63.345063 mm` at context observation `12111` / decision tick `12112` (`t=100.933333s`); terminal sample is `63.571453 mm` at context observation `12483` / tick `12484` (`t=104.033333s`).

## Bounded interpretation

There is a real sampled peak-relative decrease of about `5.747 mm`, so a gate comparing only against the low `28.662 mm` entry baseline cannot express that local improvement. However, the gap never approaches or improves upon entry, no RR TOP contact/capture occurs, and the run ends `INCOMPLETE_CONTROLLER_BLOCKED`. This is not evidence that the motion was sufficient for capture.

The RR final knee target changed from roughly `−45°` toward `−51.611°` before the hip-only assist anchored, across the nearby geometry-exit/late handoff. Exact attribution awaits the per-tick layer audit: the sealed timing currently says context `9519` / decision `9520`, while the late source starts from observation `9520`, and the late RR pair may only be reissuing an existing value. Body/link motion and wheel commands also co-evolve. Therefore neither the knee change nor negative hip motion is assigned independent causal credit here. Exact per-tick geometry, initial `waiting_actual_tracking` interval, body/hip heights, wheel-center body/world coordinates, and four-wheel channel layers will be added only after the active Isaac video has ended.
