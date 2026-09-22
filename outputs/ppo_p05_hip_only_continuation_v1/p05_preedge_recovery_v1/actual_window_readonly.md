# Real P05 window: candidate predicate only

The new gate would **first qualify at episode tick 6000, simulation 50.000 s, P05 age 30.000 s** on the existing CP216448 deterministic trace. The corresponding new nominal could first enter the next dispatch, episode tick 6001. No candidate control was applied.

Fixed interval: ticks 5992–6480 (489 actual observations), with neighboring capture rows 5991/6481. P05 entry is recorded at tick 2400 / 20.000 s. The 62 saved full task snapshots at eight-tick endpoints exactly match the predicate evaluated from per-physics evidence.

| Check | Passed / total | Rejection |
|---|---:|---|
| Absolute stage-age [30,40) | 481 / 489 | Only ticks 5992–5999, before age 30 |
| Prior endpoint + next consecutive source tick | 489 / 489 | None |
| No fresh authored wheel owner | 489 / 489 | None |
| All ten other validity/history/geometry/support checks | Each 489 / 489 | None |

All 481 ticks from 6000–6480 qualify. Current FL is AIR, its conservative gap is 7.415–8.522 mm, front distance is −34.291 to −33.873 mm, and FR/RL/RR are verified other supports. No cross or placement is invented.

Why the source gate does not permanently reject after the endpoint:

- Real capture/dispatch context identifies every consecutive source observation tick, and both endpoint-issued and previous-endpoint-dispatched flags are true throughout this interval.
- Production `_continuous_advisory` calls `_sequence_permission` and sets `advanced_this_tick=True` for P01–P05 each physical tick, including after their endpoints. It replaces the sample with a fresh `MotionExecutor.tick()` result.
- `MotionExecutor` increments its source tick after the endpoint. Atomic groups are emitted only when the current source tick equals their authored tick, so the old stop does not remain a fresh event forever. Reconstructing only these clocks from the recorded phase-entry ticks gives P05 source ticks 3592–4080 and **zero fresh wheel groups for all five existing layers**. No new P06 layer is needed.

Limit: mutable source-layer objects/`advanced_this_tick` are not directly serialized. Their conclusion uses unchanged source code and exact phase clocks, not fabricated live flags. Missing per-physics task fields use the next dispatch context with an exact matching source tick; evidence status follows the existing current-load-validity formula, and all fields were cross-checked at the 62 complete task endpoints. The candidate itself still labels its endpoint evidence as **not independently ACK verified**.

This confirms the initial scheduling opportunity, not the subsequent physical effect. Once new wheel advice actually changes motion, eligibility may change. Production, runtime/configs, optimizer and video were untouched.
