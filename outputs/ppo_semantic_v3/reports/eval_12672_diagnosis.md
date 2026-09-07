# C checkpoint 12672: P06–P09 RR diagnosis

## Outcome and scope

This **fresh P01**, deterministic C evaluation failed at **49.458333 s / tick 5935**, with `WHEEL_ONLY_CLIMB`. It is not a suffix-success or full-task-success result. RR had one `whole_body_initial_clearance` event, but **no qualified active lift, legitimate front crossing, or placement**. FR and FL placement histories were true; RR and RL were false.

- Runtime: `5036e11622ac2e9b9eca86a3991f18781fe155e0`.
- Checkpoint: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000012672.pt`.
- Recorded checkpoint SHA256: `2c05f08e5d5910506646137517351ae6df5f3e65c34b6d4297635bb00c6b7134`.
- Run: `runs/ppo_semantic_v3/validation/20260906T0349586939252Z_g5036e11622ac_86eb1a3b6506415c92de0cb07efdbe08`.
- Sources: that run's `physical_observations.jsonl` (120 Hz), `residual_and_projection_audit.jsonl` (decision-end nominal/residual/drive targets), `stage_transition_evidence.jsonl`, and `evaluation_manifest.json`. Read with PowerShell only; no simulation, Python, tests, or runtime edits.

## Physical sequence

Distances below are RR wheel center x minus the actual obstacle front x. Bottom height is the recorded live-collider wheel bottom in world coordinates; obstacle top is **50 mm**. Load fraction uses active wheel-pair normal-force magnitudes, matching the evaluator; it is not a vertical-load-only estimate.

| Time s / tick | Event | RR geometry and contact |
|---|---|---|
| 22.666667 / 2720 | P05→P06 | Center −528.079 mm; bottom −0.575 mm; GROUND; load 22.50%. |
| 43.666667 / 5240 | P06→P07 | Center −192.577 mm; bottom −0.117 mm; GROUND; load 7.05%. P06 lasted 21 s / 2520 ticks. |
| 43.733333 / 5248 | P07→P08 | Center −191.570 mm; bottom −0.162 mm; GROUND; load 5.72%. P07 lasted 8 ticks / 0.066667 s. |
| 43.800000 / 5256 | P08→P09 | Center −189.877 mm; bottom −0.291 mm; GROUND; load 8.43%. P08 lasted 8 ticks / 0.066667 s. |
| 44.275000–44.450000 / 5313–5334 | Longest RR AIR interval in this rear-task interval | 22 consecutive samples, preceded and followed by GROUND. Earlier AIR detections were brief and near ground height, not qualified lifts. |
| 44.350000 / 5322 | `whole_body_initial_clearance` | Measured upward excursion 3.073 mm; bottom 2.449 mm; center −210.923 mm; no ground/obstacle contact. This is initial evidence only. |
| 44.416667 / 5330 | Maximum AIR bottom height | **4.607 mm**, still **45.393 mm below the top**; center −220.745 mm; support count 2. RR actual hip/knee 46.119°/−6.846°; commands 52.740°/−6.846°. |
| 44.458333 / 5335 | Definite return to ground | Ground pair active, 13.746 N; bottom 1.861 mm; center −226.077 mm. RR actual hip/knee 48.797°/−6.947°. Initial qualification is cleared on ground return; there never was an above-top qualification to revoke. |
| 45.633333 / 5476 | First obstacle pair contact | GROUND_AND_OBSTACLE; center −49.783 mm; bottom −0.071 mm. Obstacle force vector approximately (−28.446, 0, 0) N; contact point x=0.520823 m, z=0.048955 m, versus front x=0.521312 m/top z=0.05 m. These measured point/force/geometry values support front-wall contact, not a free-air clearance. |
| 45.850000–49.458333 / 5502–5935 | Continuous obstacle contact | 434 consecutive OBSTACLE samples, no AIR. Wheel bottom rises from approximately −0.105 mm to 48.843 mm as its center moves from −49.772 mm to +0.376 mm. |
| 49.458333 / 5935 | Center crosses front; terminal | Contact point has reached near-top geometry, but RR active-lift history remains false. Evaluator correctly does not convert current top contact into a successful active crossing. |

P07 completed on current RR/RL workspace plus available other supports. P08 completed on current RR workspace and load readiness. Their short duration is **not itself proof of a skipped motor program**: continuous nominal inheritance must be checked independently, below. Neither completion claimed RR lift or placement.

## Nominal actions were issued and continued after phase labels changed

These times are **15 Hz decision-end samples**; an onset between two samples is bounded to at most one 0.066667 s interval. Angles are logical command degrees; wheel values are rad/s. Actual joint/drive data above are separate from these suggestions.

| Channel | Recorded nominal sequence after P06 |
|---|---|
| FL hip | 22.8° at 43.6667; **31.55° in P07** at 43.7333; **37.6° in P08** at 43.8; continues in P09 to **49.2° by 44.0**. Holds to 45.0667, returns to 38.6° by 45.4, holds to 49.2, then finite tail returns toward home (−0.15° at terminal). |
| FL knee | −13.4° through 49.2; finite tail reaches −31.4° by 49.3333. |
| RL hip | 6.9° through P07; **14.3° at P08 end**; continues in P09 to **31.2° by 44.2**, holds through 49.2; tail reaches 15.4° by 49.3333. |
| RL knee | 0° through 49.2; tail reaches 19.4° by 49.3333. |
| RR hip | 0° at P09 entry; 1.6° at 43.8667; rises to **55.6° by 44.4667**; holds through 44.9333; decreases to **−6.9° by 45.7333** and holds to terminal. |
| RR knee | Nominal remains **0° through 44.4667**; −4.9° at 44.5333, then **−37.8° by 44.9333**, held to terminal. The old knee-decrease dispatch pause is not present in this trace. |
| FR wheel | Inherited +0.3 through 44.0; +0.1 at 44.0667, crosses negative, reaches **−0.63 by 44.3333**, holds through 45.0; returns to 0 by 45.2667, then +0.3 by 45.8667. Tail returns to 0 by 49.2667. |
| FL/RL/RR wheels | Inherited **+0.3 through 49.1333**, including the failed initial-lift attempt and subsequent wall contact. Tail takes RL/RR to 0 by 49.2667; FL reaches −0.675 at terminal. |

The PPO residual was physically relevant, not zero: at P06 exit RR hip/knee residuals were about **−1.920°/−7.161°**. During the initial AIR attempt, nominal RR knee was 0° while the actual target/joint remained about −6.8°. By 45.0, nominal RR knee was −37.8°, residual −6.108°, recorded final drive target −45.158°, and actual joint −41.025°.

The **44.4583 s ground return precedes even the 44.4667 s sample whose nominal knee remains 0°**. Thus this failed first lift cannot be attributed to the later knee-decrease sequence starting too early, or to that sequence being suppressed by the removed pause. The RR hip genuinely moved tens of degrees, but the whole-body configuration produced only 4.607 mm of free-air wheel-bottom height.

Source inspection agrees with the trace: `NominalMotionProvider._continuous_advisory` advances each finite layer every tick, preserves its changed channels across ordinary phase changes, and does not wait for front arrival or a joint-sign condition before dispatch. This bounded review found **no remaining explicit missing-dispatch/stalled-layer bug in this attempt**. It does not prove every nominal suggestion is optimal, nor isolate the effects of continuous entry state, whole-body support motion, existing residual offsets, and tracking lag. An A trajectory's unique pose or timing is not a justified new hard gate.

## Next training block

**Prioritize P06 continuation** for the current failure: this exposes the RR precursor state, approach, unloading, real lift attempt, sustained clearance, and carry/placement in one continuous trajectory. A larger P06 block can collect useful failures without treating initial clearance as success. This is a training choice, not a new pretraining gate or proof that 4096 decisions will solve the task.

**Use P10 separately for RL coverage**, with honest teacher-prefix provenance. Its valid start already requires RR placement, so it bypasses the RR mechanism that failed here. P10 suffix improvement cannot be reported as repair of this full-P01 RR failure or as full-task success. No reward, observation, safety, dynamics, or success criteria changes are implied by this diagnosis.
