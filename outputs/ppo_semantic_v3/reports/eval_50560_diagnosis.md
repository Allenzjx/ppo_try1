# C50,560 deterministic natural-P01 evaluation — P09 incomplete

Status: finalized execution `SUCCEEDED`; **task/controller success=false**. This is a physical-task incompletion, not a new successful checkpoint or paired improvement.

## Bound run and outcome

Run: `runs/ppo_semantic_v3/validation/20260906T1200142665931Z_g9d70aae58243_bb0d667703da4eca80a5c99292560192`.

Runtime `9d70aae58243b9fb2248d64561959e6a67901b44`, v3/N1/seed2001, `semantic_residual_eval`, natural P01, saved heteroscedastic deterministic mean. Source is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000050560.pt`, SHA-256 `6ccf5b26784e4db7056b1ba6521d0ac34c3c49cf915ff618d4410c9f14d7fd99`. There is no teacher prefix or optimizer update during this evaluation.

Final manifests record **1,166 decisions /9,328 physics ticks /77.73333333333333s**, `INCOMPLETE_CONTROLLER_BLOCKED` in P09. Shared physical evaluation is `valid=true`, `success=false`, `termination_reason=null`, with an empty physical-failure reason. `window_ended_before_task_terminal=false`: this is the actual P09 deadline, not a truncated recorder window. No wheel-only-climb, body-collision or interface failure is recorded. Absence of a physical safety failure is not task completion.

Decision counts P01–P13 are **1,186,4,1,150,372,1,1,450,0,0,0,0**, summing to1,166. Training remains50,560/360/7,200; evaluation adds no optimization credit.

## P06 succeeds on measured workspace, not historical phase labels

`stage_transition_evidence.jsonl` records P05→P06 at tick2,736/22.8s, P06→P07 at **5,712/47.6s**, P07→P08 at5,720/47.666666667s and P08→P09 at5,728/47.733333333s. The P06 completion value is `rear_approach=1`; the following P07 completion explicitly has `workspace_RR=workspace_RL=support_RR=1`.

At the actual P06 completion tick:

| Current measurement | RR | RL |
|---|---:|---:|
| Wheel-center distance from front plane |−219.232258mm |−217.444714mm |
| Margin above unchanged workspace lower bound−220mm |0.767742mm |2.555286mm |
| Wheel-bottom clearance above obstacle top |−50.604212mm |−49.592892mm |
| Measured load fraction |0.08192161 |0.50251386 |

Both rear legs satisfy the configured workspace at this boundary; this does **not** mean either has lifted or crossed. FL is actually AIR/load0, clearance+9.126437mm, while FR carries0.41556453. Historical FR/FL placement is retained, but is not current FL support. P09 begins with RR front−216.935929mm, clearance−50.365920mm and load0.09478015; FL is AIR/load0 with clearance+32.708546mm.

## First unfinished task: corresponding RR lift, carry and placement

P09 lasts its full30s: ticks5,729–9,328,3,600 physical samples and450 decisions. Its completion target is actual `placed_RR`. RR has **no qualified lift, no front crossing and no placement**, and its center never reaches the front plane. The earlier prerequisite already missing within this task is above-top active-lift qualification, not a P10 entry or final-stop condition.

Three `whole_body_initial_clearance` events are real early motion evidence, not qualified lift:

| Event tick / time | RR front distance | RR clearance above top | Own RR joint motion / whole-body joint motion in recorded evidence window | Current FL load |
|---|---:|---:|---:|---:|
|6,198 /51.65s |−36.515173mm |−19.199467mm |0.672778° /6.290395° |0 |
|6,495 /54.125s |−52.713599mm |−46.238604mm |0.815346° /2.742422° |0 |
|6,604 /55.033333333s |−55.982735mm |−46.243994mm |2.967044° /8.623794° |0 |

RR is AIR/load0 at all three event ticks. The corresponding upward-excursion fields are10.341607,3.323026 and3.180224mm; the first event's command-motion evidence is25.825350°. These whole-body timing measurements are not proof that a particular motor caused the lift, and none asserts above-top qualification.

A single bounded PowerShell streaming pass over the **actual120Hz P09 raw observations** finds:

- Maximum RR AIR clearance: **−6.232580875mm** at tick6,312/52.6s, front−22.881266mm. Its wheel bottom is43.767419mm above ground, still below the50mm obstacle top. Current FL has genuine small contact load0.02521791 /0.717230856N, clearance+1.690877mm.
- Closest RR center to the front plane: **−22.421267201mm** at tick6,317/52.641666667s, clearance−6.651661mm, AIR/load0. This is194.514662mm forward of P09 entry, but still not a crossing. FL is then AIR/load0.
- RR is AIR for359 of3,600 ticks. FL is AIR for3,348 of3,600 ticks, including252 of those359 RR-AIR ticks. Thus FL support is usually absent but **not absent at every RR lift sample**; its brief real contact must not be replaced with a CoM proxy or an unsupported causal explanation.
- Final RR is GROUND, front−122.911480mm, clearance−49.311328mm, load0.05007624. RL is GROUND, front−280.198526mm, clearance−51.711866mm, load0.47864871. FL is AIR/load0, clearance+52.014609mm; FR is current TOP/load0.47127505. No rear Q/C/P has been earned.

The decisive geometric deficit is measured: RR never reaches above-top AIR clearance or the front plane. This report does not infer whether policy, nominal timing, support redistribution or action authority is its sole cause, and proposes no relaxed completion gate.

## Finite nominal retirement is not a video/codec failure

The15Hz command audit's last different full nominal is tick6,632. From decision-end tick6,640 through9,328, all337 logged nominal vectors are the same finite vector:

`[-18.5,-31.4,0,31.1,15.4,19.4,-6.9,-37.8, 0,0,0,0]`.

Only the four nominal wheels are zero; the eight servo suggestions remain nonzero/static. The P06 retirement diagnostic ends with peak_fraction1 and wheel_gain0. This is not an all12-channel zero command or a closed residual path. At the final tick the projected residual is approximately `[-.846,-6.414,-4.110,-16.566,-4.287,-.599,-5.003,-8.496, .107197,-.101350,-.125020,.054715]`; actual targets remain distinct, and the terminal native audit reports `verified=true`,12 changed target channels and no in-episode state writes. Native ACK tick9,507 includes the backend's initialization offset and is not substituted for episode tick9,328.

This evaluation is not a video encode attempt. Finite nominal retirement cannot be called a codec failure, and P09 incompletion must not be hidden by technical execution success. No new video, successful policy or paired stability improvement is credited.

## Evidence and scope

Read-only sources are this run's `evaluation_manifest.json`, finalized `run_manifest.json`, small `stage_transition_evidence.jsonl`, the bounded P09 window of `physical_observations.jsonl`, and selected15Hz `residual_and_projection_audit.jsonl` fields. Geometry is live collider wheel-bottom/center data; AIR/load uses actual verified ground/obstacle pairs. Reported loads sum positive active pair normal forces across the four wheels, matching the evaluator definition. No full log-probability audit, new rollout, Python process, runtime edit, old-run rewrite or commit was performed for this diagnosis. C46,464's different P06 incomplete and all prior A/B/C outcomes remain preserved, not retrospectively improved.
