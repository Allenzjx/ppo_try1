# First-episode P06→P09 continuity — fixed completed window

2026-09-06; read-only PowerShell inspection. Runtime `4d268fc547b7`.
Run: `runs/ppo_semantic_v3/train/20260906T1308472858273Z_g4d268fc547b7_dcefe730da37448599db863a5f261d47`.
Sources: `optimizer_updates.jsonl` and the first 640 complete rows of `residual_and_projection_audit.jsonl` only. No Python, Isaac launch, production edit or commit was performed for this report.

## Fixed boundary and scope

The optimizer log contains completed updates **377–381**, ending at global **53248**. Each reports 20 optimizer steps and changed actor parameters: this report covers **640 credited decisions / 5 updates / 100 optimizer steps**, global **52609–53248**. It does not count the still-running block's planned remainder.

All 640 records belong to prefix attempt 0, with the same credit start: requested/actual **P06**, tick **3584**, time **29.866667 s**, `teacher_initialized_suffix`, `from_P01_current_policy=false`. `prefix_teacher_data_in_ppo_storage=false` in every row. The earlier FR/FL task history is genuine prefix history, not PPO credit or evidence of natural full-P01 success by this policy.

| Issued-policy phase | Decisions in this fixed window | Global decisions | Decision-end ticks |
|---|---:|---|---|
| P06 | 313 | 52609–52921 | 3592–6088 |
| P07 | 1 | 52922 | 6096 |
| P08 | 2 | 52923–52924 | 6104–6112 |
| P09 | 324 | 52925–53248 | 6120–8704 |

Every decision executes 8 physics ticks. Episode ticks **3585–8704** and command ticks **3764–8883** are both consecutive, without gaps; their offset is 179 ticks, not a reset. There are **5120/5120 verified native-audit ticks**, 5120 actual native-effect ticks, and **5117 own-phase-request-effect ticks**; the other three are the incoming handoff ticks below. All root-pose/root-velocity/force-or-impulse/gravity write counters remain zero.

## Ordinary transitions are not episode terminals

| Transition event | Global decision ending there | Tick / time | Recorded completion of the outgoing phase |
|---|---:|---|---|
| P06→P07 | 52921 | 6088 / 50.733333 s | `rear_approach=1` |
| P07→P08 | 52922 | 6096 / 50.800000 s | RR workspace=1, RL workspace=1, other support for RR=1 |
| P08→P09 | 52924 | 6112 / 50.933333 s | RR workspace=1, RR load-ready=1 |

At all three events, and throughout all 640 rows: **terminal=false, termination_reason=null, time_outs=false, task_success=false**. `terminal_bootstrap_allowed=true` remains set. Production `SemanticEpisodeEnv.step` derives done only from the task/safety terminal reason; the RSL adapter returns `done=step.terminated` and resets only in that branch (`semantic_env.py:191–242`, `semantic_training.py:189–206`). Prefix wrapping does not turn phase transitions into done (`semantic_prefix.py:389–432`). Thus these transitions preserve the continuing return/value chain; they are not three successful episodes.

P07 lasts **0.066667 s** and P08 **0.133333 s**. This is explained by measured completion, not by an unconditional skip or a required playback duration. `TaskStageSupervisor` checks current goals on tick multiples of 8 and permits at most one phase advance per observation (`semantic_supervisor.py:603–624`). At P06 exit, both rear wheels are already in workspace: RR front **−211.255 mm**, RL **−219.563 mm**; both remain there through P07. FR and RL supply the required two *other* supports for RR.

P08 does not simply pass on entry: at tick 6104, RR load fraction rises to **0.249823**, so load-ready is **0.937721**, below completion. At 6112 it falls to **0.056409**, giving load-ready=1. The existing formula is `(1−load)/(1−0.20)`, clipped to [0,1], provided two other supports exist (`semantic_supervisor.py:531–536`, v3 task spec support section). No fixed FL load or historical pose is required.

## One tick of handoff, then the new policy request acts

| Issued decision | Incoming hold tick | Own-request effect ticks | End residual wheels FL/FR/RL/RR, rad/s |
|---|---:|---:|---|
| 52922 / P07 | 6089 | 7/8 | +0.057880 / −0.192651 / −0.125424 / +0.131851 |
| 52923 / P08 | 6097 | 7/8 | +0.106040 / −0.251465 / −0.182349 / +0.026851 |
| 52924 / P08 | none | 8/8 | −0.013960 / −0.217773 / −0.062349 / +0.132923 |
| 52925 / first P09 | 6113 | 7/8 | +0.091040 / −0.168437 / −0.087170 / +0.117523 |
| 52926 / second P09 | none | 8/8 | +0.052492 / −0.091745 / −0.049568 / +0.092196 |

The three bridge records retain the prior full12 residual exactly, with no forbidden-channel drop, phase-cap clipping or hard-safety modification. Maximum applied-action handoff jumps are **8.88e−16 degrees** for servos and **2.78e−17 rad/s** for wheels—floating-point roundoff, not a reset to zero. The bridge substitutes a holding raw action only when detecting that first phase-change tick (`phase_action_masks_v2.py:343–372`).

Two directional checks additionally show that the rest of each short decision is not merely old carried residual:

- P07 FR wheel: prior raw maps to residual **−0.087651**. The new raw **−0.404605** maps to **−0.230329** before rate limiting; the actual residual reaches **−0.192651**, exactly 7 × 0.015 rad/s below its prior value. Holding the old request would not command that reduction.
- First P08 RR wheel: prior request maps to **+0.149323**; new raw **−0.046880** maps to **−0.028108**. Its residual changes from **+0.131851 to +0.026851**, again the seven-tick downward slew. This direction contradicts continuing the old positive request.

These use the existing wheel cap 0.6 and slew 1.8 rad/s² / 120 Hz. Native-effect records bind the same raw request/source phase to actual float32 dispatch, exclude the hold tick, and compare against the same-state zero-policy counterfactual. They do **not** constitute a physical alternate-trajectory replay or quantify independent causal benefit of each joint/channel.

## Wheel targets, mapper/filter continuity, and current FL load

The four nominal wheel targets decline continuously at the sampled ends:
tick 6088 **0.279915**, 6096 **0.253499**, 6104 **0.236998**, 6112 **0.155076**, 6120 **0.031121**, 6128 **0** rad/s. This is not an all-wheel zero imposed at each phase boundary: the recorded actual final wheel targets remain, respectively,

| Tick | Actual final canonical wheel commands FL/FR/RL/RR, rad/s |
|---:|---|
| 6088 | +0.306877 / +0.192264 / +0.157409 / +0.306766 |
| 6096 | +0.311379 / +0.060848 / +0.128075 / +0.385350 |
| 6104 | +0.343038 / −0.014467 / +0.054649 / +0.263849 |
| 6112 | +0.141116 / −0.062696 / +0.092727 / +0.288000 |
| 6120 | +0.122161 / −0.137317 / −0.056049 / +0.148644 |
| 6128 | +0.052492 / −0.091745 / −0.049568 / +0.092196 |

Even when nominal reaches zero, the residual continues to affect native commands. In the six sampled final-tick servo ACKs, the largest previous-final→new-final changes are **0.5, 1.25, 1.15, 0.7, 0.5, 1.25 degrees**, consistent with the existing final slew. These are command targets, not measured joint/wheel velocities.

Implementation evidence is also continuous: the same backend adapter/reader is reused by ordinary `step_physics`; the bridge and observation builder are reset by episode reset, not phase change; `_history` updates each physics tick rather than being zeroed; Euler/body-omega finite differences retain their preceding timestamp/state (`semantic_env.py:112–124,150–182`; `semantic_observation.py:161–186`). **Limitation:** this JSONL stores per-tick audit summaries plus decision-end native fields, not every mapper compensation/feedback state or sensor-filter history buffer. Therefore it supports no observed reset and continuous dispatched targets, but not an independent bitwise proof of every hidden mapper/filter state. Tracking compensation may legitimately evolve; continuity does not mean constant targets or constant internal values.

FL illustrates why task history must remain separate from current load. Its history keeps qualified/crossed/placed=true and the original events Q2461/C3115/P3583 across all three handoffs. At ticks 6088/6096/6104/6112/6120/6128, however, FL is **AIR=true, support=false, load_fraction=0**, with clearance rising from **+32.349 mm at 6088 to +50.055 mm at 6112**. Its AIR counter continues 532→540→548→556 samples instead of restarting at phase change. The current support set is FR/RL/RR through 6112, then FR/RL at 6120–6128. Historical FL placement was not counted as present FL support.

This fixed window ends in **P09 at 72.533333 s**, without terminal or task success. It demonstrates cross-phase control/data continuity during a teacher-initialized suffix, not full-task success, improved stability, or a paired causal performance improvement from nominal geometry projection.
