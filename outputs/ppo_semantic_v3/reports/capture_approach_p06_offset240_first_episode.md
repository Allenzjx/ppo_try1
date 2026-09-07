# Capture-approach revision: first P06 offset-240 PPO episode

## Fixed completed window and correct credit

Run `runs/ppo_semantic_v3/train/20260906T1738509257629Z_gc34262abffc1_04bbf6558046407990f85358d738cc17`, runtime `c34262abffc16847ff32d15ecbf790dd60803e0a`, source checkpoint 72320, seed 1001. This report covers only completed episode 0, not the rest of the planned 2048 decisions.

- Requested P06 with 240 teacher-offset decisions; actual credit starts **P06 at tick 5504 / 45.866667 s**, requested phase still active, prefix attempt 0. This is a teacher-initialized suffix, not a natural current-policy P01 episode.
- **361 PPO decisions, global 72321–72681**, consume ticks 5505–8385: 360 complete eight-tick decisions plus a one-tick terminal interval, **2,881 credited physics ticks / 24.008333 s**.
- Physical duration including the prefix is **69.875 s**. Physical core count 1049 consists of **688 reset-only prefix decisions + 361 PPO decisions**. `prefix_teacher_data_in_ppo_storage=false`; the 688 are not optimizer credit.
- Completed optimizer boundary **533 / global 72704** covers this episode: 384 decisions / three updates after the 72320 origin, including 23 decisions of episode 1 which this report excludes.
- Terminal **P06 `INCOMPLETE_CONTROLLER_BLOCKED`**, stage age 40.008333 s. Physical evaluator valid true, physical termination reason null. Task and full-task success false; terminal bootstrap false, next potential zero, terminal event -40.

All 361 policy requests are P06. **No P06→P07, P07→P08 or P08→P09 transition occurs after credit begins.** The first unfinished task is current RL rear-workspace approach, not capture or a collision.

## Prefix history is not new PPO completion

The histories in the first and final PPO records have identical hard Q/C/P events:

| Leg | Qualified tick | Cross tick | Placed tick | Attribution |
| --- | ---: | ---: | ---: | --- |
| FR | 71 | 1665 | 1695 | Before credit; excluded prefix history |
| FL | 2461 | 3115 | 3583 | Before credit; excluded prefix history |
| RR | none | none | none | No completed hard rear event |
| RL | none | none | none | No completed hard rear event |

The measured history remains valid context, but the PPO episode must not be credited with the teacher's front-leg completion. The physical P06 stage clock also continues rather than restarting at the credit boundary: the first decision ends at tick 5512 with stage age 16.066667 s.

## RR really moves, but initial clearance is not qualification

The first retained decision-end RR AIR sample is tick **5568 / 46.4 s**, AIR count 2, clearance **-50.178098 mm**, initial=false. Its current uninterrupted AIR suffix therefore began at 5567; this does not rule out an earlier brief unretained contact gap between decision boundaries.

The first recorded PPO RR initial-clearance event is **5609 / 46.741667 s**: upward excursion **3.300529 mm**, RR own-joint motion 1.296553°, whole-body joint motion 8.221924° and command motion 26.065284°. In total, 20 RR initial events are logged after credit starts, with intervening current qualification resets/contact changes. These are genuine small clearance processes, not hard qualified lift or placement.

The highest **decision-end** RR clearance is tick **7296 / 60.8 s**, **-41.145268 mm** relative to the top, while AIR. No RR hard qualified event is recorded at any physics tick by the evaluator; there is consequently no rear crossing or placement. The log does not preserve every raw geometry sample, so the decision-end maximum must not be advertised as an independently reconstructed full-120-Hz height maximum.

RR is AIR in 150 and ground-contacting in 211 of the 361 decision-end records; RL remains ground-contacting in all 361. These are sample counts, not time-weighted contact occupancy. No body/wheel physical failure is recorded for this episode.

## Current workspace: RL remains the limiting leg

Existing workspace lower bound is **-220 mm**. The paired goal needs both rear legs currently eligible, not their individual best positions at unrelated instants.

| Decision-end sample | Tick / time | RR front (mm) | RL front (mm) | Rear approach |
| --- | --- | ---: | ---: | ---: |
| First PPO decision | 5512 / 45.9333 s | -254.887 | -273.181 | .787276 |
| RR closest to front | 6504 / 54.2000 s | **-102.922** | -233.356 | .946578 |
| RL best / joint goal best | **8032 / 66.9333 s** | -113.218 | **-222.931918** | **.988272** |
| Terminal | 8385 / 69.8750 s | -168.086 | -227.768984 | .968924 |

At its best retained sample, RL is **2.931918 mm behind** the unchanged lower workspace bound; terminal shortfall is **7.768984 mm**. RR has already entered the workspace, but RL prevents the paired completion at these boundaries. P06 never actually transitions, so it is incorrect to report P07 preparation or RR execution as covered. Without the raw 120 Hz geometry stream this report does not claim that no sub-decision transient ever approached the boundary.

## Nominal, residual, native controls and load distribution

Every credited decision's consumed nominal wheel vector is `[+.3,+.3,+.3,+.3]` rad/s. This is a continuing finite P06 suggestion: the newly constructed semantic P06 source's endpoint flag remains false, with retirement peak 0 / gain 1; no post-finite-source tail activation occurs before the preserved task deadline. There is no actual nominal-to-zero event to blame for this episode.

Means below are **decision-end command samples**, not wheel velocities or time-integrated motion. Wheel order is FL, FR, RL, RR:

- Mean projected residual: `[+.123702,-.308527,-.329202,-.050812]` rad/s.
- Mean actual logical drive target: `[+.423702,-.008527,-.029202,+.249188]` rad/s.
- At best RL sample 8032, residual `[+.178043,-.383347,-.324679,-.028184]` gives actual logical drive `[+.478043,-.083347,-.024679,+.271816]`.
- At terminal, residual `[+.177868,-.257709,-.407440,+.044069]` gives actual logical drive `[+.477868,+.042291,-.107440,+.344069]`.

Thus FR/RL policy residuals can offset or reverse their .3 nominal suggestion, while FL/RR commands remain positive on average. The actual rear geometry nevertheless advances from the initial state before retreating slightly from its best approach. This is a verified interaction between nominal and learned control—not proof that a particular wheel residual alone caused the deadline. No later P07/P09 channel owner or qualified carry override was exercised.

Historical FL placement does not mean current FL support. FL supports in only eight decision-end samples; its terminal AIR count is 2669, with clearance +44.466024 mm, front +322.297696 mm and load zero. At terminal FR is TOP/supporting (load .456670), RL GROUND/supporting (.518466), RR GROUND/supporting (.024864). At the best RL sample, RR and FL are AIR with zero load, while FR and RL carry approximately .516771 and .483229. These measured load distributions describe actual whole-body behaviour without imposing a fixed support template.

## New capture signal was not reached

The new approach-plus-contact helper requires an **unplaced, predecessor-eligible leg with hard qualified lift and hard crossing**. FR and FL are already placed and use the unchanged placed-retention branch. RR never obtains hard Q or C; RL remains predecessor-blocked by RR placement. Therefore **no leg in this PPO episode enters the new first-capture branch**, and the corresponding post-cross unload retirement is also not sampled.

The episode can inform approach/preparation sampling, but cannot establish whether the new capture shaping helps or hurts touchdown. A later genuine P07/offset-0 curriculum could expose the currently unsampled P07/P08/RR sequence while still starting before RR qualification; reaching the requested physical start and later outcomes must be recorded, not assumed. This observation is not a new training prerequisite or authorization to change the active run.

## Evidence limits and stop

All **2,881** credited physics ticks verify actual native effects and own-phase request effects. All decisions verify zero in-episode root-pose, root-velocity, force/impulse and gravity writes. The compact training log has continuous evaluator histories and tick-effect flags but no full 120 Hz raw contact/geometry stream; no exact force-vector, contact-point or hidden per-tick kinematic claim is invented.

Sources: first completed episode, first 361 policy audit rows, their current/evaluator histories, recorded curriculum start, and completed optimizer boundary 533/72704. Only PowerShell read access and this independent report write were used. No Python, new simulation, production edits, repeated hashes, parameters or gates were added. Reporting stops at the first completed PPO episode.
