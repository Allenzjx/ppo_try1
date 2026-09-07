# Capture-retention run, episode1: P09 BODY_COLLISION

Scope: only the second completed episode (zero-based episode_index1) of `runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189`. Read-only PowerShell parsed exactly its677 decision rows, globals55,596–56,272, and its completed-episode summary. Other episodes were not analyzed. The running8,192-decision training block was not interrupted or counted as complete.

## Recorded outcome and evidence limit

The first recorded authoritative collision **detection/termination** is episode tick **5,409 /45.075s**, global56,272, P09 age0.675s. The prior decision-end observation at5,408 /45.066666667s has no terminal reason. The final issued action ran only one physics tick:676×8+1=5,409 actual ticks, not677×8. Its native dispatch label is5,588 for source episode tick5,408; that command label is not the observed collision tick.

The completed summary reports `BODY_COLLISION`, task_success=false, full_task_success=false. The shared TaskEvaluator remains valid=true and reports `TASK_FAILURE_BODY_COLLISION`, reason `central body/obstacle collision`. This is a recorded hard failure, **not** P09's30s deadline, a rollout-window cutoff, wheel-only crossing, missing legacy entry, codec failure or reset/software execution failure. Terminal bootstrap is disabled; terminal finite-observation fallback=false. All5,409 native target ticks are verified and every decision reports no in-episode state writes. Those actuator checks do not independently validate a collision sensor.

**The persisted training logs cannot independently rule out a sensor false positive.** This run stores semantic/task summaries and native-target audits but no `physical_observations.jsonl` or complete raw contact stream. None of these677 decision records retains the base contact's force, actual collider path pair, contact point/normal, consecutive pair count, `BodyCollisionStatus.real_pair_active/persistent/geometry_penetration_m`, base collider clearance, or complete chassis pose/RPY. The first physical contact may precede5,409; only the first hard-detection tick is established. No force, penetration depth, impact face, roll/pitch angle or causal contact reconstruction is invented here.

The unchanged source supplies a narrower consistency check: `SensorReader.read` passes actual classified contacts and live base-obstacle collider penetration into `BodyCollisionDetector.evaluate` (`sensor_reader.py:285`). That detector requires the verified exact **base_link ↔ /World/Obstacle** pair, active and BODY-role, plus either at least2 persistent physics samples or at least1mm live collider penetration (`body_collision_detector.py:42–69`). Leg/wheel contacts or a CoM-support estimate alone cannot trigger it. `TaskEvaluator` then directly consumes this authoritative detected boolean (`semantic_supervisor.py:292–296`). Thus the recorded branch is consistent with the existing physical hard detector, and there is no logged evidence of a software/sensor-contract failure; however the omitted raw detector inputs prevent an independent confirmation of which corroboration branch actually fired. Preserve the recorded failure without promoting this code-path consistency to an independently measured contact proof.

## P07 → P09 physical task history

The complete policy phase ledger is P01=1, P02=167, P03=3, P04=1, P05=148, P06=341, P07=4, P08=1, P09=11; P10–P13=0. All samples are current natural-P01 PPO, not teacher-prefix credit.

| Actual transition | Tick / time | Recorded current completion values |
|---|---|---|
| P06→P07 |5,288 /44.066666667s |rear_approach=1 |
| P07→P08 |5,320 /44.333333333s |workspace_RR=1, workspace_RL=1, support_RR=1 |
| P08→P09 |5,328 /44.4s |workspace_RR=1, load_ready_RR=1 |

All three entries are valid with no entry reasons and retain FR/FL placement history. FR qualified/crossed/placed at66/1,348/1,368; FL at1,450/2,488/2,559. There is **no RR or RL hard qualification, front crossing or placement** in this episode. Rear whole-body initial-clearance diagnostics are not substituted for qualification; no new rear lift-attempt event appears after5,200 in the terminal event history.

At the actual P06→P07 transition, FL was already AIR/load0, despite historical placement, with clearance+7.055101mm. FR was current TOP/load0.407717; RR/RL were GROUND/load0.159405/0.432878. RR front distance was−219.416585mm and RL−128.203746mm. Passing measured workspace/load preparation is not a promise of sustained wheel support or successful rear lift.

## Collision-adjacent measured state

Clearance below is **wheel-bottom relative to obstacle top**, not chassis clearance or penetration. Loads are normalized among the measured wheel supports, not an absolute force balance including chassis contact.

| Episode tick | FL clearance / wheel state | FR wheel state / load | RR clearance / wheel state | RL state / load | Support count |
|---|---|---|---|---|---:|
|5,328 P09 entry |AIR, historical placed |current TOP |GROUND |GROUND |3 |
|5,376 |+143.315708mm /AIR |TOP /0.448290 |−49.875169mm /AIR |GROUND /0.551710 |2 |
|5,400 |+171.040674mm /AIR |TOP /0.370385 |−47.594498mm /GROUND |GROUND /0.435849 |3 |
|5,408, no terminal |+181.540330mm /AIR |TOP /0.778564 |−47.148480mm /AIR |GROUND /0.221436 |2 |
|5,409, BODY_COLLISION |+182.007512mm /AIR |AIR /0 |−45.090805mm /AIR |GROUND /1 |1 |

At5,409, actual wheel front distances are FL+494.138304mm, FR+552.208487mm, RL−110.272017mm, RR−198.322054mm. FR clearance is+2.834163mm and RL−52.830743mm. FL has136 consecutive AIR samples, FR1, RR2. These readings show loss of current front-wheel support and absent rear placement at the recorded failure; historical FR/FL placement is not current support. The one-tick FR contact change is temporally adjacent to the body-collision signal, but the retained evidence does not establish whether it caused, followed or shared a cause with chassis contact.

Available body diagnostics at5,408→5,409 are forward-progress coordinate0.220707778→0.221016291m, linear-speed norm0.079323776→0.063647537m/s, and angular-speed norm0.494526897→0.269522869rad/s. Angular speed is not an attitude angle; no full body pose or base/obstacle signed clearance is retained here. FL's zero actual contact is not inferred from CoM: it is explicitly AIR, obstacle_pair_active=false and load0 in the task's current-leg evidence.

The last source-tick geometry receipt is `identity_within_descent_allowance` with all12 nominal-geometry adjustment components0; actual native PPO effect is verified on11 channels. This proves no direct geometry correction at that final dispatch, not that preceding geometry use or PPO actions had no earlier influence. No isolated collision cause or paired improvement is inferred.

## Disposition

Keep episode1 as the recorded **hard BODY_COLLISION failure** with the above raw-contact evidence limitation; do not reclassify it as incomplete or success, relax a detector, or make additional task-success checks an optimizer prerequisite. A definitive sensor-false-positive review would require the missing same-tick base-pair/penetration/pose inputs; it cannot be manufactured from this summary. No production edits, Python, extra Isaac, commits or changes to historical runs were performed. Only this outputs report was added.
