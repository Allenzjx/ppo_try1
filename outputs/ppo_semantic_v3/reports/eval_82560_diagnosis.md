# C82,560 natural-P01 evaluation — confirmed P09 body collision

Status: finalized real evaluation, not successful. This is a physical task failure, not an I/O, video, interface or phase-timeout failure. No training counts or historical run files are changed by this report.

Run: `runs/ppo_semantic_v3/validation/20260906T2034252033630Z_ge99fde1b3e83_4253287d07a646d6abe47a4f3354075d`.

- Runtime `e99fde1b3e8366f0ff1d484140b82877df745f0c`; immutable `checkpoints/history/checkpoint_step_000082560.pt` reloaded, deterministic heteroscedastic mean, N1, seed2001, natural P01, no teacher.
- Execution `SUCCEEDED`, root confirmed exit0. Task success=false; physical evaluation valid=true with `TASK_FAILURE_BODY_COLLISION`, reason `central body/obstacle collision`. Zero optimizer updates.
- **764 decisions /6,107 physics ticks /50.8916666667s**, ending in P09. The final decision executes3ticks, explaining 764×8−5; this is a real terminal, not a cut evaluation window, and terminal bootstrap is disabled.
- Once-only read of the finalized decision, native and raw physical streams: 6,108 finite, contiguous raw observations at ticks0–6,107; 6,107 contiguous verified native records. No tensors, Python, simulator or repeated checkpoint hashes were used.

## Stage and task history

Policy-request phase counts P01–P13 are **[1,189,3,1,151,407,1,1,10,0,0,0,0]** (764).

| Completed transition | Tick | Episode seconds | Recorded physical goal completion |
|---|---:|---:|---|
| P01→P02 |8 |0.066667 |load_ready_FR=1 |
| P02→P03 |1,520 |12.666667 |lifted_FR, clear_FR, approach_FR=1 |
| P03→P04 |1,544 |12.866667 |placed_FR=1 |
| P04→P05 |1,552 |12.933333 |workspace_FL, support_FL=1 |
| P05→P06 |2,760 |23.0 |placed_FL=1 |
| P06→P07 |6,016 |50.133333 |rear_approach=1 |
| P07→P08 |6,024 |50.2 |workspace_RR, workspace_RL, support_RR=1 |
| P08→P09 |6,032 |50.266667 |workspace_RR, load_ready_RR=1 |

P09 therefore runs only ticks6,033–6,107: **75ticks /0.625s**, with nine full decisions and one three-tick terminal decision. It does not reach the P09 deadline.

| Leg | Qualified lift Q | Front crossing C | Placement P | Interpretation |
|---|---:|---:|---:|---|
| FR |45 |1,530 |1,544 |Real natural-P01 policy events |
| FL |1,628 |2,557 |2,755 |Real capture history; not perpetual current support |
| RR |none |none |none |No lift-attempt/initial-clearance event either |
| RL |none |none |none |No rear completion |

RR initial_clearance remains false at all ten P09 decision ends and at termination. During the75 P09 raw ticks, RR has60 ground-active and15 AIR samples; its highest AIR bottom clearance is still **−44.732659mm below obstacle top**. This episode is **never-qualified RR**, not a legal rear crossing or placement subsequently lost. AIR alone is not qualified clearance.

## Exact first body-pair contact and persistence

The full raw stream contains exactly two active `base_link` / `/World/Obstacle` pair samples, both pair_verified=true. Only the second is marked detected/persistent. The normal-force direction is effectively world +Z.

| Tick / seconds | Pair state | Force Z (N) | Consecutive active ticks | BODY detected / persistent | Geometry penetration |
|---|---|---:|---:|---|---:|
|6,105 /50.875 |inactive |0 |0 |false /false |0 |
|6,106 /50.883333 |verified active |32.738990784 |1 |false /false |0 |
|6,107 /50.891667 |verified active |12.651303291 |2 |true /true |0 |

At6,106 the recorded reason is `single-tick exact BODY pair awaits persistence or geometry corroboration`; at6,107 it is `exact base_link/obstacle pair persisted`. Real contact, exact identity and two-tick persistence support the existing hard-failure classification. Zero geometry-penetration diagnostic does **not** negate a measured pair force or its persistence.

First contact point is `[0.561080575,-0.232293382,0.050076775]m`; terminal point is `[0.561072171,-0.232297421,0.049994137]m`, near the recorded obstacle top0.05m. These are recorded contact positions, not invented face labels or proof of a single contact mechanism. An inactive contact-position value at6,105 is not counted as a collision.

Base origin at6,106 is `[0.687753618,0.007355809,0.090801805]m`, terminal `[0.687964022,0.007293418,0.090624094]m`. Approximate roll/pitch derived from the recorded quaternion are12.2295°/−15.3661° and12.1676°/−15.3936° respectively. Base-origin height is not lowest-body clearance. Terminal linear velocity is `[0.019055415,-0.012300897,-0.016504455]m/s`. The recorded CoM-inside flag remains true (margin47.332591mm); that separate diagnostic does not disprove an exact body contact. Generic support_count=3 is not relabelled as three supporting wheels.

## FL support really changes before collision

FL placement has genuine force evidence: obstacle pair is active at2,754/2,755 with normal forces5.168342/8.213777N. During P06 ticks2,761–6,016, FL obstacle contact is active on3,238 of3,256 samples; at P06 completion6,016 its force is13.070292N. This is not a permanently unsupported FL episode.

FL switches to its final uninterrupted AIR stretch at **6,023** and stays AIR for85samples through6,107. All75 P09 samples have FL obstacle force0. Historical placed_FL=true must not be substituted for current support during this interval.

| Terminal leg | Current state | Normalized load | Front distance (mm) | Bottom clearance above top (mm) |
|---|---|---:|---:|---:|
| FR |TOP |0.397502763 |+535.842124 |+1.865803 |
| FL |AIR, no obstacle pair |0 |+445.439223 |+177.357038 |
| RL |GROUND |0.602497237 |−177.628961 |−52.030796 |
| RR |AIR, unqualified |0 |−221.839752 |−44.732659 |

FL within_top_xy is true but top_geometry/current support are false; its large positive clearance is not a loaded capture. The measured load is carried by FR and RL at the terminal sample. The temporal sequence documents loss of FL contact and increasing body tilt, but is not a controlled causal attribution to a specific actor output, headroom change or wheel owner.

## Terminal command and dispatch remain real

Canonical Full12 order is FL/FR/RL/RR hip-knee pairs (degrees), followed by FL/FR/RL/RR wheel velocities (rad/s). Final requested nominal is:

`[49.2,-13.4,0,31.1,31.2,0,55.6,0,0,-0.63,0,0]`.

Selected exact same-tick arithmetic illustrates the executed path, not a counterfactual physical cause:

| Channel | Logical nominal | Mapper native | Projected residual | Controller bias | Final drive target |
|---|---:|---:|---:|---:|---:|
| RR hip, deg |55.6 |55.6 |+5.453242569 |0 |61.053242569 |
| RR knee, deg |0 |0 |+3.917300189 |0 |3.917300189 |
| FR wheel, rad/s |−0.63 |−0.63 |+0.168737963 |0 |−0.461262037 |

Final drive Full12 is `[50.598115706,-7.587167946,2.159330468,34.901291763,32.887000582,-6.320598880,61.053242569,3.917300189,-0.195433591,-0.461262037,0.216263740,0.167490001]`. Frozen wheel signs produce actual float32 setter targets `[+0.195433587,-0.461262047,-0.216263741,+0.167490005]rad/s`. The nonzero final command is dispatched; it is not a fabricated zero hold or a missing control write.

All6,107 native records pass verified, staged/dispatched equality, frozen actual reconstruction and same-tick counterfactual checks. Decision accounting reports6,099 own-policy ticks, excluding8 incoming handoff holds. All four in-episode state-write categories are0. Every native record uses `same_tick_post_mapper_servo_margin_v1`;14 carry geometry audit records are present. Final geometry status is `identity_within_descent_allowance`, with all12 nominal geometry adjustments0. This report does not infer absence of every projection clip from that identity status or treat a first-order geometry prediction as a physical outcome.

## Bounded conclusion

The completed deterministic policy reaches real FR and FL capture and completes P06 preparation, then incurs a verified, persistent central-body collision0.625s into P09, before **any RR initial/qualified/cross/place event**. It is not a previous valid rear placement lost, a timeout, a codec problem or an optimizer failure. Exact contact evidence establishes the hard failure; the record alone does not isolate which control change caused it. No full/suffix success, paired stability improvement or successful video is claimed. Training remains82,560/610/12,200; this evaluation adds0 updates.

Sources within the immutable run: `run_manifest.json`, `evaluation_manifest.json`, `stage_transition_evidence.jsonl`, `physical_observations.jsonl`, `native_tick_audit.jsonl`, `residual_and_projection_audit.jsonl`. Existing failed A/B/C and training reports are preserved.
