# C84,992 natural-P01 evaluation — P12 incomplete, RL never qualified

Final real evaluation: **1,353 decisions /10,824 physics ticks /90.2s**, P12 `INCOMPLETE_CONTROLLER_BLOCKED`, task_success=false. Execution finalized `SUCCEEDED`; root confirmed session88717 exit0/CLOSED. Physical evaluator valid=true and physical termination_reason=null: this is an unfinished task at the P12 age30s deadline, not a body collision, wheel-only crossing, interface or video failure. The200s global horizon was not reached.

Run: `runs/ppo_semantic_v3/validation/20260906T2132007864973Z_ge99fde1b3e83_62520cdf88e549ae9929b9916bd92336`. Saved `checkpoint_step_000084992.pt`, unchanged e99fde1, N1/seed2001, deterministic fixed heteroscedastic mean, natural P01, no teacher and0 optimizer updates. The evaluation is not cut short by a sampling window; terminal bootstrap=false.

## Phases and actual earned history

Manifest and decision stream match phase P01–P13 **[1,197,3,1,150,457,1,1,88,3,1,450,0]**. Every decision executes8ticks, so1,353×8=10,824. Self-to-self task-history event records are not counted as extra phase transitions.

| Transition | Tick | Seconds |
|---|---:|---:|
| P01→P02 |8 |0.066667 |
| P02→P03 |1,584 |13.2 |
| P03→P04 |1,608 |13.4 |
| P04→P05 |1,616 |13.466667 |
| P05→P06 |2,816 |23.466667 |
| P06→P07 |6,472 |53.933333 |
| P07→P08 |6,480 |54.0 |
| P08→P09 |6,488 |54.066667 |
| P09→P10 |7,192 |59.933333 |
| P10→P11 |7,216 |60.133333 |
| P11→P12 |7,224 |60.2 |
| P12 deadline |10,824 |90.2 |

Real policy event history: FR Q45/C1,594/P1,608; FL Q1,693/C2,655/P2,816; **RR Q7,149/C7,184/P7,191**. All occurred under this natural-P01 deterministic policy, not a teacher prefix. RL has no hard qualification, crossing or placement anywhere in this episode. Thus this differs from a revoked RL qualified crossing or a P13 stop failure: the first unfinished rear task is RL active lift/cross/place, with final `placed_RL=0` and P13 never entered.

## Current P10/P12 entry is not the fixed A-teacher state

Fixed decision/raw observations around7,192–7,240 show real preparation rather than an exact historical entry restore:

| Endpoint | Current phase after observation | RL front gap /load | RR front gap /contact | Actual readiness |
|---|---|---|---|---|
|7,192 /59.933333s |P10 |−245.673683mm /0.187044, GROUND |+8.248333mm /obstacle19.921835N |workspace_RL0.897305; support_RL1 |
|7,208 /60.066667s |P10 |−220.007267mm /0.042968 |+17.764810mm /load0.429236 |workspace_RL0.999970931, still below1 |
|7,216 /60.133333s |P11 |−201.298889mm /0.223052, GROUND |+11.867646mm /obstacle4.865866N |workspace passed; load_ready_RL0.971185 |
|7,224 /60.2s |P12 |−195.143994mm /0.037123, GROUND0.976787N |+3.853205mm /obstacle13.813702N |load_ready passed; RL still not qualified |

At7,224 FR is AIR/force0 while FL has real obstacle force11.521586N; RR and FL provide actual support. RL first becomes AIR on7,225, immediately after P12 entry, but its bottom is still49.437003mm below obstacle top. Relative unloading is not qualified lift.

Known A-teacher P10 initialization in the completed `p10_headroom_block_84992.md` always occurs at7,584/63.2s after948 teacher decisions, with teacher RR Q6,938/C7,109/P7,579. Current C enters P10 **392ticks/3.266667s earlier**, despite a later RR qualification, and earns RR placement just7ticks after crossing rather than the teacher's470ticks. This comparison documents distinct histories and current entry geometry; it does not require either delay, assert equal physical states or claim that short placement latency caused failure. A common phase label is not an interchangeable physical initialization.

The selected C command record also differs from a static standstill: FL nominal wheel progresses−0.30 at7,192 to−1.05 at7,224 and−1.07 at7,232. Its actual canonical drive at7,224 is−1.029447rad/s; other wheels are−0.236196/−0.269149/+0.161941. These are real nominal-plus-policy dispatches, not a copied teacher state. No causal conclusion or new cap/mapper choice follows from this brief window.

## RL attempts never reach top clearance

The first hard RLQ is absent, not merely missing from a final truncated event list. Complete task history has initial-clearance events, including four during P12 at **7,236/7,303/7,580/7,597**, but no `qualified_measured_upward_lift` or crossing/placement for RL. Earlier initial events during natural P01/P06 are not elevated to hard qualification.

Across all **3,600 actual P12 ticks7,225–10,824**:

- RL ground-active3,450; AIR150; obstacle-active0; center-front crossings0; AIR samples at/above obstacle top0. Exact ground/obstacle pair verification is present throughout.
- Highest AIR bottom is at **7,254/60.45s**, clearance **−33.886751mm**, front **−232.675191mm**, force/load0. This is the maximum of the airborne samples, not a useful above-top lift. RR and FL are then genuinely obstacle-supported,16.437492N and14.939277N respectively; FR is AIR. Therefore the lack of adequate initial RL lift cannot simply be described as “RR support was already absent.”
- Closest AIR center is7,225, front−195.286330mm, clearance−49.437003mm. Closest RL center of any contact state is7,800, front−165.922996mm, but that is GROUND with14.167935N and clearance−51.630681mm—not crossing or AIR carry success.
- Final RL is GROUND, front−503.132364mm, clearance−51.096199mm, force14.229776N/load0.497784; current initial_clearance=false, hardQ/C/P=false.

These are genuine small airborne attempts followed by ground-dominated motion, not an unlogged successful lift. Body X retreats from0.697498858m at P12 entry to0.230385497m at termination (−0.467113361m), documenting substantial physical regression without assigning a single control cause.

## RR earned placement does not guarantee current platform support

RR placement7,191 has real obstacle force21.465583N, front+7.576827mm and bottom clearance+0.001655mm. It is not a fabricated historical placement. Subsequent selected events distinguish later loss from the earned event:

- 7,281: first center-front negative after placement,−0.414905mm; obstacle contact remains active6.893254N, so crossing the center back alone is not yet loss of obstacle contact.
- 7,428: first no-obstacle-contact sample after placement, AIR at front−11.392411mm/clearance−3.040044mm. This is a first gap, not a claim of no later recontact.
- 7,595: first GROUND after placement, front−51.605519mm/clearance−50.390645mm with30.748957N ground force. RL is then also GROUND.

Terminal RR remains GROUND/front−445.589181mm/clearance−50.196214mm/load0.145687. Historical RRplaced=true is retained as an earned task event, not current TOP support. FL's final AIR stretch starts7,793 and lasts3,032raw samples; terminal FL front−168.880131mm/clearance+21.041979mm/load0. FR alone remains TOP withfront+25.117282mm/clearance+0.762514mm/load0.356529. Current final_region_valid=false and final_support_available=false; generic support_count3 includes two ground-supported rear wheels, not three platform captures.

## Native evidence and terminal target

One final read each of the three streams finds10,825 raw observations at0–10,824, all finite/contiguous; body detected0 and exact body-pair-active0. All1,353 decision indices and10,824 native indices are contiguous. **10,824 native ticks pass** verified/staged-dispatch equality/frozen reconstruction/same-tick counterfactual; own-phase10,813 excludes11incoming holds. All four in-episode state-write category sums are0. Every native record uses the current same-tick post-mapper servo-margin mode.

Terminal finite nominal Full12 is `[-18.5,-31.4,3.7,31.1,-10.1,-18.7,-6.9,-27.2,0,0,0,0]` (servo degrees, wheels rad/s); controller bias is0. Mapped/native plus delivered residual gives actual drive `[-17.025652,-35.226165,2.958494,6.920351,-11.561487,-22.283037,-11.797954,-33.180876,0.028734387,-0.324726297,-0.364317100,0.106808831]`. Terminal servo headroom audit lists no clipped indices and requested/effective policy residual equality. Frozen wheel signs yield actual float32 setter targets `[-0.028734388,-0.324726284,+0.364317089,+0.106808834]`. This verifies real dispatch at termination; it does not prove that no earlier projection or nominal geometry adjustment occurred.

Finite wheel-nominal retirement, a delivered nonzero residual and a valid30s stage deadline are not a codec or execution-chain error. The report does not claim that a different reward, mapper, teacher state or future saved-C prefix would ensure success. No new version or curriculum rollout is certified here.

## Final account

The deterministic natural-P01 policy genuinely earns RRQ/C/P, then never obtains RL above-top qualification and loses useful earlier platform support before P12 times out. **No full/suffix success or paired stability improvement** is recorded. Training stays23 finalized blocks,84,992/629/12,580; this evaluation adds0 updates. C82,560 and every earlier failed/incomplete A/B/C outcome remain preserved.

Sources: this run's finalized manifests, stage transition evidence, once-read physical/decision/native streams; existing `p10_headroom_block_84992.md` and `p13_headroom_first_episode_readonly.md` supply the fixed teacher-history comparison. No new training/evaluation run was scanned, no tensors/hashes were computed and no production was modified.
