# Completed Block2: first current-policy P09 FALL evidence

Fixed scope: HEAD a8b148463115, Block1 source138496→139520 and Block2 source139520→139904. This is completed stochastic PPO sampling, not deterministic natural-P01 evaluation or a paired comparison. No active P07 data, production changes, simulation, Torch/PT, or parameter changes.

## Actual coverage and continuous preparation

Block1 sampled P06=247 but P07/P08/P09=0. Its three completed episodes ended FALL in P05/P06/P06; its fourth segment is a nonterminal P06 tail. The first recorded current-policy P09 samples therefore belong to Block2.

Block2: P06=363, P07=2, P08=2, P09=17 optimized decisions; the two completed episodes have178/117 decisions, and89 further P06 decisions form an unfinished update-boundary tail. All three resets used accepted448-decision/3584-tick frozen-FSM prefixes, excluded from PPO.

| Recorded quantity | Block2 episode0 | Block2 episode1 |
|---|---:|---:|
| Terminal global decision |139698|139815|
| Terminal episode tick / time, including prefix |5008 /41.733333333s|4519 /37.658333333s|
| P09 policy decisions / physics ticks |14 /112|3 /23|
| First RR policy initial event |4121 (another4869)|4465|
| RR policy qualification Q tick |4873|4477|
| P06→P07 / P07→P08 / P08→P09 ticks |4880 /4888 /4896|4480 /4488 /4496|
| P06 completion-value rear_approach at handoff |0.5709444659166195|0.16073191932962916|
| Terminal RR Q / C / P |true /false /false|true /false /false|
| Terminal RR front distance, m |−0.36479535358119364|−0.28513781385589|
| Terminal RR clearance above obstacle top, m |+0.1451234615736406|+0.014063075888126012|
| Terminal body linear speed norm, m/s |0.05685390402302465|0.5226522267332205|
| Terminal body angular speed norm, rad/s |0.3519729675556283|1.3884081315659076|

All six phase transitions explicitly record `continuous_takeover=true`, entry valid, and “qualified downstream motion already active; continuous takeover.” RR Q occurred before the P06 exit. Thus the one-decision P07/P08 durations do **not** prove all spatial preparation predicates reached1: the recorded exception accepts already-active qualified motion. This is the intended continuous-motion path, not evidence of history reset or a newly diagnosed software fault. `role_prepared_RR=1` / `transfer_ready_RR=1` at those transitions must not be reinterpreted as independently measured stable receiver support.

Coverage reports P07/P08 each16 native-effect ticks and14 own-phase request-effect ticks across its two decisions; all phase handoffs were nonterminal. These are recorded request-effect checks, not proof of smooth motion, successful coordination, or full Cartesian authority.

## Current support is not placement history

- Episode0 terminal: RR AIR/load0/supportfalse; FL AIR/load0/supportfalse despite historicalFL placed. FR verified TOP support/load0.6641908130754431; RL verified GROUND support/load0.33580918692455686.
- Episode1 terminal: RR AIR/load0/supportfalse; FR AIR/load0/supportfalse despite historicalFR placed. FL verified TOP support/load0.204653814707244; RL verified GROUND support/load0.795346185292756.
- Both retain RR Q but no crossing/placement, and RR remains substantially behind the front plane. RL has no Q/C/P. The first unfinished physical rear task is therefore RR crossing/capture, not missing RR initial lift. The episodes ended independent safety FALL before task completion/deadline.

Terminal nominal wheels are `[+.3,+.3,+.3,+.3]` in canonical FL/FR/RL/RR order. Actual canonical drive targets are `[+.495917,−.463495,−.364673,+.860667]` and `[−.899997,−.736261,−.515457,+.744713]`; measured wheel velocities are `[+.495319,−.654459,−.383989,+.861038]` and `[−.902096,−.736489,−.223748,+.745426]`rad/s. Thus zero policy participation or inability to oppose every nominal wheel is contradicted by these samples. These endpoint values do not establish which command caused the fall.

## Concrete FALL threshold: available rule, missing terminal measurements

Both terminal records report `FALL`, `PHYSICAL_SAFETY`, physical validitytrue, evidence status`VERIFIED`; the evaluator's generic text is “fall or physics explosion.” The actual backend separates the conditions:

- FALL: actual base-link position z `<0.015m` **or** IMU `projected_gravity_b.z >−0.30` (dimensionless).
- Explosion: base z`>1m`, linear speed`>5m/s`, angular speed`>20rad/s`, or the separate combined-guard fallback.

The recorded speed norms above are below explosion limits; low angular velocity does not exclude a large already-existing tilt. However, exact terminal base z, projected-gravity z, IMU roll and pitch are **UNAVAILABLE** in these summaries and both `completed_episodes.jsonl` terminal records. Consequently the specific FALL disjunct cannot be identified from the saved evidence inspected here. No reclassification is made.

The explicitly requested `physical_observations.jsonl` was checked at the exact Block2 run path and **does not exist**. No physical-raw scan was possible; this was not a file-size rejection. Terminal current body AABB minimum z values0.050002954057275374 /0.0533010533828692m are available, but are not base-link origin z and are not substituted for it or used to reconstruct tilt.

The backend computes `termination_mapping.physics_guard_values`; the current compact `SemanticEpisodeEnv.step` info does not pass that mapping, raw base pose, or IMU orientation into saved terminal info. This is a concrete reporting limit, not proof that the live sensor/guard failed. No threshold/reward/nominal fix follows from the available evidence alone.

## Sources and read scope

- Existing `outputs/ppo_all_stage_acceptance_v1/block1_summary/{coverage_table.json,training_block_summary.json}` and equivalent Block2 files: coverage, histories, current contacts, event attribution.
- Only the two completed rows in `runs/ppo_all_stage_acceptance_v1/train/20260909T0327591294869Z_ga8b148463115_1996f4a3ba204ea7b30a7bf6b1038046/completed_episodes.jsonl`: selected terminal values and retained transition evidence. No residual audit rescan.
- Exact existence check of that run's `physical_observations.jsonl`: absent.
- `src/wlr50_clean/ppo/isaac_fsm_backend.py::_fall_and_explosion`; `semantic_backend.py::_termination_signals`; `semantic_env.py::step`; `semantic_training.py` compact audit/episode writers: rule and serialization scope only.

Conclusion: actual qualified rear lifting with incomplete forward crossing and changing real support is demonstrated. The specific height-versus-tilt FALL trigger, complete posture trajectory, and causal contribution of individual commands are not demonstrated. No current-policy full-task success or improvement claim is supported.
