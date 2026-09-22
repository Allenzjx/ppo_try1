# Isolated physical-quality candidate

No production edits or physical/PPO calls. Four candidates under
`candidate/src/wlr50_clean/ppo/`: semantic_reward.py, semantic_supervisor.py,
semantic_observation.py, and new semantic_task_quality.py. Root owns adoption,
configuration, migration and legal execution boundary.

## Exact changes

- New reward objective/revision `task_conditioned_hip_wheel_quality_v1` binds
  body-family/epsilon .06, all other quality families0. Existing front settings
  stay unchanged; front_fraction=.5 keeps the actual P01/P02 coefficient
  .03*(1-.5*physical_transfer_fraction), range.015–.03/s. Front sample audit
  reports this coefficient and its own contribution, not the geometry cost.
- geometry_fraction=.5 adds at most.03/s in P01/P02/P05/P06/P07/P08/P09.
  It uses existing current-pose body collider AABB and the live obstacle's six
  world planes. The Euclidean AABB separation is a conservative lower bound,
  not base_z, exact mesh distance or a collision test. Cost is
  `min(1,max(0,(.020-distance)/.020))**2`. No reward for greater height or space
  above the finite20mm engineering margin. No ground-height cost, joint-motion
  cost, force bonus, residual penalty, imitation or auxiliary optimizer.
- Per-physics-tick geometry audit contains geometry, eligible/valid/reason,
  raw cost, actual dt, effective coefficient and weighted cost. Missing eligible
  nonterminal geometry raises a measurement error, never masquerades as zero.
  Already classified terminal with unavailable geometry keeps the existing
  failure/success event and terminalPhi0; geometry cost fields remainnull with
  `terminal_measurement_omitted=true`. No new sensor or hard evaluator rule.
- Body-family raw total is at most1. Existing failure-avoidance check computes
  `(.06+.02)/15/(1-.9985)+5 = 8.555555555555683 < failure_cost40`.
  The guard itself is unchanged.
- Stage revision `task_conditioned_hip_wheel_v1` adds only the root-specified
  rolling_capture_retention dict; old capture_retention_semantics is retained.
  Only FR/FL with both fronts placed, RR unplaced and no current valid RR lift
  receive the rolling modifier. Weight is
  `clip((-.22-max(RL_front_distance,RR_front_distance))/.05,0,1)`.
  Existing region retention is multiplied by `(1-weight)+weight*usable`, where
  usable=.5/(1+max(0,gap)/.003)+.5*current_verified_TOP_bearing_support.
  The existing .2 capture budget is reused; greater force receives no benefit.
  Preparation fades this modifier by measured distance, not a phase reset or
  fixed hold timer. No permanent two-front-contact rule during RR transfer.
- Observation build appends `obstacle_planes_world_m` to diagnostic/reward
  metrics only. Actor/critic groups, encoding and schema remain unchanged.

Allowed AST scopes: reward._validate_task_priority and
SemanticRewardCalculator.evaluate; supervisor.load_task_spec and
TaskStageSupervisor._current_capture_retention; observation.SemanticObservationBuilder.build.
Only new reward import is from semantic_task_quality (OBJECTIVE aliased
TASK_CONDITIONED_QUALITY_OBJECTIVE, SPACE_CONFIG, task_space_quality_sample).
TaskEvaluator and NominalMotionProvider AST match production exactly.

## Tests

`test_task_physical_quality.py`: 44 narrow CPU tests pass; receipt
`task_physical_quality_tests.xml`. Tests directly compare candidate with original
production modules in one CPU process, without replacing production imports.
All old reward result dictionaries match exactly across13 phases and3 existing
profiles. New front audit and body cost match the old front profile bitwise when
geometry deficit is absent. Tests cover bounds/axes/finite geometry, omitted
terminal evidence, illegal missing live data, phase/dt/PBRS, real-support versus
placed semantics, retirement continuity, no force/penetration incentive,
unchanged actor encoding, protected evaluator/nominal AST and config rejection.

The initial test run found three test-expectation errors (literal.021 rounding
and a schematic obstacle plane used instead of the real fixture plane). These
were corrected in tests; algorithm code was not changed to satisfy them.
No physics success, effective policy improvement or PPO update is claimed.
