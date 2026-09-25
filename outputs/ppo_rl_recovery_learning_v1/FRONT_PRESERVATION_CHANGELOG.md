# Urgent CP225280 front preservation: implementation and attribution

The event-aligned CP225280/CP230144 comparison confirmed physical front descent
before FL crossed the edge. It did not isolate one joint as the cause. Restore
the already demonstrated coupled front policy, not a new fixed-height controller
or additional FL descent script.

Production changes at the sealed CP231680 update boundary:

- semantic_front_preservation.py: explicit independent f6d439 CP225280 source,
  full-state zero-learning publication, fresh512 storage and ordinary save/load.
  Original source identity and old later branch remain separate and immutable.
- semantic_front_replay.py: front-only reference conditional-Gaussian KL from
 100 real fit observations;99 heldout observations are evaluated, not fitted.
  No pose imitation, rear targets, live teacher, extra samples or optimizer steps.
- semantic_training.py: sum this gradient with the actual PPO actor gradient
  before native clipping, in the existing20 Adam minibatches. Log actual rows,
  phase exposure, gradient use, before/after heldout KL and separate counters.
- semantic_front_retention439.py / semantic_rear_policy_timing_migration.py:
  route only the explicit new sibling identity through its own validated contract.

Unchanged: actual439/full12 actor architecture, HISTORY and conditional sampling,
N, action mapper, actuator dispatch, rate and physical limits, sensor acceptance,
FL capture assist ON, rear completion assist OFF, all assets and physical ability.
The existing72e RR-retention reward is carried explicitly; no new front reward.
Adam/LR/Identity normalization and RNG come from CP225280, no head reset.

First real block: successful_nominal real continuous P01 prefix to P12 plus8
decisions, then512 student decisions. Prefix cannot enter PPO storage. This is
rear-suffix training, not a claim that the student executed its own front.
After save/reload, deterministic evaluation starts at natural P01 using one
student for the entire episode. Old N remains a labelled historical comparison.

Replay protection is an objective, not a guarantee of closed-loop retention.
Physical front recovery and RL progress must be reported from the new run.
