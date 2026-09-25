# Staged reward-only implementation (not applied)

Use **v2/semantic_reward.py** and **v2/semantic_reward.patch**. The root-level first draft is superseded: v2 additionally pins the exact current experiment path, so another experiment with similar settings is not silently changed.

Runtime target is only `src/wlr50_clean/ppo/semantic_reward.py`. No YAML, supervisor, observation, actor, HISTORY, mapper, controller or asset edits. Semantic marker: `rr_recapture_retention_reward_only_v1`.

| Identity | SHA256 |
| --- | --- |
| Frozen production source | b2ed7047870a776c6c0102576300559565fe18ee4581d601690df42428ecabde |
| Staged v2 target | 7d06694bc5ec1abf24d8a55795fd647e59822d1cdbcdb271b5c4985013ef43c3 |
| Single-file unified diff | dec6a834c3b70c47d8c165f699afacf4abdf0b461a97e1f21f627719d41915de |

Activation requires the exact selected `configs/ppo_rr_rl_timing_policy_learning_v1/reward_config.yaml` path plus existing cooperative v5. Construction binds adjacent task-spec bytes once; schema, old potential/retention/capture modes, rear v4, owner mode and original gap/TOP/force scales are checked exactly. Other paths/old mode use original reward. Missing/mismatched selected binding fails instead of silently choosing a default. Step-time helpers do not read files.

The change is one reward-local endpoint replacement of `.85/4 × .2 × (new−old)`. It does not write shared `task_progress_potential` or groups. Old legal AIR/TOP follows the **exact original arithmetic**. Outside legal TOP, the contact half is zero; unqualified/ground geometry uses only one quarter of the existing geometric half. Existing qualified AIR/EDGE receives that old bounded geometric half, not support evidence. Current real TOP bearing remains maximal. Actual task history and pure-wheel-climb/body/safety classification are untouched.

Terminal `Phi_after=0` is selected before inspecting invalid current geometry. Last finite previous state still uses the same reward revision. Current frames are not mutated. Audit distinguishes shared observed Phi from reward Phi, per-endpoint old/new retention, same evaluator tick and the existing global share. Ordinary phase/rollout boundaries do not acquire terminal semantics.

## Checks completed without Torch, PXR or Isaac

`test_reward_only_stdlib.py`: **17 passed**. It AST-extracts the staged pure helpers and **actual full evaluate method**, and the unchanged production `SemanticObservationSchema.encode`. It does not import the production module/dependencies. The added case also executes the exact new staged sequence-test bodies (ground and qualified-outside recovery, plus RL-swing scope exit) via AST.

Coverage includes exact legal AIR/TOP values; ground/edge/ambiguous/high-force negative examples; XY/gap continuity for current qualified motion; actual-contact loss; scoped-out front/RL continuation; invalid evidence; all439 encoded input values and frame dictionaries unchanged; task families not double-counted; one-versus-eight physical ticks produce the same endpoint shaping delta; terminal invalid current geometry skipped; no phase bonus; stationary/closed-state-loop nonpositive potential; selected path/schema/scale binding; original runtime hash unchanged; all75 bounded RR-ground log samples exactly matching the offline v2 proposal.

This is not model/physics success. `test_reward_frozen_checkpoint_boundary.py` is **deferred**, not executed with a model. It imports Torch only after both explicit no-Isaac boundary and immutable checkpoint path environment keys exist. It uses the **normally imported, applied production calculator and actual selected reward config**, plus the real observation schema; it loads the actual saved439 actor/critic and compares all439 inputs, deterministic and same-RNG stochastic raw actions, means/sigmas and critic values before/after reward evaluation, with unchanged weights. The sampled log-prob field is compared **only for stochastic=True**; deterministic audit does not contain that field and the test asserts its absence. Frames are bounded synthetic fixtures, not a new physical run. This is an inference-only boundary test, no PPO/AUX update and no claim about actuator response or new policy capability.

The prototype quality-family fixture in the stdlib full-evaluate test deliberately avoids runtime dependencies; later boundary regression must also run real existing reward/cooperative/terminal tests against the merged production implementation. The deferred checkpoint test does not bypass the separate exact identity migration/save-reload checks prepared by `same439_boundary_plan`.

Use **`reward_unit_tests_v2.apply_patch` / `_v2.patch`** for the small ordinary-import `tests/unit/test_semantic_rr_retention_reward_only.py`; the earlier unversioned test patch is superseded. AST and apply-check pass; normal production imports have **not run** while Isaac is active. It covers exact accepted AIR/TOP arithmetic, ground/contact negatives, unchanged actual439 schema encoding, same-share endpoint equation, poisoned terminal current state, unplaced RR/legitimate RL continuation, selected-path/schema/old-mode/task-scale binding. Added sequences explicitly test TOP→ground/outside→closer→TOP as exactly one PBRS share and nonpositive same-state discounted cycle; outside→real RL swing retires delta with bounded `−delta_before`, not a new failure/event penalty. It adds no runtime file beyond the single reward module.

The active CP226048 natural-P01 video currently has not established the first RR lift at the reported t66.53. This patch's RR-unplaced scope returns zero; it cannot repair that first-lift problem or be reported as doing so. `SEALED_CP226048_BOUNDED_REVIEW_PLAN.md` in the parent output directory describes only a sealed-video bounded comparison, not a new physical run.

No new learning credit. After review and only at a legal code boundary, migration must preserve same439 weights/Adam/LR/normalizer/RNG, clear unfinished old-reward rollout, and recollect. A frozen checkpoint cannot gain new motion merely because reward changed.
