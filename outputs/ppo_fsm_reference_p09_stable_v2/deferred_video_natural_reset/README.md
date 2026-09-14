# Deferred current-video natural reset guard — baseline64c0324

READY FOR REVIEW / NOT APPLIED / TESTS NOT RUN.
Baseline:64c03243ac05e43e7a5843eaee252eef1136c925.
No production edits, Python/Torch/Isaac execution, process operations or published
report changes were made while ordinary deterministic P01 evaluation continued.

## Concrete failure, not task failure

Preserved real C source:
runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/
20260910T0930063395586Z_g64c03243ac05_4bb994da17ed47bea5c9e0b04c3a2a41/source/
semantic_video_source_manifest.json.

Its reset_count=1, reset_options={}, training_phase_snapshot="P01";
checkpoint_load_provenance=null, issued_policy_decisions=0, episode_physics_ticks=null,
physical_task_success=null. The acceptance error is the first-natural-reset guard.
It is a video initialization classification error, not observed PPO/task failure.
Nothing here relabels or overwrites that failed run.

SemanticIsaacBackend.reset:205–210 constructs mode=semantic_natural_P01,
requested_phase=P01 after the ordinary scene reset and original180-step settle,
without invoking a phase snapshot loader. Inherited _make_reset_metadata:2881–2884
copies requested_phase into training_phase_snapshot. The latter is a label, not
proof of historical restoration. The old video check at:510–512 conflates them.

## Minimal patch

Apply artifact: natural_reset_64c0324.apply_patch (327 lines including new tests).
One production path changes: src/wlr50_clean/ppo/semantic_video.py.
Two test paths: update test_semantic_video_fsm_reference_v2.py's incorrect semantic
fixture marker=None; add test_semantic_video_natural_reset.py.
No backend, physics, sensing, controller, task, MDP, reward, action, checkpoint
loader or migration helper change. Main owns any next exact video migration plan
from its actual latest complete checkpoint.

Only fsm_reference_p09_stable_v2 uses the new helper/proof. The original None-only
capture guard and replay behavior for older experiments remain untouched.

The common A/B/C criterion remains first committed reset, seed4001, options{},
actual P01/tick0/decision0/nonterminal entry, zero existing reset_prime_tick_count
and entry clocks, and an episode-relative sensor clock. All those metadata fields
are already emitted by the actual backend; missing required evidence fails closed.
Explicit P01 snapshot options are rejected just like P09 snapshot options.

The two existing backend provenance encodings are recognized, not two task rules:

- A: normal_p01_reset, requested_phase=None, marker=None, snapshot_validated=False,
  no semantic execution-mode claim.
- B/C: semantic_natural_P01, requested_phase=P01, marker=P01, actual
  execution_mode=semantic_B_or_C and supervisor_schema=task_semantic_v2,
  historical_state_equality_required=False and policy_credit_excludes_settle=True.

Both reject loaded/validated snapshot flags, snapshot file/state/source-tick fields,
historical/unknown modes, missing requested-phase proof, replay/offset clocks,
nonempty options, repeated resets and inconsistent actual entry state. Merely
changing the acceptance expression to marker in(None,"P01") is intentionally avoided.

The capture stores schema wlr50_clean.current_video_natural_reset.v1 as
natural_reset_proof, including the original complete phase_snapshot_restoration,
required reset metadata and actual entry state. Independent source validation
reconstructs the same proof through the same helper and cross-checks the compact
reset_evidence fields. It does not trust only a fresh_process boolean. Missing or
altered current proof is rejected before artifacts/replay publication. Old source
files lacking this new schema remain under their original experiment routing.

## Unrun test scope

Positive tests call REAL A and semantic backend reset implementations with existing
FakeRuntime physical dependencies and SemanticFrameStub; no Isaac is implied.
They verify actual None-vs-P01 markers,180 dispatches and no extra physics/reader
operations from proof construction or JSON-round-trip validation.

Negative cases cover all three roles: actual/explicit/hidden P01 snapshot,
historical phase restoration, missing evidence, repeated reset, replay tick,
sensor offset, entry clock, mismatched marker/execution provenance and snapshot
file/source tick. Additional tests reject noninitial actual entry state, persisted
proof/summary tampering and missing current proof before artifact processing.
Older v2/v3/old-experiment validators are checked not to invoke the new guard.

The existing fake success/failure capture fixture now mirrors the real semantic
P01 receipt and validates that the new proof survives into its source manifest.
Existing done-only and terminal encoder-failure tests continue to use it.

No tests have been run. Review and apply only after the main agent's current
ordinary P01 evaluation completes and Isaac exits; this draft is not a training
or success-video publication gate.
