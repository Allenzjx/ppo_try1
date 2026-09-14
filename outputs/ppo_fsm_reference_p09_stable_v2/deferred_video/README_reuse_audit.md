# Existing deferred video draft: reuse audit, not application

Status: **FOUND_REUSABLE_DRAFT / NOT_APPLIED / NOT_TESTED**.
Current production inspected: 28609010db4e57c5b34304a4ae2563c69f9d00b9.
Production src/configs/scripts/tests remains unchanged. No Torch/Isaac/test imports,
no recording, checkpoint mutation, fabricated video, or new migration plan occurred.
Current training and natural P01 evaluation do not depend on this report.

## Existing reusable artifact

Use the already existing:
outputs/ppo_all_stage_acceptance_v1/deferred_video_patch/

Its README.md and application_inventory.json identify baseline
a8b1484631156bad6e6ef0c96bd79a2fbf12e308, six mirrored production files and one unrun
test draft. The original source hashes were copied from its recorded a8 manifest;
they are not current target hashes. Do not copy its old complete CLI/training/
migration files over 286090: doing so would erase later FSM/P09 and stop-boundary work.

The six intended paths are semantic_video.py, semantic_video_cli.py,
scripts/run_semantic_video.ps1, semantic_cli.py, semantic_training.py and
semantic_migration.py. Existing publication/encoding/physical endpoint functions
are reused, not replaced with a new framework.

## What already exists in that draft

- semantic_video.py:91 routes video_configuration through explicit experiment_id.
  :340 accepts the selector for capture; :524 verifies source/runtime selector and
  captured evaluator configuration records during independent replay.
- semantic_video_cli.py:31–35 passes preflight-verified observation layout to the
  real runner and calls load_semantic_checkpoint. :45–47 invokes the actual saved
  actor with stochastic_output=False, including its history-conditioned kernel.
  :51–62 records load provenance, verified372 policy/layout and zero updates.
  No reset of actor, std, critic, Adam or preprocessing is substituted.
- semantic_video_cli.py:69–77 enforces natural P01, N1, no phase prefix/warm-start,
  locked video seed and a full single-episode budget. semantic_video.py:374 checks
  the first actual natural P01 reset, then :414 loads on its real initial observation.
  Capture loops over that one live episode; completion must come from the common
  physical evaluator. The manifest explicitly records fresh process, episode_count=1,
  from_phase=P01, no stitching/interpolation/speed change, and model-unchanged proof.
  These are proposed runtime checks, not proof that this draft has been exercised.
- semantic_video_cli.py:89 replaces only A's reader dependency for the explicitly
  selected all-stage experiment, matching the existing independent A evaluation recipe.
  Frozen A controller, mapper, reset/settle remain untouched.
- semantic_migration.py:902 proposes exact same-HISTORY372/N1 video observation
  receipt. The loader preserves the ordinary Adam/actual LR/normalizer/RNG/counter
  restoration path and fresh empty rollout.
- tests/unit/test_semantic_video_all_stage_deferred.py:61/:80/:109/:123/:144
  covers routing, A-reader-only selection, exact372 plan/preflight and negative
  metadata/schema/factor cases. :184 creates a real CPU128-decision PPO source with
  nonempty Adam, saves/reloads through video checkpoint_loader, compares all actor/
  critic/Adam/normalizer/LR/RNG/budget state and actions with raw-history0 and0.7.
  None of these test drafts has been run; no passing receipt is claimed.

## Minimum rebase points for the current experiment (not implemented here)

1. Reuse the two video source diffs and launcher diff against current production.
   Current semantic_cli.version_paths already supports fsm_reference_p09_stable_v2;
   preserve it instead of the old a8 implementation.
2. Add fsm_reference_p09_stable_v2 to the deferred launcher ValidateSet (currently
   only transfer_roles_v1/all_stage_acceptance_v1), and include the same explicit
   current experiment in A's common all_stage_v1 reader branch at video_cli.py:89.
   This must not modify frozen A control or scene/physical properties.
3. The old _video_observation_contract folder expression (:926) special-cases only
   all_stage_acceptance_v1 and otherwise falls back to ppo_semantic_v3. Resolve the
   verified current experiment namespace instead. Require the exact same SIX selected
   configuration bindings in old/new runtime, including action_schema.json.
   The video evaluation_configuration currently records only five consumed files;
   the sixth action schema is in runtime_contract.selected_configuration, not a sixth
   direct video consumer. A current-experiment test must verify all six bindings and
   their actual pinned bytes; do not pretend the existing five-entry field contains six.
4. Parameterize the existing routing/A-reader/plan/preflight/CPU-load tests for both
   all_stage_acceptance_v1 and fsm_reference_p09_stable_v2; add current-six-config
   missing/extra/path/byte mismatch rejection. Preserve v2/v3 default324 tests.
   Actual fresh-process single-episode capture and post-hold behavior still require
   later live verification; a CPU reload test cannot certify video or task success.

Because a reusable draft already exists, no duplicate full source mirrors or second
video framework were generated for this bounded task. These are rebase instructions,
not a claim that the old draft applies cleanly or satisfies current video acceptance.

## Conflict with the newer exact372 instrumentation draft

The distinct existing allowlists must remain distinct:

- INSTRUMENTATION_FILES includes semantic_cli.py / semantic_training.py /
  semantic_migration.py and the other already reviewed instrumentation paths.
- VIDEO_FILES contains semantic_video.py / semantic_video_cli.py /
  scripts/run_semantic_video.ps1. These three are NOT instrumentation-only permission.
- Production's explicit video_review branch already authorizes VIDEO_FILES together
  with INSTRUMENTATION_FILES while prohibiting task/execution/configuration factors.

The newer deferred_runtime_identity mirror:
semantic_migration.py:1028 rejects any video_review or VIDEO_FILES delta for its
instrumentation372 factor, and :1095 invokes that helper before the video branch.
Its semantic_training.py:466 expects top-level instrumentation_observation_contract.

The older video mirror instead builds a nested
video_instrumentation_factor.observation_contract and its training.py:462 consumes
that receipt. Blindly merging the two mirrors therefore either rejects a legitimate
video boundary or discards one branch's validation.

Minimal future reconciliation is an explicitly reviewed exclusive branch:
instrumentation-only uses its strict current factor; explicit video_review uses the
existing narrow video permission and verified same372 video receipt. The loader
must accept exactly the verified receipt matching that branch, not merely width372
or either arbitrary dictionary. Keep mutual exclusion with task, execution and
new-MDP factors, complete policy/schema verification, empty storage and exact state
restoration. Do not silently add VIDEO_FILES to INSTRUMENTATION_FILES or remove the
mixed-factor rejection. No NewMdpWarmStart or fresh-Adam migration is justified.

If instrumentation is applied first, record its exact resume from the latest
completed checkpoint, then treat later video integration as its own reviewed video
boundary. If reviewed together, it is a video-specific boundary with the existing
video allowlist and reconciled same-layout receipt, not an instrumentation waiver.
No checkpoint path, target HEAD/hash or future plan is invented by this report.

## Known unchanged limits

The draft deliberately keeps PRE_TICKS=64, POST_TICKS=184 and MAX_FRAMES=3000.
Full extra video context limits the episode itself to 23751 ticks (197.925s), despite
the task's200s horizon. It also uses post-success physical hold rather than continued
policy. Current final-ACK geometry/headroom/tracking and P13 semantics require the
already documented narrow post-hold check before recording can be certified.
Do not crop/accelerate/relabel current training output to hide these limitations.

No successful checkpoint, successful video, paired improvement or physical equivalence
has been established by this reuse audit.

