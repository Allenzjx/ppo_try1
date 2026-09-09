# Video namespace routing — unapplied proposal

Status: **UNAPPLIED / UNTESTED / NO CAPTURE**. Prepared against the three existing video files at frozen `2f27c6f5065a`. Only output artifacts were written; no Python/Torch/Isaac, replay, decode, hash recomputation or test was run.

## Scope

The patch touches exactly `semantic_video_cli.py`, `semantic_video.py`, and `scripts/run_semantic_video.ps1` (all already in `semantic_migration.VIDEO_FILES`).

- CLI propagates optional `args.experiment_id` into the contract used by preflight, checkpoint loader, capture and end-of-run consistency check.
- Independent source validation rebuilds the same experiment contract from its recorded value; omission preserves legacy behavior.
- PowerShell accepts only `-ExperimentId transfer_roles_v1 -SemanticVersion v3`, forwards it, and routes run/launcher files to `runs/ppo_transfer_roles_v1/video_eval/<kind>/<run_id>`. Defaults retain the existing v2/v3 directories and shared `runs/ppo_semantic_v2/.single_process.lock`.
- Existing v3 configuration paths, 324/12 policy, history kernel, official checkpoint loader, mapper, evaluator, camera,15fps/120Hz sampling,180-settle-tail capture with0 extra ticks, post-hold, codec/PTS checks and success eligibility are unchanged.

## Apply and test — future completed boundary only

No action is requested during the live block. At an explicitly selected completed training boundary, review `git diff` and use `git apply --check outputs/ppo_transfer_roles_v1/proposed_video_namespace_routing.patch` before applying. Do not apply blindly if source has moved.

Existing targeted CPU regression command (not executed; from repository root):
```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' -m pytest -q tests/unit/test_semantic_video.py tests/unit/test_semantic_video_v3.py tests/unit/test_semantic_video_migration.py tests/unit/test_semantic_policy_cli.py::test_video_auto_selects_saved_actor_type_without_conversion tests/unit/test_semantic_policy_cli.py::test_video_explicitly_forbids_conversion_before_source_validation
```

The existing PowerShell-directory assertion remains valid because the default path expressions are retained. Before release, add small CPU/static regressions to those existing video test files: (1) main preflight sees experiment_id for transfer v3, omitted for ordinary v2/v3, and recheck uses the same options; (2) source validator rebuilds the recorded experiment contract and rejects a mismatched one without bypassing earlier success checks; (3) PS opt-in routes both output paths and forwards the flag, rejects v2+experiment, preserves the common lock; (4) existing saved-policy auto-selection remains in use. A stub can stop before native launch; no new trainer or simulator test lane is needed. No test count or pass claim is made here.

## Existing migration path, no new MDP lane

After review/tests and commit, build the existing `semantic_migration.build_migration_plan` using the actual immutable source checkpoint and target `runtime_contract(expected_head=<new committed HEAD>, semantic_version="v3", experiment_id="transfer_roles_v1")`. Use `allowed_changed_files=sorted(VIDEO_FILES)` only if the actual delta is exactly these three files, and `video_review={"reason": <reviewed routing-only rationale>}`. Then validate it with the existing `validate_migration_plan` and pass its unique file via `-ResumeMigration`.

Keep source and target experiment/config/topology identical; this is a video-only source-boundary plan, **not NewMdpWarmStart**. Existing plan/load semantics preserve actor/std/critic/Adam/normalizer/RNG/budget while discarding stale rollout and starting legal P01 physics; the review binds hashes, not success or physical equivalence. Do not bypass a failed contract comparison, mix reward/config changes, reinitialize learned state, mutate the source checkpoint, or create a new migration implementation.

Future C invocation adds only `-SemanticVersion v3 -ExperimentId transfer_roles_v1` to the existing video command, with actual `-ExpectedHead`, immutable `-Checkpoint`, `-ResumeMigration`, and `-Mode semantic_residual_eval`. Video retains its existing seed4001/P01/N1 rule; the prior natural evaluation at seed2001 does not guarantee the new capture succeeds.

This patch is preparation only. An honestly successful actual P01 result can motivate capture; the capture itself still must satisfy current physical and technical eligibility. No A5/5/probe/full-C readiness gate is introduced. Existing incomplete A remains untouched/unpaired; no success file, PPO-improved name, paired comparison or superiority claim is created.

## Read-only publication compatibility finding

**The three-file routing patch does not make preserved incomplete/unpaired A eligible for the existing comparison publisher.** This is an artifact-interface limitation, not a requirement that A succeed or be rerun.

| Existing path | Actual restriction incompatible with the allowed historical reference |
|---|---|
| `semantic_video.validate_semantic_video_source` (553–660), used by `publish_success_source` and `publish_success_comparison` | Requires success_candidate=true, diagnostic_only=false, successful physical replay and all184 real post-success ticks. Historical A is incomplete with0 post-success ticks. Thus even the neutral-looking `fsm_original_baseline.mp4` name cannot use this success-only publisher. |
| `semantic_video.publish_success_comparison` (698–730) | Validates both A and C as successes, then requires equal seed/camera/runtime_contract/semantic_version/pre-action source/extra ticks; its only filename is `fsm_vs_ppo_success.mp4`. Historical v2 A180+64 and future transfer-v3 C180+0 are not matched even if both video seeds are4001. |
| Unexecuted output-only `reports/render_outcome_comparison.py` | Allows incomplete A, but `managed_source` requires new managed v3 sources, current runtime and zero-extra180-settle evidence; `main` still requires matching contracts/reset conditions. C checkpoint/output paths are also fixed to old `ppo_semantic_v3`. Its older README's fresh-A recipe is not applicable to the latest allowance and should not be treated as a rerun instruction. |

Merely using the existing neutral title/filter is insufficient: `comparison_filter` calls `frames_for_episode`, which assumes184 success-post ticks for both sides. Preserved A actually has989 frames; that formula would expect1012. Do not fabricate the23 absent post-roll frames or relabel incomplete A as completed success. The draft helper's count based on each source's actual performed post ticks and explicit `SOURCE ENDED` hold is the reusable, non-retiming primitive.

A's stored decode/PTS record reports989 frames at15fps,65.933334s; container duration286331153s is invalid. Existing source technical validation deliberately permits bad container duration, while produced publications/comparisons require a sane container duration and full output validation. Stored decode evidence is not certification of a future rendered file. No decode, remux, repair, replay or A capture was performed here.

Conclusion: no existing public function can currently combine these exact sources truthfully without a separately reviewed, narrow **historical-reference handling** change to the existing artifact helper. Keep C's complete success/load/physical/technical checks intact; retain A's actual failure, legacy initialization and evaluator provenance; label the comparison **UNPAIRED REFERENCE — A INCOMPLETE / C actual outcome**, not a success pair or improvement. Use actual per-source frame counts and a visible end-frame hold, never cropped/retimed physics. This check implements nothing and is never an optimizer gate; original A can remain linked as a clearly caveated diagnostic meanwhile.
