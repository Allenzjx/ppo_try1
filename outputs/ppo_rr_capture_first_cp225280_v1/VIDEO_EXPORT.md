# RR-first local video export

This adapter exports one sealed direct-recorder evaluation from the isolated
`ppo_rr_capture_first_cp225280_v1` route. It does not treat the direct manifest
as an old semantic-video `official_load` receipt. The source/run manifests,
the actually reloaded composite checkpoint and its sidecar are explicit CLI
bindings.

The left comparison is the accepted original49eb CP225280 evaluation. It is
not `N_ref`. A zero-update diagnostic must pass `--allow-diagnostic` and is
labelled `DIAGNOSTIC`; it is not formal PPO success. A first formal 2048-local
checkpoint displays as `CP227328+local2048`, which remains distinct from the
older soft-KL checkpoint with the same numeric global step.

Run only after the direct source and its parent `run_manifest.json` are sealed:

```powershell
$repo = 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
$python = 'C:\Users\kskzz\miniconda3\python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
& $python "$repo\outputs\ppo_rr_capture_first_cp225280_v1\export_rr_capture_first_video.py" `
  --source '<SEALED_SOURCE_ABSOLUTE_PATH>' `
  --destination "$repo\outputs\ppo_rr_capture_first_cp225280_v1\video_review\<UNIQUE_REVIEW_NAME>" `
  --source-manifest-sha256 '<SEALED_SOURCE_MANIFEST_SHA256>' `
  --source-run-manifest-sha256 '<SEALED_RUN_MANIFEST_SHA256>' `
  --expected-head '<FROZEN_RUNTIME_HEAD>' `
  --checkpoint '<ACTUALLY_RELOADED_COMPOSITE_CHECKPOINT>' `
  --checkpoint-sha256 '<CHECKPOINT_SHA256>' `
  --checkpoint-manifest-sha256 '<CHECKPOINT_SIDECAR_SHA256>' `
  --expected-local-policy-decisions <0_OR_ACTUAL_LOCAL_COUNT> `
  --expected-local-ppo-updates <0_OR_ACTUAL_LOCAL_UPDATES> `
  --expected-local-optimizer-steps <0_OR_ACTUAL_LOCAL_ADAM_STEPS>
```

Append `--allow-diagnostic` only for a sealed
`INDEPENDENT_DIRECTION_DIAGNOSTIC` source. The adapter refuses a formal source
when that flag is present, and refuses a diagnostic source when it is absent.

## HUD command lineage

Every encoded endpoint joins three sealed records by the episode tick:

- `video_policy_decisions.jsonl`: actor selected raw/tanh, issued raw and the
  completed step's projected physical request;
- `native_tick_audit.jsonl`: the actually dispatched source command, mapped N,
  raw native clock and verified actuator mapping;
- `capture_assist_ticks.jsonl`: same-dispatch FINAL, measured actual state and
  contact evidence.

Actor raw/tanh is dimensionless. `projected_residual_full12` and
`independent_policy_residual_requested_full12` are physical-unit REQUESTs;
they are never compared as if they were raw policy latents. A diagnostic raw
override is shown separately with PPO credit zero. The capture row's top-level
`nominal_full12` is a post-step source hint, not the command dispatched for the
just-completed interval, and is labelled accordingly.

Phase evidence also remains clock-specific. `capture_assist_ticks.phase` is
the endpoint phase. `native_tick_audit.native_audit.source_phase_id` must match
the completed step's actuator-audit source phase, which can already be the new
phase when a transition occurs inside the eight-tick decision. Its
`policy_request_phase` must instead match the decision-start request phase.

The HUD's `training ledger` and `training-prefix` values are cumulative values
loaded from the saved composite checkpoint. They are not counts collected by
the deterministic evaluation episode being shown.

## Explicit v1/v2 families

Old sources keep their original447/v1 interpretation. New448 sources must
match this complete family, not merely an accepted observation dimension:

| Binding | Old v1 | New v2 |
|---|---|---|
|Direct video schema|`wlr50_clean.frozen_prior_rr_capture_video.v1`|`wlr50_clean.frozen_prior_rr_capture_video.v2`|
|Checkpoint schema|`wlr50_clean.frozen_prior_rr_capture_checkpoint.v1`|`wlr50_clean.frozen_prior_rr_capture_checkpoint.v2`|
|Task schema|`wlr50_clean.rr_capture_local_task.v1`|`wlr50_clean.rr_capture_local_task.v2`|
|Request schema|`wlr50_clean.actual_rr_capture_local_request.v1`|`wlr50_clean.actual_rr_capture_local_request.v2`|
|Actor policy|`frozen_cp225280_rr_capture_local_history_v1`|`frozen_cp225280_rr_capture_local_history_v2`|
|Profile policy|`frozen_cp225280_raw_head_plus_local447_v1`|`frozen_cp225280_raw_head_plus_local448_lineage_v2`|
|Actor layout|`role439_rr_capture_local_v1`|`role439_rr_capture_local_v2`|

The pure-standard-library family table is shared from
`export_formal_from_sealed.py`; the media adapter reuses it. Unknown families,
mixed task/request/checkpoint versions, changed runtime/hash bytes and
`legacy447_migration_only=True` formal actor configurations are rejected.
Each decoded decision must match its sealed source family; the wrapper cannot
silently reinterpret a v1 request under a v2 video label.

The v2 contract additionally binds both runtime profile and visible control
contributions to
`rr_local_defer_p09_late_and_new_p12_until_terminal_v2`. This is the declared
source-clock permission revision for this local RR task: pending P09 late
and initial new P12 unload are deferred. It is not a rear task completion
helper. Original FL assist remains ON; rear capture/geometry/wheel/owner
helpers remain OFF. No exporter changes robot state or model behavior.

For v2, the ninth task feature (`obs9[8]`, full observation column447) is
`current_attempt_capture_eligible`, consistent with endpoint metrics. The HUD
separates request-start qualification from endpoint qualification; a physical
event may change it during the eight-tick decision. Current TOP/bearing and
historical placed remain distinct. Claimed v2 local success requires current
eligible, non-GROUND TOP bearing in the sealed terminal evidence; old placed
or crossing history cannot substitute. The exporter does not relabel a
physical failure or alter the task's hold threshold.

The v2 HUD separately prints cumulative `local_*` training and
`task_v2_policy_decisions` / `task_v2_ppo_updates`. A weights-only447→448
migration can therefore show2048 inherited local decisions but **zero v2
training**. Both remain distinct from this deterministic evaluation's0 updates
and the uncredited physical prefix. The source-dispatch revision is visible
and recorded in the export receipt; controller improvement is not silently
claimed as new PPO learning.

Use the sealed wrapper for a formal source (defaults to validating/printing;
`--execute` is required to encode):

```powershell
& 'C:/Users/kskzz/miniconda3/python.exe' `
  outputs/ppo_rr_capture_first_cp225280_v1/export_formal_from_sealed.py `
  --source '<SEALED_V1_OR_V2_SOURCE>' `
  --destination '<NEW_REVIEW_DIRECTORY>' `
  --execute
```

No Torch/PXR/Isaac import is used by either tool. The media adapter uses PIL
and FFmpeg for actual frames. Standard-library wrapper tests and the bounded
PIL endpoint/HUD tests preserve old v1 data, exercise v2, and reject mixed
families, altered checkpoint bytes, missing qualification, wrong source rules
and diagnostic overrides masquerading as formal policy actions.
