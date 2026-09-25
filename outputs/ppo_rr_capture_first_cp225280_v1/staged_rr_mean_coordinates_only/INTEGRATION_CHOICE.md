# Cold integration — historical proposal; subsequently applied

Status2026-09-25: after the referenced0ff CP228864 DET sealed incomplete, root
applied production0d0f8948999222f9b8710b0eb913a9419537be83 and gain-aware output
validation at a cold boundary. Actual complete CP228864 was strictly migrated
and reloaded; initial mean/logp equivalence and actual Adam mapping passed.
The rest of this document retains pre-implementation decisions, not live status.
See parent RECOVERY.md for exact artifacts, zero-credit migration and fresh run.

Nothing here is applied or executed. No new live files/checkpoint were read.
The future CP228864 is an explicit **expected source**, not a claimed artifact.

Small integration consists of:

1. Apply the staged actor patch and `route_config_aux_guard.apply_patch.txt` only
   at a normal cold version boundary, if root decides the DET/update evidence
   warrants this experiment.
2. Copy the final reviewed `rr_mean_coordinates_candidate.py` as the new tracked
   `src/wlr50_clean/ppo/semantic_rr_mean_coordinates.py` using the normal patch
   workflow. Its stdlib validators and coordinate helper form one narrow module.
3. Commit the new runtime. Use `publish_rr_mean_coordinates_cold.py.txt` as an
   independent cold publisher, not a new route mode or a chain of migrations.

Exact old source is HEAD `0ff03eafeeb75ba8505a99478cea2a6243cf93f5`, with sealed
CP228864/local3584/PPO7/Adam140/AUX64. The invocation must supply actual checkpoint,
checkpoint SHA, sidecar SHA, COMPLETE source-run directory, new expected HEAD,
new isolated publication directory, and `--isaac-stopped`. No automatic live-pointer
lookup or old-CP fallback. If actual next source differs, this draft refuses it;
root must explicitly revise the binding based on the real sealed state.

Source whitelist permits exactly four changed files: local route, local actor,
local AUX helper (identity-only recipe guard), and local_training.json; exactly
one new file: semantic_rr_mean_coordinates.py. Other code, selected control configs,
libraries, physics, HISTORY and task semantics remain hash-identical. Local config
may only add the twelve gain values; old0ff must lack that field and only means
identity. A new runtime or checkpoint may never omit it.

`make_runner` explicitly passes the runtime setting to actor. A narrow
`coordinate_source_config` plus exact `coordinate_source_head` constructs the
old identity actor using its complete saved config; every other config field
must match. Normal training/evaluation uses the explicit gain10 default from the
new settings. `load` and `save` compare runtime, constructed actor and saved runner
configuration, rejecting mixed units even though state_dict keys are unchanged.

Publisher checks source CP/sidecar/payload/state/step/LR and COMPLETE run binding,
then uses ordinary strict `load(..., old_runtime)`, never a relaxed rebind. The
existing helper transforms the same real Adam, appends a disclosed coordinate
receipt, retains AUX ledger64/counters, and saves a unique new-git-version name
without moving the latest pointer. A newly constructed gain10 runner must strictly
reload all state/RNG/lineage before the pointer moves. Existing old0ff code, CP,
videos and artifacts remain untouched for rollback; this does not broaden the new
ordinary loader to execute older incompatible code automatically.

This rollback statement means running the retained old source/configuration with
its matching old checkpoint. It does **not** mean the old `initialize`, `migrate`
or `rebind` CLI branches remain executable with the new gain10 default. The staged
binding deliberately requires a coordinate receipt before gain10 load/save:

- `initialize` has no receipt and would currently fail at save.
- Legacy447 construction itself remains identity: the saved legacy configuration
  replaces the generated gain10 configuration. But the existing447-to448 migrator
  neither transforms destination coordinates nor creates the required receipt;
  its active-output equivalence check and gain10 save are not a valid route.
- The existing `rebind` source/config whitelists do not authorize0ff-to-gain10.
- Normal train/eval/diagnostic are only for a checkpoint already published through
  the independent cold coordinate publisher and then strictly reloaded.

The minimal integration guard is now included in the staged route patch:
when current runtime gain is nonidentity, reject initialize/migrate/rebind and
require an explicit checkpoint for train/eval/diagnostic, before run directory
creation, Torch imports or Isaac startup. Error text should name the independent
cold publisher and matching old-source rollback option. Do not silently fall back
to initialization or combine the two migrations.

The identity-only AUX guard rejects reuse of the old64/.005 SGD recipe on gain10.
It does not delete or reinterpret its previous receipt or64 updates. New coordinate
migration gives zero PPO decisions, PPO/Adam updates and AUX steps. Fresh PPO and
new deterministic physical/video evaluation remain necessary; numerical equivalence
at migration is not a learned or physical success.

Root's remaining choices: whether DET warrants gain10 at all; approving the exact
source patch/hash set and source CP after it exists; applying the staged CLI/video
coordinate checks together with that optional version change.
The raw/action/observation v2 schemas remain448/12, while the new committed runtime,
actor configuration, request gain field and coordinate receipt distinguish this
parameterization. No existing AUX recipe or actor mean is reset.

Video review finding: producer records the complete audited policy request, so its
new gain field is preserved. Existing exporter accepts extra request fields but
does not check this one. A stdlib-only synthetic probe accepted both the intended
gain10 vector and an inconsistent `[99]*12`; Torch/PXR were not imported. Existing
source/checkpoint runtime/hash checks remain useful but do not establish equality
between runtime gain, actor config gain and each request gain. Therefore this draft
must not claim strict gain-aware compatibility for the current unchanged exporter.
The new isolated draft adds explicit producer disclosure and a shared check binding
runtime, actor config, control disclosure, coordinate receipt and each request. Missing
gain may retain identity meaning only for the already supported immutable old
source families; new gain10 artifacts must carry and match it. No broad release
gate or new physical run is needed for that metadata fix.

Prepared additions, not installed:

- `route_config_aux_guard.apply_patch.txt` now contains the early CLI rejection,
  evaluation binding check, settings-derived gain and migration ledger in
  `control_contributions`. The source whitelist still has the same four changed
  files; no extra production module or mode was added.
- `coordinate_video_validator_candidate.py` is an independent stdlib helper. If
  selected, copy it as `outputs/ppo_rr_capture_first_cp225280_v1/coordinate_video_validator.py`
  and apply `exporter_coordinate_binding.apply_patch.txt`. Current output tools
  are untouched; the next0ff DET still uses its existing exporter.
- Missing gain is identity only for four exact existing source HEADs: ecf205e v1;
  5f8487b/1e10d39/0ff03ea v2. Unknown future missing fields are rejected. Current0ff
  needs no new fields. New gain10 requires explicit runtime, actor, video control
  and every-decision gain, plus identical source/checkpoint migration ledger and
  the hash-bound passed receipt bytes. Migration gives zero new learning credit
  and unchanged counters; subsequent actual PPO counters may advance. Existing
  checkpoint/run/runtime/media hash validation remains mandatory. This does not
  independently prove the receipt's numerical claims or physical success.
- Per-row checking uses `context['checkpoint']['version_family']['mean_coordinates']`,
  from actual checkpoint validation, not the bare schema family. It covers active
  and inactive rows, formal and diagnostic alike.

`test_coordinate_video_stdlib.py`: six tests PASS. Covers gain10 and current0ff;
missing/mixed runtime/actor/control/request gains; wrong destination/credit,
missing ledger, changed or failed receipt; early CLI rejection; in-memory patch
syntax. Only temporary synthetic receipt files were used, no live reads/Torch/PXR.
Tests neither deploy source nor run the publisher.
