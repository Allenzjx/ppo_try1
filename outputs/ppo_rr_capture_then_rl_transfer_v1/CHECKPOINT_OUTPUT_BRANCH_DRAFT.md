# Unapplied checkpoint-output branch implementation draft

Status: TEXT-ONLY proposal while the v5 video is active. No production file,
configuration, checkpoint or pointer has been changed; no Python/tests were
run for this draft. This is not approval to train the ancestor candidate.
Its existing control-evaluation-only provenance remains unchanged.

## Existing supported mechanism and missing entry point

`semantic_training.train_semantic(..., output_root=Path, resume_infos=...)`
already saves `output_root/checkpoints/history/checkpoint_step_NNNNNNNNN.pt`
and publishes last/copy/sidecar exclusively beneath that output root.
`save_semantic_checkpoint` rejects an existing immutable checkpoint.

The CLI and PowerShell wrapper do not expose this root. `_request_paths`
currently selects one experiment root, and `--run-dir` changes only logs.
Do not monkeypatch `_request_paths`, change experiment IDs, copy/relabel an
old checkpoint sidecar, or use NewMdpWarmStart to evade this limitation.

## Proposed narrow interface

Add `--checkpoint-output-branch ancestor220544_v5` to `semantic_cli` and
`-CheckpointOutputBranch ancestor220544_v5` to `run_semantic_ppo.ps1`.
Keep the default absent behavior exactly unchanged. Initially permit only
the existing RR v3/N1 experiment with an explicit checkpoint and commands
`train` or residual `eval`. Natural P01 training and ordinary
`checkpoint_policy` suffix training retain the existing stage/prefix rules.

Use a lowercase single directory identifier, e.g. regex
`[a-z0-9][a-z0-9_-]{0,63}`; reject traversal, absolute paths, separators,
Windows reserved device names, reparse/symlink escapes, and an output root
equal to the experiment root. Resolve and verify containment under the
canonical experiment `branches` directory. This is not an arbitrary
`--output-root` escape hatch.

Canonical example (repository root is
`C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`):

```text
outputs/ppo_rr_capture_then_rl_transfer_v1/
  checkpoints/                         # main history and main last untouched
  branches/ancestor220544_v5/
    checkpoints/
      history/checkpoint_step_000220672.pt
      checkpoint_last.pt
      checkpoint_last_pointer.json
      resume_state.json
```

Keep `runs/ppo_rr_capture_then_rl_transfer_v1/...` and all configuration
paths unchanged. Log the explicit branch/root in the existing run manifest.

## Minimal code touch points

1. `scripts/run_semantic_ppo.ps1`: add one optional validated string
   parameter and forward it as `--checkpoint-output-branch`. Do not change
   locking, resource checks, run IDs, source checkpoint resolution or HEAD
   cleanliness checks.
2. `semantic_cli.py`: add the parser option and two small pure path helpers,
   `_checkpoint_output_root(args)` and
   `_checkpoint_source_root(args, output_root)`. Do **not** change
   `version_paths` or `_request_paths`; those must continue selecting the
   same namespace, config and runtime contract.
3. In `validate_request`, determine the selected checkpoint source root
   before invoking existing `_resolved_checkpoint`. A branch resume must
   not accidentally enter the older cross-experiment RR append-migration
   fallback. That fallback remains unchanged when no branch is requested.
4. In `dispatch_live`, use `_checkpoint_output_root(args)` only for the
   resumed training save destination passed to `train_semantic`. The
   backend, runner, legal reset, prefix binding, action loop and PPO update
   path remain existing production functions.
5. In `train_semantic`, accept an optional small `checkpoint_output_routing`
   record, validate it against `output_root` and inherited routing (when
   present), and copy it into ordinary checkpoint infos and the final run
   result. No optimizer, observations, rewards, counters or sampling rules
   change. The existing v5 migration receipt continues to carry separately.

Suggested routing record (artifact provenance, not a new policy branch):

```json
{
  "schema": "wlr50_clean.checkpoint_output_routing.v1",
  "branch": "ancestor220544_v5",
  "output_root": "<resolved canonical branch root>",
  "main_latest_pointer_promotion": false
}
```

Use this record's own key, `checkpoint_output_routing`; do not name it
`*_branch` with a counter_origin, which would invoke unrelated historical
training-branch counter validators. The already preserved v5 receipt has
the exact source selection, source hashes and learning counter origin.

## Source and pointer validation

Initial branch run: allow only an explicitly provided immutable checkpoint
under the parent experiment `checkpoints/history`, after the ordinary exact
checkpoint/runtime/full-state checks. Require the source's v5 receipt and
its source-selection role to be present. If the selected branch already
contains checkpoints/a last pointer, reject a new parent-source start and
require explicit same-branch resume instead of replaying colliding counts.

Subsequent run: allow only the selected branch's `checkpoints` subtree.
Pass that branch root to existing `_resolved_checkpoint` so its
`checkpoint_last.pt` copy, pointer hash, sidecar hash and immutable history
containment checks still run unchanged. Reject a source from a different
branch, wrong namespace or arbitrary external directory. Direct immutable
paths and the branch-local last pointer must both work.

Once `args.checkpoint` is resolved, existing checkpoint-policy prefix code
receives that exact file. Do not alter prefix generation, physical reset,
prefix credit exclusion, or the source actor binding. Residual evaluation
through `semantic_cli` can use the same explicit branch option. Any separate
video exporter/launcher path resolver must be checked before later use;
do not assume it accepts the new branch automatically.

Preserve main CP221184 and its pointer byte-for-byte. The ancestor starts at
220544/1688/33760; a real 128-decision update would produce
220672/1689/33780, and +2048 would produce 222592/1704/34080. These are
conditional arithmetic, not completed training. Never add the separate
CP221184 lineage's 640/5/100 as if the ancestor received those updates.

The existing main-history 220672/220800/220928/221056/221184 files already
exist. Branch output does not remove, overwrite or rename any of them.

## Runtime binding is mandatory before applying

CLI, training and wrapper edits change the frozen runtime fingerprint even
though output routing does not change the control MDP. Existing v5 strict
migration will reject that unreviewed change, correctly. Do not bypass it
with a changed expected hash, stale plan, monkeypatch or source manifest edit.

At the later authorized boundary, create a narrow explicit same410
artifact-routing migration (not another control/reward revision). Bind the
actual sealed v5 source checkpoint/manifest/HEAD and actual committed target
HEAD. Expected runtime delta is the three routing files above, the existing
`semantic_migration.py` route, and one dedicated small migration module.
All six config bytes, actor/codec, assist/context/controller, reward,
physics, raw-Gaussian/HISTORY/caps/sigma files remain unchanged. Require an
exact reviewed file/hash set, not a wildcard whitelist.

Preserve all source weights, critic, Adam groups/moments/step/effective LR,
Identity, full RNG, all previous receipts and AUX/origins; discard rollout;
add zero decisions/PPO/Adam/AUX. Preserve `checkpoint_output_routing` if the
source later contains it. Publish to a new unique filename with official
save and independent reload, without promoting the main pointer. No source
hash or target HEAD is invented by this draft. The existing candidate-only
selection receipt is historical provenance, not retrospective training
approval; an actual branch training run requires the root's later decision.

## Minimal tests after authorization, not run now

- Existing no-option parser, source resolution and output paths unchanged.
- Branch-name/path negative cases above, wrong source branch/namespace,
  parent-source restart into an occupied branch, and pointer/hash mismatch.
- Parent-source first resume and branch-source subsequent resume, with
  natural P01 and checkpoint-policy suffix configurations. Prefix samples
  remain uncredited and the exact source checkpoint is retained.
- Reuse `test_semantic_rr_capture_training_audit.py` CPU fixture for one
  actual synthetic 128-sample/1-PPO/20-minibatch update. Populate main-history
  collision files and main pointer sentinels first; assert all their bytes
  remain unchanged while only branch outputs appear. Do not count this as
  real Isaac training or task success.
- Official load/save/fresh-reload verifies branch routing receipt, existing
  v5 source selection, all AUX/old origins, actual count increments and raw
  Gaussian/logp integrity. Same-branch last pointer resolves to the verified
  immutable pair. Control config/source bytes are identical across the
  artifact-routing migration.

No generic experiment registry expansion, new training engine, model reset,
borrowed update credit, global best-model promotion or publication framework
is part of this proposal.
