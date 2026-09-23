# Next same410 carry/handoff migration — design only

No implementation, checkpoint publication, optimizer step or source selection is performed by this plan. The active block has a legal stop request; its final counts and checkpoint hash remain unconfirmed here. Use the actually sealed latest learned checkpoint under the e24a runtime, not the zero-update CP220544 publication. Do not pre-credit the expected stop quantity.

## Existing mechanisms and smallest reuse

The official loader already supports an exclusive reviewed same-layout observation factor, exact actor/critic/Adam/Identity/full-RNG loading, empty128×1×410/12 storage, and a post-load receipt hook. `semantic_migration.validate_migration_plan` dispatches by a named schema. Ordinary training has an explicit receipt carry list. These are usable without new architecture or a broad resume waiver.

The existing knee-v3 factor is **not** reusable as an authorization factor: it pins the exact26db zero-update source, its one-time new module and five-path delta, and its old profile transition. Rewriting those constants, impersonating its schema, or wrapping the new plan as a v3 factor would invalidate its meaning and ancestor evidence. Keep old v2/v3 modules and receipts intact.

Recommended small implementation after authorization:

1. Add one thin dedicated controller factor module with a new explicit schema/factor/receipt name. It owns the exact source-version predicate, final changed-file set, allowed profile/spec key differences and new public control meanings. It can reuse existing `_contract`, `_version_bytes`, checkpoint/manifest integrity, `digest`, `file_sha`, and existing complete `preserved_keys` utilities.
2. Put only a small non-authorizing exact-state verification utility in `semantic_training.py`, alongside its existing loaders. It checks loaded actor/critic hashes, full Adam/groups/steps/effectiveLR, Identity state, restored full RNG and empty storage against a **previously verified** factor. A dedicated hook validates its own schema, calls this utility, then appends its new receipt. This avoids copying the entire publication/loader implementation while leaving historical validators untouched. Do not let arbitrary plan keys or caller-supplied allowlists select authority.
3. Add the new factor to the existing same-layout exclusivity/observation-contract chain, exact-schema validator dispatch and ordinary-save receipt carry list. No CLI relaxation, new warm-start path or alternate PPO implementation is needed.
4. Use an outputs-only publisher driver built from the established official construct/load/save/fresh-independent-load protocol. It is not necessary to duplicate another large production publisher. It uses the actual source device and unique output names, never promotes latest until ordinary learning saves normally.

If runtime/config checking is factored into a small shared helper in `semantic_migration.py`, it must remain a byte/delta checker only: the dedicated module supplies a hard-coded reviewed set and exact allowed semantic differences. The plan itself must never authorize its own paths or arbitrary JSON changes.

## Source and target bindings

After the stop is sealed, bind the source checkpoint and sidecar SHA, embedded runtime contract, actual global decisions/PPO/Adam counts, effective LR and complete v3 receipt. Require the v3 receipt's target runtime/contract to match the source's e24a runtime. Unlike the previous zero-update boundary, source counters may now exceed the v3 migration origin; require internally consistent real branch deltas instead of falsely demanding zero previous learning. The target HEAD and target source bytes are bound only after the approved implementation is committed and frozen.

Candidate behavior paths are `semantic_residual_adapter.py`, `semantic_backend.py`, `semantic_rr_capture_context.py`, `semantic_supervisor.py`, and the selected `execution_profile.yaml`. Add the dedicated new module and existing migration/training hook paths to the exact runtime delta. This list is provisional until the actual implementation is reviewed, not an already granted broad whitelist. If independent counterfactual reconstruction requires `actuator_target_effect.py`, or an explicit timeout opt-in belongs in `stage_task_spec.yaml`, include that concrete path/key deliberately; do not silently accept it. No arbitrary other source paths, removals or extra modules.

Compare every selected config binding against runtime file hashes and read old bytes from the source git revision. Preserve all configs byte-for-byte except the finalized explicitly allowed profile fields (revision and wheel/timeout semantic markers), plus a narrowly approved task-spec marker if needed. Preserve all reward, action-schema, observation-schema, quality, physical limits, rate and sampling coefficients. For changed configs require parsed equality after removing only the exact permitted keys; retaining only matching hashes supplied by a plan is insufficient.

Use AST literal checks for finalized control-version constants and the unchanged14 RR feature names/scales, seven context names/order and410 policy contract. Literal checks establish declared semantics, not proof of control behavior; positive/negative dispatch and timeout tests remain necessary. Keep the assist's existing knee-v3 rules, full search budget and feedback reference unchanged unless a separately reviewed concrete delta becomes necessary.

## Public semantics, not a silent same-MDP claim

The existing observation shape may suffice: X409 `fl_wheel_guidance_active` becomes an actual shared post-policy wheel projection flag, and X408 `rr_capture_recovery_allowed` may include narrowly bounded current TOP/HOLD/handoff permission. Their locations and numerical encoding remain unchanged, but their meaning/use changes, so explicitly record observation/control/termination semantics changes. Preserve the numeric policy mapping for the **same numeric input**; do not claim the same physical state or trajectory gives identical inputs/actions. The canonical policy contract's legacy execution-description text stays an ancestor descriptor; the new factor must state the effective wheel/capture execution rules.

The factor should state `same_mdp_claimed:false`, `controller_transition_semantics_changed:true`, unchanged weights/kernel/sigma/caps/reward/physical dynamics, and raw Gaussian samples/logp unchanged by downstream projection. It must not call projected final targets Gaussian samples or use a residual mask to zero nominal. Shared real and counterfactual dispatch must apply the same explicit projection. The single existing observation flag is sufficient only if every action-deciding operand/state is already observable or deterministically derivable; an unobserved new timer/latch would block the no-new-dimension claim.

The proposed timeout change must not certify contact from history: fresh same-tick feedback plus actual valid TOP/bearing/XY/support is required, with the existing cumulative HOLD clock bounded to10/120 seconds and a narrowly defined next-decision handoff. Preserve global200 seconds and physical failures. The synthetic old-placed-AIR case already has `placed_RR==1`; `all goals ==1` alone must not authorize a new contact-hold exemption.

## Identity publication and persistent lineage

Preserve all compatible actor/critic parameters and buffers, all Adam states/groups/step counters and actual adaptive LR, Identity normalizers, full CPU/CUDA RNG, every branch origin/count, complete AUX objects and the entire prior v2/v3 migration receipts. No head reset, zeroing, partial optimizer restart or 389→410 reappend. Discard unconsumed rollout storage and collect fresh observations under the new controller version.

Append a new migration receipt with actual source counter origin and zero added decisions/PPO/Adam/AUX. A new training branch is unnecessary; keep the RR branch's original origin. Carry the new receipt through ordinary later saves along with all ancestors. Store compact preservation digests and source bindings rather than copying historical fit reports into yet another new ledger. New checkpoint filenames derive from the actual sealed global decision count and actual target HEAD; no source hash, future count or future commit is supplied by this draft.

Before physical use, official source-device save and independent fresh reload must prove exact preserved state and zero new learning. This is a migration, not PPO. Only subsequent sealed128-decision updates add PPO/Adam credit. Publish/reload under the new runtime before a frozen checkpoint-policy prefix; keep the existing strict prefix source/effective/target binding, without a waiver.

## Bounded verification at the legal boundary

- Reject extra/missing runtime paths, altered protected configs, wrong source receipt/runtime, changed410 field order/scales, mixed factors, reused migration receipt and an existing target output.
- Synthetic same-input Gaussian μ/σ and critic outputs remain exact; complete Adam/LR/Identity/RNG and all ancestor/AUX objects remain exact across official publication/reload; storage is empty.
- Verify real and counterfactual wheel dispatch, explicit ownership/flag, nominal preservation and legal stops; do not alter an independent safety stop.
- Reuse the existing synthetic TOP1→TOP2/placed→next-decision counterexamples, plus current ground/AIR, stale feedback, exhausted cumulative HOLD and global deadline negatives. No historical outcome edits or fabricated contact.
- Check one later ordinary saved PPO update carries the new receipt/ancestors and actual raw Gaussian credit correctly. Do not require a repeated full-task success gate before learning.

This document is only a proposal. Root selects the finalized control delta, sealed source and legal publication/training boundary.
