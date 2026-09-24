# Inactive learned-branch v3 migration candidate

Only draft text/code was written under this directory. No production edit, Torch import, test, model forward, publication or simulator was run. The same-version deterministic evaluation continues independently. `CONTROL_REVIEW_COMPLETE=False` deliberately blocks construction of a formal migration plan until the final physical-evidence patch is reviewed.

## Exact sealed source — retain all new learning

Source: `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000221568.pt`.

- Checkpoint SHA256: `1827b5d59935b31e1e27cc7ae0c364dd0203a9d7f77a7d9e9502189bbd31c7d5` (root sealed-run evidence).
- Sidecar SHA256: `b27d373edb11ed5d9455bc3d6b32dda4657c9b08b08f91d9d42de0d425686b75` (independent PowerShell read/hash).
- Runtime: `44219b4fdc4d36d33be489b833c03b897766045b`; counters221568/1696/33920, effective LR1e-5, official save/load true.
- Original419 origin remains220544/1688/33760; branch learning retained1024/8/160. Do not substitute initial220544 or main222080. No borrowed1536 or historical640 credit.

Current branch name is retained; a semantic successor does not rename or fork this already learned output branch. The exact source/sidecar pins prevent a silent source choice. Future target HEAD is not fabricated; actual clean frozen runtime is supplied at legal publication.

## Small implementation surface

`semantic_rear_live_swing_migration.py` is the new module draft. It reuses `_preserved` and `_verify_identity` from the existing recapture module, the official loader/saver and source-device runner construction. It does not rewrite the old v2 receipt, factor or strict validator. `live_swing_integration.patch` contains the few namespace/CLI/generic-load/normal-save carry hooks and the two explicit config mode changes. The control patch is owned independently by handoff_review.

Candidate control mode: `rr_live_swing_evidence_v3`. Confirmed proposed control paths are `semantic_rear_policy_timing.py` and `semantic_supervisor.py`; backend imports the registered mode tuple and currently needs no change. Actor/profile codec/distribution/reward-config/caps/nominal files are not allowed runtime deltas. No media wildcard or arbitrary old-source waiver is added. If the actual final fix differs, root must review and amend the narrow draft scope before publication.

The current RL qualification latch must be reset by actual grounding and made visible in the existing RL active-lift observation slot, as proposed by the physical-evidence review. Thus419 dimension/order remain unchanged, but current RL evidence, dependent rear flags and task-potential semantics are explicitly versioned. No old placed/crossing history alone can certify new current AIR. Exact source/target numeric-input Gaussian mapping remains identical; the same physical trajectory need not produce identical encoded observations or actions.

## Lineage and state preservation

New metadata key: `rear_live_swing_migration`; schema `wlr50_clean.rear_live_swing_same419.v3`; factor `rear_live_swing_same419_factor`. Its revision origin is the learned source221568/1696/33920; the original branch origin is untouched. All migration-added PPO/AUX counts are0. Normal save carries the new receipt and the old recapture/append receipts unchanged.

To validate the historical v2 receipt without copying another large runtime, `_previous_contract` reverses the exact bound changed-file hashes and the two changed configuration bindings from the current contract. Source HEAD/content digest plus the reconstructed full contract digest must agree. It then calls the unmodified old namespace/recapture validator. This also fails on unrelated contract-field drift. The current output route must equal its source digest exactly.

Publication builds the same419 actor/critic on the recorded source device, loads all weights/buffers/full Adam/options/steps/effective LR/Identity/full RNG, checks empty rollout, saves a unique file within the existing branch history and independently reloads it. No main or branch pointer is promoted by the publisher. Suggested eventual unique filename: `checkpoint_rear_live_swing_CP221568_g<actualHEAD12>.pt`; not created now.

## Necessary bounded verification after authorization

1. Synthetic populated419 state: zero-learning migration publication and independent official load preserve every actor/critic tensor, full Adam/LR/Identity/RNG/counters/origins/AUX/route; empty rollout. New receipt only, no source overwrite.
2. One synthetic128 PPO update and official normal save carry both old and new receipts, count128/1/20 only, and use the same branch/prefix resolver; main pointer stays unchanged. Synthetic work gets no robot credit.
3. Reject wrong source/sidecar, initial220544 rollback, foreign/main route, tampered parent receipt, missing/extra control/config changes or changed419 policy/kernel. Reconstruct predecessor contract exactly and reject stale current runtime.
4. Test physical producer/consumer together using the independently prepared RL lift→AIR→ground→new-attempt positive/negative cases. A missing current-evidence field must not be hidden by substituting historical active_lift.

Do not apply, execute the draft, set the review flag or publish until root confirms the evaluation has naturally sealed and the actual control patch is accepted. No additional success gate is introduced.
