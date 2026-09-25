# CP225280 front-preserving branch: bounded recovery recommendation

Status: isolated implementation draft only. No production file, active Isaac process, existing checkpoint/pointer, or optimizer state was changed. Checkpoint binaries were streamed only for SHA256; no Torch/PXR/model was loaded.

## Recommended source, and what the physical evidence actually proves

Use the existing **f6d439 CP225280 publication**, not original422 weights forced into a current439 actor and not a relabelled CP229xxx/65a receipt:

`outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_rear_owner_CP225280_gf6d1d2df8d87.pt`

- Checkpoint SHA: `fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f`.
- Manifest SHA: `f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d`.
- Runtime: `f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b`; content `d1561db383c0b359526889f3a2844dbcf97b2ef60a9f30741ecd9c8767a43dba`.
- Counters: 225280 decisions /1725 PPO updates /34500 Adam minibatch steps; effective LR `1e-5`, Identity normalizer,439 inputs,12 outputs, historical128 collection.
- Actor hash `48d70c95eb271325b0064cd33b91e04f76942455ff1eff026356995494b427e7`; critic `e95dc6b76d96b1fce866840060362f7cb6f33e247218d711863e851bf892ad1d`; Adam `e4faa601952d96f6ebde5c9762f545742066235782edd610682ccbb5f29c8bcb`.

Original `checkpoint_step_000225280.pt` (SHA `21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf`, manifest `6e05b4e279c1d8ac99447f91d69b7db3f6dd6afaab6a326e610283f783b1f185`) is422 under `49eb23163a6e20bc56301dbafb59b137ecebce66`. Its accepted natural deterministic video source is `runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T0214316632402Z_g49eb23163a6e_530cc3c96d834dc3ba84b0b19d8db4f1/source`: FR placed23.008333s, FL placed43.041667s/P06 at43.066667s; RR capture subsequently incomplete. Those are actual front results, not complete obstacle success.

`publish_rear_owner.py` already published422→439 with all old422 actor/critic columns and full Adam state preserved; new17 first-layer columns and corresponding moments are zero. Publication has zero learning credit and an independent reload receipt. This preserves the same-numeric-input conditional mean/HISTORY/σ, not a claim of identical future physics.

The independently useful good-front student probe `runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe/20260924_owner439_student_entry_v2` actually used **b0ff439 CP225280**, SHA `e1c1d5e200f5d471fa4b32b4ac82ae41d7a8c42e13d0415b403318a6ca0dec74`, manifest `4a3ac8bf48cf5b914cc7e7c293fa08702b18f86019e97a50d1b3f78f3063ac02`. b0ff and f6d manifest actor/critic/Adam/normalizer/LR/RNG/policy/runner are equal. Their only runtime differences are `semantic_checkpoint_prefix.py` accepting439 and its migration allowlist entry; natural P01 execution is not changed by that suffix-prefix acceptance. Its pre-intervention P05 age2/4/6s observations therefore support f6d as a compatible candidate; the later probe intervention does not become policy success evidence.

## Front execution compatibility, not just weights

f6d→59e has exactly10 changed runtime files: reward, rr-retention migration, video CLI, policy metadata resolver, training, rear-policy route validator, return-profile collection handling, front-retention metadata, CLI, and migration dispatcher. The policy resolver diff adds only collection512 constructor selection.

All selected configuration hashes are unchanged f6d→59e, including action schema, execution profile, observation schema, task spec, reward YAML and curriculum. Actual N/mapper/residual adapter/FL assist, observation encoder, rear-owner actor, HISTORY and Gaussian kernels are unchanged. Source439 policy uses `(1-rho)*base_mean + rho*observable request/raw history`; history request scales `[4,4,4,6,4,4,4,4,.12,.12,.12,.12]`, temperature `.25`, full12 channels. Existing FL capture assist remains declared ON; rear completion assists remain OFF.

Physical environment state is not saved. Natural P01 regenerates the actual contact/action history. Loading compatible HISTORY semantics does not restore an arbitrary prior pose or guarantee the old43s FL success.

## Why a new identity is necessary

Original owner publisher requires exact225280 source and only its reviewed422→439 file set. Current59e adds files outside that set. The later reward migration requires a completed f6d learned update (the zero-update439 publication retains its pre-append last-update actor hash);65a identity additionally requires its own latest72e pointer; collection512 requires the latest complete65a lineage and immutable AUX ledger. None is the right authority to rename old CP225280 as their descendant.

Original route validators also bind `ancestor220544_recapture_v2`; arbitrary sibling CLI paths cannot be used without a new explicit identity. Do not edit old manifests/receipts, swap latest pointers, or relax their validators globally.

Preferred path: publish new `cp225280_front_preserved_v1` identity from the exact f6d source, retaining the original full manifest/receipts nested and old historical receipts at top for existing counters. The new validator separately validates original source under its old contract/route, then validates target439/512 and its own counter origin. Alternative original422+49e isolated runtime is possible but loses newer rear-owner semantics and is not the minimal useful continuation. Neither option overwrites the active/latest branch.

## Draft API and executable preconditions

Implementation: `tree/src/wlr50_clean/ppo/semantic_front_preservation.py`.

- `build_front_preservation_plan(checkpoint, contract, *, expected_source_sha256, expected_manifest_sha256, reason, front_replay_spec, project_root=None)`.
- `publish_front_preservation_checkpoint(checkpoint, contract, plan_path, output_checkpoint)`.
- `validate_front_preservation_lineage(metadata, contract, output_root, *, checkpoint_output_routing=None)`.
- `build_front_preservation_output_routing(metadata, contract, destination)`.
- `front_preservation_collection_options(metadata, *, experiment_id)`.
- `front_preservation_branch_counts(metadata)`; `expected_front_replay_counts(ppo_updates)` is expected accounting, not an assertion that replay actually ran.

Root integration must dispatch this identity before old namespace/collection checks; preserve identity and measured replay counts on every save and invoke its validator during ordinary load/save. Existing historical source loads still use their original validators. No branch-wide bypass is needed: publication writes the new branch directly.

Before any publication: let sole Isaac seal its requested complete update; freeze the integrated new HEAD and actual runtime contract; bind real replay dataset path/hash and spec `schema=wlr50_clean.front_replay_regularizer.v1`, optional version `cp225280_front_gaussian_kl_v1`, coefficient1.0, minibatch_size32. Replay helper separately validates real front-only data and train/heldout partition semantics.

The allowlist is fixed:10 existing f6d→59e files plus this module and `semantic_front_replay.py`. Every other file/config is immutable; reward code is specifically pinned to existing72e hash `7d06694bc5ec1abf24d8a55795fd647e59822d1cdbcdb271b5c4985013ef43c3`. Plan requires new frozen HEAD, full inventory digest, current bytes and source hashes. It cannot run against unchanged59e missing these new files.

Publication constructs original128 and target512 through official factories, loads original checkpoint officially, transfers full actor/critic/Adam/LR/Identity/RNG, requires empty512×1×439 storage/512×1×12 actions, saves a unique file, then independently reloads through the ordinary target loader. Adds0 decisions/PPO/Adam/AUX; writes no pointer. Later updates must satisfy +512 decisions/+1 PPO/+20 official Adam and actual replay accounting +20 minibatches/+640 row exposures,0 extra PPO samples/0 separate AUX steps.

Publication also evaluates the immutable real front fit/heldout observations with the shared pure HISTORY/conditional-Gaussian kernel, both before save and after reload. Its receipt reports partition counts and max absolute conditional mean/sigma error against the recorded CP225280 reference, with2e-6 float32 batch-kernel tolerance; full actor/critic/Adam state, RNG and the original distribution-cache object/content must remain unchanged. No replay loss, optimizer, teacher or physical success credit is applied.

Descendant validation checks actual last-update replay report:20 minibatches,32 immutable-dataset rows each, correct deterministic cyclic/source indices, real per-phase counts restricted to P01–P06, one-use gradient evidence,0 extra draws/optimizer/PPO credit, aggregate640 exposures and bound spec. It does not gate on improved KL or a manufactured success score.

These are necessary compatibility checks, not a new physical success gate. After publication/reload, run natural P01 promptly to verify retained front behaviour; retain any failure tail. Replay fit and controller identity cannot substitute for physical FR/FL→P06 evidence.

Tests: `test_front_preservation_stdlib.py`, source-lineage calls explicitly mocked, no real model/tensor reload claimed. Targeted cases cover source snapshot, sibling route,439/512 recognition, protected N/mapper/physics/config, reward identity, immutable dataset, no borrowed65a/AUX lineage, zero-state identity,512/20 counters, replay credit and no Torch/PXR import.
