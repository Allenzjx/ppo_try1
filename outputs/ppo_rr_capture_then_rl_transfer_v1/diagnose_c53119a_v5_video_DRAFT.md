# v5 sealed capture → follow analysis: text-only preparation

Prepared for the fixed `c53119ab332fe048668e66f0543889a128443ca5` video run ending in `886689c9a357416293bc8242856d0be6/source`. **Not executed or syntax-tested.** Do not execute until the root confirms natural sealing and Isaac process exit. Preparation reads writer code and a small historical source manifest, not live large JSONL streams. No runtime/config/test changes, Python/helper, model, simulator or encoder execution.

The new script is `diagnose_c53119a_v5_video_draft.py`. It reuses SHA-pinned standard-library functions from the already executed `diagnose_57ae41e_sealed_fixed_com.py`, without changing that prior helper or its results. It does not import any current production code or media dependency.

## Deliberate schema differences from headless

- Seal evidence is parent `run_manifest.json` + `source/semantic_video_source_manifest.json`, not headless evaluation_manifest. HEAD, natural P01 single-episode, zero optimizer updates, and actual official-loader checkpoint SHA are checked. Declared expected SHA: `308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895`. If the actual loaded CP differs, stop for explicit provenance review; a random matching nested ancestor hash is not accepted.
- Real decision rows are `video_policy_decisions.jsonl` envelopes, with actual `row.step_info` when `environment_step_returned=true`. A physically interrupted final video decision can have **no** step_info. The script preserves that row, uses real per-physics streams for its endpoint, and keeps issued versus completed evaluation decisions separate. No synthetic policy sample or learning credit.
- `capture_assist_ticks.jsonl` supplies every post-step current leg/contact/force, placed history, assist snapshot and task context. Native audit supplies the complete wheel projection/source-stop proof and the new separately replayed actual/zero-policy/geometry-zero audit states. These are recorded evidence, not newly replayed policy/controller operations.
- Optional `height_diagnostics.jsonl` supplies actual USD/live-parent RR hip mount when the sample is exact-tick, valid and clock-unchanged. A previous sparse sample remains separately timestamped; no interpolation or silent replacement. Upper-link origin is never relabelled as hip mount. Startup native joint names and full recorded native q/qd remain available through selected height rows.

## Selected response and task milestones

First P07/P09/P10/P11/P12/P13, real RR lift/cross/place history, first actual TOP/current TOP-bearing and subsequent loss/AIR, capture-assist modes including `DESCEND_PROGRESS` and `CAPTURED_FOLLOW`, first source/request delta consumption, actual release, first source-end wheel envelope/projection, and final two-second endpoints. Unreached events stay absent; RR or RL success is never predeclared.

Each selected point keeps source N, mapper N, REQUEST, mask, effective residual, captured target/owner, candidate and final targets, native targets, canonical actual q/qd and current four-leg contacts/forces. Source authored stop and scoped projection remain distinct. Exact decision metadata can be absent on non-15Hz ticks; later values are not substituted.

Two independent fixed references: first P07 mass COM0 → FL wheel center0; first P10 mass COM0 → FR wheel center0. No rolling re-anchor, and receiver displacement is separate. It includes world forward/lateral CoM displacement and body yaw change to expose the fact that a diagonal fixed projection can be dominated by forward traversal or altered body heading; it is not proof of load transfer. Missing P10 means no RL→FR reference.

RR wheel position is also transformed using the recorded base origin/quaternion into the **base body frame**, explicitly not a CoM frame. Wheel-minus-mass-CoM is a separately labelled world-axis vector. Hip/knee response and RR world gap/front changes are observations of the whole closed-loop system, not isolated joint derivatives.

Future authorized command (new destination only):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe outputs/ppo_rr_capture_then_rl_transfer_v1/diagnose_c53119a_v5_video_draft.py --owner-confirmed-sealed --output outputs/ppo_rr_capture_then_rl_transfer_v1/video_v5_c53119a_capture_follow_review
```

The reader performs one pass per specified JSONL, counts actual parsed lines, hashes read evidence against the sealed source artifact hashes, and rejects incomplete/changing/non-contiguous streams. Live Windows `Length=0` alone is never interpreted as absent logging. It produces diagnostic JSON/Markdown only, not video or task-success certification.
