# Isolated P06 exhausted FL capture-owner retirement candidate

Status: **not applied; not approved for physical deployment; no model loaded**. Production HEAD remains `72e63592bdf412d375abfda29b03cf456fbf4f7e`. The active formal video is untouched. This is a declared controller-ownership change, not learned behavior, a pose trajectory, or a reward change.

## Bounded observed first-layer difference

Source: `runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0938059083357Z_g72e63592bdf4_889c2d62e7fe433ea0dc3e99787a26ca/residual_and_projection_audit.jsonl`, global decision `228683`, episode endpoint physics tick `8792`.

The same recorded dispatch chain has pre-step episode observation tick `8791` and backend dispatch tick `8971`; the startup-offset clocks must not be conflated. Values below come from that row's actual `actuator_target_effect_audit.capture_assist_evidence`, not a separately recomputed nominal.

| FL channel, degrees | Actual composite before assist | After assist / final servo target | Assist correction |
| --- | ---: | ---: | ---: |
| hip | +36.6799207372 | +4.6197350870 | −32.0601856503 |
| knee | −43.1476619609 | −31.9856757962 | +11.1619861646 |

Owner indices are `[0,1]`; the other ten components are identical before/after assist, and `final_slew_or_clamp_indices=[]`. Helper state is `BLOCKED`, `finite_hip_travel_or_margin`, `hip_entry=24.619735°`, `hip_target=4.619735°` (original entry minus 20°), with prior contact seen and historical FL placed. The pre-step FL gap is `27.858858 mm`; current top contact and obstacle pair are false. The bounded endpoint report records FL AIR, zero bearing, gap `27.622 mm`; the other legs are FR TOP `12.206 N`, RL ground `13.246 N`, RR ground `3.567 N`.

This proves a substantial composite request is overridden. It does **not** identify the before-assist target as pure policy: it includes the actual mapped nominal/controller/residual composition. Nor does it prove useful FL preparation was suppressed. Releasing this row exposes a more-positive hip **and a more-negative knee**, not an already demonstrated positive-knee recovery. No inference of improved support, CoM motion, or physical safety follows from the target delta alone.

## Exact candidate

One production runtime path only: `src/wlr50_clean/ppo/semantic_capture_assist.py`.

The new transition requires all of: P06; physical evaluator valid; historical placed FL; current helper `BLOCKED` with reason `finite_hip_travel_or_margin`; explicit current FL AIR; explicit current FL support false; no current top/obstacle contact. Missing AIR/support data remains unknown and does not activate the rule.

At that transition, anchor the two release targets once to the **actual previous Full12 final target**, not nominal or an unclamped helper target. Reuse existing mode `RELEASE`, `release_fraction += dt / .75`, and eventual `RELEASED`; preserve contact/search history and all ten other channels. This begins with the same first 1/90 blend increment as existing P07 release. The existing final physical projection/slew and atomic write remain authoritative. Once RELEASED, the helper cannot rearm before ordinary episode reset.

P05 initial capture; P06 unplaced pending capture; current TOP/HOLD; other blocked reasons; and the existing P07 release are unchanged. A HOLD-to-AIR frame may enter BLOCKED under old search logic; it cannot take this new exit until a subsequent frame actually presents the required blocked state. No new XY, support-count, angle, or duration gate is added. This is not an authorization to load/unload other legs.

New `retirement_revision=p06_placed_air_exhausted_release_v1` is nonnumeric audit metadata; the previous recontact feedback revision remains accurate. Only two passive context fields expose existing evaluator facts (`current_FL_air`, `current_FL_support`). There is no new hidden latch, timer, observation column, mask, assist, or policy action.

## Existing observability and support risk

The present 439 layout already observes phase one-hot, FL placed history, verified wheel pair-active/force information, the twelve capture helper state features including mode/reason/release fraction/anchors, and final drive targets. See `semantic_observation.py:365–395,408–445` and the existing observation schema. `semantic_supervisor.py:815–829,1024–1038` derives AIR from both verified pairs inactive and computes support from physical bearing. Under that valid-sensing contract, FL AIR implies zero current bearing; the added context bits are not new information unavailable to actor/critic. The capture numeric feature names/scales remain byte-for-byte unchanged.

The guard therefore does not remove a currently bearing FL target on its initiating frame. Nevertheless, changing an AIR leg's geometry can perturb the floating body and other contacts; future front support can still worsen. The knee request in the logged example makes that risk concrete. The 0.75 s blend is a continuity measure, **not a physical safety proof**. As with the existing release, a contact returning during RELEASE does not rearm the helper; that behavior is deliberately not redesigned here. A later natural-prefix physical comparison is required before declaring benefit.

## Tests performed without model imports

- Eleven focused stdlib `unittest` cases pass: exact first-step blend and actual-final reanchoring; P05/unplaced/contact/HOLD/missing-data/safety negatives; other blocked reasons; pending capture; original P07 release; no rearm; replay using the same twelve numeric fields; invalid final targets; actual context separation of historical placement and current contact.
- Three old-versus-candidate synthetic traces, 900 ticks each, have exactly identical numeric state, receipts apart from revision metadata, and applied targets: P05 initial/recontact capture, P06 unplaced pending capture, and P07 release.
- Candidate/test AST parse and `git apply --check candidate.patch` pass. No Torch, PXR, Isaac, Omni, model, checkpoint, or physics was loaded. These are synthetic wiring/state-machine results, not physical success.

Commands (from repository root):

```powershell
& 'C:\Program Files\Python313\python.exe' outputs/ppo_rl_recovery_learning_v1/staged_p06_capture_retirement/assemble_candidate.py
& 'C:\Program Files\Python313\python.exe' outputs/ppo_rl_recovery_learning_v1/staged_p06_capture_retirement/check_candidate_stdlib.py
git apply --check outputs/ppo_rl_recovery_learning_v1/staged_p06_capture_retirement/candidate.patch
```

`candidate.apply_patch` is the explicit future apply_patch payload; `candidate.patch` is its standard unified-diff equivalent; `candidate/src/.../semantic_capture_assist.py` is the complete isolated runtime candidate. The intended normal test path is `tests/unit/test_semantic_p06_capture_retirement.py`. See `candidate_manifest.json` and `stdlib_test_result.json` for hashes/results.

## Application and runtime contract boundary — not implemented

Do not hot-apply this into the running episode/rollout. Although input/output dimensions, reward, raw Gaussian likelihood, weights and HISTORY are unchanged, the controller's state-to-applied-action mapping changes. It requires a declared same-shape **control** revision, not the existing reward-only migration and not a contract bypass.

The current `semantic_rr_retention_migration.py` accepts exactly its six historical f6d-to-72e reward files and claims `control_projection_changed=False`. It must remain an immutable historical receipt/validator, not be rewritten to accept this capture change. Ordinary resume into changed runtime bytes is also not a legal replacement.

If root approves deployment, the smallest compatible pattern is a new narrowly scoped outer control-identity migration module (not included here), plus routing in these existing paths: `semantic_migration.py` (dispatch/record), `semantic_cli.py` (explicit migration selection), `semantic_training.py` (load and every-save carry), `semantic_rear_policy_timing_migration.py` (outer namespace validation/history inversion). Together with this one helper file, that would be six reviewed runtime delta paths. Whether to combine it with another already approved boundary is root's later decision; no new generic framework or runner is warranted.

Source must be the actual latest complete compatible 72e immutable checkpoint and sidecar selected with explicit hashes, not a fixed older CP. Preserve actor/critic, complete Adam options/steps/actual LR, Identity normalizer, RNG, counters, existing rr-retention and owner receipts, and applicable auxiliary ledger. At the boundary retain weights but discard incomplete old rollout and start a fresh legal physical prefix/reset; do not claim bitwise simulator continuation. One-time publication changes no PPO/AUX counters; subsequent ordinary updates must inherit the new receipt while learned tensors remain mutable. Required flags include same-shape identity, `same_mdp_claimed=False`, `control_projection_changed=True`, `reward_changed=False`, `HISTORY_changed=False`, `old_rollout_inherited=False`. A CPU model migration test and later physical run are still required after a permitted code-switch boundary. None was run or implemented by this isolated candidate task.
