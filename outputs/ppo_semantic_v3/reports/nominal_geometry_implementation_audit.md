# Nominal geometry — recovery protection and attribution audit

Status: **committed implementation with841 passing CPU tests; actual new-MDP initialization and first128-decision update verified**. The protection inspection began on2026-09-06 at13:06:27Z against9d70 while the implementation was dirty. It was subsequently committed as `4d268fc547b704c590c5f21563ea4b1970b021a1`; its ten-file commit inventory is the six production files and four tests listed below. The new P06 block remains RUNNING; this report is deliberately fixed at saved **52,736/377/7,540**, not a completed2,048-decision block. No task success, successful physical geometry intervention or PPO stability improvement is inferred from this first optimization boundary.

## 1. Checkpoint52,608 remains exactly the old recovery point

PowerShell `Get-FileHash` on the actual checkpoint and manifest matches both previously reported hashes:

| File | Actual SHA-256 |
|---|---|
| `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000052608.pt` |`6956b02aaec6ed6ae69dd7821fcd6c2bffc266b8e3455c5301ee740a24fc8582` |
| `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000052608_manifest.json` |`353a9b329cb53e5d0869489c367e6165f3b8eb2b0f265e7b85be7ea0ba7a4caa` |

The checkpoint SHA also equals the sidecar's bound `checkpoint_sha256`. Its preserved metadata remains:

- `global_policy_decisions=52608`, `ppo_updates=376`, `optimizer_steps=7520`, `save_load_round_trip=true`.
- Source run `runs/ppo_semantic_v3/train/20260906T1211309264074Z_g9d70aae58243_42ace02dcac84aaa83ee7e1ee578f58e`, source runtime commit9d70; original new-MDP origin10,112.
- Current spending full_episode16,896 /phase_suffix25,600 /smoke0.
- Actor-state digest `7b7996a8555baa1289a842b6301020ad30120c773d10f43b507fdffd87a5c3d5`; critic digest `180db1155f101fac765a73b6f3434929786530c33ca3a0b44713808283d22457`; identity-normalizer digest `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`.

Byte identity protects the complete saved payload, not only these selected metadata fields. This audit **did not deserialize the network, rerun save-load, reset Adam/RNG, rewrite the sidecar, or mutate the recovery files**. `round_trip=true` is the unchanged recorded result of the completed run, not a newly performed roundtrip.52,608 remains a real trained checkpoint from the old9d70 execution semantics, not a checkpoint already adapted to the new nominal rule.

## 2. All29 frozen files remain unchanged

Baseline inventory: `artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json`, actual inventory SHA-256 `c36ab3e603b8881c463dd72ec6a84092b892edb0ca62c35d39c1bc6ec26a18e2`.

For **every one of its29 `protected_files` entries**, the current file was hashed and compared with three independent stored bindings: that baseline inventory, checkpoint52,608's `runtime_contract.frozen_A_files`, and this completed P10 run's started-manifest `runtime_contract.frozen_A_files`. All three inventories contain29 entries; mismatches/missing files=0. This is a current content check, not merely a clean-git-status inference.

| Frozen group | Files | Current actual hash matches baseline and both runtime bindings |
|---|---:|---|
| Environment/FSM/motion configuration |4 |4/4 |
| `src/wlr50_clean/fsm` |10 |10/10 |
| `src/wlr50_clean/infrastructure` |7 |7/7 |
| `src/wlr50_clean/sensing` |8 |8/8 |
| **Total** |**29** |**29/29** |

This covers the exact named29-file inventory, including old FSM/controller/guards/motion executor, frozen mapper/RobotAdapter/command batch, scene factory and contact/geometry readers. It is not a claim that every unrelated asset or external runtime binary was rehashed. `semantic_supervisor.py` and the v3 task spec have no working-tree diff against9d70 at this audit point; no old result or fixed completion criterion is being relabeled by the geometry work.

## 3. Scope actually present in the dirty implementation

The execution-profile opt-in is `nominal_geometry_advisory: preplace_world_down_nominal_advisory_v1`. Current changes are four existing PPO/config files plus two new PPO helpers; none is one of the29 frozen files:

| Current production file | SHA-256 of inspected dirty content |
|---|---|
| `configs/ppo_semantic_v3/execution_profile.yaml` |`8020972778b63b64a14e916b059bb43d4f756e1e3510c72d22897ad23cf4ed60` |
| `src/wlr50_clean/ppo/semantic_nominal_projection.py` |`3c75ff06be900cc3eee7cb74bbcf826add2ed8faff557b4c2e6fd2533f5f1c9d` |
| `src/wlr50_clean/ppo/semantic_nominal_geometry.py` |`218db2636e3723df7c3217111f55795e0390ad9bb64a7f9fdde926d177e99e8b` |
| `src/wlr50_clean/ppo/semantic_backend.py` |`01cc554858d1815bb9d5195a2b8087946bbe07ac03190b3358bb2c5c58c7c739` |
| `src/wlr50_clean/ppo/semantic_residual_adapter.py` |`ba9286b281e9684d563d8c2067dd819789f4a194ccd78079588c0dd1ddb2b274` |
| `src/wlr50_clean/ppo/actuator_target_effect.py` |`7ec65a2e425681d2e172caa409ae0735983d451bdf7346fa0cecba63cb268721` |

These are content identities from the initial dirty inspection. The subsequent4d268fc commit inventory contains exactly these six production paths plus the four new CPU-test files. The inspected path is N1: current tensor/DOF checks require a single articulation row; this report does not extend that evidence to vector geometry control.

`capture_nominal_geometry_context` (`semantic_nominal_geometry.py:55`) is active only for RR in P09 or RL in P12. It consumes the same-tick actual observation, physical evaluator and source frame, validates their tick/time agreement, mapped joints and wheel-link geometry, and reads the world-frame PhysX COM Jacobian. It shifts the COM Jacobian to the same wheel link-origin reference used by the existing geometry reader, and checks same-state COM/link velocity identities. These checks validate data/model coordinates; they are not historical joint-entry or task-success gates.

It bypasses when the current wheel is already in place-XY, is grounded, or is outside the lateral span; invalid/terminal task state also bypasses. It does not retain a prior peak airborne pose when a leg returns to ground. The clearance margin is the existing task-spec15mm value, not a tuned new threshold. Its two-DOF prediction treats base coordinates as fixed for this nominal target calculation; real floating-base/contact motion is expressly not guaranteed.

`correct_nominal_geometry` (`:198`) reconstructs the zero-current-policy final nominal via the frozen `bounded_drive_feedback_step`, using the real mapper output, existing controller bias and previous actual final drive target. The target box uses the existing hard joint limits and existing per-tick final slew. It computes physical-radian displacement from **current measured q**, then applies `project_nominal_downward` (`semantic_nominal_projection.py:198`) with available descent `max(0,current_clearance−existing_margin)`.

Only the active rear hip/knee nominal pair may change. The helper has no current policy-residual argument; the residual remains independently added afterward through the original hard/slew dispatch. `semantic_residual_adapter.py:71` retains one mapper advance, followed by the existing position setter, velocity setter and `write_data_to_sim` at119–121. The context/helper performs no extra simulator step, pose reset, force write or mapper advance.

## 4. Why this is not the old knee-sign pause or fixed entry requirement

The new calculation does **not** classify the first decreasing knee command as descent. Its descent estimate is the world vertical derivative `Jz·(q_nominal−q_actual)`. A negative knee increment that continues upward/forward carry is therefore not automatically blocked. It does not wait for a historical servo angle, velocity, motion endpoint or clock time before continuing the phase.

The projection preserves requested world-x motion exactly when feasible; otherwise it chooses the closest feasible non-reversing x value and then the smallest joint-radian change. The other leg, front legs, wheels and residual are not frozen. A mathematically constrained case can reduce one nominal forward component to zero, or require a small forward-magnitude increase, but this is an explicitly recorded same-state target solution—not a paused phase/motion clock or a blanket hold of the whole leg. The solver explicitly reports that no universal no-amplification guarantee exists.

If the hard/slew box and desired descent constraint have no compatible solution, the integration records `degraded_bypass_*` and sends the original bounded nominal rather than silently inventing a feasible pose, freezing a remembered peak or adding a new task failure/reset. **That degraded branch provides no clearance guarantee.** Missing/nonfinite/stale geometry is separately an explicit interface-contract error, not permission to fabricate a target. `physical_motion_guaranteed=false` remains in every geometry proof.

`SemanticIsaacBackend._atomic_apply` (`semantic_backend.py:220`) does not modify the supervisor's phase or nominal-source clock. Teacher mode, bias TAKEOVER, the exact receipt-handoff tick, and reset paths without a semantic plan bypass geometry; READY continuation uses the same B/C route. Frozen A actions and legacy controller private state are not rewritten.

## 5. Attribution: B changes too; current PPO action effect stays separate

The geometry path is entered even when current PPO residual is zero if a valid geometry context exists. Consequently **new B is different from old B** in eligible states. A future better rollout after enabling this helper cannot be called PPO training improvement merely because it uses a PPO checkpoint. This changes nominal execution/effective closed-loop semantics and must be versioned separately from policy-only learning; the old52,608 checkpoint contract remains9d70 until an explicit reviewed migration is recorded elsewhere.

The native audit now distinguishes three same-pre-tick target constructions:

| Branch | Contents | Meaning |
|---|---|---|
| Raw zero-policy nominal |Raw mapper nominal + controller bias |Old nominal counterfactual at this same state |
| Geometry zero-policy nominal |Geometry-adjusted nominal + controller bias |New B target at this same state |
| Actual |Geometry-adjusted nominal + controller bias + current projected PPO residual |Actually dispatched target |

All use the same pre-tick state, frozen bound/conversion logic and actual target dtype. Existing `changed_channels_full12` compares **actual versus geometry zero-policy**, so geometry-only motion is not falsely credited as PPO effect. Added `raw_nominal_native_targets`, `geometry_nominal_native_targets` and `nominal_geometry_changed_channels_full12` separately expose the geometry contribution (`actuator_target_effect.py:197–217`). Raw mapper fields keep their original meaning; `geometry_adjusted_native_full12` and `nominal_geometry_adjustment_full12` are explicit additions.

These are same-tick counterfactual target differences, not separately observed physical trajectories or an additive causal decomposition of long-run stability. Previous PPO actions may already have changed the current measured state; this helper merely avoids consuming the **current** residual to create its nominal adjustment. Future A/B/C reports must label controller/nominal changes separately, preserve matched conditions and actual outcomes, and never convert old incomplete results into new successes. No extra A-success/probe-success prerequisite is created by this reporting requirement.

## 6. Final CPU evidence; live launch is not a physical result

Existing XML receipts were read, not rerun:

| Receipt | Tests | Failures/errors/skipped | Recorded duration |
|---|---:|---|---:|
| `C:/robotics_sim/wlr_robot/nominal_geometry_dispatch_20260906_tickbind.xml` |72 |0/0/0 |5.227s |
| `C:/robotics_sim/wlr_robot/nominal_geometry_partial_cpu.xml` |124 |0/0/0 |5.458s |
| `C:/robotics_sim/wlr_robot/nominal_geometry_real_chain_20260906.xml` |207 |0/0/0 |5.718s |
| `C:/robotics_sim/wlr_robot/semantic_geometry_final_cpu.xml` |841 |0/0/0 |83.416s |

The final841-test suite has now completed; its actual XML attributes were read with PowerShell and confirm0 failures,0 errors,0 skipped and83.416s. These overlapping batches are **not summed as unique tests**. The test code includes negative-knee/upward-forward identity, world-coordinate sign invariance, same-tick geometry validation, teacher/handoff bypass, active-pair-only mutation, original controller-bias envelope, explicit infeasible bypass, real helper→mapper→dispatch→three-branch audit, and zero-policy geometry-only effect separation. Those are CPU/interface proofs; no real body/contact response, physical task outcome, finite-difference live Jacobian check or new video is claimed here.

### Preserved actual P06 launch snapshot — superseded by initialization/update receipt below

The main task launched `runs/ppo_semantic_v3/train/20260906T1308472858273Z_g4d268fc547b7_dcefe730da37448599db863a5f261d47`. Its actual `run_manifest.started.json` is `RUNNING`, command=train, runtime commit4d268fc, v3/N1/seed1001, `phase_suffix`, P06/teacher offset0, **2,048 planned decisions** and checkpoint interval4 updates. It explicitly selects old `checkpoint_step_000052608.pt`, `heteroscedastic_log_v1`, `new_mdp_warm_start=true`, `policy_distribution_migration=false`, `resume_migration=null`.

This is an explicit execution/MDP boundary, not exact same-MDP resume or another Gaussian architecture conversion. The started manifest is not evidence that initialization/restoration has finished, Adam has actually updated, the new Jacobian path is physically valid, or the task succeeded. No new initial-checkpoint preservation receipt, optimizer update or executed allocation is inferred here. The main task may append those facts when actual records exist.

Current completed training therefore remains **52,608/376/7,520** and latest completed natural-P01 evaluation remains **C50,560's P09 incomplete**. No target endpoint is prefilled, and no new A/B/C outcome or successful video is claimed. This subtask used only read-only PowerShell/code inspection and this outputs-report edit; it did not alter production files, recovery checkpoints, baselines, old runs, or start Python/Isaac/commits. The main task owns the separately recorded commit and live launch.

## 7. Actual new-MDP initial preservation and first saved update

The real new initial checkpoint is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000052608_s6956b02aaec6_g4d268fc547b7_64b915174f5f4edd7dde25494fbceb18f907c1979a0731c0e46b1beb6047a9fe.pt`. Its sidecar has the same stem plus `_manifest.json`. PowerShell measured checkpoint SHA-256 **`3f0854ff48013b52ca8572a0fe5446f9bb8adf6334f8f10b3af4e5c3e9ae1310`**, equal to the sidecar; `save_load_round_trip=true`.

Direct source/initial sidecar comparison verifies:

| State | Actual new initial evidence |
|---|---|
| Actor, including all learned heteroscedastic mean/log-std parameters |Same actor digest `7b7996a8555baa1289a842b6301020ad30120c773d10f43b507fdffd87a5c3d5` |
| Critic |Same digest `180db1155f101fac765a73b6f3434929786530c33ca3a0b44713808283d22457` |
| Identity normalizer |Same digest `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` |
| Stored training RNG |Whole `training_rng_state` objects compare equal, not only seed1001 |
| Lifetime counters |52,608 decisions /376 PPO updates /7,520 optimizer steps, unchanged at initialization |
| Origin/spending |Origin10,112; full_episode16,896 /phase_suffix25,600 /smoke0, unchanged |
| Architecture |`heteroscedastic_log_v1`,324 observations/12 actions; no Gaussian conversion |
| Adam |Source digest `5e8b4906a15b55620bc033809af5a9933fd80142cd8c495a4d9bbfaf60708137` → fresh digest `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`; all moments reset |
| Initial optimizer LR |Source effective1e-5 → new initial3e-5 |
| Old rollout / old physical state |Both explicitly not inherited; fresh new-MDP rollout after legal reset |

The migration record names source checkpoint SHA6956b02a…8582 and source manifest SHA353a9b32…4caa, `exact_mdp_resume=false`, preservation of all network parameters, source RNG restoration, retained spent budgets and excluded teacher roll-in credit. This is a genuine declared execution/MDP transition; it does not silently rewrite the old52,608 checkpoint or call changed B nominal behavior PPO learning.

The actual first new optimizer record is **update377/global52,736**,128 new decisions /20 optimizer steps. Actor digest changes from the preserved source digest to `2f3752cbf394c64324b819c1966cf502bf8ff242547ce09bfdb9e27d9e72fc2c`; `actor_parameters_changed=true`, `finite_nonzero_gradient_observed=true`, observed gradient norm range1.002740108–1.366367730. The record's **actual effective LR is1e-5**, distinct from initial3e-5; this report does not assert a constant3e-5 throughout training or infer an unobserved reason for the update's effective LR.

Actual immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000052736.pt` exists with `save_load_round_trip=true`; measured SHA-256 **`3ef19a21a812663373a517dbf3aced46bdc6d1738f3f2e2c6701035ad6609617`** matches its manifest. It binds the new4d268fc run and52,736/377/7,540, with spending full_episode16,896 /phase_suffix25,728 /smoke0. This fixes actual optimized/saved credit at+128/+1/+20; the2,048-request block is not yet finalized in this report. Later progress, task outcomes and geometry response are intentionally left uncounted pending separate reporting authorization.
