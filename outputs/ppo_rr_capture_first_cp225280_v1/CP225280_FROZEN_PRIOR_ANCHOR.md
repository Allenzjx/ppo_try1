# CP225280 frozen-prior anchor inventory (read-only)

Scope: only the named CP225280/CP227328 reports, their small manifests, the
branch-local pointer, and the already-produced successful-N reference report
were read. No model, Torch/PXR, Isaac, Recording tree, or growing log was
opened. Repository HEAD was clean at
`2bdbc58a33102153771b2b4069e00faa52cae819`; no `AGENTS.md` exists under
`C:\robotics_sim`.

## Three distinct policy artifacts

| Artifact | Immutable checkpoint / manifest | Runtime and policy interface | Credit and compatibility boundary |
|---|---|---|---|
| Accepted CP225280 video model | `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000225280.pt`; CP SHA `21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf`; manifest SHA `6e05b4e279c1d8ac99447f91d69b7db3f6dd6afaab6a326e610283f783b1f185` | HEAD `49eb23163a6e20bc56301dbafb59b137ecebce66`; `rear_cooperative_prep_history_v1`; `role419_p02_progress_v1`, 422 observations, 12 raw actions. Actor SHA `c57185f9…bec51`; Identity normalizer SHA `c230b0db…4552`. | Lifetime `225280/1725/34500`. This is the accepted deterministic physical anchor, not an RR-placement success. It uses the original 422 observation and original 49eb action transform. |
| Zero-learning 439 compatibility migration | `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_rear_owner_CP225280_gf6d1d2df8d87.pt`; CP SHA `fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f`; manifest SHA `f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d` | HEAD `f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b`; `rear_owner_recovery_history_v1`; `role422_rear_owner_recovery_v1`, 439 observations. Old 422 first-layer columns and optimizer state are declared preserved; 17 new columns/moments are zero. | Same `225280/1725/34500`; added decisions/PPO/Adam/AUX all zero; round-trip true. It is a compatibility bridge, **not the identical CP225280 controller**: issued-owner suspension, RL-edge recovery and versioned task proxies were added. Actor hash therefore differs (`48d70c95…427e7`). |
| Evaluated CP227328 descendant | `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/cp225280_front_preserved_v1/checkpoints/history/checkpoint_step_000227328.pt`; CP SHA `5b871af2df8d6b9c0180f7a51863b18b2153148d8dc2a94876f21f5bd73791b4`; manifest SHA `0200daa9a22652e1f1c64e5c5ff82a0c977f0418c508158455b3a5ce85bac308` | HEAD `892385cba8a7089b52567bb7f558c018fac82a77`; same 439 owner policy/layout as f6d, but actor SHA `02cdc3e6…cc14` after real PPO. | Lifetime `227328/1729/34580`; independent branch added `2048/4/80`. Front replay contributed 2560 row exposures in 80 normal PPO minibatches, zero extra PPO samples and zero separate AUX. It is not a frozen copy of either prior artifact. |

At inspection time the **branch-local**
`branches/cp225280_front_preserved_v1/checkpoints/checkpoint_last_pointer.json`
points to the explicit CP227328 SHA above. This is recorded only as current
pointer state; it is not treated as a repository-wide latest/best model, and
all future bindings should use the explicit checkpoint and manifest hashes.

## Accepted CP225280 deterministic opportunity window

Source:
`runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T0214316632402Z_g49eb23163a6e_530cc3c96d834dc3ba84b0b19d8db4f1/source`
(source-manifest SHA
`6414be0c81b091effc2b07fbc8287d0b4e4866d18a4d83ac4b095e3f7bc69310`).

| Event | Physical tick / time | Meaning for a frozen-prior gate |
|---|---:|---|
| P07 entry | `6736 / 56.133333 s` | Rear-task window begins; not yet a capture opportunity. |
| P09 entry / carry | `6752 / 56.266667 s` | Original policy continues as the accepted prior. |
| RR qualified | `6995 / 58.291667 s` | Real lift is observed; still not crossed. |
| RR crossed | `7979 / 66.491667 s` | First physical crossed event. Use the live event plus legal XY, not a fixed 66.49 s timer, as any new local-branch gate. |
| Episode end | `10940 / 91.166667 s` | RR never TOP/placed; terminal AIR gap `56.118904 mm`, force `0 N`. |

The source manifest binds deterministic conditional-mean evaluation, FL assist
`ON` with mode `p05_hip_only_continuation_v1` and
`is_policy_learning=false`; rear task assists and rear wheel projection are
`OFF`. Policy version is `rear_cooperative_prep_history_v1`, observation
dimension 422. This is the control/observation anchor the user accepted.

### What f6d does and does not preserve before activation

- On the same old 422 numerical input, the migration declares exact initial
  mean/sigma from the preserved old columns; the appended 17 policy columns
  and their Adam moments are zero. This is useful evidence for the frozen
  network function.
- The f6d controller is not byte- or semantics-identical to 49eb. Its owner
  projector can replace FL/RL servo candidates (indices 0,1,4,5) with an
  entry-FINAL anchor plus bounded policy-request change when
  `winning_late_owner && live && !support_transfer_permitted`; this condition
  does **not** require RR crossing. Wheels, explicit stops, nominal-provider
  output and RR hip/knee are not changed by this projector.
- In the accepted CP225280 run, P09 source stayed at cursor 648/status
  `holding`, `late_group_start_tick=null`; therefore the winning late owner did
  not start in that recorded pre-cross trajectory and the f6d owner projector
  would be inactive there. That single run does not prove it stays inactive in
  every new closed loop.
- Consequently, zeroing a new RR branch is not by itself a hard pre-cross
  guarantee if the whole f6d controller remains live. A composite package must
  either retain the original 49eb pre-activation action transform or explicitly
  keep the f6d owner projector inert until the versioned RR-cross + legal-XY
  activation condition.

## Current-source compatibility of the isolated 49eb mode

The six copied control/physics configuration files in
`configs/ppo_rr_capture_first_cp225280_v1` have the exact Git-blob hashes of
their 49eb counterparts: action, execution, observation, quality, reward and
stage/task. With those values, current source selects the retained v3 rear
timing and v4 cooperative-preparation branches. The v5 proxy path, edge-v4
timing path, owner projector and reward-only retention term are all opt-in and
remain inactive here. Current observation/policy-distribution additions are
also version-gated, so the original 422 layout remains a supported path.

This is a static dispatch/configuration result, not a closed-loop bitwise
replay claim. The isolated directory intentionally adds `local_training.json`
but currently does not contain the original `curriculum_plan.json`; moreover,
the copied execution profile still names the old namespace's curriculum path.
A custom composite route must bind that difference explicitly (and own its
512-sample collection contract) rather than presenting the directory as the
unchanged generic seven-file 49eb configuration.

## Successful N reference: reusable evidence, not a compatible policy prior

Source:
`runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source`
(manifest SHA
`03549f2590759fe22c09ec83ec12282f17f06de722a15ece547873140a083a20`).
It is git `ee5a9651591d20bea48be8ceba075fa66594cb36`, role B,
`nominal_without_learned_residual`, N+0, zero optimizer updates, and a real
73.808333 s task success. It is a different runtime/controller, not a
same-policy teacher or comparison anchor.

The existing read-only reference script/report cites these continuous source
logs directly (no Recording rescan):

- `stage_transition_evidence.jsonl` — event history;
- `video_policy_decisions.jsonl` — nominal/final/action and task evidence;
- `physical_observations.jsonl` — 120 Hz commanded/actual/contact response;
- `height_diagnostics.jsonl` — recorded RR mount/body/CoM evidence;
- `semantic_video_source_manifest.json` — provenance and outcome.

Known RR contact window: lift `5434/45.283333 s`; first recorded legal-XY
qualified endpoint `6112/50.933333 s` already has TOP contact `4.097 N`; RR
crossed and placed at `6155/51.291667 s`. The inspected 15 Hz window has no
pre-placement `AIR + legal XY` endpoint (the first such AIR row is 6272, after
placement). Therefore the 6112–6160 source rows are appropriate for studying
continuous joint/wheel/load response, but must not be relabelled as a direct
CP225280 high-gap AIR entry or on-policy target. The already-produced reference
is `outputs/ppo_rr_rl_timing_policy_learning_v1/successful_N_RR_RL_physical_reference.{md,json}`.

## CP227328 comparison boundary

Its sealed deterministic source is
`runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1842131087811Z_g892385cba8a7_c074d1ebec6947cc830fb3e0ec86dcec/source`
(source-manifest SHA
`4cab08446b22e19af76866cfc4d7a80cde4ea8d57ead29ea9f3677437c160ead`).
It restored a front completion (`FL placed 5232/43.600 s`) and reached RR
qualified `7143`, crossed `8230`, but never RR TOP/placed; it ended
`11092/92.433333 s`, AIR/0 N. FL assist remained the same declared non-learning
mode and rear task assists remained off. This is evidence about the trained
439 descendant, not evidence that CP225280, f6d and CP227328 are identical.
